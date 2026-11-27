# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Drained, in-place checkpoint update for the owned NPU V2 runtime."""

from collections.abc import Callable, Iterable
from contextlib import suppress
from copy import copy
from typing import Any

import torch


def invalidate_owned_graphs(runner: Any) -> None:
    """Invalidate captures after their device consumers have been fenced."""
    registry = runner._owned_graph_registry
    if registry is not None:
        # Retain the graph owner if close raises; never advertise successful
        # invalidation of resources that may still be live.
        registry.close()
        runner._owned_graph_registry = None
    runner._graph_generation += 1


def _require_drained(runner: Any) -> None:
    if (
        runner.execute_model_state is not None
        or runner._decode_advance is not None
        or runner.request_state.requests
    ):
        raise RuntimeError("Pause and drain owned NPU V2 requests before updating weights")
    owner = getattr(runner, "_gr_npu_stage_runner", None)
    if owner is not None and (not owner.closed or getattr(owner, "pending_plan", None) is not None):
        raise RuntimeError("Retire the active GR Beam owner before updating weights")
    for attribute in ("_beam_context", "_gr_npu_eager_context", "_gr_npu_graph_context"):
        context = getattr(runner, attribute, None)
        if context is not None and context.registry.num_active_sessions:
            raise RuntimeError("Release all GR Beam sessions before updating weights")


def _bindings(model: torch.nn.Module) -> dict[tuple[str, str], tuple[Any, ...]]:
    """Record storage layout and aliases of every live parameter and buffer."""
    bindings = {}
    for kind, tensors in (
        ("parameter", model.named_parameters(remove_duplicate=False)),
        ("buffer", model.named_buffers(remove_duplicate=False)),
    ):
        for name, tensor in tensors:
            bindings[(kind, name)] = (
                tensor.device,
                tensor.dtype,
                tuple(tensor.shape),
                tuple(tensor.stride()),
                tensor.storage_offset(),
                tensor.data_ptr(),
            )
    return bindings


def run_weight_update(
    runner: Any,
    write_weights: Callable[[torch.nn.Module], None],
    *,
    synchronize: Callable[[], None],
    commit: Callable[[], None] | None = None,
) -> None:
    """One exclusive transaction, with no rollback claim after native writes.

    The engine's paused/drained Core protocol must prevent new admission while
    this RPC runs. The runner additionally proves its own state has drained.
    No asynchronous execution is permitted to use the model during reload.
    """
    lock = runner._weight_update_lock
    if not lock.acquire(blocking=False):
        raise RuntimeError("Owned NPU V2 weight update is already in progress")
    try:
        model = runner.get_model()
        _require_drained(runner)
        before = _bindings(model)
        projection = getattr(runner, "_gr_npu_stage_projection", None)
        if projection is not None and projection.closed:
            raise RuntimeError("Owned NPU V2 stage projection was closed before reload")
        runner._weight_update_active = True
        try:
            synchronize()
            invalidate_owned_graphs(runner)
            if projection is not None:
                projection.begin_weight_update()
            write_weights(model)
            if _bindings(model) != before:
                raise RuntimeError("Owned NPU V2 in-place reload changed model storage or layout")
            if projection is not None:
                projection.finish_weight_update()
                if not projection.weights_valid:
                    raise RuntimeError("Owned NPU V2 stage projection refresh did not complete")
            synchronize()
            if commit is not None:
                commit()
        except BaseException as error:
            # Native layerwise reload may already have written parameters
            # or rebuilt temporary layer state. A fresh runner is required.
            runner._failed = error
            if projection is not None:
                with suppress(Exception):
                    projection.fail_weight_update()
            raise
        finally:
            runner._weight_update_active = False
    finally:
        lock.release()


@torch.inference_mode()
def reload_owned_checkpoint(
    runner: Any,
    weights_iterator: Iterable[tuple[str, torch.Tensor]] | None = None,
    weights_path: str | None = None,
    is_checkpoint_format: bool = True,
) -> None:
    """Use vLLM 0.30's model loader and layerwise reload contract directly."""
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.model_executor.model_loader.reload import (
        finalize_layerwise_reload,
        initialize_layerwise_reload,
    )

    if not is_checkpoint_format:
        raise ValueError("Owned NPU V2 only supports checkpoint-format weight reload")
    if weights_path is not None and (not weights_path.strip() or weights_iterator is not None):
        raise ValueError("Supply one checkpoint path or one weights iterator")

    # Prepare a new source config without changing the live version on
    # failure. The new checkpoint retains the original model architecture.
    reload_config = copy(runner.model_config)
    if weights_path is not None:
        reload_config.model = weights_path
        reload_config.revision = None
    loader = (
        None if weights_iterator is not None else get_model_loader(runner.vllm_config.load_config)
    )
    if loader is not None and not callable(getattr(loader, "get_all_weights", None)):
        raise ValueError("Native loader does not expose checkpoint weight iteration")

    def write(model: torch.nn.Module) -> None:
        expected = {name for name, _ in model.named_parameters()}
        load = getattr(model, "load_weights", None)
        if not callable(load):
            raise RuntimeError("Loaded model does not support native checkpoint reload")
        source = weights_iterator
        if source is None:
            assert loader is not None
            source = loader.get_all_weights(reload_config, model)
        try:
            with set_current_vllm_config(runner.vllm_config):
                initialize_layerwise_reload(model)
                loaded = load(source)
                finalize_layerwise_reload(model, reload_config)
        finally:
            # The loader-created iterator owns its checkpoint file handles;
            # an iterator supplied by the caller retains caller ownership.
            close = getattr(source, "close", None)
            if weights_iterator is None and callable(close):
                close()
        # For the admitted dense unquantized models, require proof that
        # every unique parameter was loaded. Tied weights appear once in
        # named_parameters(). Native warning-only reload is insufficient.
        if not isinstance(loaded, set):
            raise RuntimeError("Checkpoint loader did not report loaded parameter names")
        missing = expected - loaded
        if missing:
            raise RuntimeError(f"Checkpoint is missing required parameters: {sorted(missing)}")

    def commit() -> None:
        runner.model_config.model = reload_config.model
        runner.model_config.revision = reload_config.revision

    run_weight_update(
        runner,
        write,
        synchronize=lambda: torch.npu.synchronize(runner.device),
        commit=commit,
    )
