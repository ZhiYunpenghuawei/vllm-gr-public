# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Host contracts for exclusive owned NPU V2 checkpoint updates."""

from contextlib import nullcontext
from threading import Lock
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.v1.worker.npu.v2.lifecycle import reload_owned_checkpoint, run_weight_update
from vllm_gr.v1.worker.npu.v2.model_runner import NPUModelRunnerV2
from vllm_gr.v1.worker.npu.v2.request_state import NativeRequestState

pytestmark = [pytest.mark.l1, pytest.mark.cpu_test]


class _Projection:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.closed = False
        self.weights_valid = True

    def begin_weight_update(self) -> None:
        self.calls.append("projection-invalid")
        self.weights_valid = False

    def finish_weight_update(self) -> None:
        self.calls.append("projection-refresh")
        self.weights_valid = True

    def fail_weight_update(self) -> None:
        self.calls.append("projection-failed")
        self.weights_valid = False


def _runner(calls: list[str]) -> Any:
    # The production constructor requires a device and native model config.
    # These tests isolate the transaction protocol using a CPU model owner.
    runner = object.__new__(NPUModelRunnerV2)
    runner.model = torch.nn.Linear(2, 2, bias=False)
    runner._closed = False
    runner._failed = None
    runner._weight_update_lock = Lock()
    runner._weight_update_active = False
    runner._owned_graph_registry = SimpleNamespace(close=lambda: calls.append("graph-close"))
    runner._graph_generation = 3
    runner.request_state = NativeRequestState(1, 8)
    runner.execute_model_state = None
    runner._decode_advance = None
    runner._gr_npu_stage_projection = _Projection(calls)
    return runner


def test_checkpoint_update_orders_fence_invalidation_and_commit() -> None:
    calls: list[str] = []
    runner = _runner(calls)
    address = runner.model.weight.data_ptr()

    def write(model: torch.nn.Module) -> None:
        assert runner._weight_update_active
        with pytest.raises(RuntimeError, match="update is in progress"):
            runner.get_model()
        assert not runner._gr_npu_stage_projection.weights_valid
        calls.append("write")
        with torch.no_grad():
            model.weight.fill_(3)

    run_weight_update(
        runner,
        write,
        synchronize=lambda: calls.append("sync"),
        commit=lambda: calls.append("commit"),
    )
    assert calls == [
        "sync",
        "graph-close",
        "projection-invalid",
        "write",
        "projection-refresh",
        "sync",
        "commit",
    ]
    assert runner.model.weight.data_ptr() == address
    assert runner.model.weight.tolist() == [[3, 3], [3, 3]]
    assert runner._owned_graph_registry is None and runner._graph_generation == 4
    assert runner._failed is None and not runner._weight_update_active


@pytest.mark.parametrize("kind", ["request", "pending", "decode", "owner", "session"])
def test_checkpoint_update_rejects_undrained_state_before_any_write(kind: str) -> None:
    calls: list[str] = []
    runner = _runner(calls)
    if kind == "request":
        runner.request_state.requests["req"] = object()
    elif kind == "pending":
        runner.execute_model_state = object()
    elif kind == "decode":
        runner._decode_advance = object()
    elif kind == "owner":
        runner._gr_npu_stage_runner = SimpleNamespace(closed=False)
    else:
        runner._beam_context = SimpleNamespace(registry=SimpleNamespace(num_active_sessions=1))
    with pytest.raises(RuntimeError, match="before updating weights"):
        run_weight_update(
            runner, lambda model: calls.append("write"), synchronize=lambda: calls.append("sync")
        )
    assert calls == []
    assert runner._failed is None and not runner._weight_update_active
    assert runner._owned_graph_registry is not None


def test_partial_native_write_latches_failure_and_blocks_inference() -> None:
    calls: list[str] = []
    runner = _runner(calls)

    def write(model: torch.nn.Module) -> None:
        with torch.no_grad():
            model.weight[0].fill_(5)
        raise RuntimeError("partial write")

    with pytest.raises(RuntimeError, match="partial write"):
        run_weight_update(
            runner,
            write,
            synchronize=lambda: calls.append("sync"),
            commit=lambda: calls.append("commit"),
        )
    assert "commit" not in calls and "projection-refresh" not in calls
    assert calls[-1] == "projection-failed"
    assert runner._failed is not None and not runner._gr_npu_stage_projection.weights_valid
    with pytest.raises(RuntimeError, match="cannot continue"):
        runner.get_model()


def test_duplicate_weight_update_is_rejected_before_fencing() -> None:
    calls: list[str] = []
    runner = _runner(calls)
    with runner._weight_update_lock:
        with pytest.raises(RuntimeError, match="already in progress"):
            run_weight_update(
                runner,
                lambda model: calls.append("write"),
                synchronize=lambda: calls.append("sync"),
            )
    assert calls == []
    assert runner._failed is None


def test_pointer_change_or_refresh_failure_cannot_commit() -> None:
    for replace_pointer in (True, False):
        calls: list[str] = []
        runner = _runner(calls)

        def write(model: torch.nn.Module) -> None:
            if replace_pointer:
                model.weight = torch.nn.Parameter(model.weight.detach().clone())

        if not replace_pointer:
            runner._gr_npu_stage_projection.finish_weight_update = lambda: None
        with pytest.raises(RuntimeError, match="storage or layout|refresh did not complete"):
            run_weight_update(
                runner,
                write,
                synchronize=lambda: calls.append("sync"),
                commit=lambda: calls.append("commit"),
            )
        assert "commit" not in calls
        assert runner._failed is not None


def test_graph_close_failure_retains_owner_and_prevents_weight_write() -> None:
    calls: list[str] = []
    runner = _runner(calls)

    def fail() -> None:
        raise RuntimeError("graph still live")

    registry = SimpleNamespace(close=fail)
    runner._owned_graph_registry = registry
    with pytest.raises(RuntimeError, match="graph still live"):
        run_weight_update(
            runner, lambda model: calls.append("write"), synchronize=lambda: calls.append("sync")
        )
    assert "write" not in calls
    assert runner._owned_graph_registry is registry and runner._graph_generation == 3
    assert runner._failed is not None


def test_native_layerwise_api_and_checkpoint_identity_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.config as config
    import vllm.model_executor.model_loader as loader
    import vllm.model_executor.model_loader.reload as native_reload

    calls: list[str] = []
    runner = _runner(calls)
    runner.device = torch.device("cpu")
    runner.model_config = SimpleNamespace(model="old", revision="old-revision")
    runner.vllm_config = SimpleNamespace(load_config=object(), model_config=runner.model_config)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(synchronize=lambda device: calls.append("sync")),
        raising=False,
    )
    monkeypatch.setattr(config, "set_current_vllm_config", lambda cfg: nullcontext())

    def get_weights(model_config: Any, model: torch.nn.Module) -> Any:
        assert model_config.model == "new" and model_config.revision is None
        assert runner.model_config.model == "old"
        calls.append("native-source")
        return [("weight", torch.ones_like(model.weight))]

    def load(weights: Any) -> set[str]:
        calls.append("native-load")
        for name, weight in weights:
            assert name == "weight"
            runner.model.weight.copy_(weight)
        return {"weight"}

    runner.model.load_weights = load
    monkeypatch.setattr(
        loader, "get_model_loader", lambda cfg: SimpleNamespace(get_all_weights=get_weights)
    )
    monkeypatch.setattr(
        native_reload, "initialize_layerwise_reload", lambda model: calls.append("native-init")
    )
    monkeypatch.setattr(
        native_reload,
        "finalize_layerwise_reload",
        lambda model, cfg: calls.append("native-finalize"),
    )
    reload_owned_checkpoint(runner, weights_path="new")
    assert calls == [
        "sync",
        "graph-close",
        "projection-invalid",
        "native-source",
        "native-init",
        "native-load",
        "native-finalize",
        "projection-refresh",
        "sync",
    ]
    assert runner.model_config.model == "new" and runner.model_config.revision is None
    # Missing required weights must not mark another version as committed.
    runner.model.load_weights = lambda weights: set()
    with pytest.raises(RuntimeError, match="missing required parameters"):
        reload_owned_checkpoint(runner, weights_iterator=[], weights_path=None)
    assert runner.model_config.model == "new" and runner._failed is not None
