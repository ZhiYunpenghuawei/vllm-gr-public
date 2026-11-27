# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Startup-only capture provisioning and replay dispatch for owned NPU V2."""

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm_gr.v1.worker.npu.owned_graph import OwnedNPUGraphRegistry
from vllm_gr.v1.worker.npu.owned_graph_inputs import (
    ExactGraphCall,
    make_beam_graph_call,
    runner_graph_resources,
)


@dataclass(frozen=True)
class PreparedGraphPlan:
    phase: str
    build: Callable[[Any, Any, Mapping[str, torch.Tensor]], ExactGraphCall | None]


def _beam_call(
    runner: Any, prepared: Any, resources: Mapping[str, torch.Tensor]
) -> ExactGraphCall | None:
    if prepared.beam_info is None:
        return None
    from vllm_gr.v1.attention.backends.beam_attn import _get_npu_decode_execution

    return make_beam_graph_call(
        prepared,
        generation=runner._graph_generation,
        resources=resources,
        device=runner.device,
        xllm=_get_npu_decode_execution() == "xllm",
    )


BEAM_DECODE_CAPTURE = PreparedGraphPlan("beam_decode", _beam_call)


def validate_owned_graph_requirements(gr_config: Any) -> None:
    """Block advertised graph modes until their real startup/device gates pass.

    Exact-metadata capture support is narrower than the existing GR padded
    Prefill and dynamic Beam graph contracts. The runtime API below is usable
    by a qualification driver with explicit startup KV and memory reservation.
    """
    if gr_config.beam_graph_active:
        raise ValueError(
            "Owned NPU V2 Beam graph capture is not qualified; set gr.beam_graph_enabled=false"
        )
    if gr_config.deployment.prefill_graph_enabled:
        raise ValueError(
            "Owned NPU V2 Prefill graph capture is not qualified; set gr.prefill_graph_enabled=false"
        )


def reserve_owned_graph_memory(runner: Any, available_bytes: int, requested_bytes: int) -> int:
    """Subtract an independently chosen graph budget before final KV sizing."""
    if type(available_bytes) is not int or type(requested_bytes) is not int:
        raise TypeError("Graph and KV budgets must be integer bytes")
    if requested_bytes < 0 or available_bytes <= requested_bytes:
        raise ValueError("Owned NPU graph reservation leaves insufficient KV memory")
    if getattr(runner, "_owned_graph_registry", None) is not None:
        raise RuntimeError("Graph memory cannot be re-reserved while captures exist")
    runner._owned_graph_memory_budget_bytes = requested_bytes
    return available_bytes - requested_bytes


@dataclass(frozen=True)
class _MemorySnapshot:
    free: int
    allocated: int
    reserved: int

    def growth(self, after: "_MemorySnapshot") -> int:
        # Include allocator growth even when capture consumes previously
        # reserved blocks; include device-free loss for non-allocator memory.
        return max(
            0,
            self.free - after.free,
            after.allocated - self.allocated,
            after.reserved - self.reserved,
        )


def _memory_snapshot(device: torch.device) -> _MemorySnapshot:
    with torch.npu.device(device):
        torch.npu.synchronize(device)
        free, _ = torch.npu.mem_get_info()
        return _MemorySnapshot(
            int(free),
            int(torch.npu.memory_allocated(device)),
            int(torch.npu.memory_reserved(device)),
        )


@dataclass
class _CaptureSession:
    registry: OwnedNPUGraphRegistry
    budget_bytes: int
    baseline: _MemorySnapshot
    plans: dict[str, PreparedGraphPlan] = field(default_factory=dict)


@contextmanager
def owned_graph_capture_session(runner: Any, *, max_entries: int) -> Iterator[None]:
    """Explicit startup reservation; the Worker guarantees no admitted work.

    The budget must have been reserved before allocating final KV storage.
    No request-serving path opens this context or captures on a graph miss.
    """
    if (
        getattr(runner, "_owned_graph_registry", None) is not None
        or getattr(runner, "_owned_graph_capture_session", None) is not None
    ):
        raise RuntimeError("Owned graph capture session already exists")
    if runner.model is None or not runner.kv_caches:
        raise RuntimeError("Graph capture requires finalized model and KV storage")
    budget = getattr(runner, "_owned_graph_memory_budget_bytes", 0)
    if type(budget) is not int or budget <= 0:
        raise RuntimeError("Reserve an explicit graph memory budget before KV allocation")
    baseline = _memory_snapshot(runner.device)
    if baseline.free < budget:
        raise RuntimeError("Reserved NPU graph memory is no longer available")
    registry = OwnedNPUGraphRegistry(
        runner.device, generation=runner._graph_generation, max_entries=max_entries
    )
    session = _CaptureSession(registry, budget, baseline)
    runner._owned_graph_registry = registry
    runner._owned_graph_capture_session = session
    runner._owned_graph_capture_plans = ()
    try:
        yield
        if not session.plans or registry.capture_count == 0:
            raise RuntimeError("Owned graph startup captured no usable shape")
        runner._owned_graph_capture_plans = tuple(session.plans.values())
    except BaseException:
        # Keep references if synchronization/close fails. The Worker must
        # remain failed and release them only after safe process teardown.
        registry.close()
        runner._owned_graph_registry = None
        raise
    finally:
        runner._owned_graph_capture_session = None


def capture_owned_prepared(
    runner: Any,
    prepared: Any,
    eager: Callable[[Any], torch.Tensor],
    *,
    reserved: PreparedGraphPlan,
) -> bool:
    """Capture only inside the explicit startup session using reserved KV slots."""
    session = getattr(runner, "_owned_graph_capture_session", None)
    if session is None:
        raise RuntimeError("Owned graph capture requires an active startup reservation")
    if session.registry.generation != runner._graph_generation:
        raise RuntimeError("Stale owned graph capture generation")
    previous = session.plans.get(reserved.phase)
    if previous is not None and previous != reserved:
        raise ValueError("Conflicting owned graph phase builders")
    call = reserved.build(runner, prepared, runner_graph_resources(runner))
    if call is None:
        return False
    if call.key.phase != reserved.phase or call.key.generation != runner._graph_generation:
        raise RuntimeError("Owned graph builder returned an invalid dispatch identity")
    if not call.capture(session.registry, eager):
        return False
    consumed = session.baseline.growth(_memory_snapshot(runner.device))
    if consumed > session.budget_bytes:
        raise RuntimeError(
            f"Owned graph capture exceeded reserved bytes: {consumed} > {session.budget_bytes}"
        )
    session.plans[reserved.phase] = reserved
    return True


def forward_with_owned_graphs(
    runner: Any, prepared: Any, eager: Callable[[Any], torch.Tensor]
) -> torch.Tensor:
    """Replay complete matching startup graphs; misses use the checked eager path."""
    registry = getattr(runner, "_owned_graph_registry", None)
    if registry is None:
        return eager(prepared)
    registry.assert_usable()
    if getattr(runner, "_owned_graph_capture_session", None) is not None:
        raise RuntimeError("Serving dispatch is unavailable during owned graph capture")
    if registry.generation != runner._graph_generation:
        raise RuntimeError("Stale owned graph registry generation")
    plans = getattr(runner, "_owned_graph_capture_plans", ())
    if not plans:
        raise RuntimeError("Owned graph registry has no completed startup provisioning")
    resources = runner_graph_resources(runner)
    for plan in plans:
        call = plan.build(runner, prepared, resources)
        if call is None:
            continue
        output = call.try_replay(registry)
        if output is not None:
            return output
    runner._owned_graph_forward_miss_count = (
        getattr(runner, "_owned_graph_forward_miss_count", 0) + 1
    )
    return eager(prepared)
