# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Host lifecycle tests for owned graph dispatch, without pretending to run NPU."""

from collections.abc import Mapping
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_gr.v1.worker.npu.owned_graph import GraphKey, OwnedNPUGraphRegistry

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


class _GraphRuntime:
    """Models runtime ownership and writes into captured output addresses."""

    def __init__(self):
        self.active_graph = None
        self.replays = 0
        self.fail_replay = False
        self.fail_sync = False
        self.reserved = 0

    def pool_handle(self):
        return object()

    def memory_reserved(self):
        return self.reserved

    def new_graph(self):
        return {}

    @contextmanager
    def capture(self, graph, pool):
        assert pool is not None
        self.active_graph = graph
        self.reserved += 100
        try:
            yield
        finally:
            self.active_graph = None

    def synchronize(self):
        if self.fail_sync:
            raise RuntimeError("sync failed")

    def replay(self, graph):
        if self.fail_replay:
            raise RuntimeError("replay failed")
        self.replays += 1
        graph["output"].copy_(graph["compute"]())

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        # Simulate a device graph returning the same output addresses.
        output = inputs["tokens"] * 2
        if self.active_graph is not None:
            self.active_graph.update(output=output, compute=lambda: inputs["tokens"] * 2)
        return output


def _new_registry(runtime: Any, *, capacity: int = 2) -> OwnedNPUGraphRegistry:
    return OwnedNPUGraphRegistry(
        torch.device("cpu"), generation=1, max_entries=capacity, runtime=runtime
    )


def test_replay_uses_new_input_values_without_overwriting_old_output() -> None:
    runtime = _GraphRuntime()
    registry = _new_registry(runtime)
    key = GraphKey("beam_decode", 1, (2, 128), ("fixed",))
    inputs = {"tokens": torch.tensor([1, 2])}
    resources = {"kv": torch.empty(2)}
    assert registry.capture(key, inputs, resources, runtime.forward)
    assert registry.capture_count == 1

    first = registry.try_replay(key, {"tokens": torch.tensor([3, 4])}, resources)
    assert first is not None
    second = registry.try_replay(key, {"tokens": torch.tensor([5, 6])}, resources)
    assert second is not None
    assert first.tolist() == [6, 8]
    assert second.tolist() == [10, 12]
    assert registry.replay_count == runtime.replays == 2
    assert registry.graph_reserved_bytes_observed == 100


def test_changed_geometry_bindings_and_generation_miss() -> None:
    runtime = _GraphRuntime()
    registry = _new_registry(runtime, capacity=1)
    key = GraphKey("beam_decode", 1, (1, 2), ("fixed",))
    inputs = {"tokens": torch.ones(2)}
    resources = {"kv": torch.empty(4)}
    assert registry.capture(key, inputs, resources, runtime.forward)
    assert not registry.capture(
        GraphKey("beam_decode", 1, (1, 3)), inputs, resources, runtime.forward
    )
    assert registry.try_replay(GraphKey("beam_decode", 2, (1, 2)), inputs, resources) is None
    assert (
        registry.try_replay(GraphKey("beam_decode", 1, (1, 2), ("new",)), inputs, resources) is None
    )
    assert registry.try_replay(key, {"tokens": torch.ones(3)}, resources) is None
    assert (
        registry.try_replay(key, {"tokens": torch.ones(2, dtype=torch.float16)}, resources) is None
    )
    assert registry.try_replay(key, inputs, {"kv": torch.empty(4)}) is None
    assert registry.miss_count == 5
    assert runtime.replays == 0


def test_repeated_input_alias_is_preserved_and_distinct_views_decline() -> None:
    runtime = _GraphRuntime()
    registry = _new_registry(runtime)
    key = GraphKey("beam_decode", 1, (1,))
    value = torch.ones(2)
    assert not registry.capture(key, {"tokens": value, "other": value[:1]}, {}, runtime.forward)
    assert registry.capture(key, {"tokens": value, "other": value}, {}, runtime.forward)
    assert registry.try_replay(key, {"tokens": value, "other": value.clone()}, {}) is None
    assert registry.try_replay(key, {"tokens": value, "other": value}, {}) is not None


def test_replay_failure_latches_runtime_and_shutdown_must_drain() -> None:
    runtime = _GraphRuntime()
    registry = _new_registry(runtime)
    key = GraphKey("beam_decode", 1, (1,))
    inputs = {"tokens": torch.ones(1)}
    assert registry.capture(key, inputs, {}, runtime.forward)
    runtime.fail_replay = True
    with pytest.raises(RuntimeError, match="replay failed"):
        registry.try_replay(key, inputs, {})
    assert registry.replay_count == 0
    with pytest.raises(RuntimeError, match="runtime is invalid"):
        registry.try_replay(key, inputs, {})

    runtime.fail_sync = True
    with pytest.raises(RuntimeError, match="sync failed"):
        registry.close()
    assert registry.graph_reserved_bytes_observed == 100
    runtime.fail_sync = False
    registry.close()
    registry.close()
    with pytest.raises(RuntimeError, match="registry is closed"):
        registry.try_replay(key, inputs, {})


def test_failed_capture_is_never_published_or_reexecuted() -> None:
    runtime = _GraphRuntime()
    registry = _new_registry(runtime)
    key = GraphKey("beam_decode", 1, (1,))
    inputs = {"tokens": torch.ones(1)}

    def fail(_inputs):
        raise RuntimeError("partial capture")

    with pytest.raises(RuntimeError, match="partial capture"):
        registry.capture(key, inputs, {}, fail)
    assert registry.capture_count == 0
    with pytest.raises(RuntimeError, match="runtime is invalid"):
        registry.capture(key, inputs, {}, runtime.forward)
    registry.close()


def _runtime_fixture(monkeypatch: pytest.MonkeyPatch):
    from vllm_gr.v1.worker.npu import owned_graph_runtime as dispatch
    from vllm_gr.v1.worker.npu.owned_graph_inputs import ExactGraphCall

    runtime = _GraphRuntime()
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        model=torch.nn.Linear(2, 2),
        kv_caches=[torch.empty(2)],
        _graph_generation=1,
    )
    monkeypatch.setattr(
        dispatch,
        "OwnedNPUGraphRegistry",
        lambda device, generation, max_entries: _new_registry(runtime, capacity=max_entries),
    )
    monkeypatch.setattr(
        dispatch,
        "_memory_snapshot",
        lambda device: dispatch._MemorySnapshot(
            1000 - runtime.reserved, runtime.reserved, runtime.reserved
        ),
    )

    def build(runner, prepared, resources):
        key = GraphKey("probe", runner._graph_generation, tuple(prepared.tokens.shape))
        return ExactGraphCall(
            key,
            {"tokens": prepared.tokens},
            dict(resources),
            lambda inputs: SimpleNamespace(tokens=inputs["tokens"]),
        )

    def eager(prepared):
        return runtime.forward({"tokens": prepared.tokens})

    return dispatch, runtime, runner, dispatch.PreparedGraphPlan("probe", build), eager


def test_runtime_only_replays_explicitly_provisioned_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatch, runtime, runner, plan, eager = _runtime_fixture(monkeypatch)
    prepared = SimpleNamespace(tokens=torch.tensor([2, 3]))
    with pytest.raises(RuntimeError, match="active startup reservation"):
        dispatch.capture_owned_prepared(runner, prepared, eager, reserved=plan)
    assert dispatch.reserve_owned_graph_memory(runner, 1000, 200) == 800
    with dispatch.owned_graph_capture_session(runner, max_entries=1):
        assert dispatch.capture_owned_prepared(runner, prepared, eager, reserved=plan)
    registry = runner._owned_graph_registry
    assert registry.capture_count == 1
    assert dispatch.forward_with_owned_graphs(runner, prepared, eager).tolist() == [4, 6]
    assert runtime.replays == 1
    miss = SimpleNamespace(tokens=torch.tensor([9]))
    assert dispatch.forward_with_owned_graphs(runner, miss, eager).tolist() == [18]
    assert runtime.replays == 1 and registry.capture_count == 1
    assert runner._owned_graph_forward_miss_count == 1
    runner._graph_generation += 1
    with pytest.raises(RuntimeError, match="Stale owned graph registry"):
        dispatch.forward_with_owned_graphs(runner, prepared, eager)
    registry.close()


def test_runtime_memory_budget_failure_does_not_publish_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatch, _, runner, plan, eager = _runtime_fixture(monkeypatch)
    dispatch.reserve_owned_graph_memory(runner, 1000, 50)
    with pytest.raises(RuntimeError, match="exceeded reserved bytes"):
        with dispatch.owned_graph_capture_session(runner, max_entries=1):
            dispatch.capture_owned_prepared(
                runner, SimpleNamespace(tokens=torch.ones(1)), eager, reserved=plan
            )
    assert runner._owned_graph_registry is None
    assert runner._owned_graph_capture_session is None


def test_unqualified_graph_flags_are_not_claimed_as_supported() -> None:
    from vllm_gr.v1.worker.npu.owned_graph_runtime import validate_owned_graph_requirements

    config = SimpleNamespace(
        beam_graph_active=True, deployment=SimpleNamespace(prefill_graph_enabled=True)
    )
    with pytest.raises(ValueError, match="beam_graph_enabled=false"):
        validate_owned_graph_requirements(config)
    config.beam_graph_active = False
    with pytest.raises(ValueError, match="prefill_graph_enabled=false"):
        validate_owned_graph_requirements(config)
    config.deployment.prefill_graph_enabled = False
    validate_owned_graph_requirements(config)
