# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Owned NPU graph storage, capture, replay and lifetime.

The caller supplies the complete host geometry and all persistent bindings.
Only tensor values may change on replay. The captured function receives owned
input buffers and must not mutate them. Captures must run against startup or
other explicitly reserved KV resources, never an uncommitted live request.
"""

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class GraphKey:
    """Complete caller-provided immutable host description of one forward."""

    phase: str
    generation: int
    geometry: tuple[int, ...]
    host_metadata: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class _TensorLayout:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device

    @classmethod
    def of(cls, tensor: torch.Tensor) -> "_TensorLayout":
        return cls(tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device)


@dataclass(frozen=True, slots=True)
class _ResourceBinding:
    layout: _TensorLayout
    address: int
    storage: int
    storage_offset: int

    @classmethod
    def of(cls, tensor: torch.Tensor) -> "_ResourceBinding":
        return cls(
            _TensorLayout.of(tensor),
            tensor.data_ptr(),
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
        )


def _input_layouts(inputs: Mapping[str, torch.Tensor]) -> dict[str, _TensorLayout]:
    return {name: _TensorLayout.of(tensor) for name, tensor in inputs.items()}


def _resource_bindings(resources: Mapping[str, torch.Tensor]) -> dict[str, _ResourceBinding]:
    return {name: _ResourceBinding.of(tensor) for name, tensor in resources.items()}


def _input_aliases(inputs: Mapping[str, torch.Tensor]) -> tuple[tuple[str, ...], ...]:
    """Preserve repeated references; reject distinct views of shared storage."""
    groups: dict[int, list[tuple[str, _ResourceBinding]]] = {}
    empty = []
    for name, tensor in sorted(inputs.items()):
        bind = _ResourceBinding.of(tensor)
        if tensor.numel() == 0:
            empty.append((name,))
            continue
        groups.setdefault(bind.storage, []).append((name, bind))
    result = list(empty)
    for views in groups.values():
        first = views[0][1]
        if any(binding != first for _, binding in views):
            raise ValueError("Graph inputs contain distinct views of aliased storage")
        result.append(tuple(name for name, _ in views))
    return tuple(sorted(result))


@dataclass(slots=True)
class _Entry:
    graph: Any
    pool: Any
    forward: Callable[[Mapping[str, torch.Tensor]], torch.Tensor]
    inputs: dict[str, torch.Tensor]
    resources: dict[str, torch.Tensor]
    input_layouts: dict[str, _TensorLayout]
    aliases: tuple[tuple[str, ...], ...]
    resource_bindings: dict[str, _ResourceBinding]
    output: torch.Tensor | None = None
    reserved_bytes_delta: int = 0


class NPUGraphRuntime:
    """The actual vendor runtime, isolated here for host-only contract tests."""

    def __init__(self, device: torch.device):
        if device.type != "npu" or device.index is None:
            raise ValueError("An explicit NPU device is required for graph execution")
        self.device = device

    def pool_handle(self) -> Any:
        with torch.npu.device(self.device):
            return torch.npu.graph_pool_handle()

    def new_graph(self) -> Any:
        with torch.npu.device(self.device):
            return torch.npu.NPUGraph()

    @contextmanager
    def capture(self, graph: Any, pool: Any) -> Iterator[None]:
        with (
            torch.npu.device(self.device),
            torch.npu.graph(graph, pool=pool, capture_error_mode="global"),
        ):
            yield

    def replay(self, graph: Any) -> None:
        with torch.npu.device(self.device):
            graph.replay()

    def synchronize(self) -> None:
        torch.npu.synchronize(self.device)

    def memory_reserved(self) -> int:
        return int(torch.npu.memory_reserved(self.device))


class OwnedNPUGraphRegistry:
    """Own a bounded registry for one model/resource generation.

    Exact geometry/binding misses return None, allowing the caller to use its
    checked eager path. A capture/replay execution failure poisons the registry:
    the model may already have written KV and cannot safely repeat the forward.
    """

    def __init__(
        self, device: torch.device, *, generation: int, max_entries: int, runtime: Any = None
    ):
        if generation < 0 or max_entries <= 0:
            raise ValueError("Invalid graph generation or capacity")
        self.device = device
        self.generation = generation
        self.max_entries = max_entries
        self._runtime = NPUGraphRuntime(device) if runtime is None else runtime
        self._entries: dict[GraphKey, _Entry] = {}
        self._incomplete: list[_Entry] = []
        self._lock = RLock()
        self._closed = False
        self._failure: Exception | None = None
        self.capture_count = 0
        self.replay_count = 0
        self.miss_count = 0

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("NPU graph registry is closed")
        if self._failure is not None:
            raise RuntimeError(
                "NPU graph runtime is invalid after an execution failure"
            ) from self._failure

    def assert_usable(self) -> None:
        """Fail closed even when the current forward would miss every graph."""
        with self._lock:
            self._ensure_open()

    def _valid_inputs(self, inputs: Mapping[str, torch.Tensor]) -> bool:
        return bool(inputs) and all(
            t.device == self.device and t.is_contiguous() for t in inputs.values()
        )

    def capture(
        self,
        key: GraphKey,
        inputs: Mapping[str, torch.Tensor],
        resources: Mapping[str, torch.Tensor],
        forward: Callable[[Mapping[str, torch.Tensor]], torch.Tensor],
        *,
        warmup: Callable[[Mapping[str, torch.Tensor]], Any] | None = None,
    ) -> bool:
        """Capture an explicit reserved forward; return False before mutation on unsupported geometry."""
        with self._lock:
            self._ensure_open()
            if (
                key.generation != self.generation
                or key in self._entries
                or len(self._entries) >= self.max_entries
            ):
                return False
            if not self._valid_inputs(inputs) or any(
                t.device != self.device for t in resources.values()
            ):
                return False
            try:
                aliases = _input_aliases(inputs)
            except ValueError:
                return False
            # Inputs are read-only staging buffers. Their storage must not also
            # be writable persistent state referenced directly by the graph.
            resource_bindings = _resource_bindings(resources)
            resource_storage = {binding.storage for binding in resource_bindings.values()}
            if any(
                t.numel() and t.untyped_storage().data_ptr() in resource_storage
                for t in inputs.values()
            ):
                return False
            entry = _Entry(
                graph=None,
                pool=None,
                forward=forward,
                inputs={},
                resources=dict(resources),
                input_layouts=_input_layouts(inputs),
                aliases=aliases,
                resource_bindings=resource_bindings,
            )
            self._incomplete.append(entry)
            try:
                self._runtime.synchronize()
                reserved_before = self._runtime.memory_reserved()
                for group in aliases:
                    stable = inputs[group[0]].detach().clone()
                    for name in group:
                        entry.inputs[name] = stable
                if warmup is not None:
                    warmup(entry.inputs)
                    self._runtime.synchronize()
                # Independent graph pools permit arbitrary replay order. Sharing
                # a pool would require additional allocation-lifetime and replay
                # ordering guarantees between captures.
                entry.pool = self._runtime.pool_handle()
                entry.graph = self._runtime.new_graph()
                self._runtime.synchronize()
                with self._runtime.capture(entry.graph, entry.pool):
                    entry.output = forward(entry.inputs)
                if not isinstance(entry.output, torch.Tensor) or entry.output.device != self.device:
                    raise RuntimeError("Captured forward must return a tensor on the graph device")
                self._runtime.synchronize()
                entry.reserved_bytes_delta = max(
                    0, self._runtime.memory_reserved() - reserved_before
                )
            except Exception as exc:
                self._failure = exc
                raise
            self._incomplete.pop()
            self._entries[key] = entry
            self.capture_count += 1
            return True

    def try_replay(
        self,
        key: GraphKey,
        inputs: Mapping[str, torch.Tensor],
        resources: Mapping[str, torch.Tensor],
    ) -> torch.Tensor | None:
        """Stage fresh tensor contents and replay only the exact captured contract."""
        with self._lock:
            self._ensure_open()
            entry = self._entries.get(key)
            if entry is None or not self._valid_inputs(inputs):
                self.miss_count += 1
                return None
            try:
                aliases = _input_aliases(inputs)
            except ValueError:
                self.miss_count += 1
                return None
            if (
                _input_layouts(inputs) != entry.input_layouts
                or aliases != entry.aliases
                or _resource_bindings(resources) != entry.resource_bindings
            ):
                self.miss_count += 1
                return None
            try:
                # Serialize staging and replay with previous users of these
                # addresses. Keep an independent output for the next consumer.
                self._runtime.synchronize()
                for group in aliases:
                    name = group[0]
                    entry.inputs[name].copy_(inputs[name])
                self._runtime.synchronize()
                self._runtime.replay(entry.graph)
                self._runtime.synchronize()
                assert entry.output is not None
                output = entry.output.clone()
                self._runtime.synchronize()
            except Exception as exc:
                self._failure = exc
                raise
            self.replay_count += 1
            return output

    @property
    def graph_reserved_bytes_observed(self) -> int:
        """Observed allocator reservation growth, not a total graph memory bound."""
        with self._lock:
            return sum(entry.reserved_bytes_delta for entry in self._entries.values())

    def close(self) -> None:
        """Drain work before releasing captured graph buffers and resource references."""
        with self._lock:
            if self._closed:
                return
            # A failed synchronization cannot establish that storage is safe to
            # release. Leave references intact and let the Worker fail closed.
            try:
                self._runtime.synchronize()
            except Exception as exc:
                self._failure = exc
                raise
            self._entries.clear()
            self._incomplete.clear()
            self._closed = True
