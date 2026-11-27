# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Build exact graph input contracts from owned ModelRunner V2 forwards."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace
from math import isfinite
from typing import Any

import torch

from vllm_gr.v1.worker.npu.owned_graph import GraphKey, OwnedNPUGraphRegistry


class _UnsupportedGraphInput(ValueError):
    """The forward needs an input representation not covered by static replay."""


@dataclass(slots=True)
class ExactGraphCall:
    key: GraphKey
    inputs: dict[str, torch.Tensor]
    resources: dict[str, torch.Tensor]
    build: Callable[[Mapping[str, torch.Tensor]], Any]

    def capture(
        self, registry: OwnedNPUGraphRegistry, forward: Callable[[Any], torch.Tensor]
    ) -> bool:
        # Keep the reconstructed metadata and its workspace alive until graph
        # retirement. Warmup and capture use the same stable descriptor.
        build = self.build
        prepared = None

        def run(stable_inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
            nonlocal prepared
            if prepared is None:
                prepared = build(stable_inputs)
            return forward(prepared)

        return registry.capture(self.key, self.inputs, self.resources, run, warmup=run)

    def try_replay(self, registry: OwnedNPUGraphRegistry) -> torch.Tensor | None:
        return registry.try_replay(self.key, self.inputs, self.resources)


def runner_graph_resources(runner: Any) -> dict[str, torch.Tensor]:
    """Identify native dense model and KV storage, including tied weight aliases."""
    model = runner.model
    if model is None:
        raise RuntimeError("Graph capture requires a loaded model")
    resources = {
        f"parameter:{name}": value for name, value in model.named_parameters(remove_duplicate=False)
    }
    resources.update(
        {f"buffer:{name}": value for name, value in model.named_buffers(remove_duplicate=False)}
    )
    resources.update({f"kv:{i}": cache for i, cache in enumerate(runner.kv_caches)})
    for name, layer in model.named_modules():
        cache = getattr(layer, "kv_cache", None)
        if isinstance(cache, torch.Tensor):
            resources[f"layer_kv:{name}"] = cache
    return resources


class _InputTree:
    """Immutable host descriptor plus copied host constants and dynamic tensor leaves."""

    def __init__(self, device: torch.device, metadata_type: type, session: Any = None):
        self.device = device
        self.metadata_type = metadata_type
        self.session = session
        self.inputs: dict[str, torch.Tensor] = {}
        self.host_tensors: dict[str, torch.Tensor] = {}

    def describe(self, value: Any, path: tuple[str, ...]) -> tuple[Any, ...]:
        name = repr(path)
        if self.session is not None and value is self.session:
            return ("session",)
        if isinstance(value, torch.Tensor):
            if value.device == self.device:
                self.inputs[name] = value
                return ("input", name, tuple(value.shape), tuple(value.stride()), str(value.dtype))
            if value.device.type == "cpu" and value.numel() <= 65536:
                snapshot = value.detach().clone()
                self.host_tensors[name] = snapshot
                return (
                    "host",
                    name,
                    tuple(snapshot.shape),
                    str(snapshot.dtype),
                    _immutable(snapshot.tolist()),
                )
            raise _UnsupportedGraphInput(
                "Graph metadata contains an unsupported tensor device or host extent"
            )
        if value is None or type(value) in (str, int, bool):
            return ("value", type(value), value)
        if type(value) is float and isfinite(value):
            return ("value", float, value)
        if type(value) is dict and all(type(key) is str for key in value):
            return (
                "dict",
                tuple((key, self.describe(value[key], (*path, key))) for key in sorted(value)),
            )
        if type(value) in (list, tuple):
            return (
                type(value).__name__,
                tuple(self.describe(item, (*path, str(i))) for i, item in enumerate(value)),
            )
        if type(value) is self.metadata_type:
            return (
                "metadata",
                tuple(
                    (field.name, self.describe(getattr(value, field.name), (*path, field.name)))
                    for field in fields(value)
                ),
            )
        raise _UnsupportedGraphInput(f"Unsupported graph metadata value: {type(value).__name__}")

    def rebuild(self, node: tuple[Any, ...], inputs: Mapping[str, torch.Tensor]) -> Any:
        kind = node[0]
        if kind == "session":
            return self.session
        if kind == "input":
            return inputs[node[1]]
        if kind == "host":
            return self.host_tensors[node[1]]
        if kind == "value":
            return node[2]
        if kind == "dict":
            return {key: self.rebuild(value, inputs) for key, value in node[1]}
        if kind == "list":
            return [self.rebuild(value, inputs) for value in node[1]]
        if kind == "tuple":
            return tuple(self.rebuild(value, inputs) for value in node[1])
        if kind == "metadata":
            return self.metadata_type(
                **{key: self.rebuild(value, inputs) for key, value in node[1]}
            )
        raise AssertionError(f"Unknown graph input kind: {kind}")


def _immutable(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_immutable(item) for item in value)
    if type(value) in (bool, int) or (type(value) is float and isfinite(value)):
        return value
    raise _UnsupportedGraphInput("Invalid host tensor metadata value")


def build_exact_graph_call(
    prepared: Any,
    *,
    phase: str,
    generation: int,
    geometry: tuple[int, ...],
    resources: Mapping[str, torch.Tensor],
    device: torch.device,
    session: Any = None,
    session_descriptor: tuple[int, ...] = (),
) -> ExactGraphCall | None:
    """Capture every host-affecting value, or decline before using a graph.

    PreparedForward.input_batch belongs to scheduler/sample state, and the
    forward_prepared contract does not read it. Graphs only capture the model
    forward. The metadata and slot mappings consumed by attention stay bound.
    """
    from vllm_gr.v1.attention.backends.beam_attn_metadata import BeamAttentionMetadata

    tree = _InputTree(device, BeamAttentionMetadata, session)
    try:
        descriptor = tree.describe(
            {
                "input_ids": prepared.input_ids,
                "positions": prepared.positions,
                "attention_metadata": prepared.attention_metadata,
                "slot_mappings": prepared.slot_mappings,
                "beam_info": prepared.beam_info,
            },
            (),
        )
        key = GraphKey(phase, generation, geometry, (descriptor, session_descriptor))
        hash(key)
    except (_UnsupportedGraphInput, TypeError):
        return None
    # Release transient source devices from the builder closure. The graph's
    # persistent staging buffers are allocated by OwnedNPUGraphRegistry.
    inputs = tree.inputs
    tree.inputs = {}
    prepared_type = type(prepared)

    def build(stable_inputs: Mapping[str, torch.Tensor]) -> Any:
        return prepared_type(**tree.rebuild(descriptor, stable_inputs), input_batch=None)

    return ExactGraphCall(key, inputs, dict(resources), build)


def make_beam_graph_call(
    prepared: Any,
    *,
    generation: int,
    resources: Mapping[str, torch.Tensor],
    device: torch.device,
    xllm: bool,
) -> ExactGraphCall | None:
    """Describe pure xLLM Beam Decode, keeping host lengths and step exact."""
    from vllm_gr.v1.worker.npu.beam_execution_batch import NPUBeamExecutionBatch

    if not xllm or not prepared.beam_info:
        return None
    info = prepared.beam_info
    if set(info) != {"beam_data", "req_ids", "beam_session"}:
        return None
    session = info["beam_session"]
    if type(session) is not NPUBeamExecutionBatch or not session.is_active:
        return None
    context = session.context
    if not getattr(context, "_gr_uses_xllm_buffers", False):
        return None
    step_buffer = getattr(context, "decode_step_1based_buf", None)
    slot_buffer = getattr(context, "xllm_block_table_buf", None)
    if not isinstance(step_buffer, torch.Tensor) or not isinstance(slot_buffer, torch.Tensor):
        return None
    if not prepared.attention_metadata or any(
        not metadata.is_beam_decode
        or metadata.beam_session is not session
        or metadata.std_indices is not None
        or metadata.prefix_indices is None
        or metadata.prefix_seqlens_kv_device is None
        or metadata.prefix_block_table is None
        or not metadata.npu_prefix_rows_are_identity
        or metadata.suffix_max_q_len != 1
        or metadata.num_actual_tokens != session.attention_width
        for metadata in prepared.attention_metadata.values()
    ):
        return None

    # The model forward only consumes the already built metadata. Normalize
    # request names in its context, preserving ordered lengths/widths/steps;
    # user identities stay with scheduler and result processing, not the graph.
    req_ids = tuple(info["req_ids"])
    beam_data = info["beam_data"]
    if (
        len(req_ids) != session.batch_size
        or len(set(req_ids)) != len(req_ids)
        or set(beam_data) != set(req_ids)
    ):
        return None
    values = [beam_data[req_id] for req_id in req_ids]
    if any(
        set(value) != {"prefix_len", "cache_len", "is_beam_decode", "beam_width", "decode_step"}
        for value in values
    ):
        return None
    step = session.decode_step
    if any(
        value["is_beam_decode"] is not True
        or value["beam_width"] != session.request_width
        or value["decode_step"] != step
        or value["prefix_len"] <= 0
        or value["cache_len"] != value["prefix_len"]
        for value in values
    ):
        return None
    graph_ids = [str(i) for i in range(session.batch_size)]
    normalized = replace(
        prepared,
        beam_info={
            "beam_data": dict(zip(graph_ids, values)),
            "req_ids": graph_ids,
            "beam_session": session,
        },
    )
    binding = dict(resources)
    binding["beam:steps"] = step_buffer
    binding["beam:slots"] = slot_buffer
    for i, pool in enumerate(session.pools):
        binding[f"beam:{i}:key"] = pool.unshared_key
        binding[f"beam:{i}:value"] = pool.unshared_value
    return build_exact_graph_call(
        normalized,
        phase="beam_decode",
        generation=generation,
        geometry=(session.batch_size, session.request_width, step),
        resources=binding,
        device=device,
        session=session,
        session_descriptor=(
            session.slot,
            session.attention_width,
            context.max_decode_steps,
            *(member.slot for member in session.sessions),
        ),
    )
