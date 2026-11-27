# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Provision exact Prefill and ordinary FULL graph shapes before admission."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

from vllm_gr.v1.worker.npu.owned_graph_inputs import ExactGraphCall
from vllm_gr.v1.worker.npu.owned_graph_runtime import (
    PreparedGraphPlan,
    capture_owned_prepared,
    owned_graph_capture_session,
)
from vllm_gr.v1.worker.npu.owned_prefill_graph import (
    make_ordinary_full_graph_call,
    make_prefill_graph_call,
)


def _tnd(
    runner: Any, prepared: Any, resources: Mapping[str, torch.Tensor]
) -> ExactGraphCall | None:
    return make_prefill_graph_call(
        prepared,
        generation=runner._graph_generation,
        resources=resources,
        device=runner.device,
        layout="TND",
    )


def _bsnd(
    runner: Any, prepared: Any, resources: Mapping[str, torch.Tensor]
) -> ExactGraphCall | None:
    return make_prefill_graph_call(
        prepared,
        generation=runner._graph_generation,
        resources=resources,
        device=runner.device,
        layout="BSND",
    )


def _ordinary(
    runner: Any, prepared: Any, resources: Mapping[str, torch.Tensor]
) -> ExactGraphCall | None:
    return make_ordinary_full_graph_call(
        prepared, generation=runner._graph_generation, resources=resources, device=runner.device
    )


PREFILL_TND_CAPTURE = PreparedGraphPlan("prefill_tnd", _tnd)
PREFILL_BSND_CAPTURE = PreparedGraphPlan("prefill_bsnd", _bsnd)
ORDINARY_FULL_CAPTURE = PreparedGraphPlan("ordinary_full", _ordinary)
_PLANS = {
    plan.phase: plan for plan in (PREFILL_BSND_CAPTURE, PREFILL_TND_CAPTURE, ORDINARY_FULL_CAPTURE)
}


@dataclass(frozen=True)
class StartupGraphSpec:
    """Exact unpadded request lengths; not a variable-length graph bucket."""

    phase: Literal["prefill_tnd", "prefill_bsnd", "ordinary_full"]
    query_lengths: tuple[int, ...]
    sequence_lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.phase not in _PLANS:
            raise ValueError(f"Unsupported startup graph phase: {self.phase}")
        if not self.query_lengths or len(self.query_lengths) != len(self.sequence_lengths):
            raise ValueError("Graph needs paired nonempty query and sequence lengths")
        if any(
            type(q) is not int or type(s) is not int or q <= 0 or s < q
            for q, s in zip(self.query_lengths, self.sequence_lengths)
        ):
            raise ValueError("Invalid exact graph query or sequence length")
        if self.phase != "ordinary_full" and self.query_lengths != self.sequence_lengths:
            raise ValueError("Prefill capture requires complete fresh prompts")
        if self.phase == "prefill_bsnd" and len(set(self.query_lengths)) != 1:
            raise ValueError("BSND capture requires equal unpadded request lengths")


def exact_prefill_startup_specs(gr_config: Any) -> tuple[StartupGraphSpec, ...]:
    """Interpret existing capture sizes as explicit exact-length probe shapes.

    No native bucket or arbitrary request lengths are promised. This helper
    is for the qualification driver while public graph admission is gated.
    """
    from vllm_gr.v1.worker.npu.npu_prefill_graph_runner import parse_bsnd_capture_specs

    deployment = gr_config.deployment
    buckets = deployment.prefill_token_buckets
    if not buckets:
        raise ValueError("Exact Prefill graph capture requires explicit prefill token sizes")
    bsnd = parse_bsnd_capture_specs(deployment.npu_bsnd_capture_specs, buckets)
    result = [
        StartupGraphSpec("prefill_bsnd", (total // batch,) * batch, (total // batch,) * batch)
        for total, batch in bsnd
    ]
    result.extend(StartupGraphSpec("prefill_tnd", (total,), (total,)) for total in buckets)
    return tuple(result)


@dataclass(frozen=True)
class _DummyLayout:
    block_tables: tuple[tuple[int, ...], ...]
    positions: tuple[int, ...]
    slots: tuple[int, ...]
    last_block: int


def _dummy_layout(spec: StartupGraphSpec, *, block_size: int, num_blocks: int) -> _DummyLayout:
    if block_size <= 0 or num_blocks <= 1:
        raise ValueError("Owned graph dummy input requires initialized non-null KV blocks")
    block_tables = []
    positions = []
    slots = []
    next_block = 1  # Preserve the native null block at index zero.
    for query_len, seq_len in zip(spec.query_lengths, spec.sequence_lengths):
        blocks_needed = (seq_len + block_size - 1) // block_size
        blocks = tuple(range(next_block, next_block + blocks_needed))
        next_block += blocks_needed
        if next_block > num_blocks:
            raise ValueError("Exact graph startup geometry exceeds reserved KV capacity")
        block_tables.append(blocks)
        for position in range(seq_len - query_len, seq_len):
            positions.append(position)
            slots.append(blocks[position // block_size] * block_size + position % block_size)
    if slots[-1] >= 2**31:
        raise ValueError("Owned graph slot mapping exceeds int32")
    return _DummyLayout(tuple(block_tables), tuple(positions), tuple(slots), next_block - 1)


def capture_startup_graphs(
    runner: Any,
    specs: Sequence[StartupGraphSpec],
    eager: Callable[[Any], torch.Tensor],
) -> int:
    """Capture against temporarily reserved physical blocks before any request.

    The enclosing Worker must have reserved graph bytes before computing KV
    capacity. Every captured weight/KV address remains the real serving address.
    The dummy blocks are cleared before capture and after each successful graph.
    On a capture failure stop startup instead of reusing potentially invalid
    graph or KV state. This function never provisions a live-request graph.
    """
    if not specs:
        return 0
    if len(set(specs)) != len(specs):
        raise ValueError("Duplicate startup graph shapes")
    scheduler = runner.vllm_config.scheduler_config
    max_len = runner.vllm_config.model_config.max_model_len
    block_size = runner.vllm_config.cache_config.block_size
    caches = runner.kv_caches
    if not caches:
        raise RuntimeError("Startup graph capture requires finalized KV caches")
    num_blocks = min(cache.shape[0] for cache in caches)
    planned = []
    for spec in specs:
        if (
            len(spec.query_lengths) > scheduler.max_num_seqs
            or sum(spec.query_lengths) > scheduler.max_num_batched_tokens
        ):
            raise ValueError("Startup graph exceeds scheduler request or token capacity")
        if max(spec.sequence_lengths) > max_len:
            raise ValueError("Startup graph exceeds model context capacity")
        planned.append((spec, _dummy_layout(spec, block_size=block_size, num_blocks=num_blocks)))

    device = runner.device

    def clear_blocks(last: int) -> None:
        for cache in caches:
            cache[1 : last + 1].zero_()
        torch.npu.synchronize(device)

    with torch.inference_mode(), owned_graph_capture_session(runner, max_entries=len(planned)):
        for spec, layout in planned:
            clear_blocks(layout.last_block)
            prepared = runner._prepare_forward(
                torch.zeros(sum(spec.query_lengths), dtype=torch.int32, device=device),
                torch.tensor(layout.positions, dtype=torch.long, device=device),
                spec.query_lengths,
                spec.sequence_lengths,
                layout.block_tables,
                torch.tensor(layout.slots, dtype=torch.int32, device=device),
            )
            if not capture_owned_prepared(runner, prepared, eager, reserved=_PLANS[spec.phase]):
                raise RuntimeError(f"Exact startup graph is not capturable: {spec}")
            # Capture completed and drained successfully. Dummy writes cannot
            # escape into the next reserved shape or subsequent scheduler use.
            clear_blocks(layout.last_block)
    return runner._owned_graph_registry.capture_count
