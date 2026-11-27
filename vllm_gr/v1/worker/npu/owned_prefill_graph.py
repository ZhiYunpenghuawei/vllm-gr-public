# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Exact-length Prefill and ordinary FULL graphs for owned NPU V2.

Host attention lengths participate in the graph key. No task update is needed
when the lengths are unchanged. BSND is restricted to complete fresh prompts
with equal lengths and no padding. General TND length-bucket updates and padded
BSND execution remain separate device validation work.
"""

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Literal

import torch

from vllm_gr.v1.worker.npu.owned_graph_inputs import ExactGraphCall, build_exact_graph_call


def make_prefill_graph_call(
    prepared: Any,
    *,
    generation: int,
    resources: Mapping[str, torch.Tensor],
    device: torch.device,
    layout: Literal["TND", "BSND"],
) -> ExactGraphCall | None:
    """Select a complete fresh-prefill shape without modifying the live forward.

    The owned runner must have built actual sequence lengths and query offsets
    without scheduler padding. Any cached prefix causes an eager fallback.
    """
    if (
        layout not in ("TND", "BSND")
        or prepared.beam_info is not None
        or not prepared.attention_metadata
    ):
        return None
    if prepared.input_ids.ndim != 1 or prepared.positions.shape != prepared.input_ids.shape:
        return None
    tokens = prepared.input_ids.shape[0]
    if tokens <= 0:
        return None
    replacement = {}
    request_lengths: tuple[int, ...] | None = None
    for name, metadata in prepared.attention_metadata.items():
        if (
            metadata.is_beam_decode
            or metadata.beam_session is not None
            or metadata.prefix_indices is not None
            or not metadata.causal
            or metadata.npu_causal_atten_mask is None
            or metadata.num_reqs <= 0
            or metadata.num_actual_tokens != tokens
            or metadata.slot_mapping.ndim != 1
            or metadata.slot_mapping.shape[0] != tokens
        ):
            return None
        qsl = metadata.query_start_loc_cpu
        seq_lens = metadata.seq_lens_cpu
        if qsl is None or seq_lens is None:
            return None
        if len(qsl) != metadata.num_reqs + 1 or len(seq_lens) != metadata.num_reqs:
            return None
        if qsl[0] != 0 or qsl[-1] != tokens:
            return None
        lengths = tuple(qsl[i + 1] - qsl[i] for i in range(metadata.num_reqs))
        if any(length <= 0 for length in lengths) or lengths != tuple(seq_lens):
            # Cached prefix, decode, or partial/chunked prefill. The dense
            # BSND path would otherwise omit earlier KV.
            return None
        if request_lengths is not None and request_lengths != lengths:
            return None
        request_lengths = lengths
        if layout == "BSND" and len(set(lengths)) != 1:
            return None

        # Keep native physical KV writes and exact host metadata. The GR
        # prefill fast path runs TND paged attention or dense BSND from Q/K/V.
        # The captured metadata owns its workspace; do not mutate eager state.
        replacement[name] = replace(
            metadata,
            is_prefill_graph_exec=True,
            prefill_fia_layout=layout,
            bsnd_resources={},
            std_indices=None,
            std_cu_seqlens_q_device=None,
            std_seqlens_kv_device=None,
            std_block_table=None,
            std_seqlens_q_cpu=None,
            std_cu_seqlens_q_cpu=None,
            std_seqlens_kv_cpu=None,
        )
    assert request_lengths is not None
    adapted = replace(prepared, attention_metadata=replacement)
    return build_exact_graph_call(
        adapted,
        phase=f"prefill_{layout.lower()}",
        generation=generation,
        geometry=(len(request_lengths), tokens, *request_lengths),
        resources=resources,
        device=device,
    )


def make_ordinary_full_graph_call(
    prepared: Any,
    *,
    generation: int,
    resources: Mapping[str, torch.Tensor],
    device: torch.device,
) -> ExactGraphCall | None:
    """Ordinary forward with identical host lengths and physical layout."""
    if prepared.beam_info is not None or not prepared.attention_metadata:
        return None
    tokens = prepared.input_ids.shape[0]
    if tokens <= 0 or prepared.positions.shape != prepared.input_ids.shape:
        return None
    lengths = None
    for metadata in prepared.attention_metadata.values():
        if (
            metadata.is_beam_decode
            or metadata.beam_session is not None
            or metadata.prefix_indices is not None
        ):
            return None
        if metadata.prefill_fia_layout != "TND" or metadata.bsnd_resources:
            return None
        if metadata.num_actual_tokens != tokens:
            return None
        qsl = metadata.query_start_loc_cpu
        seq_lens = metadata.seq_lens_cpu
        if qsl is None or seq_lens is None or metadata.num_reqs <= 0:
            return None
        if (
            len(qsl) != metadata.num_reqs + 1
            or len(seq_lens) != metadata.num_reqs
            or qsl[0] != 0
            or qsl[-1] != tokens
        ):
            return None
        query_lengths = tuple(qsl[i + 1] - qsl[i] for i in range(metadata.num_reqs))
        if any(query <= 0 or sequence < query for query, sequence in zip(query_lengths, seq_lens)):
            return None
        current = tuple(qsl), tuple(seq_lens)
        if lengths is not None and lengths != current:
            return None
        lengths = current
    assert lengths is not None
    return build_exact_graph_call(
        prepared,
        phase="ordinary_full",
        generation=generation,
        geometry=(len(lengths[1]), tokens),
        resources=resources,
        device=device,
    )
