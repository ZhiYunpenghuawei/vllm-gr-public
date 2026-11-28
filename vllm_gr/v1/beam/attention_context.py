# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""BeamAttentionContext — worker-level shared resource container for incremental decode.

Holds the KV pools, session registry, and suffix temporary buffers needed
by the attention layer during beam decode.  All per-request state (decode_step,
beam_width, prefix_len, is_last_step) is managed by BeamSessionRegistry,
NOT by this context.

This context is a worker-level singleton.  BeamSearchSession instances
reference it but do not own it.
"""

from __future__ import annotations

import torch

from vllm_gr.v1.beam.attention_pool import BeamAttentionPool, _validate_pool_dtype
from vllm_gr.v1.beam.session_registry import BeamSessionRegistry


class BeamDecisionBuffers:
    """Device-resident beam-decision state.

    Holds ``sequence`` / ``beam_scores`` / ``group_token_num`` at stable
    addresses for Beam decisions. Buffers are sized to
    (max_batch, max_beams, max_decode_steps) and never reallocated.
    """

    def __init__(
        self,
        max_batch: int,
        max_beams: int,
        max_decode_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.max_beams = max_beams
        self.device = device
        # Sequence history: [max_batch, max_beams, max_decode_steps]
        self.sequence = torch.zeros(
            (max_batch, max_beams, max_decode_steps),
            dtype=torch.int32,
            device=device,
        )
        # Cumulative log-probs per beam: [max_batch, max_beams]
        self.beam_scores = torch.zeros((max_batch, max_beams), dtype=torch.float32, device=device)
        # Per-target-beam source-beam mapping (prefix sums consumed by the
        # fused select_unshared_kv kernel): [max_batch, max_beams]
        self.group_token_num = torch.zeros((max_batch, max_beams), dtype=torch.int32, device=device)
        # Host-side token count for the worker decision (next sequence slot to
        # write).  Mirrors the session decode_step but lives with the buffers
        # so the decision module owns it entirely and reset() clears it.
        self.worker_decode_steps = 0

    def reset(self, batch_index: int | None = None) -> None:
        """Clear all legacy state, or only one physical device row.

        A row reset leaves the legacy shared host counter untouched. The
        persistent runner owns separate per-row device counters.
        """
        if batch_index is None:
            self.sequence.zero_()
            self.beam_scores.zero_()
            self.group_token_num.zero_()
            self.worker_decode_steps = 0
        else:
            if isinstance(batch_index, bool) or not isinstance(batch_index, int):
                raise TypeError("batch_index must be an integer")
            if not 0 <= batch_index < self.sequence.shape[0]:
                raise ValueError("batch_index exceeds decision capacity")
            self.sequence[batch_index : batch_index + 1].zero_()
            self.beam_scores[batch_index : batch_index + 1].zero_()
            self.group_token_num[batch_index : batch_index + 1].zero_()


class BeamAttentionContext:
    """Worker-level shared resource container for incremental decode.

    Responsibilities:
    - Own the per-layer KV pools (shared across all sessions).
    - Own the BeamSessionRegistry (session_id → slot mapping + state).
    - Own the suffix KV temporary buffers (shared, partitioned by slot).
    - Provide get_suffix_buf(slot, beam_width) for per-session buffer slices.

    Exposed to:
    - context.pools[i]            → forward / cascade_attn
    - context.registry            → session management
    - context.get_suffix_buf(...) → cascade_attn
    """

    _gr_execution_owner: object | None

    def __init__(
        self,
        max_batch: int,
        max_beams: int,
        num_kv_heads: int,
        max_decode_steps: int,
        head_dim: int,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        _validate_pool_dtype(dtype)

        self.max_batch = max_batch
        self.max_beams = max_beams
        self.max_decode_steps = max_decode_steps
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype

        # Per-request session registry
        self.registry = BeamSessionRegistry(max_batch)

        # Per-layer KV pools (shared across all sessions)
        self.pools: list[BeamAttentionPool] = [
            BeamAttentionPool(
                max_batch,
                max_beams,
                num_kv_heads,
                max_decode_steps,
                head_dim,
                device,
                dtype,
            )
            for _ in range(num_layers)
        ]

        # Suffix KV temporary buffers (for cascade attention)
        # Partitioned by slot: each slot gets max_beams * max_decode_steps rows
        self.suffix_k_buf = torch.zeros(
            (max_batch * max_beams * max_decode_steps, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.suffix_v_buf = torch.zeros(
            (max_batch * max_beams * max_decode_steps, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )

        # Device-side step scalar, filled before model forward / graph replay.
        self.decode_step_buf = torch.zeros(1, dtype=torch.int32, device=device)

        # Separate per-layer scratch for reorder.
        self.aux_key_buf = torch.zeros(
            (num_layers, max_beams, num_kv_heads, max_decode_steps, head_dim),
            dtype=dtype,
            device=device,
        )
        self.aux_value_buf = torch.zeros(
            (num_layers, max_beams, num_kv_heads, max_decode_steps, head_dim),
            dtype=dtype,
            device=device,
        )

        # Device-resident beam-decision state .
        self.decision = BeamDecisionBuffers(max_batch, max_beams, max_decode_steps, device, dtype)

    def reorder(
        self,
        parent: torch.Tensor,
        slot: int,
        decode_step: int,
        *,
        layout_prevalidated: bool = False,
    ) -> None:
        from vllm_gr.v1.beam.reorder import reorder_unshared_kv

        reorder_unshared_kv(
            parent,
            slot,
            self.pools,
            decode_step,
            aux_key=self.aux_key_buf,
            aux_value=self.aux_value_buf,
        )

    def get_suffix_buf(self, slot: int, beam_width: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get suffix KV buffer slice for a specific slot.

        Returns (k_buf, v_buf) each of shape [beam_width * max_decode_steps, H, D].
        """
        offset = slot * self.max_beams * self.max_decode_steps
        size = beam_width * self.max_decode_steps
        return self.suffix_k_buf[offset : offset + size], self.suffix_v_buf[offset : offset + size]
