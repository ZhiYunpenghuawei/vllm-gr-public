# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Reference KV pool reorder for incremental Beam Decode."""

from __future__ import annotations

import torch


def select_unshared_kv(
    beam_index: torch.Tensor,  # [W_cur] int32, source beam index for each beam
    slot: int,  # batch slot index
    unshared_key_caches: list[torch.Tensor],  # per-layer pool key
    unshared_value_caches: list[torch.Tensor],  # per-layer pool value
    decode_step: int,  # 0-indexed, reorder range [0:decode_step+1]
    aux_key_bufs: list[torch.Tensor] | None = None,  # pre-allocated [W, H, max_steps, D]
    aux_value_bufs: list[torch.Tensor] | None = None,  # pre-allocated [W, H, max_steps, D]
) -> None:
    """In-place reorder pool KV according to beam_index for a specific slot.

    ``beam_index[i] = j`` means: new beam i's KV history should come from
    old beam j.

    Only the first ``n = len(beam_index)`` pool slots are reordered; the
    trailing slots (leftover from a larger beam count on an earlier step) are
    left stale and unreferenced.  ``beam_index`` values index into the *full*
    previous pool, so a shrinking beam count correctly rescues the surviving
    beams' histories into the front slots.  When the beam count stays constant
    (the fixed-width case) ``n`` equals the pool width and this is a full
    reorder.

    When aux_key_bufs/aux_value_bufs are provided, they are used instead of
    .clone() to avoid per-step new allocations (graph compatibility).
    Otherwise, the original .clone() path is used.

    Note: select_unshared_kv is called outside the graph (after beam
    decision), so decode_step remains a host int.
    """
    n = beam_index.shape[0]
    for layer_idx, (key_cache, value_cache) in enumerate(
        zip(unshared_key_caches, unshared_value_caches)
    ):
        # key_cache: [max_batch, W, H, max_steps, D]
        # Only reorder [0:decode_step+1] range for the given slot
        if aux_key_bufs is not None and aux_value_bufs is not None:
            # Graph-compatible path: use pre-allocated aux_buf instead of .clone()
            aux_key = aux_key_bufs[layer_idx]
            aux_value = aux_value_bufs[layer_idx]
            # Copy only the [0:decode_step+1] range into aux_buf
            aux_key[:, :, : decode_step + 1, :].copy_(key_cache[slot, :, :, : decode_step + 1, :])
            aux_value[:, :, : decode_step + 1, :].copy_(
                value_cache[slot, :, :, : decode_step + 1, :]
            )
            # Reorder by beam_index and write back (only first n beams, only valid steps)
            key_cache[slot, :n, :, : decode_step + 1, :] = aux_key[
                beam_index, :, : decode_step + 1, :
            ]
            value_cache[slot, :n, :, : decode_step + 1, :] = aux_value[
                beam_index, :, : decode_step + 1, :
            ]
        else:
            # Original path: .clone() creates new allocation (not Graph-compatible)
            old_key = key_cache[slot, :, :, : decode_step + 1, :].clone()
            old_value = value_cache[slot, :, :, : decode_step + 1, :].clone()
            key_cache[slot, :n, :, : decode_step + 1, :] = old_key[beam_index]
            value_cache[slot, :n, :, : decode_step + 1, :] = old_value[beam_index]


def reorder_unshared_kv(
    parent: torch.Tensor,
    slot: int,
    pools: list,
    decode_step: int,
    aux_key_bufs: list[torch.Tensor] | None = None,
    aux_value_bufs: list[torch.Tensor] | None = None,
    aux_key: torch.Tensor | None = None,
    aux_value: torch.Tensor | None = None,
) -> None:
    """Reorder every layer's written suffix KV in place.

    The optional per-layer staging tensors preserve the original history
    until all children have gathered their own parent.
    """
    if aux_key is not None and aux_value is not None:
        aux_key_bufs = [aux_key[layer] for layer in range(len(pools))]
        aux_value_bufs = [aux_value[layer] for layer in range(len(pools))]
    select_unshared_kv(
        parent,
        slot,
        [p.unshared_key for p in pools],
        [p.unshared_value for p in pools],
        decode_step,
        aux_key_bufs=aux_key_bufs,
        aux_value_bufs=aux_value_bufs,
    )


__all__ = ["reorder_unshared_kv", "select_unshared_kv"]
