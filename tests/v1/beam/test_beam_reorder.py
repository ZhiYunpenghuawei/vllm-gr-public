# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Reorder dispatcher tests.

Cover arbitrary request-local parent orderings (permutation, duplicated
source and shrink) against the reference reorder.
"""

from __future__ import annotations

import pytest
import torch

from vllm_gr.v1.beam.reorder import reorder_unshared_kv


class _Pool:
    def __init__(self, key: torch.Tensor, value: torch.Tensor) -> None:
        self.unshared_key = key
        self.unshared_value = value


def _make_pools(
    shape: tuple[int, ...],
    layers: int,
    device: torch.device,
    seed: int,
) -> list[_Pool]:
    torch.manual_seed(seed)
    return [
        _Pool(
            torch.randn(shape, dtype=torch.float16).to(device),
            torch.randn(shape, dtype=torch.float16).to(device),
        )
        for _ in range(layers)
    ]
    # note: randn on CPU then moved, so all paths share identical inputs


def _reference_reorder(
    parent: torch.Tensor, pools: list[_Pool], slot: int, decode_step: int
) -> None:
    """Replay the cascade reference on CPU."""
    from vllm_gr.v1.beam.reorder import select_unshared_kv

    select_unshared_kv(
        parent.cpu(),
        slot,
        [p.unshared_key.cpu() for p in pools],
        [p.unshared_value.cpu() for p in pools],
        decode_step,
    )


_MAX_BEAMS, _KV_HEADS, _STEPS, _HEAD_DIM, _LAYERS = 4, 2, 3, 8, 2
_POOL_SHAPE = (1, _MAX_BEAMS, _KV_HEADS, _STEPS, _HEAD_DIM)


@pytest.mark.parametrize(
    "parent",
    [
        torch.tensor([0, 1, 2, 3], dtype=torch.int32),  # identity
        torch.tensor([2, 0, 1, 3], dtype=torch.int32),  # permutation (unsorted)
        torch.tensor([0, 0, 1, 2], dtype=torch.int32),  # duplicated source
        torch.tensor([3, 1], dtype=torch.int32),  # shrink 4 -> 2
    ],
)  # type: ignore[misc]
def test_reorder_cascade_mode_matches_beam_kernels(
    monkeypatch: pytest.MonkeyPatch,
    parent: torch.Tensor,
) -> None:
    """The dispatcher in cascade mode must equal the original kernel."""

    seed = 10 + int(parent.sum())
    pools = _make_pools(_POOL_SHAPE, _LAYERS, torch.device("cpu"), seed)
    expected = _make_pools(_POOL_SHAPE, _LAYERS, torch.device("cpu"), seed)

    decode_step = 2
    _reference_reorder(parent, expected, slot=0, decode_step=decode_step)
    reorder_unshared_kv(parent, 0, pools, decode_step)

    for pool, ref in zip(pools, expected):
        torch.testing.assert_close(pool.unshared_key, ref.unshared_key, rtol=0, atol=0)
        torch.testing.assert_close(pool.unshared_value, ref.unshared_value, rtol=0, atol=0)


pytestmark = pytest.mark.cpu_test
