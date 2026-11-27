# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Host tests for exact host-metadata and layout eligibility of NPU graphs."""

from dataclasses import dataclass
from typing import Any, Literal

import pytest
import torch

from vllm_gr.v1.attention.backends.beam_attn_metadata import BeamAttentionMetadata
from vllm_gr.v1.worker.npu.owned_prefill_graph import (
    make_ordinary_full_graph_call,
    make_prefill_graph_call,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def test_graph_qualification_requires_configuration_before_kv_allocation():
    from types import SimpleNamespace

    from vllm_gr.v1.worker.npu.owned_prefill_graph_runtime import StartupGraphSpec
    from vllm_gr.v1.worker.npu.v2.model_runner import NPUModelRunnerV2

    runner = SimpleNamespace(kv_cache_config=None, request_state=SimpleNamespace(requests={}))
    spec = StartupGraphSpec("prefill_tnd", (8,), (8,))
    NPUModelRunnerV2.configure_prefill_graph_qualification(runner, (spec,), memory_bytes=1024)
    assert runner._graph_qualification_specs == (spec,)
    assert runner._graph_qualification_budget == 1024
    runner.kv_cache_config = object()
    with pytest.raises(RuntimeError, match="before KV allocation"):
        NPUModelRunnerV2.configure_prefill_graph_qualification(runner, (spec,), memory_bytes=2048)


@dataclass(frozen=True)
class _PreparedForward:
    input_ids: torch.Tensor
    positions: torch.Tensor
    attention_metadata: dict[str, BeamAttentionMetadata]
    slot_mappings: dict[str, torch.Tensor]
    beam_info: dict[str, Any] | None = None
    input_batch: object | None = None


def _prepared(query: list[int], sequence: list[int] | None = None) -> _PreparedForward:
    sequence = query if sequence is None else sequence
    starts = [0]
    for length in query:
        starts.append(starts[-1] + length)
    tokens = starts[-1]
    slots = torch.arange(tokens)
    metadata = BeamAttentionMetadata(
        num_actual_tokens=tokens,
        max_query_len=max(query),
        query_start_loc=torch.tensor(starts),
        max_seq_len=max(sequence),
        seq_lens=torch.tensor(sequence),
        block_table=torch.zeros((len(query), 2), dtype=torch.int32),
        slot_mapping=slots,
        npu_causal_atten_mask=torch.zeros((4, 4), dtype=torch.bool),
        is_prefill_graph_exec=True,
        num_reqs=len(query),
        query_start_loc_cpu=starts,
        seq_lens_cpu=sequence,
        num_heads=2,
        num_kv_heads=1,
    )
    return _PreparedForward(
        torch.arange(tokens),
        torch.cat([torch.arange(length) for length in query]),
        {"layer": metadata},
        {"layer": slots},
        input_batch=object(),
    )


def _call(prepared: _PreparedForward, layout: Literal["TND", "BSND"] = "TND"):
    return make_prefill_graph_call(
        prepared, generation=1, resources={}, device=torch.device("cpu"), layout=layout
    )


def test_tnd_key_includes_exact_host_lengths_but_not_dynamic_token_values() -> None:
    first = _prepared([2, 3])
    second = _prepared([2, 3])
    second.input_ids.add_(10)
    a, b = _call(first), _call(second)
    assert a is not None and b is not None
    assert a.key == b.key
    c = _call(_prepared([1, 4]))
    assert c is not None and c.key != a.key
    stable = b.build(b.inputs)
    assert stable.input_ids.tolist() == second.input_ids.tolist()
    assert stable.input_batch is None
    assert stable.attention_metadata["layer"].query_start_loc_cpu == [0, 2, 5]
    assert stable.attention_metadata["layer"].slot_mapping is stable.slot_mappings["layer"]
    # Capture metadata must not overwrite the live descriptor.
    stable.attention_metadata["layer"].query_start_loc_cpu[1] = 1
    assert second.attention_metadata["layer"].query_start_loc_cpu == [0, 2, 5]


def test_bsnd_requires_equal_fresh_prompt_lengths() -> None:
    prepared = _prepared([2, 2])
    call = _call(prepared, "BSND")
    assert call is not None
    stable = call.build(call.inputs)
    assert stable.attention_metadata["layer"].prefill_fia_layout == "BSND"
    assert prepared.attention_metadata["layer"].prefill_fia_layout == "TND"
    assert _call(_prepared([1, 3]), "BSND") is None
    assert _call(_prepared([2, 2], [3, 2]), "BSND") is None
    assert _call(_prepared([2, 2], [3, 2]), "TND") is None


def test_ordinary_full_keeps_cached_prefix_lengths_in_key() -> None:
    def ordinary(prepared):
        return make_ordinary_full_graph_call(
            prepared, generation=1, resources={}, device=torch.device("cpu")
        )

    first = ordinary(_prepared([1, 1], [8, 6]))
    second = ordinary(_prepared([1, 1], [8, 7]))
    assert first is not None and second is not None
    assert first.key.phase == "ordinary_full"
    assert first.key != second.key
    assert ordinary(_prepared([1, 1], [0, 8])) is None


def test_unknown_metadata_does_not_create_graph_contract() -> None:
    prepared = _prepared([2])
    prepared.attention_metadata["layer"].scheduler_metadata = {"object": object()}
    assert _call(prepared) is None


def test_startup_dummy_blocks_are_disjoint_and_protect_native_null_block() -> None:
    from vllm_gr.v1.worker.npu.owned_prefill_graph_runtime import StartupGraphSpec, _dummy_layout

    spec = StartupGraphSpec("ordinary_full", (2, 1), (5, 3))
    layout = _dummy_layout(spec, block_size=2, num_blocks=6)
    assert layout.block_tables == ((1, 2, 3), (4, 5))
    assert layout.positions == (3, 4, 2)
    assert layout.slots == (5, 6, 10)
    assert layout.last_block == 5
    with pytest.raises(ValueError, match="exceeds reserved KV"):
        _dummy_layout(spec, block_size=2, num_blocks=5)


def test_startup_prefill_requires_fresh_and_uniform_bsnd() -> None:
    from vllm_gr.v1.worker.npu.owned_prefill_graph_runtime import StartupGraphSpec

    with pytest.raises(ValueError, match="complete fresh prompts"):
        StartupGraphSpec("prefill_tnd", (2,), (3,))
    with pytest.raises(ValueError, match="equal unpadded"):
        StartupGraphSpec("prefill_bsnd", (2, 3), (2, 3))
