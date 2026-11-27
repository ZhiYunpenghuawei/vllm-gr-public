# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Host-side contracts for the owned NPU V2 eager runner.

These tests exercise logical state and physical input construction only. They
do not establish model, allocator, or attention support on actual hardware.
"""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from vllm_gr.v1.worker.npu.v2.model_runner import ExecuteModelState, InputBatch, NPUModelRunnerV2
from vllm_gr.v1.worker.npu.v2.request_state import NativeRequestState, RequestState

pytestmark = [pytest.mark.l1, pytest.mark.cpu_test]


def _new(req_id: str, tokens: Any, blocks: tuple[int, ...], computed: int = 0) -> Any:
    return SimpleNamespace(
        req_id=req_id,
        prompt_token_ids=tokens,
        prefill_token_ids=tokens,
        block_ids=(list(blocks),),
        num_computed_tokens=computed,
        mm_features=[],
        prompt_embeds=None,
        lora_request=None,
        pooling_params=None,
        sampling_params=object(),
    )


def _output(*, new=(), counts=None, cached=(), finished=(), table_updates=None) -> Any:
    return SimpleNamespace(
        scheduled_new_reqs=list(new),
        num_scheduled_tokens=counts or {},
        total_num_scheduled_tokens=sum((counts or {}).values()),
        scheduled_cached_reqs=SimpleNamespace(
            resumed_req_ids=set(),
            req_ids=[row[0] for row in cached],
            num_computed_tokens=[row[1] for row in cached],
            new_block_ids=[row[2] for row in cached],
        ),
        finished_req_ids=set(finished),
        preempted_req_ids=None,
        block_table_updates=table_updates,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        kv_connector_metadata=None,
        ec_connector_metadata=None,
        has_structured_output_requests=False,
        new_block_ids_to_zero=None,
        kv_cache_block_copies=None,
    )


def _apply(state: NativeRequestState, output: Any) -> None:
    state.publish(state.prepare(output, block_size=4, num_blocks=16))


def test_prefill_state_is_detached_and_advances_once() -> None:
    state = NativeRequestState(2, 20)
    prompt = np.array([7, 8, 9, 10, 11], dtype=np.int64)
    frame = _output(new=[_new("a", prompt, (1, 2), 4)], counts={"a": 1})
    _apply(state, frame)
    prompt[:] = 0
    frame.scheduled_new_reqs[0].block_ids[0][:] = [14, 15]
    assert state.requests["a"] == RequestState((7, 8, 9, 10, 11), (1, 2), 4)
    state.advance({"a": 4}, {"a": 1})
    assert state.requests["a"].computed == 5
    with pytest.raises(RuntimeError, match="duplicate"):
        state.advance({"a": 4}, {"a": 1})


def test_cached_native_blocks_and_retirement_are_transactional() -> None:
    state = NativeRequestState(2, 20)
    _apply(state, _output(new=[_new("a", [1, 2, 3, 4], (1,), 0)], counts={"a": 4}))
    state.advance({"a": 0}, {"a": 4})
    before = state.requests.copy()
    invalid = _output(cached=[("a", 4, ([99],))], counts={"a": 1})
    with pytest.raises(ValueError, match="outside allocated"):
        _apply(state, invalid)
    assert state.requests == before
    valid = _output(cached=[("a", 4, ([99],))], counts={"a": 1}, table_updates={"a": ([1, 2],)})
    _apply(state, valid)
    assert state.requests["a"].blocks == (1, 2)
    state.advance({"a": 4}, {"a": 1})
    assert state.requests["a"].computed == 5
    _apply(state, _output(finished=["a"]))
    _apply(state, _output(finished=["a"]))
    assert state.requests == {}


def test_fail_closed_before_native_state_mutation() -> None:
    state = NativeRequestState(1, 20)
    _apply(state, _output(new=[_new("a", [1, 2], (1,))], counts={"a": 2}))
    before = state.requests.copy()
    frame = _output(new=[_new("b", [3], (2,))], counts={"b": 1})
    with pytest.raises(ValueError, match="capacity"):
        _apply(state, frame)
    assert state.requests == before
    frame = _output(cached=[("a", 1, None)], counts={"a": 1})
    with pytest.raises(RuntimeError, match="disagrees"):
        _apply(state, frame)
    assert state.requests == before


def test_prefill_inputs_follow_cached_prefix_and_per_request_logit_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A CPU stand-in verifies the pure tensor/index preparation algorithm.
    # Constructing and executing the production runner requires NPU.
    runner = object.__new__(NPUModelRunnerV2)
    runner.device = torch.device("cpu")
    runner._cache_spec = SimpleNamespace(block_size=4)
    runner.request_state = NativeRequestState(2, 20)
    frame = _output(
        new=[_new("a", [1, 2, 3, 4, 5, 6], (2, 3), 4), _new("b", [7, 8, 9], (4,))],
        counts={"b": 3, "a": 2},
    )
    _apply(runner.request_state, frame)
    captured: dict[str, Any] = {}

    def prepare(*args: Any, **kwargs: Any) -> Any:
        captured["args"] = args
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(runner, "_prepare_forward", prepare)
    runner.prepare_inputs(frame)
    ids, positions, query_lengths, sequence_lengths, blocks, slots = captured["args"]
    assert ids.tolist() == [7, 8, 9, 5, 6]
    assert positions.tolist() == [0, 1, 2, 4, 5]
    assert query_lengths == (3, 2)
    assert sequence_lengths == (3, 6)
    assert blocks == ((4,), (2, 3))
    assert slots.tolist() == [16, 17, 18, 12, 13]
    assert slots.dtype is torch.int32
    batch = captured["input_batch"]
    assert batch.req_ids == ("b", "a")
    assert batch.logits_indices.tolist() == [2, 4]
    assert runner.request_state.requests["a"].computed == 4


def test_pending_prefill_logits_restore_gr_order_and_commit_counts() -> None:
    runner = object.__new__(NPUModelRunnerV2)
    runner.device = torch.device("cpu")
    runner._closed = False
    runner._failed = None
    runner.model = SimpleNamespace(compute_logits=lambda hidden: hidden + 1)
    runner.request_state = NativeRequestState(2, 20)
    _apply(
        runner.request_state,
        _output(new=[_new("a", [1, 2], (1,)), _new("b", [3], (2,))], counts={"a": 2, "b": 1}),
    )
    batch = InputBatch(("a", "b"), torch.tensor([1, 2]), {"a": 0, "b": 0}, {"a": 2, "b": 1})
    runner.execute_model_state = ExecuteModelState(torch.tensor([[10], [20], [30]]), batch)
    with pytest.raises(ValueError, match="membership"):
        runner.consume_prefill_logits(("a", "a"))
    result = runner.consume_prefill_logits(("b", "a"))
    assert result.tolist() == [[31], [21]]
    assert runner.execute_model_state is None
    assert runner.request_state.requests["a"].computed == 2
    assert runner.request_state.requests["b"].computed == 1
    with pytest.raises(RuntimeError, match="No owned"):
        runner.consume_prefill_logits(("a", "b"))
