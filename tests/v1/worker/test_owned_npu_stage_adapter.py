# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from types import SimpleNamespace

import pytest
import torch

from vllm_gr.v1.worker.npu.npu_beam_stage_adapter import NPUEagerStagePlan
from vllm_gr.v1.worker.npu.v2.stage_adapter import execute_v2_stage

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def test_prefill_preserves_native_pending_and_order():
    calls = []
    logits = torch.ones(2, 4)
    runner = SimpleNamespace(consume_prefill_logits=lambda ids: calls.append(ids) or logits)
    owner = SimpleNamespace(worker=runner)
    execution = SimpleNamespace(kind="prefill", request_ids=("b", "a"))
    plan = SimpleNamespace(execution=execution)
    output = SimpleNamespace(num_scheduled_tokens={"b": 3, "a": 5})

    def original(worker, scheduler_output):
        assert worker is runner and scheduler_output is output
        calls.append("execute")

    result, actual = execute_v2_stage(owner, plan, None, original, output)
    assert result is None and actual is logits
    assert calls == ["execute", ("b", "a")]


def test_decode_rows_expand_without_changing_logical_scheduler_counts():
    calls = []
    execution = SimpleNamespace(kind="decode", num_execution_rows=4, batch_size=2, input_width=2)
    plan = NPUEagerStagePlan(execution, (1, 1), 4, (0, 1, 2, 3), (0, 0, 1, 1), (5, 5, 8, 8))
    batch = SimpleNamespace(execution=execution)
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"a": 1, "b": 1}, total_num_scheduled_tokens=2
    )
    kv_blocks = ((1,), (2,))

    def forward(ids, positions, view, blocks, session):
        assert view is execution and blocks == kv_blocks and session is batch
        assert ids.tolist() == [11, 12, 21, 22]
        assert positions.tolist() == [5, 5, 8, 8]
        calls.append("forward")
        return ids.float().reshape(4, 1)

    runner = SimpleNamespace(
        device=torch.device("cpu"),
        begin_decode=lambda output: calls.append(output),
        forward_beam_decode=forward,
        get_model=lambda: SimpleNamespace(compute_logits=lambda x: x),
    )
    owner = SimpleNamespace(
        worker=runner, latest_token_ids=torch.tensor([[11, 12], [21, 22]]), attention_batch=batch
    )
    result, logits = execute_v2_stage(
        owner,
        plan,
        SimpleNamespace(kv_blocks=kv_blocks),
        lambda *args: pytest.fail("V2 Decode must not call native Prefill"),
        scheduler_output,
    )
    assert result is None
    assert logits[:, 0].tolist() == [11, 12, 21, 22]
    assert calls == [scheduler_output, "forward"]
    assert scheduler_output.num_scheduled_tokens == {"a": 1, "b": 1}
    assert scheduler_output.total_num_scheduled_tokens == 2


def test_missing_prefix_blocks_rejected_before_native_state():
    owner = SimpleNamespace(
        worker=SimpleNamespace(), latest_token_ids=torch.ones(1), attention_batch=object()
    )
    plan = SimpleNamespace(execution=SimpleNamespace(kind="decode"))
    with pytest.raises(RuntimeError, match="native prefix blocks"):
        execute_v2_stage(owner, plan, SimpleNamespace(kv_blocks=None), None, None)


def test_owned_runner_installation_is_idempotent():
    from vllm_gr.v1.worker.npu.v2 import runner_installer

    runner_installer.install()
    _, cls = runner_installer.discover_worker_runners()[0]
    execute = cls.execute_model
    sample = cls.sample_tokens
    assert runner_installer.is_patched(cls)
    runner_installer.install()
    assert cls.execute_model is execute
    assert cls.sample_tokens is sample
