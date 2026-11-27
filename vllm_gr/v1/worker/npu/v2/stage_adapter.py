# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Translate GR logical dispatches into owned NPU V2 model forwards."""

from collections.abc import Callable
from typing import Any

import torch


def execute_v2_stage(
    owner: Any,
    plan: Any,
    metadata: Any,
    original: Callable[..., Any],
    scheduler_output: Any,
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, torch.Tensor]:
    runner = owner.worker
    execution = plan.execution
    if execution.kind == "prefill":
        result = original(runner, scheduler_output, *args, **kwargs)
        if result is not None:
            raise RuntimeError("Owned NPU V2 Prefill must publish pending hidden states")
        return result, runner.consume_prefill_logits(execution.request_ids)
    if (
        owner.latest_token_ids is None
        or owner.attention_batch is None
        or metadata.kv_blocks is None
    ):
        raise RuntimeError(
            "Owned NPU V2 Decode requires winners, session, and native prefix blocks"
        )
    if owner.attention_batch.execution != execution:
        # Fixed-width sessions retain their first execution view; their
        # logical binding is still checked by the shared owner.
        owner.attention_batch.execution.assert_same_logical_batch(execution)
    rows = execution.num_execution_rows
    ids = torch.empty(rows, dtype=torch.int32, device=runner.device)
    positions = torch.empty(rows, dtype=torch.long, device=runner.device)
    plan.prepare_decode_inputs(ids, positions, owner.latest_token_ids)
    runner.begin_decode(scheduler_output)
    hidden = runner.forward_beam_decode(
        ids, positions, execution, metadata.kv_blocks, owner.attention_batch
    )
    logits = runner.get_model().compute_logits(hidden)
    if not isinstance(logits, torch.Tensor) or logits.shape[0] != rows:
        raise RuntimeError("Owned NPU V2 Decode returned invalid physical logits rows")
    return None, logits


def retire_v2_stage(runner: Any, owner: Any, metadata: Any, scheduler_output: Any) -> None:
    """Retire only the matching owner, after all device consumers have drained."""
    binding = None if owner is None else (owner.failed_execution or owner.previous)
    if (
        owner is None
        or binding is None
        or binding.session_id != metadata.session_id
        or binding.owner_generation != metadata.owner_generation
        or owner.pending_plan is not None
    ):
        raise RuntimeError("Owned NPU V2 retirement has no matching drained owner")
    if not set(binding.request_ids).issubset(scheduler_output.finished_req_ids):
        raise ValueError("Owned NPU V2 retirement must finish every logical request")
    torch.npu.synchronize(runner.device)
    runner.abandon_pending_dispatch()
    # A failed forward may have poisoned the runner. Release only owned state;
    # do not invoke another model operation or claim the runner is recovered.
    if getattr(runner, "_failed", None) is not None:
        for req_id in binding.request_ids:
            runner.request_state.requests.pop(req_id, None)
    else:
        runner.consume_native_state(scheduler_output)
    if not owner.closed:
        owner._release()
