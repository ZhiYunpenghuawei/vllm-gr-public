# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from types import SimpleNamespace

import pytest
import torch

from vllm_gr.v1.worker.npu.v2.stage_adapter import retire_v2_stage

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def test_retire_waits_for_device_and_removes_native_state(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(synchronize=lambda device: calls.append("fence")),
        raising=False,
    )
    runner = SimpleNamespace(
        device="npu",
        abandon_pending_dispatch=lambda: calls.append("abandon"),
        consume_native_state=lambda output: calls.append("state"),
    )
    binding = SimpleNamespace(session_id="a", owner_generation=2, request_ids=("a", "b"))
    owner = SimpleNamespace(
        previous=binding,
        failed_execution=None,
        pending_plan=None,
        closed=False,
        _release=lambda: calls.append("release"),
    )
    output = SimpleNamespace(finished_req_ids={"a", "b"})
    retire_v2_stage(runner, owner, binding, output)
    assert calls == ["fence", "abandon", "state", "release"]


def test_failed_owner_retires_without_reentering_forward(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(synchronize=lambda device: calls.append("fence")),
        raising=False,
    )
    failure = RuntimeError("device failure")
    runner = SimpleNamespace(
        device="npu",
        _failed=failure,
        request_state=SimpleNamespace(requests={"a": object()}),
        abandon_pending_dispatch=lambda: calls.append("abandon"),
        consume_native_state=lambda output: pytest.fail("must not resume failed runner"),
    )
    binding = SimpleNamespace(session_id="a", owner_generation=2, request_ids=("a",))
    owner = SimpleNamespace(
        previous=None,
        failed_execution=binding,
        pending_plan=None,
        closed=False,
        _release=lambda: calls.append("release"),
    )
    retire_v2_stage(runner, owner, binding, SimpleNamespace(finished_req_ids={"a"}))
    assert not runner.request_state.requests
    assert runner._failed is failure
    assert calls == ["fence", "abandon", "release"]


def test_stale_retirement_rejected_before_device_fence():
    binding = SimpleNamespace(session_id="a", owner_generation=2, request_ids=("a",))
    owner = SimpleNamespace(
        previous=binding, failed_execution=None, pending_plan=None, closed=False
    )
    with pytest.raises(RuntimeError, match="matching drained owner"):
        retire_v2_stage(None, owner, SimpleNamespace(session_id="a", owner_generation=1), None)
