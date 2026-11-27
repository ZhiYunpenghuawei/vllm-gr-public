# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Exercise the audited native output seam without claiming a device upgrade."""

from dataclasses import dataclass, field

import pytest

from vllm_gr.v1.worker.npu.v2 import native_contract

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


# Reduced constructor contracts, matching the published signatures. They test
# the adapter's argument and row-order handling only; these cannot substitute
# for importing and executing the independently installed R1 runtime.
@dataclass
class Output030:
    req_ids: list[str]
    req_id_to_index: dict[str, int]
    sampled_token_ids: list[list[int]] = field(default_factory=list)


@dataclass
class Output034:
    req_ids: list[str]
    sampled_token_ids: list[list[int]] = field(default_factory=list)


@pytest.mark.parametrize(("version", "output_type"), [("0.30.0", Output030), ("0.34.0", Output034)])
def test_audited_output_constructor_preserves_indices(version, output_type):
    contract = native_contract.contract_for_version(version)
    req_ids = ["second", "first"]
    rows = [[7], [3]]
    output = contract.make_output(output_type, req_ids=req_ids, sampled_token_ids=rows)
    assert output.req_ids == req_ids
    assert output.sampled_token_ids == rows
    assert contract.output_indices(output) == {"second": 0, "first": 1}
    req_ids.reverse()
    assert output.req_ids == ["second", "first"]


def test_native_r0_output_uses_actual_installed_constructor():
    from vllm.v1.outputs import ModelRunnerOutput

    assert native_contract.active_native_contract().public_version == "0.30.0"
    output = native_contract.make_model_runner_output(req_ids=["r"], sampled_token_ids=[[123]])
    assert isinstance(output, ModelRunnerOutput)
    assert output.req_id_to_index == {"r": 0}
    assert output.sampled_token_ids == [[123]]


def test_r1_seam_does_not_enable_r1_runtime(monkeypatch):
    monkeypatch.setattr(native_contract, "version", lambda _: "0.34.0")
    assert native_contract.contract_for_version("0.34.0").public_version == "0.34.0"
    with pytest.raises(RuntimeError, match="not an enabled NPU runtime"):
        native_contract.active_native_contract()
    with pytest.raises(RuntimeError, match="not an enabled NPU runtime"):
        native_contract.make_model_runner_output(req_ids=[])


def test_local_version_suffix_does_not_change_audited_contract():
    assert native_contract.contract_for_version(
        "0.30.0+vendor"
    ) == native_contract.contract_for_version("0.30.0")
    with pytest.raises(ValueError, match="No audited"):
        native_contract.contract_for_version("0.30.1")


def test_invalid_output_rows_fail_before_construction():
    contract = native_contract.contract_for_version("0.30.0")
    with pytest.raises(ValueError, match="duplicate"):
        contract.make_output(Output030, req_ids=["r", "r"])
    with pytest.raises(ValueError, match="Sampled token rows"):
        contract.make_output(Output030, req_ids=["r", "s"], sampled_token_ids=[[1]])
    with pytest.raises(ValueError, match="derived"):
        contract.make_output(Output030, req_ids=["r"], req_id_to_index={})
    with pytest.raises(ValueError, match="does not match"):
        contract.output_indices(Output030(["r"], {"r": 1}))
