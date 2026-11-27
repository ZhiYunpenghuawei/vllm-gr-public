# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from pathlib import Path

import pytest

from tools.npu import probe_owned_v2 as probe

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def test_missing_runtime_blocks_without_importing_torch(monkeypatch):
    monkeypatch.setattr(probe, "versions", lambda: dict.fromkeys(probe.PACKAGES))
    monkeypatch.setattr(probe, "ascend_contamination", lambda: [])
    result = probe.preflight("0.30.0")
    assert result["status"] == "blocked"
    assert "missing distribution: torch-npu" in result["blockers"]
    assert result["model_runner_v2"] == "not_validated"


def test_clean_metadata_is_not_model_validation(monkeypatch):
    installed = dict.fromkeys(probe.PACKAGES)
    installed.update({"vllm": "0.30.0+vendor", "torch": "2.10.0", "torch-npu": "2.10.0.post7"})
    monkeypatch.setattr(probe, "versions", lambda: installed)
    monkeypatch.setattr(probe, "ascend_contamination", lambda: [])
    result = probe.preflight("0.30.0")
    assert result["status"] == "preflight_ready"
    assert result["device_probe"] == "not_run"
    assert result["model_runner_v2"] == "not_validated"


def test_wrong_version_and_ascend_registration_are_rejected(monkeypatch):
    installed = dict.fromkeys(probe.PACKAGES, "0.34.0")
    monkeypatch.setattr(probe, "versions", lambda: installed)
    monkeypatch.setattr(probe, "ascend_contamination", lambda: ["vllm_ascend is importable"])
    result = probe.preflight("0.30.0")
    assert result["status"] == "blocked"
    assert len(result["blockers"]) == 3


def test_existing_receipt_is_preserved(tmp_path: Path, monkeypatch):
    output = tmp_path / "receipt.json"
    output.write_text("previous evidence")
    monkeypatch.setattr(probe.sys, "argv", ["probe", "--output", str(output)])
    with pytest.raises(SystemExit):
        probe.main()
    assert output.read_text() == "previous evidence"
