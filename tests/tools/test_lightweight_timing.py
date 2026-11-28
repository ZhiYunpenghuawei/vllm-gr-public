# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Exercise the timing launcher's Beam probe across checkout layouts."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]
INSTRUMENTATION = Path(__file__).resolve().parents[2] / "tools" / "AB_Tests" / "instrumentation"


@pytest.mark.parametrize(  # type: ignore[misc]
    "legacy_ascend_present", [False, True], ids=["legacy-parent-absent", "legacy-ascend-present"]
)
def test_beam_worker_decision_probe(tmp_path: Path, legacy_ascend_present: bool) -> None:
    checkout = tmp_path / "checkout"
    files = {
        "vllm/__init__.py": "",
        "vllm_gr/__init__.py": "",
        "vllm_gr/v1/__init__.py": "",
        "vllm_gr/v1/beam/__init__.py": "",
        "vllm_gr/v1/worker/__init__.py": "",
        "vllm_gr/v1/beam/beam_decision.py": (
            "def run_beam_decision_candidates():\n    return 'legacy'\n"
        ),
    }
    if legacy_ascend_present:
        files.update(
            {
                "vllm_gr/v1/worker/npu/__init__.py": "",
                "vllm_gr/v1/worker/npu/beam_decision.py": "raise AssertionError('old backend must not import')\n",
            }
        )
    for name, source in files.items():
        path = checkout / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")

    output = tmp_path / "timing"
    completed = subprocess.run(
        [
            sys.executable,
            "-S",  # Skip installed .pth hooks in this synthetic runtime.
            "-c",
            "from os import getpid\n"
            "import sitecustomize\n"
            "from vllm_gr.v1.beam.beam_decision import run_beam_decision_candidates\n"
            "assert run_beam_decision_candidates() == 'legacy'\n"
            "print(getpid())\n",
        ],
        cwd=tmp_path,
        env={
            "PYTHONPATH": os.pathsep.join((str(checkout), str(INSTRUMENTATION))),
            "VLLM_GR_LIGHTWEIGHT_TIMING": "1",
            "VLLM_GR_LIGHTWEIGHT_TIMING_DIR": str(output),
        },
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert not completed.stderr
    receipt = output / f"cpu-timing-{int(completed.stdout)}.json"
    timing = json.loads(receipt.read_text(encoding="utf-8"))
    assert timing["metrics"]["beam_worker_decision"]["count"] == 1
