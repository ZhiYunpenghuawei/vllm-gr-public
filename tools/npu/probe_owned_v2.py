# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Collect a version-bound preflight and execute an optional NPU eager probe.

This is a prerequisite probe, not evidence of a working ModelRunner V2.
It never installs packages, downloads models, or selects an unassigned device.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

PACKAGES = ("vllm", "vllm-gr", "torch", "torch-npu", "triton", "triton-ascend", "vllm-ascend")
REQUIRED = ("vllm", "torch", "torch-npu")


def versions() -> dict[str, str | None]:
    found = {}
    for name in PACKAGES:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
    return found


def ascend_contamination() -> list[str]:
    """Detect old Python package and mapped libraries, without importing it."""
    problems = []
    if importlib.util.find_spec("vllm_ascend") is not None:
        problems.append("vllm_ascend is importable")
    if any(name == "vllm_ascend" or name.startswith("vllm_ascend.") for name in sys.modules):
        problems.append("vllm_ascend is loaded")
    maps = Path("/proc/self/maps")
    if maps.is_file():
        for line in maps.read_text(encoding="utf-8").splitlines():
            path = line.split(maxsplit=5)[-1]
            if path.startswith("/") and ("/vllm_ascend/" in path or "_C_ascend" in Path(path).name):
                problems.append(f"Ascend library mapped: {path}")
    return sorted(set(problems))


def preflight(expected_vllm: str) -> dict[str, Any]:
    installed = versions()
    blockers = [f"missing distribution: {name}" for name in REQUIRED if installed[name] is None]
    if installed["vllm"] and installed["vllm"].partition("+")[0] != expected_vllm:
        blockers.append(f"expected vLLM {expected_vllm}, found {installed['vllm']}")
    if installed["vllm-ascend"] is not None:
        blockers.append("vllm-ascend is installed")
    blockers.extend(ascend_contamination())
    return {
        "schema_version": 1,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "machine": platform.machine(),
        "expected_vllm": expected_vllm,
        "packages": installed,
        "status": "blocked" if blockers else "preflight_ready",
        "blockers": blockers,
        "model_runner_v2": "not_validated",
        "device_probe": "not_run",
    }


def device_probe(index: int) -> dict[str, Any]:
    """Test actual dense eager arithmetic and device transfers on one assigned NPU."""
    import torch
    import torch_npu  # noqa: F401 - registers the NPU backend

    if not torch.npu.is_available() or not 0 <= index < torch.npu.device_count():
        raise RuntimeError(f"Assigned NPU {index} is unavailable")
    device = torch.device("npu", index)
    torch.npu.set_device(device)
    # Exactly representable fixed inputs give a deterministic CPU comparison.
    x = torch.tensor([[1.0, -2.0, 3.0, 4.0]], dtype=torch.float32)
    weight = torch.tensor([[1.0, 2.0, 0.0, -1.0], [0.0, 1.0, 2.0, 1.0]])
    checked = []
    for dtype in (torch.float16, torch.bfloat16):
        host_x, host_w = x.to(dtype), weight.to(dtype)
        gpu_x, gpu_w = host_x.to(device), host_w.to(device)
        actual = torch.nn.functional.linear(gpu_x, gpu_w)
        torch.npu.synchronize(device)
        expected = torch.nn.functional.linear(host_x.float(), host_w.float()).to(dtype)
        torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
        checked.append(str(dtype))
    pollution = ascend_contamination()
    if pollution:
        raise RuntimeError("; ".join(pollution))
    return {
        "status": "passed",
        "device": str(device),
        "device_name": torch.npu.get_device_name(device),
        "checked_dtypes": checked,
        "operation": "dense_linear",
        "model_runner_v2": "not_validated",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-version", default="0.30.0")
    parser.add_argument(
        "--device", type=int, help="Explicit assigned NPU index; omit for metadata-only preflight"
    )
    parser.add_argument("--output", required=True, type=Path, help="New JSON receipt path")
    args = parser.parse_args()
    if args.device is not None and args.device < 0:
        parser.error("--device must be nonnegative")
    if args.output.exists():
        parser.error("use a fresh receipt path")
    report = preflight(args.vllm_version)
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    report["gr_revision"] = result.stdout.strip() if result.returncode == 0 else None
    if args.device is not None and not report["blockers"]:
        try:
            report["device_probe"] = device_probe(args.device)
        except Exception as exc:
            report["device_probe"] = "failed"
            report["blockers"].append(f"device probe: {type(exc).__name__}: {exc}")
            report["status"] = "blocked"
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(bool(report["blockers"]))


if __name__ == "__main__":
    raise SystemExit(main())
