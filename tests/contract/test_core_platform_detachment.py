# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""GR core remains usable without importing the former vendor runtime."""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def test_core_import_does_not_load_legacy_ascend_runtime() -> None:
    # Check a fresh process so previously imported modules cannot hide an
    # accidental dependency. The core package must not initialize a device.
    script = """
import importlib.abc
import sys

class BlockVendorImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'vllm_ascend', 'torch_npu', 'custom_ops_lib'}:
            raise AssertionError(f'Unexpected vendor runtime import: {fullname}')
        return None

sys.meta_path.insert(0, BlockVendorImports())
import vllm_gr
from vllm_gr.config import GRConfig
assert vllm_gr.__version__
assert GRConfig(beam_max_width=8).beam_max_width == 8
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        timeout=30,
    )


def test_legacy_backend_is_not_part_of_core_source() -> None:
    package = Path(__file__).resolve().parents[2] / "vllm_gr"
    for relative in (
        "platforms/ascend/platform.py",
        "ops/npu/beam_ops.py",
        "v1/worker/npu/npu_model_runner_patch.py",
        "v1/ascend_compat.py",
        "constraints/npu_backend.py",
        "constraints/npu_tables.py",
    ):
        assert not (package / relative).exists(), relative


def test_shared_constraint_and_beam_interfaces_do_not_load_cuda_kernels() -> None:
    pytest.importorskip("torch")
    script = """
import importlib.abc
import sys
import torch

class BlockDeviceImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'vllm_ascend', 'torch_npu', 'triton'}:
            raise AssertionError(f'Unexpected device-specific runtime import: {fullname}')
        if fullname.startswith(('vllm_gr.ops.cuda', 'vllm_gr.constraints.cuda_backend')):
            raise AssertionError(f'Unexpected CUDA implementation import: {fullname}')
        return None

sys.meta_path.insert(0, BlockDeviceImports())
from vllm_gr.constraints import ConstraintOutput, load_constraint_backend
from vllm_gr.v1.beam.dynamic_width import DynamicBeamState
assert callable(load_constraint_backend)
assert ConstraintOutput is not None and DynamicBeamState is not None
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        timeout=30,
    )
