# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Subprocess checks for backend infrastructure import isolation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.helpers.paths import REPO_ROOT


@pytest.mark.l1  # type: ignore[misc]
def test_package_import_defers_runtime_checks_and_runs_opencv_before_grllm() -> None:
    code = """
import importlib.metadata
import sys
from types import ModuleType
from unittest.mock import patch


def unexpected_version(distribution):
    raise AssertionError(f"unexpected distribution lookup for {distribution}")


with patch.object(importlib.metadata, "version", unexpected_version):
    import vllm_gr

assert "vllm" not in sys.modules
assert "torch" not in sys.modules
assert "GRLLM" in dir(vllm_gr)
assert "GRLLM" not in vars(vllm_gr)

from vllm_gr import _env_check

with patch.object(_env_check, "check_opencv_env", side_effect=RuntimeError("bad ABI")):
    try:
        vllm_gr.GRLLM
    except RuntimeError as exc:
        assert str(exc) == "bad ABI"
    else:
        raise AssertionError("preflight failure should stop the lazy import")
assert "vllm_gr.entrypoints.llm" not in sys.modules
assert "GRLLM" not in vars(vllm_gr)

frontend = ModuleType("vllm_gr.entrypoints.llm")
frontend.GRLLM = object()
sys.modules[frontend.__name__] = frontend
with patch.object(_env_check, "check_opencv_env") as preflight:
    assert vllm_gr.GRLLM is frontend.GRLLM
    assert vllm_gr.GRLLM is frontend.GRLLM
    preflight.assert_called_once_with()
"""
    subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )


def test_platform_package_and_builtins_do_not_load_hardware_modules() -> None:
    project_root = REPO_ROOT
    code = """
import sys

import vllm_gr.platforms

assert "vllm_gr.config.parser" not in sys.modules
assert "vllm_gr.config.models" not in sys.modules

for module_name in (
    "torch",
    "torch_npu",
    "vllm",
    "vllm_ascend",
    "vllm_gr.v1.worker.cuda.v1.gpu_model_runner_patch",
    "vllm_gr.v1.worker.npu.npu_model_runner_patch",
    "vllm_gr.v1.attention.backends.beam_attn_gpu",
    "vllm_gr.v1.attention.backends.beam_attn_npu",
    "vllm_gr.platforms.ascend.config_patch",
    "vllm_gr.v1.worker.npu.startup_patch",
    "vllm_gr.v1.worker.npu.kv_layout_patch",
):
    assert module_name not in sys.modules, module_name

from vllm_gr.platforms.table import BUILTIN_PROBES
assert set(BUILTIN_PROBES) == {"cuda"}
assert "torch" not in sys.modules and "vllm" not in sys.modules

from tests.helpers.native_platform import native_platform
import vllm.platforms
vllm.platforms.current_platform = native_platform("CUDA", "cuda")
from vllm_gr.platforms.table import current_gr_platform
assert current_gr_platform().backend_id == "cuda"
assert "vllm_gr.platforms.ascend.platform" not in sys.modules
"""

    subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.l1  # type: ignore[misc]
def test_obsolete_installed_entry_point_cannot_replace_native_cuda(tmp_path: Path) -> None:
    # Reproduce an environment that still contains a distribution from the
    # retired plugin system. Its broken import must be irrelevant to selection.
    dist = tmp_path / "stale_gr_plugin-1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Name: stale-gr-plugin\nVersion: 1.0\n")
    (dist / "entry_points.txt").write_text(
        "[vllm_gr.platform_backends]\ncuda = missing_plugin:replacement\n"
    )
    code = """
import os
import sys
from importlib import metadata
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, sys.argv[1])
assert any(entry.value == "missing_plugin:replacement" for entry in
           metadata.entry_points(group="vllm_gr.platform_backends"))
from tests.helpers.native_platform import native_platform
import vllm.platforms as native
native.current_platform = native_platform("CUDA","cuda")
os.environ["VLLM_GR_BACKEND"] = "old-replacement"
os.environ["VLLM_GR_DISABLE_BUILTIN_BACKENDS"] = "cuda"

installed_entry_points = metadata.entry_points
def current_group_only(*, group):
    assert group == "vllm_gr.platform_plugins", group
    return installed_entry_points(group=group)
with patch.object(metadata, "entry_points", side_effect=current_group_only):
    from vllm_gr.platforms.table import current_gr_platform

    assert type(current_gr_platform()).__module__ == "vllm_gr.platforms.cuda.platform"
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.l1  # type: ignore[misc]
def test_cuda_plugin_initialization_does_not_import_ascend() -> None:
    project_root = REPO_ROOT
    code = """
import importlib.abc
import os
import sys

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["VLLM_PLUGINS"] = ""
vendor_packages = (
    "torch_npu",
    "vllm_ascend",
)
retired_gr_modules = (
    "vllm_gr.v1.ascend_compat",
    "vllm_gr.platforms.ascend",
    "vllm_gr.v1.worker.npu",
    "vllm_gr.ops.npu",
    "vllm_gr.v1.attention.backends.beam_attn_npu",
)

def forbidden(name):
    return any(
        name == mod or name.startswith(mod + ".")
        for mod in vendor_packages + retired_gr_modules
    )

# Upstream dependencies can legitimately probe whether optional vendor packages
# are installed. A find_spec call is not an import. Reject actual import events
# and check sys.modules after initialization, while retaining the import finder
# guard for the removed GR modules.
def reject_vendor_import(event, args):
    if event == "import" and forbidden(args[0]):
        raise AssertionError(f"retired backend imported: {args[0]}")

class RejectLegacyAscend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(
            fullname == mod or fullname.startswith(mod + ".") for mod in retired_gr_modules
        ):
            raise AssertionError(f"retired backend imported: {fullname}")

sys.addaudithook(reject_vendor_import)
sys.meta_path.insert(0, RejectLegacyAscend())
import vllm.platforms
from tests.helpers.cuda_platform import CudaGRPlatform
from vllm.platforms.cuda import CudaPlatform

# The real CUDA class provides concrete methods needed during vLLM startup.
platform = CudaPlatform()
assert isinstance(platform, CudaGRPlatform.native_platform_type)
if not __import__("torch").cuda.is_available():
    # No device is visible on host-only builds. Stub the capability on the
    # class: initialize_runtime installs the GR platform instance as
    # current_platform, and that instance inherits the stub through the MRO.
    CudaPlatform.get_device_capability = classmethod(
        lambda *args, **kwargs: None
    )
vllm.platforms.current_platform = platform

from vllm_gr.plugin import initialize_runtime
first = initialize_runtime()
assert first["patch_count"] > 0, first
assert first["platform_seam"] is True, first
assert set(first["patches"].values()) == {"applied"}, first
second = initialize_runtime()
assert second["patch_count"] == 0, second
assert second["platform_seam"] is False, second
assert set(second["patches"].values()) == {"already_applied"}, second
assert not any(forbidden(name) for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "module_name",
    [
        "vllm_gr.v1.worker.cuda.gpu_postprocess_workspace",
        "vllm_gr.v1.worker.cuda.v2.gpu_model_runner_v2_patch",
        "vllm_gr.v1.worker.model_runner_patch",
    ],
)
def test_shared_workspace_and_v2_do_not_import_v1_integration(module_name: str) -> None:
    project_root = REPO_ROOT
    code = """
import importlib
import importlib.abc
import os
import sys
from unittest.mock import Mock

blocked_packages = ("vllm_gr.v1.worker.cuda.v1", "vllm_gr.v1.worker.npu")

def forbidden(name):
    return name == "vllm_gr.v1.worker.model_runner_common" or any(
        name == package or name.startswith(package + ".") for package in blocked_packages
    )

class RejectInactiveIntegration(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if forbidden(fullname):
            raise AssertionError(f"unexpected integration dependency: {fullname}")

sys.meta_path.insert(0, RejectInactiveIntegration())
module_name = sys.argv[1]
if module_name == "vllm_gr.v1.worker.model_runner_patch":
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    import vllm.platforms
    from vllm.platforms.interface import Platform

    from tests.helpers.native_platform import native_platform
    platform = native_platform("CUDA", "cuda")
    vllm.platforms.current_platform = platform

    from vllm_gr.v1.worker.cuda import runner_installer as runner
    from vllm_gr.v1.worker import model_runner_patch
    class Worker:
        vllm_config = None
        use_v2_model_runner = True
        def init_device(self):
            pass

    runner.discover_worker_runners = lambda: (("CUDA Worker.init_device", Worker),)
    from vllm_gr.v1.worker.cuda.v2 import gpu_model_runner_v2_patch as v2
    native_class = type("V2", (), {})
    v2.discover_worker_runners = lambda: (("V2", native_class),)
    v2.is_patched = lambda cls: bool(getattr(cls, "_installed", False))
    install_v2 = Mock(side_effect=lambda: setattr(native_class, "_installed", True))
    v2._apply_v2_runner_patches = install_v2
    runner.install_worker_hooks = Mock()
    model_runner_patch.apply_worker_patches()
    install_v2.assert_not_called()
    Worker().init_device()
    install_v2.assert_called_once_with()
else:
    importlib.import_module(module_name)
assert not any(forbidden(name) for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", code, module_name],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )


pytestmark = pytest.mark.cpu_test


@pytest.mark.l1  # type: ignore[misc]
def test_runner_contracts_import_stays_hardware_free() -> None:
    code = """
import sys
from vllm_gr.v1.worker import cuda, model_runner_contracts, stage_binding
from vllm_gr.v1.worker.cuda import v1, v2

for package in (cuda, v1, v2):
    assert not any(
        name.startswith(package.__name__ + ".")
        and (package is not cuda or name not in (v1.__name__, v2.__name__))
        for name in sys.modules
    )
assert callable(model_runner_contracts.is_worker_runner_patched)
assert callable(stage_binding.StageBinding)
for name in ("torch", "vllm", "torch_npu", "vllm_ascend",
             "vllm_gr.v1.worker.model_runner_common",
             "vllm_gr.v1.worker.cuda.v2.gpu_model_runner_v2_patch"):
    assert name not in sys.modules, name
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )


@pytest.mark.l1  # type: ignore[misc]
def test_shared_execution_imports_no_platform_implementation() -> None:
    code = """
import importlib
import os
import sys

# Exercise GR imports independently of third-party plugin autoload on the host.
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["VLLM_PLUGINS"] = ""

for module in (
    "vllm_gr.config.models",
    "vllm_gr.platforms.admission",
    "vllm_gr.constraints.loader",
    "vllm_gr.constraints.execution",
    "vllm_gr.ops.onerec_decode",
    "vllm_gr.ops.rec_constrained_topk",
    "vllm_gr.v1.beam.attention_context",
    "vllm_gr.v1.beam.reorder",
    "vllm_gr.v1.worker.gr_execution",
    "vllm_gr.v1.worker.model_runner_common",
    "vllm_gr.v1.worker.model_runner_patch",
    "vllm_gr.v1.worker.gr_startup",
    "vllm_gr.v1.worker.runner_binding",
    "vllm_gr.v1.engine.worker_readiness",
    "vllm_gr.v1.engine.hot_update",
):
    importlib.import_module(module)

unexpected = [
    name for name in sys.modules
    if name.startswith(("vllm_gr.platforms.cuda", "vllm_gr.ops.cuda", "vllm_gr.constraints.cuda_",
                        "vllm_gr.v1.worker.cuda", "vllm_gr.v1.attention.backends.beam_attn_gpu",
                        "vllm_gr.platforms.ascend", "vllm_gr.v1.worker.npu", "vllm_gr.ops.npu",
                        "torch_npu", "vllm_ascend"))
]
assert not unexpected, unexpected
"""
    subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True, timeout=120)
