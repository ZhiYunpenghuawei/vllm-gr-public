# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Host checks for selecting and identifying a clean, owned NPU runtime."""

from importlib.util import module_from_spec, spec_from_file_location
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.helpers.paths import REPO_ROOT
from vllm_gr.platforms.ascend import environment

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


@pytest.fixture
def resolver(monkeypatch):
    spec = spec_from_file_location(
        "_owned_runtime_resolver", REPO_ROOT / "tools/ci/resolve_runtime.py"
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "package_version", lambda *_: "0.30.0")
    monkeypatch.setattr(module, "source_digest", lambda _: "source")
    return module


def test_owned_runtime_spec_excludes_ascend_but_keeps_triton_pin(resolver, monkeypatch):
    packages = {"vllm": "0.30.0", "triton-ascend": "3.6.0", "torch-npu": "2.10.0.post7"}
    monkeypatch.setattr(resolver, "inventory", lambda: packages.copy())
    monkeypatch.setenv("VLLM_GR_OWNED_NPU", "1")
    owned = resolver.resolve(REPO_ROOT, "npu")
    assert owned["owned_npu"] is True
    assert "vllm-ascend" not in owned["expected"]
    assert owned["expected"]["triton-ascend"] == packages["triton-ascend"]
    # A legacy environment must still provide the old validated pin.
    monkeypatch.delenv("VLLM_GR_OWNED_NPU")
    with pytest.raises(ValueError, match="vllm-ascend"):
        resolver.resolve(REPO_ROOT, "npu")
    packages["vllm-ascend"] = "0.30.0rc2"
    legacy = resolver.resolve(REPO_ROOT, "npu")
    assert legacy["expected"]["vllm-ascend"] == "0.30.0rc2"
    assert legacy["cache_key"] != owned["cache_key"]
    # Presence of the old package is an error even at the correct version.
    monkeypatch.setenv("VLLM_GR_OWNED_NPU", "1")
    with pytest.raises(ValueError, match="must not contain vllm-ascend"):
        resolver.resolve(REPO_ROOT, "npu")


@pytest.fixture
def environment_probe(monkeypatch):
    versions = {"vllm": "0.30.0", "triton-ascend": "3.6.0", "torch-npu": "2.10.0.post7"}
    modules = {}
    monkeypatch.setattr(environment, "sys", SimpleNamespace(modules=modules))
    monkeypatch.setattr(environment, "check_vllm_runtime", lambda: {"vllm": versions["vllm"]})
    monkeypatch.setattr(environment, "installed_version", versions.get)
    pins = Mock(
        side_effect=lambda name, expected: versions[name] if versions[name] == expected else None
    )
    imports = Mock(return_value=SimpleNamespace(__file__="installed/module.py"))
    finder = Mock(return_value=None)
    monkeypatch.setattr(environment, "require_runtime_version", pins)
    monkeypatch.setattr(environment, "import_module", imports)
    monkeypatch.setattr(environment, "find_spec", finder)
    return SimpleNamespace(
        versions=versions, pins=pins, imports=imports, finder=finder, modules=modules
    )


def test_owned_environment_checks_no_ascend_import(environment_probe, monkeypatch):
    monkeypatch.setenv("VLLM_GR_OWNED_NPU", "1")
    report = environment.check_ascend_environment()
    assert "vllm-ascend" not in report
    assert report["torch-npu"] == "2.10.0.post7"
    environment_probe.pins.assert_called_once_with("triton-ascend", "3.6.0")
    assert [call.args[0] for call in environment_probe.imports.call_args_list] == [
        "torch_npu",
        "triton",
        "triton._C.libtriton.ascend",
    ]
    environment_probe.finder.assert_called_once_with("vllm_ascend")


@pytest.mark.parametrize("presence", ["metadata", "source checkout", "loaded"])
def test_owned_environment_rejects_old_package(environment_probe, monkeypatch, presence):
    monkeypatch.setenv("VLLM_GR_OWNED_NPU", "1")
    if presence == "metadata":
        environment_probe.versions["vllm-ascend"] = "0.30.0rc2"
    elif presence == "source checkout":
        environment_probe.finder.return_value = object()
    else:
        environment_probe.modules["vllm_ascend.worker"] = object()
    with pytest.raises(RuntimeError, match="without vLLM-Ascend"):
        environment.check_ascend_environment()
    environment_probe.imports.assert_not_called()


def test_legacy_environment_keeps_exact_vendor_pin(environment_probe, monkeypatch):
    monkeypatch.delenv("VLLM_GR_OWNED_NPU", raising=False)
    environment_probe.versions["vllm-ascend"] = "0.30.0rc2"
    report = environment.check_ascend_environment()
    assert report["vllm-ascend"] == "0.30.0rc2"
    environment_probe.pins.assert_any_call("vllm-ascend", "0.30.0rc2")
    environment_probe.finder.assert_not_called()


def test_owned_environment_rejects_mapped_ascend_binary(tmp_path, monkeypatch):
    from pathlib import Path

    maps = tmp_path / "maps"
    maps.write_text("0000-0001 r-xp 00000000 00:00 1 /vendor/vllm_ascend/_C_ascend.so\n")
    monkeypatch.setattr(
        environment, "Path", lambda path: maps if path == "/proc/self/maps" else Path(path)
    )
    with pytest.raises(RuntimeError, match="binary mapped"):
        environment.assert_no_ascend_runtime_loaded()
