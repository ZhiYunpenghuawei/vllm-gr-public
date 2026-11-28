# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Contracts for the built-in CUDA integration."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest import mock

import pytest

from tests.helpers.cuda_platform import CudaGRPlatform

cuda = CudaGRPlatform()


def test_cuda_integration_lazily_returns_platform_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envs = ModuleType("vllm.envs")
    setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", False)
    monkeypatch.setitem(sys.modules, "vllm.envs", envs)
    from vllm_gr.v1.worker.cuda import runner_installer as runner

    gpu_attention = object()
    imported: list[str] = []

    def import_module(name: str) -> object:
        imported.append(name)
        return {
            "vllm_gr.v1.attention.backends.beam_attn_gpu": gpu_attention,
        }[name]

    monkeypatch.setattr(f"{CudaGRPlatform.__module__}.import_module", import_module)

    assert cuda.get_model_runner_hooks() is runner
    assert cuda.get_attention_execution() is gpu_attention
    assert imported == [
        "vllm_gr.v1.attention.backends.beam_attn_gpu",
    ]


def test_cuda_runtime_patch_is_noop() -> None:
    assert cuda.apply_runtime_patches() == 0
    assert cuda.apply_runtime_patches(force=True) == 0


@pytest.mark.parametrize(  # type: ignore[misc]
    ("field", "value"),
    [
        ("dense_prefill", "vendor"),
        ("topk_select", "vendor"),
        ("topk_keep", 257),
        ("unconstrained_head_layout", "vendor"),
        ("prefix_policy", "vendor"),
        ("prefill_flash_attn_version", 4),
    ],
)
def test_cuda_algorithm_choices_are_validated_at_backend_admission(
    field: str, value: object
) -> None:
    from vllm_gr.config import GRConfig
    from vllm_gr.config.models import DeploymentOptions

    # The shared configuration parses data without selecting a device.
    if field == "prefill_flash_attn_version":
        config = GRConfig(prefill_flash_attn_version=value)
    else:
        config = GRConfig(deployment=DeploymentOptions.from_mapping({field: value}))
    with pytest.raises(ValueError, match=field):
        cuda.create_config_service().validate_engine(None, config)


def test_cuda_environment_check_validates_platform_and_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_gr.platforms import compat

    platforms = ModuleType("vllm.platforms")
    from tests.helpers.native_platform import native_platform

    monkeypatch.setattr(
        platforms, "current_platform", native_platform("CUDA", "cuda"), raising=False
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)

    report = {"vllm": "0.30.0"}
    check_versions = mock.Mock(return_value=report)
    monkeypatch.setattr(compat, "check_vllm_runtime", check_versions)

    assert cuda.check_environment() == {
        "vllm": "0.30.0",
        "platform": "cuda",
    }
    check_versions.assert_called_once_with()

    monkeypatch.setattr(
        platforms, "current_platform", native_platform("ROCM", "cuda"), raising=False
    )
    with pytest.raises(RuntimeError, match="requires the native CUDA platform"):
        cuda.check_environment()


pytestmark = pytest.mark.cpu_test
