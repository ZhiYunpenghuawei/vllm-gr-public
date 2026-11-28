# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Device-free regressions for accelerator cleanup after engine shutdown."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.helpers.cuda_platform import CudaGRPlatform
from vllm_gr.platforms.errors import UnsupportedBackendError
from vllm_gr.utils import accelerator


@pytest.mark.parametrize("available", [False, True])  # type: ignore[misc]
def test_release_memory_only_cleans_available_cuda(
    monkeypatch: pytest.MonkeyPatch, available: bool
) -> None:
    cuda = SimpleNamespace(
        is_available=lambda: available,
        synchronize=Mock(),
        empty_cache=Mock(),
        ipc_collect=Mock(),
    )
    monkeypatch.setattr("vllm_gr.platforms.table.current_gr_platform", CudaGRPlatform)
    monkeypatch.setattr(torch, "cuda", cuda)

    accelerator.release_accelerator_memory()

    for operation in (cuda.synchronize, cuda.empty_cache, cuda.ipc_collect):
        if available:
            operation.assert_called_once_with()
        else:
            operation.assert_not_called()


def test_release_memory_delegates_to_selected_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    release = Mock()
    platform = SimpleNamespace(release_accelerator_memory=release)
    monkeypatch.setattr("vllm_gr.platforms.table.current_gr_platform", lambda: platform)
    accelerator.release_accelerator_memory()
    release.assert_called_once_with()


def test_release_memory_without_gr_platform_is_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    def unsupported() -> None:
        raise UnsupportedBackendError("No GR integration")

    monkeypatch.setattr("vllm_gr.platforms.table.current_gr_platform", unsupported)
    accelerator.release_accelerator_memory()


pytestmark = pytest.mark.cpu_test
