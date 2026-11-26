# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Import-light probes for the GR platforms shipped with this package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interface import GRPlatform


def cuda_platform_plugin() -> str | None:
    from vllm.platforms import current_platform

    if current_platform.is_cuda() and current_platform.device_type == "cuda":
        return "vllm_gr.platforms.cuda.platform.CudaGRPlatform"
    return None


def ascend_platform_plugin() -> str | None:
    from vllm.platforms import current_platform

    if current_platform.device_type == "npu":
        from .identity import qualified_class_name

        if any(
            qualified_class_name(cls) == "vllm_ascend.platform.NPUPlatform"
            for cls in type(current_platform).__mro__
        ):
            return "vllm_gr.platforms.ascend.platform.AscendGRPlatform"
    return None


BUILTIN_PROBES = {"cuda": cuda_platform_plugin, "ascend": ascend_platform_plugin}


def current_gr_platform() -> GRPlatform:
    from .discovery import selected_platform

    return selected_platform()
