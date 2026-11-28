# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""CUDA integration for the native vLLM platform."""


def cuda_platform_plugin() -> str | None:
    """Discover the native CUDA platform without loading its GR implementation."""
    from vllm.platforms import current_platform

    if current_platform.is_cuda() and current_platform.device_type == "cuda":
        return "vllm_gr.platforms.cuda.platform.CudaGRPlatform"
    return None
