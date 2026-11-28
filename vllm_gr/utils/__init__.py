# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Utility helpers for device detection and accelerator memory cleanup.

Convenience re-exports for the few symbols that are still consumed via the
``vllm_gr.utils`` top level; everything else should be imported directly from
its defining submodule (``vllm_gr.utils.accelerator`` etc.).
"""

from vllm_gr.utils.accelerator import (
    release_accelerator_memory,
)

__all__ = [
    "release_accelerator_memory",
]
