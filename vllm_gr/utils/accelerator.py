# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Accelerator resource cleanup through the selected GR platform."""

from __future__ import annotations

from vllm_gr.logger import init_logger

logger = init_logger("vllm_gr.utils.accelerator")


def release_accelerator_memory() -> None:
    """Best-effort release of resources after engine shutdown."""
    from vllm_gr.platforms.errors import UnsupportedBackendError
    from vllm_gr.platforms.table import current_gr_platform

    try:
        current_gr_platform().release_accelerator_memory()
    except UnsupportedBackendError:
        # An ordinary native-only engine can have no matching GR plugin.
        return
    except Exception:
        logger.debug("GR platform accelerator cleanup failed", exc_info=True)
