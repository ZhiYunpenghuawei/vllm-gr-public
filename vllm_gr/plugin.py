# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""vLLM general plugin entry point for vllm-gr.

Registered via ``pyproject.toml`` under ``[project.entry-points."vllm.general_plugins"]``.
vLLM calls ``load_general_plugins()`` in every process (main, EngineCore subprocess,
Worker subprocess), which discovers and executes this entry point.

This module is the formal runtime initialization boundary for vllm-gr.
It checks runtime compatibility and installs the patches for Beam Search,
sliced vocabulary projection, and Ascend integration.
"""

import logging

logger = logging.getLogger("vllm_gr.plugin")


def install_gr_platform(integration: object) -> bool:
    """Serve as the process platform so GR platform methods reach vLLM.

    The GR platform is a subclass of the selected native platform, so every
    native consumer keeps working through the MRO; the only addition
    reachable from vLLM is the Beam attention seam in
    ``get_attn_backend_cls``, which vLLM's cached selector consults lazily
    on every backend resolution. Idempotent; returns whether this call
    performed the replacement.
    """
    import vllm.platforms as vllm_platforms

    from vllm_gr.platforms.interface import GRPlatform

    if isinstance(vllm_platforms.current_platform, GRPlatform):
        return False
    if not isinstance(integration, GRPlatform):
        return False
    vllm_platforms.current_platform = integration
    return True


def initialize_runtime(*, required: bool = True) -> dict[str, object]:
    """Initialize the current process for vllm-gr runtime usage.

    Safe to call repeatedly. The underlying patch helper is idempotent, so
    this can be used both by the vLLM plugin loader and by explicit engine
    startup paths in the main process. Each process checks its native platform before installing runtime patches.
    """
    from vllm_gr.engine.startup import configure_model_runner_default
    from vllm_gr.platforms.errors import UnsupportedBackendError
    from vllm_gr.platforms.table import current_gr_platform

    try:
        integration = current_gr_platform()
    except UnsupportedBackendError:
        if required:
            raise
        return {"patch_count": 0, "patches": {}}
    platform_seam = install_gr_platform(integration)
    configure_model_runner_default()
    integration.check_environment()

    from vllm_gr.patch import get_last_beam_patch_report, re_apply_patches

    patch_count = re_apply_patches(integration=integration)

    report = get_last_beam_patch_report()
    state: dict[str, object] = {
        "patch_count": patch_count,
        "patches": report.statuses() if report is not None else {},
        "platform_seam": platform_seam,
    }
    logger.debug(
        "vllm-gr runtime initialized: patches=%d, platform_seam=%s", patch_count, platform_seam
    )
    return state


def register() -> None:
    """Entry point used by vLLM's general plugin loader."""
    state = initialize_runtime(required=False)
    logger.debug("vllm-gr plugin loaded (%s patches applied)", state["patch_count"])
