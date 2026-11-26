# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Hardware-neutral CUSTOM attention selection for GR runtime initialization."""

from threading import Lock
from typing import Any

from vllm_gr.config import GRAttentionBackend, GRConfig

_BEAM_BACKEND_PATH = "vllm_gr.v1.attention.backends.beam_attn.BeamAttentionBackend"
_REGISTRATION_LOCK = Lock()


def _get_current_gr_attention_backend() -> GRAttentionBackend | None:
    from vllm.config import get_current_vllm_config_or_none

    return GRConfig.from_vllm_config(get_current_vllm_config_or_none()).attention_backend


def _register_beam_backend(path: str = _BEAM_BACKEND_PATH) -> None:
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    custom = AttentionBackendEnum.CUSTOM
    with _REGISTRATION_LOCK:
        if custom.is_overridden():
            owner = custom.get_path()
            if owner != path:
                raise RuntimeError(
                    f"GR Beam requires CUSTOM attention, already registered to {owner}. "
                    "Load these attention backends in separate processes."
                )
        else:
            register_backend(custom, path)


def prepare_engine_attention(config: Any) -> None:
    """Claim the process-local slot before a Beam engine constructs its layers."""
    if GRConfig.from_vllm_config(config).attention_backend is not GRAttentionBackend.CUSTOM:
        return
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    if (
        getattr(getattr(config, "attention_config", None), "backend", None)
        is not AttentionBackendEnum.CUSTOM
    ):
        raise ValueError("GR CUSTOM attention must be selected before native attention selection")
    from vllm_gr.platforms.table import current_gr_platform

    platform = current_gr_platform()
    if platform.beam_attention_backend is None:
        raise ValueError(f"GR platform {platform.backend_id!r} does not support CUSTOM attention")
    _register_beam_backend(platform.beam_attention_backend)
    apply_custom_attention_backend_patch()


def clear_attention_backend_cache() -> None:
    from vllm.v1.attention.selector import _cached_get_attn_backend

    _cached_get_attn_backend.cache_clear()


def apply_custom_attention_backend_patch(force: bool = False) -> int:
    """Install the bridge on the selected vendor platform when needed."""
    from vllm.platforms import current_platform

    from vllm_gr.platforms.errors import UnsupportedBackendError
    from vllm_gr.platforms.table import current_gr_platform

    try:
        platform = current_gr_platform()
    except UnsupportedBackendError:
        return 0
    return platform.install_attention_selector(type(current_platform))
