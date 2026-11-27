# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Hardware-neutral CUSTOM attention selection for GR runtime initialization."""

from threading import Lock
from typing import Any

from vllm_gr.config import GRAttentionBackend, GRConfig

_BEAM_BACKEND_PATH = "vllm_gr.v1.attention.backends.beam_attn.BeamAttentionBackend"
_REGISTRATION_LOCK = Lock()
_ORIGINAL_SELECTOR_ATTR = "_vllm_gr_custom_attention_original"


def _get_current_gr_attention_backend() -> GRAttentionBackend | None:
    from vllm.config import get_current_vllm_config_or_none

    return GRConfig.from_vllm_config(get_current_vllm_config_or_none()).attention_backend


def _get_custom_beam_backend() -> type:
    from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionBackend

    return BeamAttentionBackend


def _register_beam_backend(
    path: str = _BEAM_BACKEND_PATH,
    *,
    bridged_owners: tuple[str, ...] = (),
) -> None:
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    custom = AttentionBackendEnum.CUSTOM
    with _REGISTRATION_LOCK:
        if custom.is_overridden():
            owner = custom.get_path()
            if owner != path and owner not in bridged_owners:
                raise RuntimeError(
                    f"GR Beam requires CUSTOM attention, already registered to {owner}. "
                    "Load these attention backends in separate processes."
                )
        # A native runtime may claim the CUSTOM slot before a GR engine starts
        # (vllm-ascend registers "ASCEND" at import time). Platform-bridged
        # claims are superseded here: the beam backend owns the slot, and the
        # platform bridges resolve Beam selection through it. Unknown owners
        # still fail loudly above.
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
    _register_beam_backend(
        platform.beam_attention_backend,
        bridged_owners=platform.bridged_custom_attention_owners,
    )
    apply_custom_attention_backend_patch()


def clear_attention_backend_cache() -> None:
    """Clear native selection results while preserving the live GR wrapper."""
    import vllm.v1.attention.selector as selector_mod

    current = selector_mod._cached_get_attn_backend
    native = getattr(current, _ORIGINAL_SELECTOR_ATTR, current)
    native.cache_clear()


def install_beam_selector_fallback() -> int:
    """Resolve CUSTOM (and the native None reset) to the imported Beam class.

    Native graph machinery records the runner's attention backend class and
    branches its replay path on that type (vllm-ascend treats only subclasses
    of its own backend as updatable-graph capable). If the selector hands the
    runner a path string, or the native backend after vllm-ascend resets
    ``attention_config.backend`` to None, FULL decode replays are routed
    through the updatable-graph machinery, which resolves zero update tasks
    for Beam metadata and leaves the stream dead. Returning the imported Beam
    class here keeps the native graph manager on the plain replay path; this
    mirrors the selector behavior the platform bridge replaced.
    """
    import vllm.v1.attention.selector as selector_mod
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    original = selector_mod._cached_get_attn_backend
    if getattr(original, _ORIGINAL_SELECTOR_ATTR, None) is not None:
        return 0

    def _patched_cached_get_attn_backend(
        backend: Any, attn_selector_config: Any, num_heads: int | None = None
    ) -> Any:
        if backend == AttentionBackendEnum.CUSTOM:
            return _get_custom_beam_backend()
        if backend is None and _get_current_gr_attention_backend() is GRAttentionBackend.CUSTOM:
            return _get_custom_beam_backend()
        return original(backend, attn_selector_config, num_heads=num_heads)

    setattr(_patched_cached_get_attn_backend, _ORIGINAL_SELECTOR_ATTR, original)
    selector_mod._cached_get_attn_backend = _patched_cached_get_attn_backend
    return 1


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
