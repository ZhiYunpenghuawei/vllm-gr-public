# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Beam attention selection helpers consumed by the platform seam.

The selection seam itself is :meth:`GRPlatform.get_attn_backend_cls`; this
module only carries the marker lookup and the Beam class import it uses,
plus the selector cache clearing that engine re-admission relies on. The
shared AttentionBackendEnum.CUSTOM registry slot is never claimed.
"""

from vllm_gr.config import GRAttentionBackend, GRConfig

_BEAM_BACKEND_PATH = "vllm_gr.v1.attention.backends.beam_attn.BeamAttentionBackend"


def _get_current_gr_attention_backend() -> GRAttentionBackend | None:
    from vllm.config import get_current_vllm_config_or_none

    return GRConfig.from_vllm_config(get_current_vllm_config_or_none()).attention_backend


def _gr_attention_is_custom() -> bool:
    """Whether the current engine's GR configuration selects Beam attention."""
    return _get_current_gr_attention_backend() is GRAttentionBackend.CUSTOM


def clear_attention_backend_cache() -> None:
    """Clear native selection results cached for earlier engine configs."""
    from vllm.v1.attention.selector import _cached_get_attn_backend

    _cached_get_attn_backend.cache_clear()
