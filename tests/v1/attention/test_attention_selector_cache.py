# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

import sys
from functools import lru_cache

import pytest
import torch

from vllm_gr.v1.attention import selector_patch

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def test_selector_cache_reset_clears_cached_resolutions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.v1.attention.selector as selector

    @lru_cache
    def lookup(key: str) -> object:
        return object()

    monkeypatch.setattr(selector, "_cached_get_attn_backend", lookup)
    cached = lookup("key")
    selector_patch.clear_attention_backend_cache()
    assert selector._cached_get_attn_backend is lookup
    assert lookup("key") is not cached


def test_selected_beam_backend_import_failure_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.platforms as native_platforms
    from vllm.v1.attention.selector import (
        AttentionSelectorConfig,
        _cached_get_attn_backend,
    )

    from vllm_gr.platforms.cuda.platform import CudaGRPlatform

    monkeypatch.setattr(native_platforms, "current_platform", CudaGRPlatform())
    monkeypatch.setattr(selector_patch, "_gr_attention_is_custom", lambda: True)
    monkeypatch.setitem(sys.modules, "vllm_gr.v1.attention.backends.beam_attn", None)
    config = AttentionSelectorConfig(
        head_size=64, dtype=torch.float16, kv_cache_dtype=None, block_size=16
    )
    selector_patch.clear_attention_backend_cache()
    try:
        with pytest.raises(ModuleNotFoundError, match="beam_attn"):
            _cached_get_attn_backend(None, config)
    finally:
        selector_patch.clear_attention_backend_cache()
