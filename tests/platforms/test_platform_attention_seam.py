# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""The platform class is the single Beam attention selection seam."""

from __future__ import annotations

from typing import Any

import pytest

from vllm.platforms.interface import Platform, PlatformEnum

import vllm_gr.v1.attention.selector_patch as selector_patch
from vllm_gr.platforms.interface import GRPlatform
from vllm_gr.plugin import install_gr_platform

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


_NATIVE_PATH = "fake.native.Backend"


class _FakeNative(Platform):  # type: ignore[misc]
    """A native parent whose backend selection returns a fixed path string."""

    _enum = PlatformEnum.OOT
    device_type = "probe"

    @classmethod
    def get_attn_backend_cls(
        cls, selected_backend: Any, attn_selector_config: Any, num_heads: int | None = None
    ) -> Any:
        return _NATIVE_PATH


class _Probe(GRPlatform, _FakeNative):  # type: ignore[misc]
    """GRPlatform over a fake native parent; exercises the seam via the MRO."""

    backend_id = "probe"
    native_platform_type = _FakeNative

    def resolve(self, request: Any) -> Any:  # pragma: no cover - abstract stub
        raise NotImplementedError


def test_seam_routes_gr_engines_to_the_beam_class(monkeypatch: pytest.MonkeyPatch) -> None:
    """A GR engine selecting Beam attention resolves the Beam backend path.

    The platform-method contract is a dotted class path that vLLM's selector
    imports. The condition is the GR configuration marker, so both an
    explicit CUSTOM request and the vendor runtime's cleared field
    (vllm-ascend resets attention_config.backend to None) reach Beam; the
    shared AttentionBackendEnum.CUSTOM registry slot is never consulted.
    """
    from vllm_gr.v1.attention.selector_patch import _BEAM_BACKEND_PATH

    monkeypatch.setattr(selector_patch, "_gr_attention_is_custom", lambda: True)
    assert _Probe.get_attn_backend_cls(None, object()) == _BEAM_BACKEND_PATH
    assert _Probe.get_attn_backend_cls("CUSTOM", object()) == _BEAM_BACKEND_PATH


def test_seam_defers_to_native_without_gr_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a GR Beam selection the native resolution is untouched."""
    monkeypatch.setattr(selector_patch, "_gr_attention_is_custom", lambda: False)
    assert _Probe.get_attn_backend_cls(None, object()) == _NATIVE_PATH
    assert _Probe.get_attn_backend_cls("FLASH_ATTN", object()) == _NATIVE_PATH


def test_plugin_installs_gr_platform_as_current(monkeypatch: pytest.MonkeyPatch) -> None:
    """The GR platform replaces current_platform exactly once per process.

    The replacement is the mechanism that routes vLLM's lazy platform
    queries (attention selection above, and future seams) through the GR
    class while every native attribute keeps resolving through the MRO.
    """
    import vllm.platforms as vllm_platforms

    from tests.helpers.native_platform import native_platform

    original = native_platform("OOT", "vendor")
    monkeypatch.setattr(vllm_platforms, "current_platform", original)

    probe = _Probe()
    assert install_gr_platform(probe) is True
    assert vllm_platforms.current_platform is probe
    # Idempotent: a second initialization must not replace the instance.
    assert install_gr_platform(probe) is False
    assert vllm_platforms.current_platform is probe
    # Non-GRPlatform objects are rejected rather than installed.
    assert install_gr_platform(original) is False
    assert vllm_platforms.current_platform is probe
