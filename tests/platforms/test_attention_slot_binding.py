# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Beam attention selection: shared-slot claims and platform routing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_gr.v1.attention.selector_patch import (
    _register_beam_backend,
    prepare_engine_attention,
)

_BEAM_PATH = "vllm_gr.v1.attention.backends.beam_attn.BeamAttentionBackend"

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def _gr_custom_config(backend) -> SimpleNamespace:
    """A minimal config whose GR marker selects CUSTOM attention."""
    return SimpleNamespace(
        attention_config=SimpleNamespace(backend=backend),
        additional_config={
            "vllm_gr": {"gr": {}, "attention-config": {"backend": "CUSTOM"}}
        },
    )


@pytest.fixture()
def custom_slot():
    """Isolate the process-wide attention override table per test."""
    from vllm.v1.attention.backends import registry

    saved = dict(registry._ATTN_OVERRIDES)
    registry._ATTN_OVERRIDES.clear()
    try:
        yield registry
    finally:
        registry._ATTN_OVERRIDES.clear()
        registry._ATTN_OVERRIDES.update(saved)


def test_unclaimed_slot_registers_beam_path(custom_slot):
    _register_beam_backend(_BEAM_PATH)
    assert custom_slot.AttentionBackendEnum.CUSTOM.get_path() == _BEAM_PATH


def test_platform_bridged_claim_is_superseded(custom_slot):
    """A native claim declared by the platform is overwritten, not rejected.

    vllm-ascend registers CUSTOM -> "ASCEND" at import time; the Ascend
    attention bridge resolves Beam selection through the registry, so the
    beam backend must become the CUSTOM owner on engine startup.
    """
    custom_slot.register_backend(custom_slot.AttentionBackendEnum.CUSTOM, "ASCEND")
    _register_beam_backend(_BEAM_PATH, bridged_owners=("ASCEND",))
    assert custom_slot.AttentionBackendEnum.CUSTOM.get_path() == _BEAM_PATH


def test_unknown_owner_still_fails(custom_slot):
    custom_slot.register_backend(
        custom_slot.AttentionBackendEnum.CUSTOM, "other.module.ConflictingBackend"
    )
    with pytest.raises(RuntimeError, match="already registered to other.module.ConflictingBackend"):
        _register_beam_backend(_BEAM_PATH, bridged_owners=("ASCEND",))


def test_idempotent_reclaim_keeps_beam_path(custom_slot):
    _register_beam_backend(_BEAM_PATH)
    _register_beam_backend(_BEAM_PATH, bridged_owners=())
    assert custom_slot.AttentionBackendEnum.CUSTOM.get_path() == _BEAM_PATH


def test_ascend_platform_declares_native_claim(custom_slot):
    """The Ascend GR platform must keep declaring the claim it bridges."""
    pytest.importorskip("vllm_ascend")
    from vllm_gr.platforms.ascend.platform import AscendGRPlatform

    assert "ASCEND" in AscendGRPlatform.bridged_custom_attention_owners


class _FakeNativePlatform:
    """A native platform whose selector the Ascend bridge installs onto."""

    native_path = "native.plugin.backend"

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, attn_selector_config, num_heads=None):
        return cls.native_path


@pytest.fixture()
def bridged_native():
    from vllm_gr.platforms.ascend.attention_bridge import install

    saved = _FakeNativePlatform.__dict__["get_attn_backend_cls"].__func__
    assert install(_FakeNativePlatform) == 1
    # A second install must be a no-op.
    assert install(_FakeNativePlatform) == 0
    yield _FakeNativePlatform
    setattr(_FakeNativePlatform, "get_attn_backend_cls", classmethod(saved))


def test_bridge_resolves_explicit_custom(bridged_native, custom_slot):
    custom_slot.register_backend(custom_slot.AttentionBackendEnum.CUSTOM, _BEAM_PATH)
    assert (
        bridged_native.get_attn_backend_cls(custom_slot.AttentionBackendEnum.CUSTOM, None)
        == _BEAM_PATH
    )


def test_beam_attention_routing_resolves_beam_class(monkeypatch):
    """vllm-ascend clears attention_config.backend to None before the native
    ModelRunner resolves its backend.

    The graph manager records that resolved class and branches its replay
    path on the type: only native backend subclasses take the updatable-graph
    machinery, which resolves zero update tasks for Beam metadata and leaves
    the stream dead. The per-process attention router must keep GR engines on
    the imported Beam class instead.
    """
    import vllm.v1.attention.selector as selector_mod
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    import vllm_gr.v1.attention.selector_patch as selector_patch
    from vllm_gr.config import GRAttentionBackend
    from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionBackend

    original = selector_mod._cached_get_attn_backend
    assert selector_patch.install_beam_attention_routing() == 1
    # A second install must be a no-op.
    assert selector_patch.install_beam_attention_routing() == 0
    patched = selector_mod._cached_get_attn_backend
    try:
        monkeypatch.setattr(
            selector_patch, "_get_current_gr_attention_backend", lambda: GRAttentionBackend.CUSTOM
        )
        assert patched(None, object()) is BeamAttentionBackend
        assert patched(AttentionBackendEnum.CUSTOM, object()) is BeamAttentionBackend
        # Without a GR CUSTOM selection the wrapper delegates to the saved
        # native resolution (structural check; the native path reads config).
        monkeypatch.setattr(selector_patch, "_get_current_gr_attention_backend", lambda: None)
        assert getattr(patched, "_vllm_gr_custom_attention_original", None) is original
    finally:
        setattr(selector_mod, "_cached_get_attn_backend", original)


def test_routing_platform_skips_slot_registration(custom_slot, monkeypatch):
    """Routing platforms resolve Beam at the per-process attention seam.

    Engine construction must leave the shared CUSTOM slot untouched and keep
    attention_config at the value the vendor runtime left it.
    """
    import vllm_gr.platforms.table as table

    monkeypatch.setattr(
        table,
        "current_gr_platform",
        lambda: SimpleNamespace(routes_attention_backend=True, backend_id="fake"),
    )
    config = _gr_custom_config(None)
    prepare_engine_attention(config)
    assert not custom_slot.AttentionBackendEnum.CUSTOM.is_overridden()
    assert config.attention_config.backend is None


def test_slot_platform_claims_shared_slot(custom_slot, monkeypatch):
    """Without routing, engine construction claims the shared CUSTOM slot."""
    import vllm_gr.platforms.table as table

    monkeypatch.setattr(
        table,
        "current_gr_platform",
        lambda: SimpleNamespace(
            routes_attention_backend=False,
            backend_id="fake",
            beam_attention_backend=_BEAM_PATH,
            bridged_custom_attention_owners=(),
            install_attention_selector=lambda native_platform: 0,
        ),
    )
    config = _gr_custom_config(custom_slot.AttentionBackendEnum.CUSTOM)
    prepare_engine_attention(config)
    assert custom_slot.AttentionBackendEnum.CUSTOM.get_path() == _BEAM_PATH


def test_bridge_delegates_none_without_gr_custom(bridged_native, monkeypatch):
    import vllm_gr.v1.attention.selector_patch as selector_patch

    monkeypatch.setattr(
        selector_patch, "_get_current_gr_attention_backend", lambda: None
    )
    assert bridged_native.get_attn_backend_cls(None, None) == _FakeNativePlatform.native_path
