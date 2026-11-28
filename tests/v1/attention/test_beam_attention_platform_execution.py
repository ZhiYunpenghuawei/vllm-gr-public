# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Platform isolation and delegation contracts for Beam Attention."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.helpers.paths import REPO_ROOT
from vllm_gr.v1.attention.backends import beam_attn


def test_facade_import_does_not_load_platform_execution_modules() -> None:
    project_root = REPO_ROOT
    code = """
import sys

import vllm_gr.v1.attention.backends.beam_attn

assert "vllm_gr.v1.attention.backends.beam_attn_gpu" not in sys.modules
assert not any(name.startswith("vllm_ascend") for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )


def test_cuda_attention_uses_its_own_execution_module() -> None:
    from vllm_gr.v1.attention.backends import beam_attn_gpu

    assert beam_attn._get_cuda_execution_module() is beam_attn_gpu


@pytest.mark.parametrize("tuple_cache", [False, True])  # type: ignore[misc]
def test_forward_preserves_cache_and_attention_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tuple_cache: bool,
) -> None:
    calls: dict[str, Any] = {}

    def reshape_and_cache(*args: Any) -> None:
        calls["reshape"] = args

    def cascade_attention(**kwargs: Any) -> None:
        calls["cascade"] = kwargs

    execution = SimpleNamespace(
        reshape_and_cache=reshape_and_cache,
        cascade_attention=cascade_attention,
    )
    monkeypatch.setattr(
        beam_attn,
        "_get_cuda_execution_module",
        lambda: execution,
    )

    impl = object.__new__(beam_attn.BeamAttentionImpl)
    impl.kv_sharing_target_layer_name = None
    impl.kv_cache_dtype = "auto"
    impl.sliding_window = (6, 0)
    impl.scale = 0.5
    impl.logits_soft_cap = 9.0
    impl.vllm_flash_attn_version = 3
    impl.prefill_flash_attn_version = impl.vllm_flash_attn_version
    impl.sinks = torch.empty(1)
    impl.alibi_slopes = torch.empty(2)

    impl.head_size = 8
    key_cache = torch.zeros(2, 16, 2, 8)
    value_cache = torch.ones_like(key_cache)
    kv_cache: Any
    if tuple_cache:
        kv_cache = (key_cache, value_cache)
    else:
        kv_cache = torch.cat((key_cache, value_cache), dim=-1).transpose(1, 2)

    key = torch.empty(1)
    value = torch.empty(1)
    query = torch.empty(1)
    output = torch.empty(1)
    slot_mapping = torch.tensor([0])
    metadata = SimpleNamespace(slot_mapping=slot_mapping, is_beam_decode=False, beam_session=None)
    k_scale = torch.tensor(1.0)
    v_scale = torch.tensor(2.0)
    layer = SimpleNamespace(_k_scale=k_scale, _v_scale=v_scale)

    returned = impl.forward(
        layer=layer,
        query=query,
        key=key,
        value=value,
        kv_cache=kv_cache,
        attn_metadata=metadata,
        output=output,
    )

    assert returned is output
    reshape_args = calls["reshape"]
    assert reshape_args[0] is key
    assert reshape_args[1] is value
    torch.testing.assert_close(reshape_args[2], key_cache)
    torch.testing.assert_close(reshape_args[3], value_cache)
    assert reshape_args[4] is slot_mapping
    assert reshape_args[5] == "auto"
    assert reshape_args[6] is k_scale
    assert reshape_args[7] is v_scale

    cascade_args = calls["cascade"]
    assert cascade_args["output"] is output
    assert cascade_args["query"] is query
    torch.testing.assert_close(cascade_args["key_cache"], key_cache)
    torch.testing.assert_close(cascade_args["value_cache"], value_cache)
    assert cascade_args["attn_metadata"] is metadata
    assert cascade_args["sliding_window_size"] == [6, 0]
    assert cascade_args["scale"] == 0.5
    assert cascade_args["logits_soft_cap"] == 9.0
    assert cascade_args["vllm_flash_attn_version"] == 3
    assert cascade_args["s_aux"] is impl.sinks
    assert cascade_args["alibi_slopes"] is impl.alibi_slopes


def test_forward_incremental_uses_beam_pool_and_skips_paged_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the incremental beam decode path uses beam_pool and
    skips reshape_and_cache for the paged KV cache."""
    from vllm_gr.v1.beam.attention_context import BeamAttentionContext
    from vllm_gr.v1.beam.session import BeamSearchSession

    calls: dict[str, Any] = {}

    def unexpected_reshape(*args: Any) -> None:
        raise AssertionError("incremental forward must not write the Paged KV Cache")

    def cascade_attention(**kwargs: Any) -> None:
        calls.update(kwargs)

    from vllm_gr.v1.beam.attention_pool import write_eager_step

    execution = SimpleNamespace(
        reshape_and_cache=unexpected_reshape,
        cascade_attention=cascade_attention,
        cache_unshared_kv=write_eager_step,
    )
    monkeypatch.setattr(beam_attn, "_get_cuda_execution_module", lambda: execution)

    # Create a session using our 3-layer architecture
    context = BeamAttentionContext(
        max_batch=1,
        max_beams=4,
        num_kv_heads=2,
        max_decode_steps=3,
        head_dim=8,
        num_layers=1,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )
    slot = context.registry.allocate("test", beam_width=0, prefix_len=0)
    session = BeamSearchSession(session_id="test", slot=slot, context=context)
    session.begin_session(beam_width=2, prefix_len=5)

    metadata = SimpleNamespace(
        is_beam_decode=True,
        beam_session=session,
        incremental_suffix_kv_len=1,
        suffix_max_q_len=1,
        prefix_max_seq_len=5,
        prefix_indices=torch.arange(2),
    )

    impl = object.__new__(beam_attn.BeamAttentionImpl)
    impl.kv_sharing_target_layer_name = None
    impl.kv_cache_dtype = "auto"
    impl.sliding_window = (-1, -1)
    impl.scale = 0.5
    impl.logits_soft_cap = 0.0
    impl.vllm_flash_attn_version = 3
    impl.prefill_flash_attn_version = impl.vllm_flash_attn_version
    impl.sinks = None
    impl.alibi_slopes = None

    query = torch.zeros(2, 4, 8, dtype=torch.float16)
    impl.head_size = 8
    key = torch.arange(2 * 2 * 8, dtype=torch.float16).reshape(2, 2, 8)
    value = key + 100
    output = torch.empty_like(query)
    key_cache = torch.empty(1, 1, 2, 8, dtype=torch.float16)
    value_cache = torch.empty_like(key_cache)

    pool = session.pools[0]
    stale_context = BeamAttentionContext(
        max_batch=1,
        max_beams=4,
        num_kv_heads=2,
        max_decode_steps=3,
        head_dim=8,
        num_layers=1,
        device=torch.device("cpu"),
        dtype=torch.float16,
    )
    stale_pool = stale_context.pools[0]
    layer = SimpleNamespace(
        layer_name="model.layers.0.self_attn",
        _k_scale=None,
        _v_scale=None,
        _beam_pool=stale_pool,
    )

    returned = impl.forward(
        layer=layer,
        query=query,
        key=key,
        value=value,
        kv_cache=torch.cat((key_cache, value_cache), dim=-1).transpose(1, 2),
        attn_metadata=metadata,
        output=output,
    )

    assert returned is output
    # Verify that cascade_attention was called with beam_pool
    assert not stale_pool.unshared_key.count_nonzero()
    assert not stale_pool.unshared_value.count_nonzero()
    assert "beam_pool" in calls
    assert calls["beam_pool"] is pool
    # Verify pool KV was written by cache_unshared_kv
    torch.testing.assert_close(pool.unshared_key[0, :2, :, 0, :], key)
    torch.testing.assert_close(pool.unshared_value[0, :2, :, 0, :], value)


pytestmark = pytest.mark.cpu_test


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "invalid",
    ["missing-session", "inactive-session", "old-metadata", "multi-query", "missing-pool"],
)
def test_incremental_forward_rejects_invalid_state_before_cache_write(
    monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    from vllm_gr.v1.attention.backends import beam_attn

    def unexpected(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("invalid beam state reached a device operation")

    monkeypatch.setattr(
        beam_attn,
        "_get_cuda_execution_module",
        lambda: SimpleNamespace(
            reshape_and_cache=unexpected, cache_unshared_kv=unexpected, cascade_attention=unexpected
        ),
    )
    session = SimpleNamespace(is_active=invalid != "inactive-session", pools=[])
    metadata = SimpleNamespace(
        is_beam_decode=invalid != "old-metadata",
        prefix_indices=torch.arange(2),
        suffix_max_q_len=2 if invalid == "multi-query" else 1,
        beam_session=None if invalid == "missing-session" else session,
    )
    impl = object.__new__(beam_attn.BeamAttentionImpl)
    impl.vllm_flash_attn_version = 2
    impl.head_size = 8
    tensor = torch.empty(2, 1, 8)
    with pytest.raises((RuntimeError, ValueError), match="Beam decode requires"):
        impl.forward(
            SimpleNamespace(layer_name="model.layers.0.self_attn"),
            tensor,
            tensor,
            tensor,
            torch.empty(1, 1, 16, 16),
            metadata,
            output=tensor,
        )
