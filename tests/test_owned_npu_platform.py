# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Host checks for the independent NPU platform and group ownership."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm import envs
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from vllm_gr.platforms.ascend.owned_platform import (
    ATTENTION_BACKEND,
    WORKER_CLASS,
    OwnedNPUPlatform,
    validate_owned_runtime_config,
)
from vllm_gr.platforms.ascend.owned_platform_plugin import register
from vllm_gr.v1.worker.npu import owned_worker

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            world_size=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            worker_cls="auto",
        ),
        scheduler_config=SimpleNamespace(async_scheduling=True),
        model_config=SimpleNamespace(dtype=torch.float16, runner_type="generate"),
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=16),
        attention_config=SimpleNamespace(backend=None),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.NONE
        ),
        use_v2_model_runner=True,
    )


def test_platform_plugin_inactive_without_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("VLLM_GR_OWNED_NPU", raising=False)
    assert register() is None


def test_owned_platform_config_and_attention(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_DISTRIBUTED_USE_SPLIT_GROUP", False)
    config = _config()
    OwnedNPUPlatform.check_and_update_config(config)
    assert config.parallel_config.worker_cls == WORKER_CLASS
    assert config.attention_config.backend is AttentionBackendEnum.CUSTOM
    assert config.cache_config.block_size == 128
    assert not OwnedNPUPlatform.is_pin_memory_available()
    assert OwnedNPUPlatform.get_attn_backend_cls(None, SimpleNamespace()) == ATTENTION_BACKEND
    with pytest.raises(ValueError, match="CUSTOM"):
        OwnedNPUPlatform.get_attn_backend_cls(AttentionBackendEnum.FLASH_ATTN, SimpleNamespace())


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("parallel_config", "tensor_parallel_size", 2, "TP/PP/DP/DCP/PCP"),
        ("parallel_config", "pipeline_parallel_size", 2, "TP/PP/DP/DCP/PCP"),
        ("parallel_config", "data_parallel_size", 2, "TP/PP/DP/DCP/PCP"),
        ("parallel_config", "prefill_context_parallel_size", 2, "TP/PP/DP/DCP/PCP"),
        ("parallel_config", "decode_context_parallel_size", 2, "TP/PP/DP/DCP/PCP"),
        ("parallel_config", "world_size", 2, "single local rank"),
        (
            "parallel_config",
            "distributed_executor_backend",
            "external_launcher",
            "external executor",
        ),
        ("model_config", "is_moe", True, "dense text"),
        ("cache_config", "cache_dtype", "fp8", "KV cache"),
        ("compilation_config", "cudagraph_mode", CUDAGraphMode.FULL, "eager execution"),
        ("", "kv_transfer_config", object(), "kv_transfer_config"),
        ("", "use_v2_model_runner", False, "VLLM_USE_V2_MODEL_RUNNER"),
    ],
)
def test_unsupported_configuration_fails_before_allocation(
    monkeypatch, section, field, value, message
):
    monkeypatch.setattr(envs, "VLLM_DISTRIBUTED_USE_SPLIT_GROUP", False)
    config = _config()
    setattr(getattr(config, section) if section else config, field, value)
    with pytest.raises(ValueError, match=message):
        validate_owned_runtime_config(config)


@pytest.fixture
def group_runtime(monkeypatch):
    events = []
    group = SimpleNamespace(device_group=object(), world_size=1, rank_in_group=0)
    dist = SimpleNamespace(
        is_initialized=Mock(return_value=False),
        is_backend_available=Mock(return_value=True),
        get_rank=Mock(return_value=0),
        get_world_size=Mock(return_value=1),
        get_backend=Mock(return_value="hccl"),
    )
    state = SimpleNamespace(
        model_parallel_is_initialized=Mock(return_value=False),
        init_distributed_environment=Mock(side_effect=lambda **_: events.append("world")),
        ensure_model_parallel_initialized=Mock(side_effect=lambda *_, **__: events.append("model")),
        destroy_distributed_environment=Mock(side_effect=lambda: events.append("destroy_world")),
        destroy_model_parallel=Mock(side_effect=lambda: events.append("destroy_model")),
    )
    for name in ("world", "tp", "pp", "dp", "pcp", "dcp"):
        setattr(state, f"get_{name}_group", Mock(return_value=group))
    monkeypatch.setattr(owned_worker, "dist", dist)
    monkeypatch.setattr(owned_worker, "parallel_state", state)
    return SimpleNamespace(events=events, dist=dist, state=state, group=group)


def test_single_rank_group_lifetime(group_runtime):
    groups = owned_worker.SingleRankGroups()
    groups.initialize(init_method="file:///tmp/unused-by-test", device_index=2)
    kwargs = group_runtime.state.init_distributed_environment.call_args.kwargs
    assert (kwargs["world_size"], kwargs["rank"], kwargs["local_rank"], kwargs["backend"]) == (
        1,
        0,
        2,
        "hccl",
    )
    assert group_runtime.events == ["world", "model"]
    group_runtime.state.ensure_model_parallel_initialized.assert_called_once_with(
        1, 1, 1, 1, backend="hccl"
    )
    with pytest.raises(RuntimeError, match="twice"):
        groups.initialize(init_method="unused", device_index=2)
    groups.close()
    groups.close()
    assert group_runtime.events == ["world", "model", "destroy_model", "destroy_world"]


def test_external_group_is_not_destroyed(group_runtime):
    group_runtime.dist.is_initialized.return_value = True
    groups = owned_worker.SingleRankGroups()
    with pytest.raises(RuntimeError, match="uninitialized distributed"):
        groups.initialize(init_method="unused", device_index=0)
    groups.close()
    assert group_runtime.events == []


def test_missing_hccl_never_falls_back_to_gloo(group_runtime):
    group_runtime.dist.is_backend_available.return_value = False
    with pytest.raises(RuntimeError, match="HCCL"):
        owned_worker.SingleRankGroups().initialize(init_method="unused", device_index=0)
    assert group_runtime.events == []


def test_partial_model_group_initialization_rolls_back(group_runtime):
    group_runtime.state.ensure_model_parallel_initialized.side_effect = RuntimeError("group failed")
    groups = owned_worker.SingleRankGroups()
    with pytest.raises(RuntimeError, match="group failed"):
        groups.initialize(init_method="unused", device_index=0)
    groups.close()
    assert group_runtime.events == ["world", "destroy_model", "destroy_world"]


def test_partial_world_group_initialization_rolls_back(group_runtime):
    group_runtime.state.init_distributed_environment.side_effect = RuntimeError("world failed")
    groups = owned_worker.SingleRankGroups()
    with pytest.raises(RuntimeError, match="world failed"):
        groups.initialize(init_method="unused", device_index=0)
    groups.close()
    assert group_runtime.events == ["destroy_world"]


def test_group_cleanup_attempts_world_teardown_on_model_failure(group_runtime):
    groups = owned_worker.SingleRankGroups()
    groups.initialize(init_method="unused", device_index=0)
    group_runtime.state.destroy_model_parallel.side_effect = RuntimeError("destroy failed")
    with pytest.raises(RuntimeError, match="destroy failed"):
        groups.close()
    assert group_runtime.events[-1] == "destroy_world"
