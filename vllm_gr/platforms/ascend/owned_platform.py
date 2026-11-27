# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Native vLLM platform for owned, single-device NPU ModelRunner V2."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from vllm.platforms import Platform, PlatformEnum
from vllm.v1.attention.backends.registry import AttentionBackendEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig


WORKER_CLASS = "vllm_gr.v1.worker.npu.owned_worker.NPUWorker"
ATTENTION_BACKEND = "vllm_gr.v1.attention.backends.beam_attn.BeamAttentionBackend"
COMMUNICATOR_CLASS = "vllm_gr.platforms.ascend.owned_communicator.NPUCommunicator"


def validate_owned_runtime_config(config: Any) -> None:
    """Validate the initial support boundary before allocating device state."""
    from vllm import envs
    from vllm.config.compilation import CompilationMode, CUDAGraphMode

    from vllm_gr.config.execution import validate_gr_execution_runtime

    validate_gr_execution_runtime(config)
    parallel = config.parallel_config
    if parallel.world_size != 1 or getattr(parallel, "nnodes", 1) != 1:
        raise ValueError("The owned NPU backend requires a single local rank")
    if getattr(parallel, "distributed_executor_backend", None) in ("ray", "external_launcher"):
        raise ValueError("The owned NPU backend does not support an external executor")
    if getattr(parallel, "assigned_physical_gpu_ids", None) is not None:
        raise ValueError("Use ASCEND_RT_VISIBLE_DEVICES to select the owned NPU device")
    if envs.VLLM_DISTRIBUTED_USE_SPLIT_GROUP:
        raise ValueError("The CUDA-specific split-group initialization path is unsupported on NPU")
    if any(
        getattr(parallel, key, False)
        for key in ("enable_expert_parallel", "enable_eplb", "enable_elastic_ep")
    ):
        raise ValueError("Expert parallelism and EPLB are unsupported on the owned NPU backend")

    model = config.model_config
    if model is None:
        raise ValueError("A model configuration is required for the owned NPU backend")
    if getattr(model, "is_moe", False) or getattr(model, "runner_type", "generate") != "generate":
        raise ValueError("The owned NPU backend requires a dense text generation model")
    if getattr(model, "enable_sleep_mode", False) or getattr(model, "enable_prompt_embeds", False):
        raise ValueError(
            "Sleep mode and prompt embeddings are unsupported on the owned NPU backend"
        )
    if (
        getattr(config, "engram_config", None) is not None
        or getattr(config, "watermark_config", None) is not None
    ):
        raise ValueError("Engram and watermarking are unsupported on the owned NPU backend")
    if config.cache_config.cache_dtype not in ("auto", "float16", "bfloat16"):
        raise ValueError("The owned NPU backend supports only unquantized FP16/BF16 KV cache")
    if not config.use_v2_model_runner:
        raise ValueError("The owned NPU backend requires VLLM_USE_V2_MODEL_RUNNER=1")
    # PR-02 establishes the eager lifecycle; graph support requires its own
    # implementation and must be admitted explicitly when it is added.
    compilation = config.compilation_config
    if compilation.mode not in (None, CompilationMode.NONE) or compilation.cudagraph_mode not in (
        None,
        CUDAGraphMode.NONE,
    ):
        raise ValueError("The owned NPU backend initially requires eager execution")


class OwnedNPUPlatform(Platform):
    """A native OOT platform that deliberately exposes only implemented capabilities."""

    _enum = PlatformEnum.OOT
    device_name = "npu"
    device_type = "npu"
    dispatch_key = "PrivateUse1"
    dist_backend = "hccl"
    ray_device_key = "NPU"
    device_control_env_var = "ASCEND_RT_VISIBLE_DEVICES"
    simple_compile_backend = "eager"

    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype) -> None:
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"Unsupported owned NPU model dtype: {dtype}")

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        torch.npu.set_device(device)

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        torch.npu.manual_seed_all(seed)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.npu.get_device_name(device_id)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        if not torch.npu.is_initialized():
            raise NotImplementedError("NPU total memory requires an initialized device")
        return int(torch.npu.mem_get_info(device_id)[1])

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        uuid = getattr(torch.npu.get_device_properties(device_id), "uuid", None)
        if not uuid:
            raise RuntimeError(f"NPU device {device_id} does not expose a UUID")
        return str(uuid)

    @classmethod
    def get_current_memory_usage(cls, device: Any = None) -> float:
        return float(torch.npu.memory_allocated(device))

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        # vLLM 0.30 treats pinning as proof of UVA. Use its copy path until
        # direct device access to pinned host tensors is independently proven.
        return False

    @classmethod
    def opaque_attention_op(cls) -> bool:
        return True

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return COMMUNICATOR_CLASS

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: AttentionBackendEnum | None,
        attn_selector_config: Any,
        num_heads: int | None = None,
    ) -> str:
        if selected_backend not in (None, AttentionBackendEnum.CUSTOM):
            raise ValueError("The owned NPU backend requires CUSTOM attention")
        if any(
            getattr(attn_selector_config, key, False)
            for key in ("use_mla", "use_sparse", "use_compress")
        ):
            raise ValueError(
                "MLA, sparse and compressed attention are unsupported on the owned NPU backend"
            )
        return ATTENTION_BACKEND

    @classmethod
    def import_kernels(cls) -> None:
        # Model layers use a separate registration owner. Do not try loading
        # the native CUDA extension or import the old Ascend plugin here.
        import torch_npu  # noqa: F401

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        validate_owned_runtime_config(vllm_config)
        parallel = vllm_config.parallel_config
        if parallel.worker_cls not in ("auto", WORKER_CLASS):
            raise ValueError("The owned NPU platform requires its own Worker")
        attention = vllm_config.attention_config
        if attention.backend not in (None, AttentionBackendEnum.CUSTOM):
            raise ValueError("The owned NPU platform requires CUSTOM attention")
        parallel.worker_cls = WORKER_CLASS
        attention.backend = AttentionBackendEnum.CUSTOM
        cache = vllm_config.cache_config
        if not getattr(cache, "user_specified_block_size", False):
            cache.block_size = 128
