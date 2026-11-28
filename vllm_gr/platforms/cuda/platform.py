# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Native CUDA integration for GR."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, cast

from vllm.platforms.cuda import CudaPlatform

from vllm_gr.config.backend_plan import (
    BackendPlan,
    BackendRequest,
    PlanData,
)

from ..interface import (
    BeamAttentionExecution,
    GRPlatform,
    ModelRunnerInstaller,
)
from ..services import require_native_plan
from . import cuda_platform_plugin

if TYPE_CHECKING:
    import torch

    from vllm_gr.constraints.backend import ConstraintBackend
    from vllm_gr.constraints.execution import ConstrainedTopKExecution
    from vllm_gr.constraints.table import ConstraintTable
    from vllm_gr.v1.worker.prefill_graph_common import BasePrefillGraphRunner

    from ..config import BackendConfigService


class CudaGRPlatform(GRPlatform, CudaPlatform):
    """GR components for the native CUDA platform."""

    # The process platform replaces native vLLM's instance. Inherit its
    # concrete CUDA implementation, not the abstract device-query base.
    native_platform_type = CudaPlatform
    beam_attention_layout = "cuda"
    backend_id: str = "cuda"
    canonical_constraint_backend = "cuda"
    default_use_v2_model_runner = False
    # Semantic ownership includes hooks still installed by shared dispatch.
    patch_ids: tuple[str, ...] = ("GR-PATCH-011",)

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend,
        attn_selector_config,
        num_heads: int | None = None,
    ):
        from vllm_gr.v1.attention.selector_patch import _gr_attention_is_custom

        if _gr_attention_is_custom():
            return "vllm_gr.v1.attention.backends.beam_attn.BeamAttentionBackend"
        return super().get_attn_backend_cls(
            selected_backend,
            attn_selector_config=attn_selector_config,
            num_heads=num_heads,
        )

    def resolve(self, request: BackendRequest) -> BackendPlan:
        """Admit the existing CUDA paths without importing a device runtime."""
        from vllm_gr.v1.worker.cuda.runner_installer import runner_id

        if request.options.to_dict():
            raise ValueError("CUDA currently accepts no additional backend options")
        gr = request.config.to_dict()
        native = request.native.to_dict()
        from .config import validate_cuda_options

        validate_cuda_options(gr)
        constraint = "canonical" if request.has_constraint_table else "none"
        return BackendPlan(
            self.backend_id,
            request,
            PlanData(
                {
                    "beam_limits": {
                        "supports_unconstrained": True,
                        "supports_dynamic_width": True,
                        "supports_online_trie_reload": True,
                        "host_policy": "cuda",
                        "runner_generation": "v2" if native["use_v2_model_runner"] else "v1",
                        "local_kv_heads": native.get("local_kv_heads"),
                        "head_size": native.get("head_size"),
                        "device_type": "cuda",
                        "max_decode_rows": None,
                        "retain_candidate_widths": False,
                        "constraint_backend": "cuda",
                    },
                    "runner": runner_id(native["use_v2_model_runner"]),
                    "attention": "vllm_gr.v1.attention.backends.beam_attn_gpu",
                    "prefill_attention": "cuda.cascade",
                    "decode_attention": "cuda.cascade",
                    "prefill_flash_attn_version": gr["prefill_flash_attn_version"],
                    "requires_attention": gr["attention_backend"] == "CUSTOM",
                    "constraint": constraint,
                    "constraint_implementation": "cuda" if constraint != "none" else None,
                    "stage": "cuda.v2" if native["use_v2_model_runner"] else "cuda.v1",
                    "graph": "capture-with-eager-fallback" if gr["beam_graph_enabled"] else "eager",
                }
            ),
        )

    def create_config_service(self) -> BackendConfigService:
        from .config import CudaConfigService

        return CudaConfigService()

    def check_environment(self) -> dict[str, str]:
        from ..compat import check_vllm_runtime

        if cuda_platform_plugin() is None:
            raise RuntimeError("CUDA integration requires the native CUDA platform")

        report = check_vllm_runtime()
        report["platform"] = "cuda"
        return report

    def get_model_runner_hooks(self) -> ModelRunnerInstaller:
        from vllm_gr.v1.worker.cuda import runner_installer

        return runner_installer

    def get_attention_execution(self) -> BeamAttentionExecution:
        return cast(
            BeamAttentionExecution,
            import_module("vllm_gr.v1.attention.backends.beam_attn_gpu"),
        )

    def validate_constraint_backend(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        max_rows: int,
        max_top_k: int,
    ) -> None:
        import torch

        from vllm_gr.ops.cuda.rec_constrained_topk import MAX_TOP_K

        if max_top_k > MAX_TOP_K:
            raise ValueError(f"CUDA constraint max_top_k must be <= {MAX_TOP_K}, got {max_top_k}")
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"CUDA constraint logits must be fp16/bf16, got {dtype}")
        if device.type != "cuda":
            raise RuntimeError(f"CUDA constraint backend requires a CUDA device, got {device}")

    def create_constraint_backend(
        self,
        table: ConstraintTable,
        *,
        device: torch.device,
        dtype: torch.dtype,
        max_rows: int,
        max_top_k: int,
    ) -> ConstraintBackend:
        from vllm_gr.constraints.cuda_loader import create_cuda_constraint_backend

        return create_cuda_constraint_backend(
            table, device=device, dtype=dtype, max_rows=max_rows, max_top_k=max_top_k
        )

    def release_accelerator_memory(self) -> None:
        import torch

        from vllm_gr.logger import init_logger

        logger = init_logger(__name__)
        try:
            if not torch.cuda.is_available():
                return
        except Exception:
            logger.debug("CUDA availability probe failed during cleanup", exc_info=True)
            return
        for name in ("synchronize", "empty_cache", "ipc_collect"):
            try:
                getattr(torch.cuda, name)()
            except Exception:
                logger.debug("CUDA %s failed during cleanup", name, exc_info=True)

    def create_constrained_topk_execution(
        self,
        *,
        device: str | torch.device | None = None,
        plan: BackendPlan | None = None,
    ) -> ConstrainedTopKExecution | None:
        """Create the production CUDA Constraint Table execution adapter."""
        from vllm_gr.constraints.cuda_execution import CUDAConstrainedTopKExecution

        if plan is not None:
            require_native_plan(self.backend_id, plan)
        return CUDAConstrainedTopKExecution(device=device)

    def get_prefill_graph_runner_class(self) -> type[BasePrefillGraphRunner]:
        from vllm_gr.v1.worker.cuda.v1.gpu_prefill_graph_runner import (
            GPUPrefillGraphRunner,
        )

        return GPUPrefillGraphRunner
