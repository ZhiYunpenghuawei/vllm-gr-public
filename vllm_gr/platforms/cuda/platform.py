# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Native CUDA integration for GR."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, cast

from vllm.platforms.cuda import CudaPlatformBase

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
from ..table import cuda_platform_plugin

if TYPE_CHECKING:
    import torch

    from vllm_gr.constraints.execution import ConstrainedTopKExecution
    from vllm_gr.v1.worker.prefill_graph_common import BasePrefillGraphRunner

    from ..config import BackendConfigService


class CudaGRPlatform(GRPlatform, CudaPlatformBase):
    """GR components for the native CUDA platform."""

    native_platform_type = CudaPlatformBase
    beam_attention_layout = "cuda"
    backend_id: str = "cuda"
    canonical_constraint_backend = "cuda"
    default_use_v2_model_runner = False
    # Semantic ownership includes hooks still installed by shared dispatch.
    patch_ids: tuple[str, ...] = ("GR-PATCH-011",)

    def resolve(self, request: BackendRequest) -> BackendPlan:
        """Admit the existing CUDA paths without importing a device runtime."""
        from vllm_gr.v1.worker.cuda.runner_installer import runner_id

        if request.options.to_dict():
            raise ValueError("CUDA currently accepts no additional backend options")
        gr = request.config.to_dict()
        native = request.native.to_dict()
        if gr["npu_decode_execution"] == "xllm":
            raise ValueError("CUDA does not support npu_decode_execution='xllm'")
        constraint = "canonical" if request.has_constraint_table else "none"
        return BackendPlan(
            self.backend_id,
            request,
            PlanData(
                {
                    "beam_limits": {
                        "supports_unconstrained": True,
                        "supports_dynamic_width": True,
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

    def create_constrained_topk_execution(
        self,
        *,
        device: str | torch.device | None = None,
        plan: BackendPlan | None = None,
    ) -> ConstrainedTopKExecution | None:
        """Create the production CUDA Constraint Table execution adapter."""
        from vllm_gr.constraints.execution import CUDAConstrainedTopKExecution

        if plan is not None:
            require_native_plan(self.backend_id, plan)
        return CUDAConstrainedTopKExecution(device=device)

    def get_prefill_graph_runner_class(self) -> type[BasePrefillGraphRunner]:
        from vllm_gr.v1.worker.cuda.v1.gpu_prefill_graph_runner import (
            GPUPrefillGraphRunner,
        )

        return GPUPrefillGraphRunner
