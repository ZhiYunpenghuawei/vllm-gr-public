# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Native Ascend integration for GR."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

from vllm_ascend.platform import NPUPlatform

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

if TYPE_CHECKING:
    import torch

    from vllm_gr.constraints.execution import NPUConstrainedTopKExecution
    from vllm_gr.v1.worker.prefill_graph_common import BasePrefillGraphRunner

    from ..config import BackendConfigService


class AscendGRPlatform(GRPlatform, NPUPlatform):
    """GR components for the native Ascend platform."""

    native_platform_type = NPUPlatform
    beam_attention_layout = "ascend"
    backend_id: str = "ascend"
    canonical_constraint_backend = "npu"
    default_use_v2_model_runner = False
    beam_attention_backend = "vllm_gr.v1.attention.backends.beam_attn.BeamAttentionBackend"
    # vllm-ascend registers AttentionBackendEnum.CUSTOM -> "ASCEND" at import
    # time and clears attention_config.backend during config finalization.
    # Beam selection is routed at the per-process attention seam from the GR
    # configuration marker instead: the shared slot stays untouched by GR,
    # attention_config keeps its native value, and the L1 slot mechanism
    # above remains available if routing is disabled.
    routes_attention_backend = True
    bridged_custom_attention_owners = ("ASCEND",)
    # Semantic ownership includes deferred hooks and existing shared dispatch.
    patch_ids: tuple[str, ...] = (
        "GR-PATCH-001",
        "GR-PATCH-002",
        "GR-PATCH-012",
        "GR-PATCH-016",
        "GR-PATCH-018",
        "GR-PATCH-019",
        "GR-PATCH-021",
    )
    vendor_distribution: str | None = "vllm-ascend"

    def beam_kv_geometry(self, config: Any) -> tuple[int | None, int | None]:
        from ..admission import local_kv_geometry

        return local_kv_geometry(config)

    def resolve(self, request: BackendRequest) -> BackendPlan:
        """Keep Ascend's V1 runner and its existing auto-decode policy."""
        if request.options.to_dict():
            raise ValueError("Ascend currently accepts no additional backend options")
        gr = request.config.to_dict()
        native = request.native.to_dict()
        constraint = "canonical" if request.has_constraint_table else "none"
        return BackendPlan(
            self.backend_id,
            request,
            PlanData(
                {
                    "beam_limits": {
                        "supports_unconstrained": True,
                        "supports_dynamic_width": True,
                        "host_policy": "ascend",
                        "runner_generation": "v2" if native["use_v2_model_runner"] else "v1",
                        "local_kv_heads": native.get("local_kv_heads"),
                        "head_size": native.get("head_size"),
                        "device_type": "npu",
                        "retain_candidate_widths": True,
                        "constraint_backend": "npu",
                        "max_decode_rows": native["scheduler_config"]["max_num_batched_tokens"],
                    },
                    "runner": "vllm_gr.v1.worker.npu.npu_model_runner_patch",
                    "attention": "vllm_gr.v1.attention.backends.beam_attn_npu",
                    "prefill_attention": "ascend.cascade",
                    "decode_attention": gr["npu_decode_execution"] or "auto",
                    "requires_attention": gr["attention_backend"] == "CUSTOM",
                    "constraint": constraint,
                    "constraint_implementation": "npu" if constraint != "none" else None,
                    "stage": "ascend.v1",
                    "graph": "capture-with-eager-fallback" if gr["beam_graph_enabled"] else "eager",
                }
            ),
        )

    def create_attention_executor(
        self, *, plan: BackendPlan, device: object
    ) -> BeamAttentionExecution:
        require_native_plan(self.backend_id, plan)
        from .decode import resolve_mode

        module = self._attention_module(resolve_mode(plan.choices.to_dict()["decode_attention"]))
        if not module.is_available():
            raise RuntimeError("Ascend attention implementation is unavailable")
        return module

    def create_config_service(self) -> BackendConfigService:
        from .config import AscendConfigService

        return AscendConfigService()

    def check_environment(self) -> dict[str, str]:
        from .environment import check_ascend_environment

        return check_ascend_environment(require_npu_platform=True)

    def apply_runtime_patches(self, *, force: bool = False) -> int:
        del force  # Installers inspect their current targets on every call.
        from .spawn_patch import apply_npu_spawn_patch

        # Preserve spawn protection before native worker imports or process
        # creation. NPUWorker initializes Ascend ops before its ModelRunner;
        # keep that delayed import before the common Beam installer runs.
        spawn_count = int(apply_npu_spawn_patch())

        from vllm_gr.v1.worker.npu.startup_patch import patch_npu_worker_init_device

        worker_count = patch_npu_worker_init_device()

        # Install the Beam attention router before any native ModelRunner
        # resolves its attention backend. vllm-ascend clears
        # attention_config.backend to None during config finalization; an
        # unpatched resolution yields the native backend, which the native
        # graph manager then records as updatable-graph capable, routing FULL
        # decode replays through machinery that cannot resolve Beam update
        # tasks. The router answers the imported Beam class for GR engines
        # from the GR configuration marker, from one early per-process
        # installation with no engine-construction-time ordering left.
        from vllm_gr.v1.attention.selector_patch import install_beam_attention_routing

        routing_count = install_beam_attention_routing()
        return spawn_count + worker_count + routing_count

    def install_attention_selector(self, native_platform: type) -> int:
        from .attention_bridge import install

        from vllm_gr.v1.attention.selector_patch import install_beam_attention_routing

        # The bridge preserves explicit CUSTOM for direct platform queries;
        # the selector-level routing keeps the runner (and the native graph
        # manager that records its backend class) on the imported Beam class.
        return install(native_platform) + install_beam_attention_routing()

    def get_model_runner_hooks(self) -> ModelRunnerInstaller:
        return cast(
            ModelRunnerInstaller,
            import_module("vllm_gr.v1.worker.npu.npu_model_runner_patch"),
        )

    def get_attention_execution(self) -> BeamAttentionExecution:
        from vllm.config import get_current_vllm_config_or_none

        from .decode import engine_mode

        return self._attention_module(engine_mode(get_current_vllm_config_or_none()))

    @staticmethod
    def _attention_module(mode: str) -> BeamAttentionExecution:
        path = "vllm_gr.v1.attention.backends.beam_attn_npu"
        if mode == "xllm":
            path += "_xllm"
        return cast(BeamAttentionExecution, import_module(path))

    def create_constrained_topk_execution(
        self,
        *,
        device: str | torch.device | None = None,
        plan: BackendPlan | None = None,
    ) -> NPUConstrainedTopKExecution:
        """Create a fresh NPU constrained top-k execution adapter.

        The adapter is stateful (holds loaded tables + digest), so a new
        instance is created per call; the ModelRunner owns it for one
        lifetime and creates another after a reset.
        """
        from vllm_gr.constraints.execution import NPUConstrainedTopKExecution

        if plan is not None:
            require_native_plan(self.backend_id, plan)
        return NPUConstrainedTopKExecution(device=device)

    def get_prefill_graph_runner_class(self) -> type[BasePrefillGraphRunner]:
        from vllm_gr.v1.worker.npu.npu_prefill_graph_runner import NPUPrefillGraphRunner

        return NPUPrefillGraphRunner
