# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Contracts used by native GR runner and attention implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

from vllm.platforms.interface import Platform

if TYPE_CHECKING:
    import torch

    from vllm_gr.config import GRConfig
    from vllm_gr.config.backend_plan import BackendPlan, BackendRequest
    from vllm_gr.constraints.backend import ConstraintBackend
    from vllm_gr.constraints.execution import ConstrainedTopKExecution
    from vllm_gr.constraints.table import ConstraintTable
    from vllm_gr.v1.beam.batch_contracts import BeamBatchRequest
    from vllm_gr.v1.worker.prefill_graph_common import BasePrefillGraphRunner

    from .admission import BeamLimits
    from .config import BackendConfigService


class ModelRunnerInstaller(Protocol):
    """Install and verify the selected native runner's own method surface."""

    def discover_worker_runners(self) -> tuple[tuple[str, type[Any]], ...]: ...

    def install(self) -> None: ...

    def is_patched(self, runner_class: type[Any]) -> bool: ...


class ModelRunnerAdapter(Protocol):
    """Per-runner callbacks with the selected native signatures and return values."""

    def prepare_inputs(self, *args: Any, **kwargs: Any) -> Any: ...

    def execute_model(self, *args: Any, **kwargs: Any) -> Any: ...

    # Some native runners complete sampling entirely inside execute_model.
    @property
    def sample_tokens(self) -> Callable[..., Any] | None: ...


class BeamAttentionExecution(Protocol):
    """Availability contract for a platform-owned Beam Attention executor."""

    def is_available(self) -> bool: ...


class GRPlatform(Platform, ABC):
    """Stateless GR policy for one platform selected by native vLLM.

    Engine admission owns the serializable choices. Each loaded worker owns
    the runner, attention and constraint operations created for those choices.
    """

    api_version = 2
    native_platform_type: type[Platform]
    beam_attention_layout: str | None = None
    canonical_constraint_backend = "platform"
    backend_id: str
    default_use_v2_model_runner: bool | None = None
    vendor_distribution: str | None = None
    patch_ids: tuple[str, ...] = ()

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend,
        attn_selector_config,
        num_heads: int | None = None,
    ):
        """The single Beam attention selection seam.

        vLLM's cached selector reaches the platform lazily on every backend
        resolution, so a platform answering its Beam backend path here covers every
        native consumer - the ModelRunner's backend resolution and attention
        layer construction - from one seam that exists from class
        definition. The shared AttentionBackendEnum.CUSTOM registry slot is
        never claimed. Nothing is installed at engine construction, and
        ``attention_config`` keeps the value selected by the native platform:
        the Beam intent travels in the GR configuration marker alone.

        The platform-method contract is a dotted class path; the selector
        imports it. A platform must override this method to supply its own
        Beam attention backend; the base rejects GR Beam selection rather than
        silently choosing a different device's implementation.
        """
        from vllm_gr.v1.attention.selector_patch import _gr_attention_is_custom

        if _gr_attention_is_custom():
            raise NotImplementedError(
                f"GR platform {cls.backend_id!r} has no Beam attention backend"
            )
        return super().get_attn_backend_cls(
            selected_backend,
            attn_selector_config=attn_selector_config,
            num_heads=num_heads,
        )

    @abstractmethod
    def resolve(self, request: BackendRequest) -> BackendPlan:
        """Admit this engine and identify the runner operations it needs."""

    def check_environment(self) -> dict[str, str]:
        from .compat import check_vllm_runtime

        return check_vllm_runtime()

    def beam_kv_geometry(self, config: Any) -> tuple[int | None, int | None] | None:
        """Return the local KV geometry when required for extended decode admission."""
        return None

    def validate_request(self, limits: BeamLimits, gr: GRConfig, request: BeamBatchRequest) -> None:
        """Optional EngineCore-only checks. Raise GRDeviceRequestError for bad requests."""

    def create_config_service(self) -> BackendConfigService:
        from .config import BackendConfigService

        return BackendConfigService()

    def apply_runtime_patches(self, *, force: bool = False) -> int:
        return 0

    def release_accelerator_memory(self) -> None:
        """Best-effort release of the selected platform's process resources."""

    def get_model_runner_hooks(self) -> ModelRunnerInstaller:
        raise NotImplementedError(f"{self.backend_id} has no GR ModelRunner installer")

    def create_model_runner_adapter(
        self, *, plan: BackendPlan, native_runner: Any
    ) -> ModelRunnerAdapter:
        from .services import native_runner_adapter

        return native_runner_adapter(self.backend_id, plan=plan, native_runner=native_runner)

    def validate_worker(self, *, plan: BackendPlan, native_runner: Any) -> None:
        from .services import validate_native_worker

        validate_native_worker(self.backend_id, plan=plan, native_runner=native_runner)

    def get_attention_execution(self) -> BeamAttentionExecution:
        raise NotImplementedError(f"{self.backend_id} has no Beam attention executor")

    def create_attention_executor(
        self, *, plan: BackendPlan, device: object
    ) -> BeamAttentionExecution:
        from .services import require_native_plan

        require_native_plan(self.backend_id, plan)
        execution = self.get_attention_execution()
        if not execution.is_available():
            raise RuntimeError(
                f"Backend {self.backend_id!r} attention implementation is unavailable"
            )
        return execution

    def validate_constraint_backend(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        max_rows: int,
        max_top_k: int,
    ) -> None:
        """Validate the requested table capacity and placement before artifact I/O."""

    def create_constraint_backend(
        self,
        table: ConstraintTable,
        *,
        device: torch.device,
        dtype: torch.dtype,
        max_rows: int,
        max_top_k: int,
    ) -> ConstraintBackend:
        """Return a ready, independently owned backend for a validated canonical artifact."""
        raise NotImplementedError(
            f"GR platform {self.backend_id!r} has no canonical constraint backend factory"
        )

    def create_constrained_topk_execution(
        self, *, device: str | torch.device | None = None, plan: BackendPlan | None = None
    ) -> ConstrainedTopKExecution | None:
        raise NotImplementedError(f"{self.backend_id} has no constrained Top-K executor")

    def get_prefill_graph_runner_class(self) -> type[BasePrefillGraphRunner]:
        """Select the GR graph runner when a V1 worker requests prefill capture."""
        raise NotImplementedError(f"{self.backend_id} has no GR prefill graph runner")
