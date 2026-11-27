# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Worker lifecycle for the project's independent NPU V2 runtime."""

from __future__ import annotations

from contextlib import ExitStack
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import parallel_state
from vllm.logger import init_logger
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase
from vllm.v1.worker.workspace import (
    init_workspace_manager,
    is_workspace_manager_initialized,
    reset_workspace_manager,
)

from vllm_gr.config.execution import require_beam_config
from vllm_gr.platforms.ascend.owned_platform import validate_owned_runtime_config

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
    from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput


logger = init_logger(__name__)


class SingleRankGroups:
    """Own native vLLM groups for a single-rank NPU worker.

    A pre-initialized distributed runtime is outside the initial deployment
    contract. Reject it rather than taking ownership of the launcher's groups.
    Register cleanup before initialization so partial failures are covered.
    """

    def __init__(self) -> None:
        self._cleanup = ExitStack()
        self._started = False
        self._closed = False

    def initialize(self, *, init_method: str, device_index: int) -> None:
        if self._started or self._closed:
            raise RuntimeError("NPU communication groups cannot be initialized twice")
        if dist.is_initialized() or parallel_state.model_parallel_is_initialized():
            raise RuntimeError(
                "The owned NPU worker requires an uninitialized distributed environment"
            )
        if not dist.is_backend_available("hccl"):
            raise RuntimeError("torch-npu did not register the HCCL backend")

        with ExitStack() as cleanup:
            # init_distributed_environment may fail after creating the default
            # process group. Both the default group and native group state
            # belong to this worker because no launcher group was present.
            cleanup.callback(parallel_state.destroy_distributed_environment)
            parallel_state.init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=init_method,
                local_rank=device_index,
                backend="hccl",
                timeout=timedelta(seconds=120),
            )
            if dist.get_rank() != 0 or dist.get_world_size() != 1:
                raise RuntimeError("Unexpected distributed rank or world size")
            world = parallel_state.get_world_group()
            if str(dist.get_backend(world.device_group)) != "hccl":
                raise RuntimeError("The NPU world group must use HCCL")
            cleanup.callback(parallel_state.destroy_model_parallel)
            parallel_state.ensure_model_parallel_initialized(1, 1, 1, 1, backend="hccl")
            for getter in (
                parallel_state.get_tp_group,
                parallel_state.get_pp_group,
                parallel_state.get_dp_group,
                parallel_state.get_pcp_group,
                parallel_state.get_dcp_group,
            ):
                group = getter()
                if group.world_size != 1 or group.rank_in_group != 0:
                    raise RuntimeError(
                        "The owned NPU backend requires one-rank model parallel groups"
                    )
                if str(dist.get_backend(group.device_group)) != "hccl":
                    raise RuntimeError("Every device communication group must use HCCL")
            self._cleanup = cleanup.pop_all()
            self._started = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._cleanup.close()


class NPUWorker(WorkerBase):
    """Connect the native executor protocol to the owned V2 runner.

    The worker owns its communication and workspace lifetime, while the
    model runner owns model, input, KV, execution and graph resources.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        super().__init__(vllm_config, local_rank, rank, distributed_init_method, is_driver_worker)
        self._cleanup = ExitStack()
        self._closed = False
        self._failed = False

    def init_device(self) -> None:
        if self._closed or self._failed or self.model_runner is not None:
            raise RuntimeError("The owned NPU worker cannot be initialized again")
        validate_owned_runtime_config(self.vllm_config)
        require_beam_config(self.vllm_config)
        if self.rank != 0:
            raise ValueError("The owned NPU worker requires rank zero")

        import torch_npu  # noqa: F401

        try:
            with ExitStack() as cleanup:
                if is_workspace_manager_initialized():
                    raise RuntimeError(
                        "The owned NPU worker requires an uninitialized workspace manager"
                    )
                if dist.is_initialized() or parallel_state.model_parallel_is_initialized():
                    raise RuntimeError(
                        "The owned NPU worker does not borrow external communication groups"
                    )
                device_index = self.current_platform.logical_device_id_to_visible_device_id(
                    self.local_rank
                )
                if not torch.npu.is_available() or not 0 <= device_index < torch.npu.device_count():
                    raise ValueError(f"Invalid or unavailable NPU device index: {device_index}")
                self.device = torch.device(f"npu:{device_index}")
                torch.npu.set_device(self.device)
                # Final cleanup runs after the model runner, workspace, and
                # process groups have released their owned references.
                cleanup.callback(torch.npu.empty_cache)
                groups = SingleRankGroups()
                cleanup.callback(groups.close)
                with set_current_vllm_config(self.vllm_config):
                    groups.initialize(
                        init_method=self.distributed_init_method, device_index=device_index
                    )
                    set_random_seed(self.model_config.seed)
                    cleanup.callback(reset_workspace_manager)
                    init_workspace_manager(self.device, num_ubatches=1)
                    from vllm_gr.v1.worker.npu.v2.model_runner import NPUModelRunnerV2

                    runner = NPUModelRunnerV2(self.vllm_config, self.device)
                    cleanup.callback(runner.shutdown)
                    self.model_runner = runner
                self._cleanup = cleanup.pop_all()
        except BaseException:
            self._failed = True
            self.model_runner = None
            raise

    def _runner(self) -> Any:
        if self._closed or self._failed or self.model_runner is None:
            raise RuntimeError("The owned NPU worker is not ready")
        return self.model_runner

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        from vllm_gr.v1.worker.npu.model_layers import (
            validate_model_layer_registration,
            validate_supported_model_config,
        )

        validate_supported_model_config(self.vllm_config)
        validate_model_layer_registration()
        with set_current_vllm_config(self.vllm_config):
            self._runner().load_model(load_dummy_weights=load_dummy_weights)
        validate_model_layer_registration()

    def get_model(self) -> torch.nn.Module:
        return self._runner().get_model()

    def determine_available_memory(self) -> int:
        return self._runner().determine_available_memory()

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self._runner().get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        with set_current_vllm_config(self.vllm_config):
            self._runner().initialize_kv_cache(kv_cache_config)

    def compile_or_warm_up_model(self) -> CompilationTimes:
        with set_current_vllm_config(self.vllm_config):
            return self._runner().compile_or_warm_up_model()

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        return self._runner().execute_model(scheduler_output)

    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        return self._runner().sample_tokens(grammar_output)

    def update_max_model_len(self, max_model_len: int) -> None:
        self._runner().update_max_model_len(max_model_len)

    def get_supported_tasks(self) -> tuple[str, ...]:
        return self._runner().get_supported_tasks()

    def reset_encoder_cache(self) -> None:
        # The platform rejects encoder and multimodal models at startup.
        return

    def take_draft_token_ids(self) -> None:
        # Speculative decoding is not admitted for this backend.
        return None

    def get_kv_connector_handshake_metadata(self) -> None:
        # KV connectors and disaggregated serving are rejected at startup.
        return None

    def update_config(self, overrides: dict[str, Any]) -> None:
        self._runner().update_config(overrides)

    def reload_weights(self, *args: Any, **kwargs: Any) -> None:
        with set_current_vllm_config(self.vllm_config):
            self._runner().reload_weights(*args, **kwargs)

    def synchronize_device(self) -> None:
        if self.device is not None:
            torch.npu.synchronize(self.device)

    def check_health(self) -> None:
        self._runner()._require_live()

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            # The runner shutdown must drain outstanding work before releasing
            # model state. ExitStack also attempts group cleanup if it fails.
            self._cleanup.close()
        except BaseException:
            self._failed = True
            logger.exception("Owned NPU Worker shutdown failed")
            raise
        finally:
            self.model_runner = None
