# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Install the admitted GR lifecycle on the project's owned NPU V2 runner."""

from functools import wraps
from typing import Any

from vllm_gr.v1.worker import gr_execution, model_runner_contracts as contracts
from vllm_gr.v1.worker.runner_lifecycle import has_runner_lifecycle, install_runner_lifecycle

RUNNER_ID = "vllm_gr.v1.worker.npu.v2.runner_installer"


def discover_worker_runners() -> tuple[tuple[str, type], ...]:
    from .model_runner import NPUModelRunnerV2

    return (("NPUModelRunnerV2", NPUModelRunnerV2),)


def is_patched(runner_class: type) -> bool:
    return contracts.is_worker_runner_v2_patched(runner_class) and has_runner_lifecycle(
        runner_class, RUNNER_ID
    )


def make_execute(original):
    @wraps(original)
    def execute(self, scheduler_output, *args, **kwargs):
        retirement = gr_execution._dispatch_retirement_frame(self, scheduler_output)
        if retirement is not None:
            return retirement
        stage = gr_execution._dispatch_stage_frame(
            self, scheduler_output, original, *args, **kwargs
        )
        if stage is not gr_execution._NOT_A_STAGE_FRAME:
            return stage
        if getattr(scheduler_output, "beam_data", None):
            raise ValueError("Legacy Beam frames are unsupported by owned NPU V2")
        return original(self, scheduler_output, *args, **kwargs)

    setattr(execute, contracts._WORKER_EXECUTE_PATCH_MARKER, True)
    return execute


def make_sample(original):
    @wraps(original)
    def sample(self, grammar_output=None):
        owner = getattr(self, "_gr_npu_stage_runner", None)
        if owner is not None and (
            owner.pending_plan is not None or owner.execution_error is not None
        ):
            return owner.sample(grammar_output)
        return original(self, grammar_output)

    setattr(sample, contracts._WORKER_V2_SAMPLE_TOKENS_PATCH_MARKER, True)
    return sample


def install() -> None:
    from vllm_gr.v1.worker.model_runner_patch import (
        install_worker_utility_hooks,
        make_load_model_with_constraint_backend,
        make_shutdown_with_constraint_backend,
    )
    from vllm_gr.v1.worker.npu.npu_model_runner_patch import (
        make_patched_load_model,
        make_patched_shutdown,
    )
    from vllm_gr.v1.worker.npu.owned_worker import NPUWorker
    from vllm_gr.v1.worker.runner_binding import bind_after_model_load
    from .model_runner import NPUModelRunnerV2

    install_worker_utility_hooks(NPUWorker)
    runner: Any = NPUModelRunnerV2
    if is_patched(runner):
        return
    if getattr(runner, "_gr_owned_npu_v2", False):
        raise RuntimeError("Owned NPU V2 hooks are partially installed or replaced")
    # Shared GR model/constraint ownership is independent of the old vendor
    # Worker. The SID wrappers borrowed here do not invoke Ascend APIs.
    runner.load_model = bind_after_model_load(
        make_patched_load_model(
            make_load_model_with_constraint_backend(runner.load_model, max_batch=4)
        )
    )
    runner.shutdown = make_patched_shutdown(make_shutdown_with_constraint_backend(runner.shutdown))
    runner.execute_model = make_execute(runner.execute_model)
    runner.sample_tokens = make_sample(runner.sample_tokens)
    install_runner_lifecycle(runner, RUNNER_ID, prepare="prepare_inputs")
    runner._gr_owned_npu_v2 = True
    if not is_patched(runner):
        raise RuntimeError("Owned NPU V2 lifecycle installation is incomplete")
