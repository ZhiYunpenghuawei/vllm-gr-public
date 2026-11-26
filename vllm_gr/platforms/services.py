# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Runner-owned services for the admitted native platform."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from vllm_gr.config.backend_plan import BackendPlan

if TYPE_CHECKING:
    from .interface import ModelRunnerAdapter

logger = logging.getLogger(__name__)


def constraint_execution_override(*, device: Any) -> Any:
    """Honor the shared CI/dev Torch oracle before calling the platform factory."""
    override = os.environ.get("VLLM_GR_CONSTRAINT_EXECUTION")
    if override is None:
        return None
    if override != "torch":
        raise ValueError(
            f"unsupported VLLM_GR_CONSTRAINT_EXECUTION={override!r}; expected 'torch' or unset"
        )
    from vllm_gr.constraints.execution import TorchConstrainedTopKExecution

    logger.info("vllm-gr: constraint execution overridden to Torch oracle for CI/dev")
    return TorchConstrainedTopKExecution(device=device)


def validate_required_services(
    plan: BackendPlan, required: dict[str, str | None], reported: dict[str, Any], worker: str
) -> None:
    for name, expected in required.items():
        actual = reported.get(name)
        if not isinstance(actual, str) or not actual.strip():
            raise RuntimeError(f"{worker} has no bound {name}")
        if expected is not None and actual != expected:
            label = (
                "canonical constraint"
                if name == "constraint" and plan.constraint_mode == "canonical"
                else name
            )
            raise RuntimeError(
                f"{worker} {label} implementation mismatch: expected {expected}, got {actual}"
            )


def require_native_plan(backend_id: str, plan: BackendPlan) -> None:
    if plan.backend_id != backend_id:
        raise ValueError(f"Expected a {backend_id!r} plan, got {plan.backend_id!r}")


def native_runner_adapter(
    backend_id: str, *, plan: BackendPlan, native_runner: Any
) -> ModelRunnerAdapter:
    from vllm_gr.v1.worker.runner_lifecycle import bind_native_runner

    require_native_plan(backend_id, plan)
    return bind_native_runner(plan, native_runner)


def validate_native_worker(backend_id: str, *, plan: BackendPlan, native_runner: Any) -> None:
    require_native_plan(backend_id, plan)
    if plan.request.config.to_dict()["beam_execution_mode"] == "v1":
        from vllm_gr.config.execution import validate_gr_execution_runtime
        from vllm_gr.v1.engine.beam_admission import validate_gr_device_config

        validate_gr_execution_runtime(native_runner.vllm_config)
        validate_gr_device_config(native_runner.vllm_config)
