# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Fixtures for GR deployment settings and constraint artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from vllm_gr.config import (
    ConstraintConfig,
    ConstraintMode,
    ConstraintTableSpec,
    GRConfig,
)
from vllm_gr.config.models import DeploymentOptions
from vllm_gr.platforms.config import BackendConfigService


def platform_config_service(device: str | None) -> BackendConfigService:
    """Supply native defaults for the platform simulated by an EngineArgs test."""
    from tests.helpers.cuda_platform import CudaGRPlatform

    if device == "cuda":
        return CudaGRPlatform().create_config_service()
    return BackendConfigService()


def table_constraint(*, max_top_k: int = 128) -> ConstraintConfig:
    return ConstraintConfig(
        mode=ConstraintMode.TABLE,
        table=ConstraintTableSpec(
            backend="cuda",
            path="/tmp/constraint_table.safetensors",
            max_top_k=max_top_k,
            artifact_digest="sha256:" + "a" * 64,
            tokenizer_digest="sha256:" + "b" * 64,
        ),
    )


def admit_test_beam_limits(
    host: Any,
    *,
    backend_id: str = "test",
    device_type: str | None = "cuda",
    constraint_backend: str | None = "cuda",
    supports_unconstrained: bool = True,
    supports_dynamic_width: bool = True,
    supports_online_trie_reload: bool = True,
    max_decode_rows: int | None = None,
    retain_candidate_widths: bool = False,
) -> Any:
    """Publish explicitly selected host-test capabilities in the driver's wire format.

    This fixture exercises frontend admission without discovering hardware.
    Worker binding tests should instead use admit_backend_plan with a platform.
    """
    from vllm_gr.config.backend_plan import (
        BACKEND_PLAN_KEY,
        BackendPlan,
        BackendRequest,
        PlanData,
    )
    from vllm_gr.config.resolved_choices import ResolvedChoices

    empty = PlanData({})
    plan = BackendPlan(
        backend_id,
        BackendRequest(empty, empty, empty, empty),
        PlanData(
            {
                "beam_limits": {
                    "device_type": device_type,
                    "max_decode_rows": max_decode_rows,
                    "retain_candidate_widths": retain_candidate_widths,
                    "constraint_backend": constraint_backend,
                    "runner_generation": "v2"
                    if getattr(host, "use_v2_model_runner", False)
                    else "v1",
                    "local_kv_heads": None,
                    "head_size": None,
                    "supports_unconstrained": supports_unconstrained,
                    "supports_dynamic_width": supports_dynamic_width,
                    "supports_online_trie_reload": supports_online_trie_reload,
                }
            }
        ),
    )
    host.additional_config = {
        **(getattr(host, "additional_config", None) or {}),
        BACKEND_PLAN_KEY: ResolvedChoices.from_plan(plan).to_wire(),
    }
    return host


def engine_args_from_file(tmp_path: Path, mapping: Mapping[str, object]) -> SimpleNamespace:
    """Prepare native EngineArgs transport from a scenario file."""
    import json

    from vllm_gr.config import gr_config_to_mapping, load_gr_config

    path = tmp_path / "scenario.yaml"
    path.write_text(json.dumps(mapping), encoding="utf-8")

    return SimpleNamespace(
        additional_config={"vllm_gr": gr_config_to_mapping(load_gr_config(path))}
    )


def prepared_engine_args(
    *, config: Path | None = None, gr: Mapping[str, Any] | None = None, **kwargs: Any
) -> Any:
    from vllm.engine.arg_utils import EngineArgs

    from vllm_gr.config.parser import read_settings
    from vllm_gr.engine.startup import prepare_engine_settings

    document = read_settings(config) if config else None
    values, _, _ = prepare_engine_settings(document, kwargs, gr)
    return EngineArgs(**values)


def worker_config(config: GRConfig, host: Any = None) -> Any:
    """Publish one worker fixture through the same snapshot used by the driver."""
    if host is None:
        host = SimpleNamespace()
    if not hasattr(host, "additional_config"):
        host.additional_config = {}
    config.write_to_vllm_config(host)
    return host


def update_worker_config(worker: Any, **changes: Any) -> GRConfig:
    config = replace(GRConfig.from_vllm_config(worker.vllm_config), **changes)
    config.write_to_vllm_config(worker.vllm_config)
    return config


def engine_config(**options: object) -> SimpleNamespace:
    """A test engine carrying a validated GR snapshot."""
    host = SimpleNamespace(additional_config={})
    set_deployment(host, **options)
    return host


def set_deployment(host: SimpleNamespace, **options: object) -> None:
    config = GRConfig.from_vllm_config(host)
    if config.beam_max_width is None:
        config = replace(config, beam_max_width=128)
    deployment = DeploymentOptions.from_mapping({**config.deployment.to_mapping(), **options})
    host._vllm_gr_parsed_config = replace(config, deployment=deployment)


def serving_beam_engine_args(
    model: str, config: dict[str, Any], gpu_memory_utilization: float
) -> dict[str, Any]:
    """Build the serving regression's real engine inputs from width schedules."""
    from vllm_gr.config.parser import gr_config_to_mapping, load_gr_config

    return dict(
        model=model,
        trust_remote_code=True,
        async_scheduling=True,
        enable_chunked_prefill=False,
        max_model_len=2048,
        max_num_batched_tokens=8192,
        # V1 owns one logical request; beam rows are managed by the worker.
        max_num_seqs=1,
        enforce_eager=not config["beam_graph_enabled"],
        enable_log_requests=False,
        shutdown_timeout=10,
        max_logprobs=2 * max(config["candidate_width"]),
        gpu_memory_utilization=gpu_memory_utilization,
        additional_config={"vllm_gr": gr_config_to_mapping(load_gr_config(overrides=config))},
        attention_config={"backend": "CUSTOM"},
    )
