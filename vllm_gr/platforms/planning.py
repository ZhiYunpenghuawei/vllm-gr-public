# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Normalize engine inputs and admit plans without storing runtime state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
from enum import Enum
from typing import Any

from vllm_gr.config.backend_plan import (
    BACKEND_PLAN_KEY,
    LEGACY_PLAN_KEY,
    BackendPlan,
    BackendRequest,
    PlanData,
)
from vllm_gr.config.resolved_choices import ResolvedChoices

from .compat import installed_version
from .identity import qualified_class_name
from .interface import GRPlatform

BACKEND_OPTIONS_KEY = "vllm_gr_backend_options"

# Only settings common to every rank belong in the engine fingerprint. Native
# allocation results, device ordinals, rank IDs and handles deliberately do not.
_NATIVE_FIELDS = {
    "model_config": (
        "dtype",
        "quantization",
        "max_model_len",
        "is_encoder_decoder",
        "is_hybrid",
        "is_multimodal_model",
    ),
    "cache_config": ("block_size", "cache_dtype", "enable_prefix_caching"),
    "scheduler_config": (
        "async_scheduling",
        "enable_chunked_prefill",
        "max_num_seqs",
        "max_num_batched_tokens",
        "long_prefill_token_threshold",
    ),
    "parallel_config": (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
        "use_ubatching",
    ),
}


def _primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return _primitive(value.value)
    if isinstance(value, Mapping):
        return {key: _primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_primitive(item) for item in value]
    return value


def integration_versions(integration: GRPlatform) -> PlanData:
    """Capture the selected implementation and package versions across ranks."""
    from vllm_gr.version import __version__

    vendor = integration.vendor_distribution
    return PlanData(
        {
            "gr": __version__,
            "vllm": installed_version("vllm"),
            "vendor_distribution": vendor,
            "vendor": installed_version(vendor) if vendor else None,
            "integration": qualified_class_name(type(integration)),
        }
    )


def normalize_backend_request(config: Any, integration: GRPlatform) -> BackendRequest:
    from vllm_gr.config import GRConfig

    native: dict[str, Any] = {}
    for section, fields in _NATIVE_FIELDS.items():
        source = getattr(config, section, None)
        values = {name: _primitive(getattr(source, name, None)) for name in fields}
        # torch.dtype is deliberately encoded by its stable public name, never
        # by a general default=str serializer for arbitrary runtime objects.
        if section == "model_config" and values["dtype"] is not None:
            values["dtype"] = str(values["dtype"])
        native[section] = values
    attention = getattr(config, "attention_config", None)
    overrides = getattr(attention, "backend_per_kind", None) or {}
    native["attention_per_kind"] = {
        kind: backend.name if isinstance(backend, Enum) else backend
        for kind, backend in overrides.items()
    }
    native["use_v2_model_runner"] = bool(getattr(config, "use_v2_model_runner", False))
    model = getattr(config, "model_config", None)
    text_config = getattr(model, "hf_text_config", None)
    native["geometry"] = {
        key: getattr(text_config, key, None)
        for key in (
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "num_hidden_layers",
            "vocab_size",
        )
    }
    additional = getattr(config, "additional_config", None)
    options = additional.get(BACKEND_OPTIONS_KEY, {}) if isinstance(additional, Mapping) else {}
    gr = GRConfig.from_vllm_config(config)
    if gr.beam_max_decode_steps > 3:
        geometry = integration.beam_kv_geometry(config)
        if geometry is not None:
            native["local_kv_heads"], native["head_size"] = geometry
    gr_fields = asdict(gr)
    gr_fields["beam_execution_mode"] = "v1" if gr.beam_max_width is not None else None
    return BackendRequest(
        config=PlanData(_primitive(gr_fields)),
        native=PlanData(native),
        compatibility=integration_versions(integration),
        options=PlanData(options),
    )


def resolve_backend_plan(integration: GRPlatform, request: BackendRequest) -> BackendPlan:
    plan = integration.resolve(request)
    if not isinstance(plan, BackendPlan):
        raise TypeError("Backend resolve() must return BackendPlan")
    if (plan.backend_id, plan.request) != (
        integration.backend_id,
        request,
    ):
        raise ValueError("Backend resolve() changed request or integration identity")
    return plan


def _normalize_with_admitted_length(
    config: Any, integration: GRPlatform, admitted_length: Any
) -> BackendRequest:
    request = normalize_backend_request(config, integration)
    native = request.native.to_dict()
    model = native["model_config"]
    actual_length = model["max_model_len"]
    if (
        type(admitted_length) is int
        and type(actual_length) is int
        and 0 < actual_length <= admitted_length
    ):
        # The plan retains the admitted limit; native load/profiling may fit a
        # smaller runtime allocation before first bind or a later full reload.
        # Do not overwrite the live config: validate_worker sees its actual size.
        # Increases, invalid lengths and every other input remain fingerprinted.
        model["max_model_len"] = admitted_length
        request = replace(request, native=PlanData(native))
    return request


def worker_plan_from_choices(
    config: Any, expected: ResolvedChoices, *, integration: GRPlatform | None = None
) -> BackendPlan:
    """Verify local inputs and consume the choices admitted by the engine."""
    from .table import current_gr_platform

    if integration is None:
        integration = current_gr_platform()
    request = _normalize_with_admitted_length(
        config, integration, expected.limits.to_dict()["max_model_len"]
    )
    local = BackendPlan(
        integration.backend_id,
        request,
        expected.choices,
        requirements=expected.requirements,
    )
    if ResolvedChoices.from_plan(local) != expected:
        raise RuntimeError(
            "Worker backend plan mismatch: local platform, config or choices differ from engine admission"
        )
    return local


def admit_backend_plan(config: Any) -> BackendPlan | None:
    """Attach only serializable plan data after EngineArgs has normalized GR."""
    from vllm_gr.config import GRConfig

    from .table import current_gr_platform

    additional = getattr(config, "additional_config", None)
    if isinstance(additional, Mapping) and LEGACY_PLAN_KEY in additional:
        raise ValueError(
            "Unsupported legacy GR plan format; use the same vllm-gr version on the engine and workers"
        )
    gr = GRConfig.from_vllm_config(config)
    if not gr.enabled and gr.beam_max_width is None:
        additional = getattr(config, "additional_config", None)
        if isinstance(additional, Mapping) and additional.get(BACKEND_OPTIONS_KEY):
            raise ValueError("Backend options require an enabled GR engine")
        if isinstance(additional, Mapping) and BACKEND_PLAN_KEY in additional:
            config.additional_config = {
                key: value for key, value in additional.items() if key != BACKEND_PLAN_KEY
            }
        return None
    integration = current_gr_platform()
    plan = resolve_backend_plan(integration, normalize_backend_request(config, integration))
    additional = getattr(config, "additional_config", None)
    if additional is not None and not isinstance(additional, Mapping):
        raise TypeError("Backend plans require mapping additional_config")
    config.additional_config = {
        **(additional or {}),
        BACKEND_PLAN_KEY: ResolvedChoices.from_plan(plan).to_wire(),
    }
    return plan
