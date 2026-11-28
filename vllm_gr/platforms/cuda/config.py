# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from ..config import BackendConfigService

if TYPE_CHECKING:
    from vllm_gr.config import GRConfig


class CudaConfigService(BackendConfigService):
    default_worker_decision = True
    supports_constraint_vocab_slicing = True

    def validate_engine(self, config: Any, gr: GRConfig) -> None:
        validate_cuda_options(asdict(gr))

    def finalize_native_config(self, config: Any, gr: GRConfig) -> None:
        compilation = getattr(config, "compilation_config", None)
        if (
            gr.beam_max_width is not None
            and not getattr(config, "use_v2_model_runner", False)
            and compilation is not None
            and compilation.backend == "inductor"
        ):
            # Fix reduction geometry across AOT loads and preserve explicit overrides.
            compilation.inductor_compile_config.setdefault("dynamic_scale_rblock", False)


def validate_prefill_flash_attn_version(value: object) -> int | str | None:
    """Validate the FlashAttention versions implemented by the CUDA backend."""
    if value is None or (isinstance(value, str) and value == "auto"):
        return value
    if type(value) is int and value in (2, 3):
        return value
    raise ValueError("beam.prefill_flash_attn_version must be null, 'auto', 2 or 3")


def validate_cuda_options(gr: Mapping[str, Any]) -> None:
    """Check the concrete algorithms supported by the CUDA implementation."""
    deployment = gr["deployment"]
    if deployment["topk_keep"] > 256 or deployment["topk_select"] not in ("i32", "i64", "flash"):
        raise ValueError("gr.topk_keep must be in 1..256 and topk_select must be i32, i64 or flash")
    if deployment["prefix_policy"] not in ("auto", "off") or deployment[
        "unconstrained_head_layout"
    ] not in (None, "transposed"):
        raise ValueError("Unsupported prefix_policy or unconstrained_head_layout")
    if deployment["dense_prefill"] not in ("off", "fa2", "cudnn"):
        raise ValueError("gr.dense_prefill must be off, fa2 or cudnn")
    validate_prefill_flash_attn_version(gr["prefill_flash_attn_version"])


def dense_prefill_mode() -> str | None:
    from vllm.config import get_current_vllm_config_or_none

    from vllm_gr.config.models import deployment_for

    config = get_current_vllm_config_or_none()
    if (
        config is None
        or config.cache_config.enable_prefix_caching
        or config.scheduler_config.enable_chunked_prefill
        # A KV connector can load earlier prompt tokens into the paged cache.
        or getattr(config, "kv_transfer_config", None) is not None
    ):
        return None
    mode = deployment_for(config).dense_prefill
    return mode if mode != "off" else None
