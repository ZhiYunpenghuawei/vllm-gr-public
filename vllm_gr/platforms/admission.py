# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Physical limits shared by Beam admission and frontend grouping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm_gr.config.backend_plan import backend_wire
from vllm_gr.config.resolved_choices import read_resolved_choices

if TYPE_CHECKING:
    from vllm_gr.v1.beam.batch_contracts import BeamBatchRequest


@dataclass(frozen=True, slots=True)
class BeamLimits:
    """Hardware placement and the native runner's Decode row budget."""

    device_type: str | None
    max_decode_rows: int | None
    retain_candidate_widths: bool
    constraint_backend: str | None
    runner_generation: str
    local_kv_heads: int | None
    head_size: int | None
    backend_id: str | None = None
    supports_unconstrained: bool = False
    supports_dynamic_width: bool = False
    supports_online_trie_reload: bool = False
    # An optional policy identity for diagnostics. The admitted capabilities,
    # not this string, determine which shared frontend operations are available.
    host_policy: str | None = None

    def uses_retained_widths(self, config: Any) -> bool:
        options = config.deployment
        return (
            self.retain_candidate_widths
            and options.candidate_width is not None
            and options.candidate_width == options.beam_width
        )

    def __post_init__(self) -> None:
        if self.device_type is not None and not isinstance(self.device_type, str):
            raise ValueError("Invalid Beam admission device_type")
        if type(self.retain_candidate_widths) is not bool:
            raise ValueError("Invalid Beam admission retain_candidate_widths")
        if self.constraint_backend is not None and not isinstance(self.constraint_backend, str):
            raise ValueError("Invalid Beam admission constraint_backend")
        if (
            type(self.supports_unconstrained) is not bool
            or type(self.supports_dynamic_width) is not bool
            or type(self.supports_online_trie_reload) is not bool
        ):
            raise ValueError("Invalid Beam admission capabilities")
        if self.host_policy is not None and not isinstance(self.host_policy, str):
            raise ValueError("Invalid admitted GR platform host policy")
        if self.runner_generation not in ("v1", "v2"):
            raise ValueError("Invalid Beam admission runner_generation")
        budget = self.max_decode_rows
        if budget is not None and (type(budget) is not int or budget < 1):
            raise ValueError("Invalid Beam admission max_decode_rows")


def local_kv_geometry(config: Any) -> tuple[int | None, int | None]:
    """Query native KV geometry after tensor-parallel placement is known."""
    try:
        model = config.model_config
        heads = model.get_num_kv_heads(getattr(config, "parallel_config", None))
        head_size = model.get_head_size()
    except (AttributeError, TypeError, ValueError):
        return None, None
    if type(heads) is not int or type(head_size) is not int:
        raise ValueError("beam_search requires positive integer KV-head geometry")
    return heads, head_size


def beam_limits(config: Any) -> BeamLimits:
    wire = backend_wire(config)
    if wire is not None:
        # Re-admission replaces the envelope. Keep the verified limits on that
        # engine, so every member of a batch need not re-hash the engine plan.
        cached = getattr(config, "_gr_beam_limits", None)
        if isinstance(cached, tuple) and cached[0] is wire:
            return cached[1]
        admitted = read_resolved_choices(config)
        assert admitted is not None
        fields = admitted.choices.to_dict().get("beam_limits")
        required = {
            "device_type",
            "max_decode_rows",
            "retain_candidate_widths",
            "constraint_backend",
            "runner_generation",
            "local_kv_heads",
            "head_size",
        }
        optional = {
            "supports_unconstrained",
            "supports_dynamic_width",
            "supports_online_trie_reload",
            "host_policy",
        }
        if not isinstance(fields, dict) or not required <= fields.keys() <= required | optional:
            raise ValueError("Invalid Beam admission limits")
        limits = BeamLimits(**fields, backend_id=admitted.backend_id)
        try:
            object.__setattr__(config, "_gr_beam_limits", (wire, limits))
        except (AttributeError, TypeError):
            pass
        return limits
    # Before admission a host config may carry placement metadata, but it
    # cannot assert any execution capability or constraint implementation.
    device = getattr(getattr(config, "device_config", None), "device", None)
    placement = str(device).split(":", 1)[0] if device is not None else None
    generation = "v2" if getattr(config, "use_v2_model_runner", False) else "v1"
    return BeamLimits(
        placement,
        None,
        False,
        None,
        generation,
        None,
        None,
    )


def validate_beam_batch(request: BeamBatchRequest, limits: BeamLimits) -> None:
    """Check the peak number of input rows across Decode stages."""
    budget = limits.max_decode_rows
    if budget is None or request.beam_params.max_tokens == 1:
        return
    params = request.beam_params
    # Prefill produces the first winners. Each remaining invocation consumes
    # the preceding stage's winners; the final output is never a model input.
    input_width = (
        params.beam_width
        if params.dynamic_beam_widths is None
        else max(params.dynamic_beam_widths[:-1])
    )
    physical_tokens = len(request.items) * input_width
    if physical_tokens > budget:
        device = limits.device_type.upper() if limits.device_type else "Native"
        raise ValueError(
            f"{device} Beam Decode requires {physical_tokens} physical tokens "
            f"(batch_size={len(request.items)} * peak_input_width={input_width}), "
            f"exceeding max_num_batched_tokens={budget}; increase the token budget "
            "or reduce the batch size or Decode input widths"
        )
