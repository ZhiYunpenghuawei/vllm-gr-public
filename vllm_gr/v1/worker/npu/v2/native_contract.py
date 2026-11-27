# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Version-bound native interfaces for the owned NPU V2 runner.

This module records audited API differences separately from runtime admission.
The v0.30.0 implementation target still requires device qualification; the
v0.34.0 contract is an upgrade audit candidate, not an enabled backend.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from vllm.v1.outputs import ModelRunnerOutput


_OutputT = TypeVar("_OutputT")
_IMPLEMENTATION_TARGET = "0.30.0"


def _request_indices(req_ids: Sequence[str]) -> dict[str, int]:
    indices = {req_id: index for index, req_id in enumerate(req_ids)}
    if len(indices) != len(req_ids):
        raise ValueError("Model runner output contains duplicate request IDs")
    return indices


@dataclass(frozen=True, slots=True)
class NativeRuntimeContract:
    """Explicit native output ABI for an audited vLLM release."""

    public_version: str
    output_has_request_index: bool

    def make_output(
        self,
        output_type: Callable[..., _OutputT],
        *,
        req_ids: Sequence[str],
        sampled_token_ids: list[list[int]] | None = None,
        **fields: Any,
    ) -> _OutputT:
        """Preserve request ordering across native constructor changes."""
        if "req_id_to_index" in fields:
            raise ValueError("Native request indices must be derived from req_ids")
        ids = list(req_ids)
        indices = _request_indices(ids)
        if sampled_token_ids:
            if len(sampled_token_ids) != len(ids):
                raise ValueError("Sampled token rows must match the request IDs")
        if sampled_token_ids is not None:
            fields["sampled_token_ids"] = sampled_token_ids
        if self.output_has_request_index:
            fields["req_id_to_index"] = indices
        return output_type(req_ids=ids, **fields)

    def output_indices(self, output: Any) -> dict[str, int]:
        """Expose a consistent request-index view to owned GR consumers."""
        indices = _request_indices(output.req_ids)
        if self.output_has_request_index and output.req_id_to_index != indices:
            raise ValueError("Native output request index does not match req_ids")
        return indices


def contract_for_version(installed_version: str) -> NativeRuntimeContract:
    """Return an audited contract; this does not admit a version for execution."""
    public_version = installed_version.partition("+")[0]
    # v0.30.0 ModelRunnerOutput requires req_id_to_index:
    # https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/outputs.py#L324-L328
    # v0.34.0 removes it; the native scheduler derives it from req_ids:
    # https://github.com/vllm-project/vllm/blob/v0.34.0/vllm/v1/outputs.py#L267-L275
    # https://github.com/vllm-project/vllm/blob/v0.34.0/vllm/v1/core/sched/scheduler.py#L2357
    if public_version == "0.30.0":
        return NativeRuntimeContract(public_version, output_has_request_index=True)
    if public_version == "0.34.0":
        return NativeRuntimeContract(public_version, output_has_request_index=False)
    raise ValueError(f"No audited native NPU runtime contract for vLLM {installed_version}")


def active_native_contract() -> NativeRuntimeContract:
    """Resolve only the current implementation target at runtime."""
    contract = contract_for_version(version("vllm"))
    if contract.public_version != _IMPLEMENTATION_TARGET:
        raise RuntimeError(
            f"vLLM {contract.public_version} has an audited migration seam but is not an enabled NPU runtime"
        )
    return contract


def make_model_runner_output(
    *,
    req_ids: Sequence[str],
    sampled_token_ids: list[list[int]] | None = None,
    **fields: Any,
) -> ModelRunnerOutput:
    """Construct the active version's native output at the owned boundary."""
    contract = active_native_contract()
    from vllm.v1.outputs import ModelRunnerOutput

    return contract.make_output(
        ModelRunnerOutput,
        req_ids=req_ids,
        sampled_token_ids=sampled_token_ids,
        **fields,
    )
