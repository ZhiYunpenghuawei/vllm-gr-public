# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Upload and verify a canonical constraint table for the CUDA backend."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numpy.typing import NDArray

from vllm_gr.constraints.backend import (
    INVALID_LOGIT,
    ConstraintBackendInfo,
    ConstraintMetadata,
)
from vllm_gr.constraints.table import (
    CONSTRAINT_TABLE_TENSOR_NAMES,
    ConstraintTable,
    token_id_window,
)

if TYPE_CHECKING:
    from vllm_gr.constraints.cuda_backend import CUDAConstraintBackend

_READINESS_BACKGROUND_LOGIT = -0.25
_READINESS_LOGPROB_ABS_TOL = 1.0e-3
_READINESS_LOGPROB_REL_TOL = 1.0e-4


@dataclass(frozen=True, slots=True)
class _ReadyCase:
    current_step: int
    prefix: tuple[int, int, int]
    candidate_count: int
    first_tokens: tuple[int, ...]
    last_tokens: tuple[int, ...]
    trie_id: int = 0


def _readiness_cases(
    table: ConstraintTable, top_k: int, trie_id: int = 0
) -> tuple[_ReadyCase, ...]:
    """Copy only bounded readiness data from one real a -> b -> c path."""

    first = table.first_token_ids[table.root_offsets[trie_id] : table.root_offsets[trie_id + 1]]
    a = int(first[0])
    row = trie_id * table.metadata.vocab_size + a
    begin1 = int(table.prefix1_offsets[row])
    end1 = int(table.prefix1_offsets[row + 1])
    if end1 <= begin1:  # Defensive: canonical validation already guarantees this.
        raise ValueError("constraint Artifact has no step-1 candidate for its first token")
    b = int(table.prefix1_values[begin1])
    pair_key = row * table.metadata.vocab_size + b
    pair_index = int(np.searchsorted(table.prefix1_pair_keys, pair_key))
    if (
        pair_index >= table.prefix1_pair_keys.size
        or int(table.prefix1_pair_keys[pair_index]) != pair_key
    ):
        raise ValueError("constraint Artifact is missing a real step-2 pair key")
    begin2 = int(table.prefix2_value_offsets[pair_index])
    end2 = int(table.prefix2_value_offsets[pair_index + 1])
    if end2 <= begin2:
        raise ValueError("constraint Artifact has no step-2 candidate for its first pair")

    def ready_case(
        step: int,
        prefix: tuple[int, int, int],
        values: NDArray[Any],
    ) -> _ReadyCase:
        count = int(values.size)
        selected = min(count, top_k)
        return _ReadyCase(
            current_step=step,
            prefix=prefix,
            candidate_count=count,
            first_tokens=tuple(int(value) for value in values[:selected]),
            last_tokens=tuple(int(value) for value in values[count - selected :]),
            trie_id=trie_id,
        )

    return (
        ready_case(0, (0, 0, 0), first),
        ready_case(1, (a, 0, 0), table.prefix1_values[begin1:end1]),
        ready_case(2, (a, b, 0), table.prefix2_values[begin2:end2]),
    )


def _upload_table_tensor(array: NDArray[Any], device: torch.device) -> torch.Tensor:
    # Canonical arrays are permanently read-only. Use a writable staging copy
    # so torch.from_numpy never observes a non-writable buffer.
    staging = np.array(array, copy=True)
    try:
        return torch.from_numpy(staging).to(device=device)
    finally:
        del staging


def _synchronize_current_cuda_stream(device: torch.device) -> None:
    torch.cuda.current_stream(device).synchronize()


def _unique_storage_bytes(tensors: dict[str, torch.Tensor]) -> int:
    """Count each owned storage once, independent of allocator state."""

    seen: set[tuple[str, int, int]] = set()
    total = 0
    for tensor in tensors.values():
        storage = tensor.untyped_storage()
        size = int(storage.nbytes())
        key = (str(tensor.device), int(storage.data_ptr()), size)
        if key not in seen:
            seen.add(key)
            total += size
    return total


def _upload_cuda_backend(
    table: ConstraintTable,
    *,
    device: torch.device,
    max_top_k: int,
    max_rows: int,
) -> tuple[CUDAConstraintBackend, tuple[_ReadyCase, ...]]:
    from vllm_gr.constraints.cuda_backend import CUDAConstraintBackend
    from vllm_gr.ops.cuda.rec_constrained_topk import validate_cuda_sort_capacity

    validate_cuda_sort_capacity(
        table.max_first_degree,
        table.metadata.max_prefix1_degree,
        table.metadata.max_prefix2_degree,
    )
    cases = tuple(
        case
        for trie_id in range(table.metadata.trie_count)
        for case in _readiness_cases(table, max_top_k, trie_id)
    )
    device_tables: dict[str, torch.Tensor] = {}
    backend: CUDAConstraintBackend | None = None
    try:
        for name in CONSTRAINT_TABLE_TENSOR_NAMES:
            device_tables[name] = _upload_table_tensor(getattr(table, name), device)
        if table.metadata.trie_count > 1:
            device_tables["first_token_offsets"] = _upload_table_tensor(table.root_offsets, device)
        _synchronize_current_cuda_stream(device)
        info = ConstraintBackendInfo(
            table_metadata=table.metadata,
            backend="cuda",
            max_top_k=max_top_k,
            max_rows=max_rows,
            host_memory_bytes=0,
            device_memory_bytes=_unique_storage_bytes(device_tables),
        )
        backend = CUDAConstraintBackend(
            first_token_ids=device_tables["first_token_ids"],
            prefix1_offsets=device_tables["prefix1_offsets"],
            prefix1_values=device_tables["prefix1_values"],
            prefix1_pair_keys=device_tables["prefix1_pair_keys"],
            prefix2_value_offsets=device_tables["prefix2_value_offsets"],
            prefix2_values=device_tables["prefix2_values"],
            info=info,
            vocab_slice=token_id_window(
                table.first_token_ids, table.prefix1_values, table.prefix2_values
            ),
            first_token_offsets=device_tables.get("first_token_offsets"),
            max_first_degree=table.max_first_degree,
        )
        return backend, cases
    except BaseException:
        if backend is not None:
            backend.close()
        device_tables.clear()
        raise


def _device_matches(actual: torch.device, expected: torch.device) -> bool:
    """Compare devices the way PyTorch resolves them for placement calls.

    An index-less ``expected`` device (e.g. ``torch.device("cuda")``) targets
    the current device of that type, so ``actual`` carrying an explicit index
    (e.g. ``torch.device("cuda:0")``) is a match — bare ``!=`` would reject it.
    """
    return actual.type == expected.type and (
        expected.index is None or actual.index == expected.index
    )


def _validate_ready_output(
    output: Any,
    *,
    expected_logprobs: list[dict[int, float]],
    candidate_count: int,
    rows: int,
    top_k: int,
    vocab_size: int,
    device: torch.device,
) -> None:
    expected_shape = (rows, top_k)
    if tuple(output.valid_mask.shape) != expected_shape:
        raise RuntimeError(
            f"readiness valid_mask shape {tuple(output.valid_mask.shape)} != {expected_shape}"
        )
    if output.valid_mask.dtype != torch.bool:
        raise RuntimeError(f"readiness valid_mask must be bool, got {output.valid_mask.dtype}")
    if not _device_matches(output.valid_mask.device, device):
        raise RuntimeError(
            f"readiness valid_mask device {output.valid_mask.device} is not the target {device}"
        )
    if not output.valid_mask.is_contiguous():
        raise RuntimeError("readiness valid_mask must be contiguous on the target device")
    expected_valid = min(candidate_count, top_k)
    for row in range(rows):
        row_mask = output.valid_mask[row]
        if int(row_mask.sum().item()) != expected_valid:
            raise RuntimeError(
                f"readiness row {row} returned {int(row_mask.sum().item())} valid slots, "
                f"expected {expected_valid}"
            )
        valid_tokens = output.token_ids[row][row_mask]
        if valid_tokens.numel() and (
            bool((valid_tokens < 0).any()) or bool((valid_tokens >= vocab_size).any())
        ):
            raise RuntimeError("readiness returned an out-of-range valid token")
        expected_row = expected_logprobs[row]
        if tuple(sorted(int(token) for token in valid_tokens.tolist())) != tuple(
            sorted(expected_row)
        ):
            raise RuntimeError("readiness returned tokens outside the expected legal Top-K set")
        valid_logprobs = output.logprobs[row][row_mask]
        if valid_logprobs.numel() and not bool(torch.isfinite(valid_logprobs).all()):
            raise RuntimeError("readiness returned a non-finite valid logprob")
        for token, actual in zip(valid_tokens.tolist(), valid_logprobs.tolist()):
            expected = expected_row[int(token)]
            if not math.isclose(
                float(actual),
                expected,
                rel_tol=_READINESS_LOGPROB_REL_TOL,
                abs_tol=_READINESS_LOGPROB_ABS_TOL,
            ):
                raise RuntimeError(
                    f"readiness row {row} token {int(token)} logprob {float(actual)} "
                    f"does not match normalized reference {expected}"
                )
        padding = output.logprobs[row][~row_mask]
        if padding.numel() and not bool((padding == INVALID_LOGIT).all()):
            raise RuntimeError("readiness padding must equal INVALID_LOGIT")


def _expected_ready_logprobs(
    selected: tuple[int, ...],
    selected_logits: torch.Tensor,
    *,
    normalization_size: int,
    background_logit: float,
    temperature: float,
) -> dict[int, float]:
    """Normalize selected scores without materializing the complete domain."""

    effective_temperature = 1.0 if temperature == 0.0 else temperature
    selected_scores = selected_logits.float() / effective_temperature
    denominator_terms = selected_scores
    remaining = normalization_size - len(selected)
    if remaining:
        repeated_background = background_logit / effective_temperature + math.log(remaining)
        denominator_terms = torch.cat(
            (selected_scores, torch.tensor([repeated_background], dtype=torch.float32))
        )
    log_denominator = torch.logsumexp(denominator_terms, dim=0)
    return {
        token: float(score - log_denominator)
        for token, score in zip(selected, selected_scores.tolist())
    }


def _run_device_readiness(
    backend: CUDAConstraintBackend,
    *,
    cases: tuple[_ReadyCase, ...],
    device: torch.device,
    dtype: torch.dtype,
    max_rows: int,
    synchronize: Callable[[torch.device], None],
) -> None:
    rows = min(max_rows, 2)
    vocab_size = backend.info.table_metadata.vocab_size
    top_k = backend.info.max_top_k
    for case in cases:
        background_logit = float(
            torch.tensor(_READINESS_BACKGROUND_LOGIT, dtype=dtype).float().item()
        )
        logits = torch.full(
            (rows, vocab_size),
            background_logit,
            dtype=dtype,
            device=device,
        )
        expected: list[dict[int, float]] = []
        temperature_values = [
            0.0 if (row + case.current_step) % 2 == 0 else 2.0 for row in range(rows)
        ]
        for row in range(rows):
            selected = case.first_tokens if row % 2 == 0 else tuple(reversed(case.last_tokens))
            if selected:
                indices = torch.tensor(selected, dtype=torch.long, device=device)
                host_values = (
                    torch.arange(
                        len(selected),
                        0,
                        -1,
                        dtype=torch.float32,
                    )
                    / len(selected)
                ).to(dtype=dtype)
                logits[row].index_copy_(0, indices, host_values.to(device=device))
                expected.append(
                    _expected_ready_logprobs(
                        selected,
                        host_values,
                        normalization_size=vocab_size,
                        background_logit=background_logit,
                        temperature=temperature_values[row],
                    )
                )
            else:  # Defensive: canonical readiness cases are always non-empty.
                expected.append({})

        prefixes = torch.tensor(
            [case.prefix] * rows,
            dtype=torch.int32,
            device=device,
        )
        temperatures = torch.tensor(
            temperature_values,
            dtype=torch.float32,
            device=device,
        )
        output = backend._select_for_readiness(  # noqa: SLF001
            logits,
            ConstraintMetadata(
                prefix_token_ids=prefixes,
                temperatures=temperatures,
                current_step=case.current_step,
                top_k=top_k,
                trie_ids=(
                    torch.full((rows,), case.trie_id, dtype=torch.int32, device=device)
                    if backend.info.table_metadata.trie_count > 1
                    else None
                ),
            ),
        )
        synchronize(device)
        _validate_ready_output(
            output,
            expected_logprobs=expected,
            candidate_count=case.candidate_count,
            rows=rows,
            top_k=top_k,
            vocab_size=vocab_size,
            device=device,
        )


def _run_cuda_readiness(
    backend: CUDAConstraintBackend,
    *,
    cases: tuple[_ReadyCase, ...],
    device: torch.device,
    dtype: torch.dtype,
    max_rows: int,
) -> None:
    # CUDA select() uses the shared full-window scoring contract (candidate
    # scores must be comparable across parent rows).  Readiness must verify
    # against the same domain, not the legacy candidate-set normalization.
    _run_device_readiness(
        backend,
        cases=cases,
        device=device,
        dtype=dtype,
        max_rows=max_rows,
        synchronize=_synchronize_current_cuda_stream,
    )


def create_cuda_constraint_backend(
    table: ConstraintTable,
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_rows: int,
    max_top_k: int,
) -> CUDAConstraintBackend:
    """Return a CUDA backend only after all canonical table probes pass."""
    backend, cases = _upload_cuda_backend(
        table, device=device, max_top_k=max_top_k, max_rows=max_rows
    )
    try:
        _run_cuda_readiness(
            backend,
            cases=cases,
            device=device,
            dtype=dtype,
            max_rows=max_rows,
        )
        backend._mark_ready()  # noqa: SLF001
        return backend
    except BaseException:
        backend.close()
        raise
