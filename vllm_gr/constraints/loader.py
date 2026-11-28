# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Construct an explicit reference or selected platform's canonical backend."""

from __future__ import annotations

import torch

from vllm_gr.config.models import (
    ConstraintBackendKind,
    ConstraintTableSpec,
    constraint_backend_name,
)
from vllm_gr.constraints.artifact import load_constraint_table_artifact
from vllm_gr.constraints.backend import ConstraintBackend, ConstraintBackendInfo
from vllm_gr.constraints.cpu_backend import CPUConstraintBackend

_LOGIT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _validate_loader_inputs(
    *,
    device: torch.device,
    vocab_size: int,
    dtype: torch.dtype,
    max_rows: int,
) -> None:
    if not isinstance(device, torch.device):
        raise ValueError(f"constraint device must be torch.device, got {device!r}")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError(f"constraint vocab_size must be positive, got {vocab_size!r}")
    if dtype not in _LOGIT_DTYPES:
        raise ValueError(f"constraint dtype must be fp16/bf16/fp32, got {dtype}")
    if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows <= 0:
        raise ValueError(f"constraint max_rows must be positive, got {max_rows!r}")


def load_constraint_backend(
    config: ConstraintTableSpec,
    *,
    device: torch.device,
    vocab_size: int,
    dtype: torch.dtype,
    max_rows: int,
) -> ConstraintBackend | None:
    """Load exactly one independent backend, or return None when disabled."""
    if not config.enabled:
        return None
    _validate_loader_inputs(
        device=device,
        vocab_size=vocab_size,
        dtype=dtype,
        max_rows=max_rows,
    )
    if config.path is None or config.artifact_digest is None or config.tokenizer_digest is None:
        raise ValueError("enabled constraint table configuration is incomplete")
    max_top_k = config.max_top_k
    assert max_top_k is not None
    backend_name = constraint_backend_name(config.backend)
    platform = None
    if backend_name != ConstraintBackendKind.CPU.value:
        from vllm_gr.platforms.table import current_gr_platform

        platform = current_gr_platform()
        declared = platform.canonical_constraint_backend
        if backend_name in (ConstraintBackendKind.AUTO.value, ConstraintBackendKind.PLATFORM.value):
            backend_name = declared
        if backend_name != declared:
            raise ValueError(
                f"constraint backend {backend_name!r} does not match the selected "
                f"GR platform {platform.backend_id!r} implementation {declared!r}"
            )
        platform.validate_constraint_backend(
            device=device, dtype=dtype, max_rows=max_rows, max_top_k=max_top_k
        )

    table = load_constraint_table_artifact(
        config.path,
        expected_digest=config.artifact_digest,
        expected_tokenizer_digest=config.tokenizer_digest,
        expected_vocab_size=vocab_size,
        expected_format=config.format,
    )
    if platform is None:
        info = ConstraintBackendInfo(
            table_metadata=table.metadata,
            backend=ConstraintBackendKind.CPU,
            max_top_k=max_top_k,
            max_rows=max_rows,
            host_memory_bytes=table.total_bytes,
            device_memory_bytes=0,
        )
        return CPUConstraintBackend(table, info)

    backend = platform.create_constraint_backend(
        table, device=device, dtype=dtype, max_rows=max_rows, max_top_k=max_top_k
    )
    # The ready backend owns its tables until close(). Check metadata before
    # publishing it to the runner or replacing the active version on reload.
    try:
        info = backend.info
        if (
            constraint_backend_name(info.backend) != backend_name
            or info.table_metadata != table.metadata
            or info.max_rows < max_rows
            or info.max_top_k < max_top_k
        ):
            raise ValueError(
                f"GR platform {platform.backend_id!r} returned incompatible constraint metadata"
            )
    except BaseException:
        backend.close()
        raise
    return backend


__all__ = ["load_constraint_backend"]
