# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
# mypy: disable-error-code="misc"

from __future__ import annotations

import gc
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from tests.constraints.test_table import make_table
from tests.helpers.cuda_platform import CudaGRPlatform
from vllm_gr.config import ConstraintBackendKind, ConstraintTableSpec
from vllm_gr.constraints import cuda_loader, loader
from vllm_gr.constraints.artifact import save_constraint_table_artifact
from vllm_gr.constraints.backend import (
    INVALID_LOGIT,
    ConstraintBackendInfo,
    ConstraintMetadata,
    ConstraintOutput,
)
from vllm_gr.constraints.cpu_backend import CPUConstraintBackend
from vllm_gr.constraints.table import CONSTRAINT_TABLE_TENSOR_NAMES, ConstraintTable

if TYPE_CHECKING:
    from vllm_gr.constraints.cuda_backend import CUDAConstraintBackend


def _config(path: Path, digest: str, **overrides: object) -> ConstraintTableSpec:
    values: dict[str, object] = {
        "enabled": True,
        "backend": ConstraintBackendKind.CPU,
        "path": str(path),
        "max_top_k": 4,
        "artifact_digest": digest,
        "tokenizer_digest": "sha256:" + "b" * 64,
    }
    values.update(overrides)
    return ConstraintTableSpec(**values)


def test_disabled_loader_returns_before_validation_or_io(monkeypatch: pytest.MonkeyPatch) -> None:
    config = ConstraintTableSpec(max_top_k=1, enabled=False)
    monkeypatch.setattr(
        loader,
        "load_constraint_table_artifact",
        lambda *args, **kwargs: pytest.fail("Artifact I/O must not run"),
    )
    assert (
        loader.load_constraint_backend(
            config,
            device="not a device",
            vocab_size=-1,
            dtype=torch.int64,
            max_rows=0,
        )
        is None
    )


def test_cpu_loader_constructs_info_from_canonical_artifact(tmp_path: Path) -> None:
    path = tmp_path / "table.safetensors"
    digest = save_constraint_table_artifact(make_table(), path)
    backend = loader.load_constraint_backend(
        _config(path, digest),
        device=torch.device("cpu"),
        vocab_size=6,
        dtype=torch.float32,
        max_rows=7,
    )
    assert backend is not None
    assert backend.info.backend is ConstraintBackendKind.CPU
    assert backend.info.host_memory_bytes == make_table().total_bytes
    assert backend.info.device_memory_bytes == 0
    assert backend.info.max_rows == 7


@pytest.mark.parametrize("selector", ["auto", "platform", "example_backend"])
def test_platform_loader_routes_opaque_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selector: str
) -> None:
    path = tmp_path / "table.safetensors"
    table = make_table()
    digest = save_constraint_table_artifact(table, path)
    backend = SimpleNamespace(close=Mock())

    def create_backend(loaded_table: ConstraintTable, **kwargs: object) -> SimpleNamespace:
        # The persisted artifact supplies the authoritative metadata, including
        # its digest. The factory must report that metadata, not pre-save data.
        backend.info = ConstraintBackendInfo(
            loaded_table.metadata, "example_backend", 4, 2, 0, loaded_table.total_bytes
        )
        return backend

    platform = SimpleNamespace(
        backend_id="example",
        canonical_constraint_backend="example_backend",
        validate_constraint_backend=Mock(),
        create_constraint_backend=Mock(side_effect=create_backend),
    )
    monkeypatch.setattr("vllm_gr.platforms.table.current_gr_platform", lambda: platform)
    device = torch.device("cpu")
    result = loader.load_constraint_backend(
        _config(path, digest, backend=selector),
        device=device,
        vocab_size=6,
        dtype=torch.float32,
        max_rows=2,
    )
    assert result is backend
    platform.validate_constraint_backend.assert_called_once_with(
        device=device, dtype=torch.float32, max_rows=2, max_top_k=4
    )
    args, kwargs = platform.create_constraint_backend.call_args
    assert args[0].metadata.artifact_digest == digest
    assert args[0].metadata.tokenizer_digest == table.metadata.tokenizer_digest
    assert kwargs == dict(device=device, dtype=torch.float32, max_rows=2, max_top_k=4)
    backend.close.assert_not_called()


def test_platform_loader_rejects_mismatch_and_closes_returned_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "table.safetensors"
    table = make_table()
    digest = save_constraint_table_artifact(table, path)
    backend = SimpleNamespace(close=Mock())

    def create_backend(loaded_table: ConstraintTable, **kwargs: object) -> SimpleNamespace:
        backend.info = ConstraintBackendInfo(
            loaded_table.metadata, "wrong_backend", 4, 2, 0, loaded_table.total_bytes
        )
        return backend

    platform = SimpleNamespace(
        backend_id="example",
        canonical_constraint_backend="example_backend",
        validate_constraint_backend=Mock(),
        create_constraint_backend=Mock(side_effect=create_backend),
    )
    monkeypatch.setattr("vllm_gr.platforms.table.current_gr_platform", lambda: platform)
    device = torch.device("cpu")
    with pytest.raises(ValueError, match="does not match the selected"):
        loader.load_constraint_backend(
            _config(path, digest, backend="wrong_backend"),
            device=device,
            vocab_size=6,
            dtype=torch.float32,
            max_rows=2,
        )
    platform.validate_constraint_backend.assert_not_called()
    platform.create_constraint_backend.assert_not_called()
    with pytest.raises(ValueError, match="incompatible constraint metadata"):
        loader.load_constraint_backend(
            _config(path, digest, backend="example_backend"),
            device=device,
            vocab_size=6,
            dtype=torch.float32,
            max_rows=2,
        )
    backend.close.assert_called_once_with()


def test_cuda_top_k_limit_fails_before_artifact_io(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("triton")
    monkeypatch.setattr("vllm_gr.platforms.table.current_gr_platform", CudaGRPlatform)
    config = ConstraintTableSpec(
        enabled=True,
        backend="cuda",
        path="/does/not/exist",
        artifact_digest="sha256:" + "a" * 64,
        tokenizer_digest="sha256:" + "b" * 64,
        max_top_k=257,
    )
    monkeypatch.setattr(
        loader,
        "load_constraint_table_artifact",
        lambda *args, **kwargs: pytest.fail("Artifact I/O must not run"),
    )
    with pytest.raises(ValueError, match="256"):
        loader.load_constraint_backend(
            config,
            device=torch.device("privateuseone"),
            vocab_size=6,
            dtype=torch.float16,
            max_rows=2,
        )


@pytest.mark.parametrize("failure_index", range(1, 7))
def test_upload_failure_rethrows_original_exception_and_releases_tensors(
    failure_index: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("triton")
    table = make_table()
    error = RuntimeError("third upload failed")
    references: list[weakref.ReferenceType[torch.Tensor]] = []
    calls = 0

    def upload(array: np.ndarray, device: torch.device) -> torch.Tensor:
        nonlocal calls
        del array, device
        calls += 1
        if calls == failure_index:
            raise error
        tensor = torch.zeros(1)
        references.append(weakref.ref(tensor))
        return tensor

    monkeypatch.setattr(cuda_loader, "_upload_table_tensor", upload)
    with pytest.raises(RuntimeError) as raised:
        cuda_loader._upload_cuda_backend(  # noqa: SLF001
            table,
            device=torch.device("cpu"),
            max_top_k=2,
            max_rows=2,
        )
    assert raised.value is error
    gc.collect()
    assert all(reference() is None for reference in references)


def test_upload_sync_failure_releases_all_created_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("triton")
    table = make_table()
    error = RuntimeError("upload stream sync failed")
    references: list[weakref.ReferenceType[torch.Tensor]] = []

    def upload(array: np.ndarray, device: torch.device) -> torch.Tensor:
        del array, device
        tensor = torch.zeros(1)
        references.append(weakref.ref(tensor))
        return tensor

    def fail_sync(device: torch.device) -> None:
        del device
        raise error

    monkeypatch.setattr(cuda_loader, "_upload_table_tensor", upload)
    monkeypatch.setattr(cuda_loader, "_synchronize_current_cuda_stream", fail_sync)
    with pytest.raises(RuntimeError) as raised:
        cuda_loader._upload_cuda_backend(  # noqa: SLF001
            table,
            device=torch.device("cpu"),
            max_top_k=2,
            max_rows=2,
        )
    assert raised.value is error
    gc.collect()
    assert all(reference() is None for reference in references)


def test_unique_storage_memory_is_deduplicated() -> None:
    base = torch.zeros(16, dtype=torch.int32)
    other = torch.zeros(3, dtype=torch.int64)
    tensors = {"base": base, "view": base[2:8], "other": other}
    assert cuda_loader._unique_storage_bytes(tensors) == (  # noqa: SLF001
        base.untyped_storage().nbytes() + other.untyped_storage().nbytes()
    )


def _readiness_backend(
    table: ConstraintTable, top_k: int
) -> tuple[CUDAConstraintBackend, CPUConstraintBackend]:
    tensors = {
        name: torch.from_numpy(np.array(getattr(table, name), copy=True))
        for name in CONSTRAINT_TABLE_TENSOR_NAMES
    }
    pytest.importorskip("triton")
    from vllm_gr.constraints.cuda_backend import CUDAConstraintBackend

    backend = CUDAConstraintBackend(
        **tensors,
        info=ConstraintBackendInfo(table.metadata, "cuda", top_k, 2, 0, table.total_bytes),
    )
    oracle = CPUConstraintBackend(
        table,
        ConstraintBackendInfo(
            table.metadata, ConstraintBackendKind.CPU, top_k, 2, table.total_bytes, 0
        ),
    )
    return backend, oracle


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_readiness_accepts_normalized_reference_at_top_k_256(
    dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = make_table(first_token_ids=np.asarray([1, 3, 4], dtype=np.int32))
    backend, oracle = _readiness_backend(table, 256)
    monkeypatch.setattr(cuda_loader, "_synchronize_current_cuda_stream", lambda device: None)
    monkeypatch.setattr(
        type(backend),
        "_select_for_readiness",
        lambda self, logits, metadata: oracle.select(logits, metadata),
    )
    try:
        cuda_loader._run_cuda_readiness(
            backend,
            cases=cuda_loader._readiness_cases(table, 256),
            device=torch.device("cpu"),
            dtype=dtype,
            max_rows=2,
        )
    finally:
        backend.close()
        oracle.close()


@pytest.mark.parametrize("failure_mode", ["raw", "ignore_temperature", "top_k_normalized"])
def test_readiness_rejects_unnormalized_logprobs(
    failure_mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = make_table(first_token_ids=np.asarray([1, 3, 4], dtype=np.int32))
    backend, oracle = _readiness_backend(table, 2)

    def bad_select(
        self: CUDAConstraintBackend, logits: torch.Tensor, metadata: ConstraintMetadata
    ) -> ConstraintOutput:
        del self
        if failure_mode == "ignore_temperature":
            return oracle.select(
                logits, replace(metadata, temperatures=torch.ones_like(metadata.temperatures))
            )
        output = oracle.select(logits, metadata)
        scores = logits.gather(1, output.token_ids.long()).float()
        if failure_mode == "top_k_normalized":
            temp = metadata.temperatures.reshape(-1, 1)
            scores = scores / torch.where(temp == 0, 1.0, temp)
            scores = scores.masked_fill(~output.valid_mask, float("-inf"))
            scores -= torch.logsumexp(scores, dim=1, keepdim=True)
        return replace(output, logprobs=torch.where(output.valid_mask, scores, INVALID_LOGIT))

    monkeypatch.setattr(type(backend), "_select_for_readiness", bad_select)
    monkeypatch.setattr(cuda_loader, "_synchronize_current_cuda_stream", lambda device: None)
    try:
        with pytest.raises(RuntimeError, match="normalized reference"):
            cuda_loader._run_cuda_readiness(
                backend,
                cases=cuda_loader._readiness_cases(table, 2),
                device=torch.device("cpu"),
                dtype=torch.float32,
                max_rows=2,
            )
    finally:
        backend.close()
        oracle.close()


def test_readiness_failure_closes_backend_and_preserves_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = make_table()
    path = tmp_path / "table.safetensors"
    digest = save_constraint_table_artifact(table, path)
    backend, oracle = _readiness_backend(table, 2)
    oracle.close()
    error = RuntimeError("readiness launch failed")

    def fail_readiness(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr("vllm_gr.platforms.table.current_gr_platform", CudaGRPlatform)
    monkeypatch.setattr(cuda_loader, "_upload_cuda_backend", lambda *args, **kwargs: (backend, ()))
    monkeypatch.setattr(cuda_loader, "_run_cuda_readiness", fail_readiness)
    with pytest.raises(RuntimeError) as raised:
        loader.load_constraint_backend(
            _config(path, digest, backend="cuda", max_top_k=2),
            device=torch.device("cuda"),
            vocab_size=6,
            dtype=torch.float16,
            max_rows=2,
        )
    assert raised.value is error
    assert backend._first_token_ids is None  # noqa: SLF001


pytestmark = pytest.mark.cpu_test
