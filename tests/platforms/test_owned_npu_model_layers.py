# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Host checks for the owned NPU native model-layer admission contract."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from vllm_gr.v1.worker.npu import model_layers

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


@dataclass
class _Priorities:
    rms_norm: list[str] = field(default_factory=lambda: ["native"])
    fused_add_rms_norm: list[str] = field(default_factory=lambda: ["native"])
    gelu_and_mul_sparse: list[str] = field(default_factory=lambda: ["native"])


@pytest.fixture
def config() -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architecture="Qwen3ForCausalLM",
            model_impl="auto",
            runner_type="generate",
            attn_type="decoder",
            dtype="torch.bfloat16",
            quantization=None,
            get_sliding_window=lambda: None,
        ),
        quant_config=None,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        kernel_config=SimpleNamespace(linear_backend="auto", ir_op_priority=_Priorities()),
    )


@pytest.mark.parametrize(
    "architecture", ["Qwen3ForCausalLM", "Qwen2ForCausalLM", "LlamaForCausalLM"]
)
@pytest.mark.parametrize("dtype", ["torch.float16", "torch.bfloat16"])
def test_dense_model_admitted(config: SimpleNamespace, architecture: str, dtype: str) -> None:
    config.model_config.architecture = architecture
    config.model_config.dtype = dtype
    model_layers.validate_supported_model_config(config)


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("architecture", "Qwen3MoeForCausalLM", "architecture"),
        ("model_impl", "transformers", "native vLLM model implementation"),
        (
            "model_class_overrides",
            {"Qwen3ForCausalLM": "custom:Layer"},
            "native vLLM model implementation",
        ),
        ("runner_type", "pooling", "decoder-only"),
        ("attn_type", "encoder_only", "decoder-only"),
        ("is_moe", True, "is_moe"),
        ("uses_mrope", True, "uses_mrope"),
        ("is_hybrid", True, "is_hybrid"),
        ("dtype", "torch.float32", "FP16 or BF16"),
        ("quantization", "awq", "quantization"),
        ("is_quantized", True, "quantization"),
    ],
)
def test_unsupported_model_rejected(
    config: SimpleNamespace, attribute: str, value: object, message: str
) -> None:
    setattr(config.model_config, attribute, value)
    with pytest.raises(ValueError, match=message):
        model_layers.validate_supported_model_config(config)


def test_resolved_quantization_and_sliding_window_rejected(config: SimpleNamespace) -> None:
    config.quant_config = object()
    with pytest.raises(ValueError, match="quantization"):
        model_layers.validate_supported_model_config(config)
    config.quant_config = None
    config.model_config.get_sliding_window = lambda: 128
    with pytest.raises(ValueError, match="sliding-window"):
        model_layers.validate_supported_model_config(config)


def test_quantized_or_mismatched_kv_cache_rejected(config: SimpleNamespace) -> None:
    for dtype in ("fp8", "fp16"):
        config.cache_config.cache_dtype = dtype
        with pytest.raises(ValueError, match="KV cache dtype"):
            model_layers.validate_supported_model_config(config)
    config.cache_config.cache_dtype = "bfloat16"
    model_layers.validate_supported_model_config(config)


def test_unsupported_kernel_dispatch_rejected(config: SimpleNamespace) -> None:
    config.kernel_config.linear_backend = "cutlass"
    with pytest.raises(ValueError, match="native Linear"):
        model_layers.validate_supported_model_config(config)
    config.kernel_config.linear_backend = "auto"
    # Native layer methods must not dispatch to a GPU provider through IR.
    config.kernel_config.ir_op_priority.fused_add_rms_norm = ["vllm_c", "native"]
    with pytest.raises(ValueError, match="fused_add_rms_norm.*native priority"):
        model_layers.validate_supported_model_config(config)
    config.kernel_config.ir_op_priority.fused_add_rms_norm = []
    model_layers.validate_supported_model_config(config)


@pytest.mark.parametrize(
    "name", ["tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"]
)
def test_parallel_model_rejected(config: SimpleNamespace, name: str) -> None:
    setattr(config.parallel_config, name, 2)
    with pytest.raises(ValueError, match="TP/PP/DP/DCP/PCP=1"):
        model_layers.validate_supported_model_config(config)


def test_clean_model_layer_registry_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.model_executor import custom_op

    class ForeignNorm:
        pass

    monkeypatch.setattr(custom_op, "op_registry_oot", {})
    model_layers.validate_model_layer_registration()
    monkeypatch.setattr(custom_op, "op_registry_oot", {"RMSNorm": ForeignNorm})
    with pytest.raises(RuntimeError, match="RMSNorm=.*ForeignNorm"):
        model_layers.validate_model_layer_registration()
    # Refusal must not silently clear state owned by another platform.
    assert custom_op.op_registry_oot["RMSNorm"] is ForeignNorm


def test_native_ir_import_and_priority(config: SimpleNamespace) -> None:
    from vllm.config.kernel import IrOpPriorityConfig
    from vllm.ir import ops

    model_layers.import_model_layer_kernels()
    config.kernel_config.ir_op_priority = IrOpPriorityConfig.with_default(["native"])
    priority = model_layers.get_default_ir_op_priority(config)
    assert priority.rms_norm == ["native"]
    assert priority.fused_add_rms_norm == ["native"]
    assert "native" in ops.rms_norm.impls
    assert "native" in ops.fused_add_rms_norm.impls
