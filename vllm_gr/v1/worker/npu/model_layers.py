# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Native model-layer contract for the owned NPU ModelRunner V2 backend.

The first backend uses the upstream dense decoder layers with native Torch
implementations. With an OOT platform, CustomOp.forward_oot delegates to
forward_native. UnquantizedLinearMethod uses F.linear on non-CUDA platforms,
and vocabulary and linear layers skip TP collectives when TP=1. This preserves
native weight loading, tied weights, RoPE, and the residual RMSNorm contract.
Device parity and performance still require the real NPU runtime.
"""

from typing import Any

SUPPORTED_ARCHITECTURES = frozenset({"LlamaForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM"})
_SUPPORTED_DTYPES = frozenset({"torch.float16", "torch.bfloat16"})


def validate_supported_model_config(vllm_config: Any) -> None:
    """Admit only the dense, unquantized native decoder layer scope."""
    model = vllm_config.model_config
    if model is None:
        raise ValueError("Owned NPU model layers require a model configuration")
    if model.architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Unsupported owned NPU architecture: {model.architecture}")
    if model.model_impl not in ("auto", "vllm") or getattr(model, "model_class_overrides", None):
        raise ValueError("Owned NPU model layers require the native vLLM model implementation")
    if model.runner_type != "generate" or model.attn_type != "decoder":
        raise ValueError("Owned NPU model layers require decoder-only text generation")
    for name in (
        "is_encoder_decoder",
        "is_multimodal_model",
        "is_hybrid",
        "is_moe",
        "is_attention_free",
        "has_noops",
        "uses_mrope",
        "is_diffusion",
    ):
        if getattr(model, name, False):
            raise ValueError(f"Owned NPU model layers do not support {name}")
    if model.get_sliding_window() is not None:
        raise ValueError("Owned NPU model layers do not support sliding-window attention")
    if (
        model.quantization is not None
        or getattr(model, "is_quantized", False)
        or vllm_config.quant_config is not None
    ):
        raise ValueError("Owned NPU model layers do not support quantization")
    if str(model.dtype) not in _SUPPORTED_DTYPES:
        raise ValueError("Owned NPU model layers require FP16 or BF16")

    parallel = vllm_config.parallel_config
    for name in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
    ):
        if getattr(parallel, name, 1) != 1:
            raise ValueError("Owned NPU model layers require TP/PP/DP/DCP/PCP=1")

    # Keep both the computation and weight layout on the native floating-point
    # path. A quantized KV cache needs a different attention implementation.
    cache_dtype = vllm_config.cache_config.cache_dtype
    dtype_aliases = (
        ("auto", "float16", "fp16", "half")
        if str(model.dtype) == "torch.float16"
        else ("auto", "bfloat16", "bf16")
    )
    if cache_dtype not in dtype_aliases:
        raise ValueError("Owned NPU KV cache dtype must match the model dtype")

    kernel = vllm_config.kernel_config
    if kernel.linear_backend != "auto" or getattr(kernel, "linear_backend_per_quant", None):
        raise ValueError("Owned NPU model layers require automatic native Linear dispatch")
    _validate_native_ir_priority(kernel.ir_op_priority)


def _validate_native_ir_priority(priority: Any) -> None:
    # vLLM's config defaults may still be empty before platform defaults are
    # appended. Do not accept provider names from other accelerator backends.
    from dataclasses import fields

    for field in fields(priority):
        providers = getattr(priority, field.name)
        if providers not in ([], ["native"]):
            raise ValueError(f"Owned NPU IR operator {field.name} requires native priority")


def get_default_ir_op_priority(vllm_config: Any) -> Any:
    """Select native IR kernels rather than importing CUDA kernel providers."""
    from vllm.config.kernel import IrOpPriorityConfig

    _validate_native_ir_priority(vllm_config.kernel_config.ir_op_priority)
    return IrOpPriorityConfig.with_default(["native"])


def import_model_layer_kernels() -> None:
    """Register native IR definitions, without vLLM's device kernel package."""
    from vllm.ir import ops  # noqa: F401


def validate_model_layer_registration() -> None:
    """Prevent an installed plugin from replacing the native layer contract.

    A process with another OOT layer registry is not a clean owned-NPU runtime.
    Do not delete registry entries: their associated state and imported kernels
    may already have been initialized by the other backend.
    """
    from vllm.model_executor.custom_op import op_registry_oot

    if op_registry_oot:
        registered = ", ".join(
            f"{name}={layer.__module__}.{layer.__qualname__}"
            for name, layer in sorted(op_registry_oot.items())
        )
        raise RuntimeError(
            f"Unexpected OOT model-layer registrations in owned NPU runtime: {registered}"
        )
