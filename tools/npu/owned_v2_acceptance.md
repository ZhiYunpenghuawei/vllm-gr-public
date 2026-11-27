# Owned NPU ModelRunner V2: development and acceptance

This development branch targets the vLLM 0.30.0 API first (R0 candidate),
then an independent vLLM 0.34.0 upgrade exercise (R1 candidate).
Neither candidate is a validated NPU runtime yet. Retain the old matched
vLLM-Ascend stack in a separate environment for baseline comparison.

Initial scope is single-device, unquantized FP16/BF16 dense decoder GR Beam,
including existing constraint, dynamic-width, and required Beam graph behavior.
TP/PP/DP/PCP/DCP must all be one. Reject KV/EC transfer, speculative decoding,
LoRA, MoE, quantization, and multimodal models. Ordinary generate, penalties,
prompt logprobs, and ordinary FULL graphs are outside this initial scope;
reject unsupported paths explicitly. The final support matrix requires device evidence.

## Task 01: feasibility

The native GPU V2 runner creates CUDA streams and uses device-specific input,
state, sampling, and graph kernels. The reference Ascend V2 integration also
patches UVA, block tables, input batches, model state, and Triton operations.
Changing the device string or inheriting the GPU runner without these replacements
does not establish compatibility. Use a composed NPU runner with native scheduler
contracts, owned input/KV implementation, and verified native dense model layers.

References:

- [Native V2 runner](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/worker/gpu/model_runner.py)
- [Native buffer utilities](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/worker/gpu/buffer_utils.py)
- [Ascend V2 runner](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0rc2/vllm_ascend/worker/v2/model_runner.py)
- [Ascend worker patches](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0rc2/vllm_ascend/patch/worker/__init__.py)

Run the preflight in the selected interpreter without installing vLLM-Ascend:

```bash
python tools/npu/probe_owned_v2.py --vllm-version 0.30.0 --output preflight.json
```

On an assigned NPU, add `--device 0` (or the assigned visible index) to a new
receipt path. The optional probe executes dense Linear on FP16/BF16 inputs and
compares actual device values with a CPU reference. It does not load a real model
or prove V2 feasibility. Never interpret metadata-only success as device success.

The development environment has no NPU device or torch-npu/CANN stack.
Device qualification is pending: record CANN/driver/device identity, xllm-ops
build identity, real model/tokenizer revision, baseline outputs and metrics,
and execute a no-Ascend native V2 model load/forward. Later implementation
commits remain experimental until the corresponding acceptance checks pass.
