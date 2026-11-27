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

## Implementation status (2026-11-27)

The ten implementation commits are based on GR revision
`a105884f3378af462b6c05575d2c087202c050a9`. Independent execution is explicitly
selected with `VLLM_GR_OWNED_NPU=1`; the legacy runtime remains the default
until the independent backend is qualified. This is a development branch,
not a production cutover or a claim that vLLM-Ascend has been deleted from
every legacy path in the repository.

| Task | Implemented evidence | Host result | Remaining acceptance |
| --- | --- | --- | --- |
| 01 | Version-bound preflight and optional FP16/BF16 NPU Linear probe | 4 passed; local preflight correctly blocks without torch-npu | Actual model/V2 feasibility, frozen NPU stack and baseline measurements |
| 02 | Owned platform, Worker, HCCL single-rank lifecycle | 23 passed | Real driver/EngineCore/Worker startup and process-group teardown on NPU |
| 03 | Native dense-model layer dispatch, scope and registration checks | 45 passed, including platform regressions | Loaded OneRec checkpoint, layer numerics and actual operator manifest |
| 04 | Owned eager request state, input/KV and pending hidden-state execution | 5 passed | Real multi-request Prefill/Decode, KV parity and measured memory |
| 05 | GR Beam stage adapter and lifecycle binding | 295 passed in the legacy/owned run plus 5 final focused checks; 1 baseline-reproduced failure | Full device constraint, dynamic-width, reorder and SID equivalence |
| 06 | Retirement fences, failure latch, checkpoint transaction | 23 passed | Real cancellation, serving, reload and failure-injection tests |
| 07 | Owned NPU graph registry, exact bindings, reserved-capture/replay API | 13 passed | Production Beam capture provisioning, dynamic replay and NPU graph proof |
| 08 | Exact-length TND/equal-length BSND and ordinary forward graph qualification; budget reservation | 15 passed | Variable-length TND updates, padded BSND and complete memory/performance qualification |
| 09 | Mode-bound environment/CI and addon provenance; old runtime isolation | 63 passed; 11 baseline-reproduced failures | Clean wheel/container installation, real Worker library/import evidence and end-to-end gates |
| 10 | Version-bound native output ABI and localized request index mapping | Final combined suite: 154 passed | Real independently installed R1 and complete upgrade audit/qualification |

These counts are per-run, not unique test counts. All device acceptance items
remain pending. The public owned runtime requires eager execution and both
`gr.beam_graph_enabled=false` and `gr.prefill_graph_enabled=false`. The
graph qualification APIs are usable for reserved startup experiments; they
do not implement the full production graph scope from the original plan.
Ordinary generation and multi-device Beam remain outside initial admission.

The host interpreter used for these checks had Python 3.12.14, vLLM 0.30.0,
Torch 2.13.0, and Triton 3.7.1, with no torch-npu, Triton-Ascend or NPU device.
This is not the required target runtime. The existing public engine config
test in `test_npu_model_runner_patch.py` and eleven stage-projection lifecycle
tests failed with vLLM `DeviceConfig` unable to infer the device type;
the same failures were reproduced using the original checkout and interpreter.
No NPU model, HCCL, ACL graph, xllm kernel, accuracy or performance run was made.

Reproduce the final combined host checks in a prepared development interpreter:

```bash
python -m pytest -q --strict-test-levels \
  tests/tools/test_owned_npu_probe.py \
  tests/test_owned_npu_platform.py \
  tests/platforms/test_owned_npu_model_layers.py \
  tests/v1/worker/test_owned_npu_v2_state.py \
  tests/v1/worker/test_owned_npu_stage_adapter.py \
  tests/v1/worker/test_owned_npu_retirement.py \
  tests/v1/worker/test_owned_npu_v2_lifecycle.py \
  tests/platforms/test_owned_npu_graph.py \
  tests/platforms/test_owned_npu_prefill_graph.py \
  tests/test_owned_npu_runtime.py \
  tests/test_owned_npu_native_contract.py \
  tests/ops/npu/test_npu_xattention_out_binding.py \
  tests/ci/test_ci_runtime.py \
  tests/contract/test_upstream_patch_contracts.py::test_patch_inventories_are_aligned_and_complete \
  tests/contract/test_upstream_patch_contracts.py::test_common_and_selected_provider_patch_ownership_is_unique \
  -m 'cpu_test and not npu_test'
```

## Task 10: version boundary

The owned runner obtains its native contract once at initialization. vLLM
0.30.0 includes `ModelRunnerOutput.req_id_to_index`, while 0.34.0 removes that
field and derives indices from `req_ids`. The output adapter validates request
order and constructs the appropriate output for the audited version. The
tests exercise the actual installed 0.30 constructor and a reduced signature
fixture for 0.34; the latter does not execute the R1 package.

- [R0 output definition](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/outputs.py#L324)
- [R1 output definition](https://github.com/vllm-project/vllm/blob/v0.34.0/vllm/v1/outputs.py#L267)
- [R1 index consumption](https://github.com/vllm-project/vllm/blob/v0.34.0/vllm/v1/core/sched/scheduler.py#L2357)

R1 stays rejected at runtime. Before enabling it, audit every native consumer
of `req_id_to_index`, the scheduler lifecycle and request metadata, KV cache
specification and hash/cache-hit alignment, model loading and platform APIs,
and the graph/stream contract. Freeze the actual compatible Torch/torch-npu,
CANN, Triton and xllm stack; compare both releases with identical model,
tokenizer, dataset, constraints, graph settings and device. Change the default
version pin only after the functional, numerical, graph and performance gates
pass on each release independently.
