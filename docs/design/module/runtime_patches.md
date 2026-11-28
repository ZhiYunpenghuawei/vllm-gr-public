# Runtime patches and upstream dependencies

Three files describe the same patch set:

| File | Role |
| --- | --- |
| [patch_inventory.yaml](../patch_inventory.yaml) | Authoritative data: installer, target, process roles, risk, tests and removal condition. |
| [Runtime patch inventory](../runtime_patch_inventory.md) | Checked index: active IDs in contract-test order, scope, backend ownership, owner, installer and risk. |
| This page | Rationale: the behavior each hook protects and the upstream assumption it depends on. |

Backend ownership identifies the shipped platform class responsible for an inventoried patch. Installation
can still happen later in a worker or during native configuration.

## GR-PATCH-000 — Runtime Plugin Activation

Scope: `core`; backend ID: `null`

The general plugin reads vLLM's native platform identity and resolves a GR platform class from installed probes and the built-in table. It sets ModelRunner defaults and checks runtime
compatibility before installing platform and shared patches. The frontend and EngineCore use their own ordered plans. Each
patch module verifies its bindings so repeated initialization can reuse an
existing wrapper. The selected GR platform is installed as the process
`current_platform`. `GRPlatform.get_attn_backend_cls` selects the Beam
backend or delegates to the native platform; GR no longer claims a global
CUSTOM registry slot.

## GR-PATCH-003 — FlatLogprobs Fast Append Override

Scope: `core`; backend ID: `null`

The frontend appends flattened log probabilities through the GR implementation
of the native fast-append method in
[logprobs_patch.py](../../../vllm_gr/v1/engine/logprobs_patch.py).
Its arguments and the upstream flattened representation must stay compatible.
The patch test checks offsets, ranks, and decoded tokens for empty and nonempty
entries and omitted optional columns.

## GR-PATCH-004 — OpenAI Beam Serving Override

Scope: `core`; backend ID: `null`

The serving frontend groups prompts into one Beam submission and formats their
ranked candidates for the OpenAI-compatible API. The wrapper depends on the
native serving and protocol methods it replaces. The patch validates the Chat
and Completion sampling methods and the Completion `vllm_xargs` schema for
integer lists. A missing binding or a mismatched schema triggers reinstallation.

## GR-PATCH-005 — EngineArgs Configuration Override

Scope: `core`; backend ID: `null`

GR serve and GRLLM pass a validated snapshot through native
`EngineArgs.additional_config`. The GR hook on `create_engine_config` resolves
device-dependent settings and admits the backend plan before worker startup.

## GR-PATCH-007 — EngineCore Client and AsyncLLM Extensions

Scope: `core`; backend ID: `null`

Client extensions submit a whole Beam session and route its wait, abort and
release operations. A rejected submission must preserve the identity that the
frontend uses to complete the request.

## GR-PATCH-008 — EngineCoreRequestType Extension

Scope: `core`; backend ID: `null`

GR adds a wire discriminator for grouped session submission to the native
request type. An upstream enum value collision could send a valid message to
the wrong handler.

## GR-PATCH-009 — EngineCore Lifecycle and Socket Processing

Scope: `core`; backend ID: `null`

EngineCore admits grouped submissions and routes their terminal results and
retirement acknowledgements. Native socket dispatch and output collection must
keep ordinary requests progressing alongside GR work.

## GR-PATCH-010 — Scheduler Beam Override

Scope: `core`; backend ID: `null`

Scheduling attaches Beam stage and retirement metadata to native scheduler
outputs. Logical token accounting and outstanding worker receipts must agree
before the next owner can be admitted.

## GR-PATCH-022 — Prefix-cache hash memoization

Scope: `core`; backend ID: `null`

Bounded per-process memoization avoids hashing identical prefix blocks and
prompts repeatedly. Reinstallation checks the wrappers and the EngineCore
binding that constructs the hasher.

## GR-PATCH-011 — GPU ModelRunner V1/V2 Overrides

Scope: `backend`; backend ID: `cuda`

CUDA ModelRunner V1 and V2 adapt native input preparation, stage execution and
output completion. Request ordering, graph-visible storage and native lifecycle
boundaries have to agree. See the [runner contract](model_runner_v2.md).

## GR-PATCH-015 — Sliced Vocabulary Projection

Scope: `core`; backend ID: `null`

The worker binds the loaded table's token window to each model's logits processor.
Processors without a window use native full-vocabulary projection. Sliced
projection also adapts log-probability collection. Head-derived caches must stay valid across
native weight reload.

## Retired patches

GR-PATCH-001 and GR-PATCH-013 were replaced by the unified platform attention
selection seam. The remaining legacy built-in Ascend hooks (002, 012, 016,
018, 019, 021) were removed with the old device integration. Refer to the
[checked inventory](../runtime_patch_inventory.md#retired-patches) for each
historical target and rationale.

Retired IDs stay in the YAML inventory for traceability. They have no active
installer.

| ID | Historical target | Why it was removed |
| --- | --- | --- |
| `GR-PATCH-006` | `OpenAIServingModels.__init__` | Legacy Catalog serving patch and its tests were removed; canonical V1 constraints load in the worker. |
| `GR-PATCH-014` | LMCache request tracking | Experimental LMCache integration removed; no supported version pin or vLLM 0.30 validation. |
| `GR-PATCH-017` | `NPUWorker.profile_memory` | vllm-ascend 0.30.0rc2 replaced the method with `log_memory_stats`; no native method remained to wrap. |
| `GR-PATCH-020` | `vllm.config.ModelConfig.maybe_untie_word_embeddings` | Model-specific tied-embedding compatibility was retired with its deployment recipe. |

## Maintenance rules

1. Update the YAML entry, the checked index and the rationale here together.
2. Keep the patch ID, scope, backend ID and ordering stable; the
   [upstream contract test](../../../tests/contract/test_upstream_patch_contracts.py)
   verifies them.
3. Run the tests or validators listed for the changed patch. Do not cite a
   test that does not exercise the hook.
4. When upgrading vLLM, recheck private methods, descriptor types, wire tags and
   process-specific bindings.

Retired IDs stay in the YAML inventory for traceability. Benchmark-only shims
belong to their benchmark and do not participate in production startup. See
the [installation guide](../../getting_started/installation/README.md) for
supported runtime requirements.
