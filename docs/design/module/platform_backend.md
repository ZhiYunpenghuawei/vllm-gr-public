# GR platform integration

vLLM selects the native platform and constructs its Worker and ModelRunner.
GR reads `vllm.platforms.current_platform` and uses an in-tree table to
load the matching implementation. The engine admits Beam choices once.
Each Worker binds the selected operations to its own Runner after model load.

## Platform selection

| GR integration | Required native identity | GR Runner |
| --- | --- | --- |
| CUDA | Platform enum `CUDA` and device type `cuda` | CUDA V1 or V2 |
| Ascend | A `vllm_ascend.platform.NPUPlatform` or subclass with device type `npu` | Ascend V1 |

ROCm also reports device type `cuda`. Class lineage distinguishes Ascend
from other out-of-tree platforms. Zero-argument built-in probes inspect the native instance and return a qualified GR class path. Each implementation inherits GRPlatform and its native platform base. The winning instance is cached in each process. [Built-in platform probes](../../../vllm_gr/platforms/table.py)

### Plugin startup

vLLM calls `vllm_gr.plugin:register` through its `vllm.general_plugins`
entry point in the frontend, EngineCore and Worker processes. The plugin
looks up the native platform. On a match, it configures the runner default,
checks the runtime and installs the platform and shared Beam hooks. Ascend
installs spawn protection before loading the native worker. On an unmatched
platform, automatic plugin initialization skips the GR patches. Explicit
GR runtime initialization reports the unsupported native identity.
[Plugin entry point](../../../vllm_gr/plugin.py)

## Engine plans

The engine places the selected platform ID, normalized configuration digest,
software versions, Beam choices and limits in
`additional_config["vllm_gr_choices"]`. Request frontends can apply those
limits without access to the engine's accelerator. Workers check the admitted
choices against their local configuration before binding. EngineCore checks
the reported stage, attention implementation, Decode mode and constraint
digest against the admitted requirements and across ranks.
[Resolved choices](../../../vllm_gr/config/resolved_choices.py) and
[worker readiness](../../../vllm_gr/v1/engine/worker_readiness.py)

A GR engine requesting Beam Attention claims vLLM's `CUSTOM` backend slot
before model load. CUDA uses native backend selection. The Ascend bridge
handles an explicit `CUSTOM` choice and delegates native choices to the
vendor selector. [Ascend attention bridge](../../../vllm_gr/platforms/ascend/attention_bridge.py)

## Adding an in-tree platform

1. Add a zero-argument probe returning the platform class path and any host-only request validation. Cover both the intended
   platform and a nearby identity, such as ROCm for a CUDA entry.
2. Implement configuration defaults and resolve serializable choices and
   physical limits for the native runner. Defer hardware imports until the
   selected integration needs them. See the shipped
   [CUDA](../../../vllm_gr/platforms/cuda/platform.py) and
   [Ascend](../../../vllm_gr/platforms/ascend/platform.py) implementations.
3. Supply platform-specific Attention and constraint operations and bind
   stage execution, sampling, retirement, cancellation and capture on the
   runner. Shared dispatch calls these bound operations.
   [Stage binding](../../../vllm_gr/v1/worker/stage_binding.py)
4. Exercise identity selection, import isolation, runner lifecycle and
   readiness in the [platform tests](../../../tests/platforms/). Validate
   output quality and performance on the target device using the
   [benchmark acceptance guide](../../contributing/ci/benchmark_acceptance.md).

## Migrating from the Provider API

RFC #851 removes the GR Provider registry, the external
`vllm_gr.platform_backends` entry-point group and the legacy Provider
adapter in the same change. Remove `VLLM_GR_BACKEND`,
`VLLM_GR_DISABLE_BUILTIN_BACKENDS` and `VLLM_GR_STRICT_PROVIDER_RANGES`
from deployment environments. vLLM's native platform now determines which
GR integration runs. Upgrade the engine and all Workers together. The
resolved choices use wire format version 3 and Workers reject older formats.
Port external GR implementations to the in-tree table and component
interfaces above. [RFC #851](https://github.com/vllm-project/vllm-gr/issues/851)
