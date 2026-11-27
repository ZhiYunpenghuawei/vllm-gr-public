# GR platform integration

vLLM selects the native hardware platform. GR selects one `GRPlatform` for
that identity, resolves execution choices for each engine, and binds the
selected operations to each loaded ModelRunner. Platform discovery runs
outside the request and token-generation paths.

## Platform selection

GR discovers its process-local platform independently. Both installed
`vllm_gr.platform_plugins` and built-ins expose a zero-argument probe.
Probes can inspect vLLM's selected native platform and return the qualified
class name of a compatible GR implementation. Only the winner is imported.

| External matches | Built-in matches | Result |
| --- | --- | --- |
| One | Any | Use the external platform, including on CUDA and Ascend. |
| Two or more | Any | Report an ambiguity with the matching plugin names. |
| Zero | One | Use the matching built-in class. |
| Zero | Two or more | Report an ambiguity. |
| Zero | Zero | Skip optional plugin activation. Explicit GR initialization reports the unsupported identity. |

A probe returns a class path or `None`. A legacy probe requiring an identity argument fails at startup. An exception loading or
calling a probe is logged with its origin. A malformed result, an incompatible
selected class or a failure constructing it is an initialization error. GR
instantiates the winning class once and caches it in the process. Spawned and
forked processes resolve their own instance.

| Built-in class | Required native identity | GR Runner |
| --- | --- | --- |
| [CudaGRPlatform](../../../vllm_gr/platforms/cuda/platform.py) | Platform enum `CUDA` and device type `cuda` | CUDA V1 or V2 |
| [AscendGRPlatform](../../../vllm_gr/platforms/ascend/platform.py) | A `vllm_ascend.platform.NPUPlatform` or subclass with device type `npu` | Ascend V1 |

ROCm also uses `cuda` for tensor placement. A vendor using the native
out-of-tree platform enum can distinguish its hardware by the qualified
class lineage. Each winner declares its native platform type, checked before instantiation.
[Identity](../../../vllm_gr/platforms/identity.py), [built-in table](../../../vllm_gr/platforms/table.py)
and [plugin discovery](../../../vllm_gr/platforms/discovery.py)

### Plugin startup

vLLM calls `vllm_gr.plugin:register` through `vllm.general_plugins`. The
selected class supplies runtime checks, runner defaults and any vendor hooks.
GR installs the shared Beam hooks and reuses the native Scheduler, EngineCore
and Worker lifecycle. A platform that needs a different Worker can supply
a qualified `worker_cls` on its configuration service. An explicit native
`--worker-cls` wins over this default.
[Startup](../../../vllm_gr/plugin.py) and [configuration service](../../../vllm_gr/platforms/config.py)

## Engine plans

The engine admits its runner generation, Beam limits, Attention, graph and
constraint choices into `additional_config["vllm_gr_choices"]`. Request
frontends consume these limits even when the local host has no accelerator.
Workers verify the admitted choices against their local runtime, then bind
the selected operations. EngineCore compares the workers' reported stage,
Attention, Decode mode and constraint digest with the admitted requirements.
For installed plugins, the plan also records the selected class, entry point,
distribution, version and platform API version. Deploy the same versions on
the driver and every worker.
[Engine admission](../../../vllm_gr/platforms/planning.py),
[wire format](../../../vllm_gr/config/resolved_choices.py) and
[readiness](../../../vllm_gr/v1/engine/worker_readiness.py)

For canonical constraint tables, the driver resolves `backend: auto` using the
selected platform. A plugin supplying its own operator declares
`beam_limits.constraint_backend="platform"` and implements
`create_constraint_backend`. The loader verifies the safetensors artifact,
tokenizer and vocabulary before passing the table to the factory. The
Worker owns the resulting backend across inference, replacement and close.
[Loader](../../../vllm_gr/constraints/loader.py) and [table backend interface](../../../vllm_gr/constraints/backend.py)

The platform's `resolve()` sets `supports_unconstrained`,
`supports_dynamic_width`, token budgets and other physical limits for the
admitted runner. A CPU-only request frontend can check these facts without
installing the device plugin. For additional rules that require vendor code,
`validate_request()` is bound when EngineCore starts. Per-process defaults,
such as `default_use_v2_model_runner`, apply before native configuration.
The platform supplies the prefill graph runner and Attention backend only
for the execution modes it implements.

Beam Attention is selected by the platform seam: the plugin installs the
selected GR platform as the process `current_platform`, and
`GRPlatform.get_attn_backend_cls` returns the imported Beam backend class
for engines whose GR configuration selects it, delegating every other
selection to the native platform through the MRO. The shared
`AttentionBackendEnum.CUSTOM` registry slot is never claimed (vendor
runtimes may squat it at import time), `attention_config` keeps the value
the vendor runtime left it, and nothing is registered or installed at
engine construction.
[Platform seam](../../../vllm_gr/platforms/interface.py)

<a id="adding-an-in-tree-platform"></a>

## Adding a platform

Implement the [GRPlatform contract](../../../vllm_gr/platforms/interface.py).
Keep the shared scheduler, EngineCore and native Worker when their interfaces
fit the hardware. The platform resolves engine-specific choices and supplies
implementations for the supported runner, Attention and constrained Top-K
paths. Per-engine or per-worker state belongs on the admitted plan or runner
binding. [Stage binding](../../../vllm_gr/v1/worker/stage_binding.py)

Register a built-in probe in `BUILTIN_PROBES`. For a separately installed package declare the probe in its
wheel's metadata, for example:

```toml
[project.entry-points."vllm_gr.platform_plugins"]
acme = "acme_gr:probe"
```

```python
def probe():
    from vllm.platforms import current_platform

    if current_platform.device_type == "acme":
        return "acme_gr.platform.AcmeGRPlatform"
    return None
```

In the implementation module, inherit the vendor platform to reuse its native
operations. The compatibility check also accepts subclasses of the vendor type.

```python
class AcmeGRPlatform(GRPlatform, AcmePlatform):
    api_version = 2
    native_platform_type = AcmePlatform
    # Implement the GR admission and execution hooks here.
```

Keep the entry-point module import-light and defer device runtime imports to
the selected implementation. The plugin can also inherit an existing GR
platform class when the native runner and operators are compatible. Test
neighboring identities, conflicts, independent worker discovery and package
version drift. Run quality and performance checks on the target hardware.
[Installation](../../getting_started/installation/README.md),
[platform tests](../../../tests/platforms/) and
[benchmark acceptance](../../contributing/ci/benchmark_acceptance.md)

## Migrating from the Provider API

RFC #851 retires the Provider registry, the `vllm_gr.platform_backends`
entry-point group and the legacy adapter together. Remove `VLLM_GR_BACKEND`,
`VLLM_GR_DISABLE_BUILTIN_BACKENDS` and `VLLM_GR_STRICT_PROVIDER_RANGES`
from deployments. Port external implementations to a platform class with a
probe in the new entry-point group. Upgrade driver and workers together and
restart the processes. [RFC #851](https://github.com/vllm-project/vllm-gr/issues/851)
