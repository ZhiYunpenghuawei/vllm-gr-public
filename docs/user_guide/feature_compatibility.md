# Feature compatibility

To run Beam search, use the [offline quickstart](../getting_started/quickstart.md)
or send an [HTTP Beam request](../serving/completions.md#request). Use the
matrix to choose settings for your device.

The core includes CUDA inference and experimental gRPC. Other devices require
a matching independent GR platform plugin. The legacy Ascend backend and
its vendor runtime integration have been removed; no built-in Ascend
inference path is provided.
CPU packages support packaging and host checks only.
The matrix describes the documented runtime paths and their configuration limits.

## Support matrix

| Feature | CUDA | Details |
| --- | --- | --- |
| vLLM ModelRunner | Library and general CLI default to V1. The direct offline Beam, HTTP and OneRec gRPC examples select V2 by default. | [ModelRunner selection](../configuration/README.md#modelrunner-selection), [example defaults](../../examples/README.md#choose-a-modelrunner) |
| Catalog-constrained Beam (constraint table) | Supported | [Configure a constraint table](../configuration/README.md#constraint-table) |
| Full-vocabulary Beam without a catalog | Supported | [Beam settings](../configuration/README.md#beam-settings) |
| Dynamic Beam widths | Supported | [Beam settings](../configuration/README.md#beam-settings) |
| Independent per-parent candidate widths | Supported | [Beam settings](../configuration/README.md#beam-settings) |
| Beam Decode graphs | Supported (`gr.beam_graph_enabled`) | [CUDA ModelRunner V2](../design/module/model_runner_v2.md) |
| Prefill graphs | Supported; explicit capture configuration | [Prefill capture](../design/feature/cuda_prefill_graph.md) |
| Asynchronous scheduling | Supported | [Configuration](../configuration/README.md#asynchronous-scheduling), [CUDA walkthrough](onerec_quickstart.md) |
| Multi-Trie selection per request | Supported | [Constraint tables](../configuration/constraint_tables.md) |
| gRPC transport | Experimental, requires the `grpc` extra | [gRPC token inputs and CUDA example](../serving/grpc.md) |
| Checkpoint updates on running instances | Supported with a drained instance | [Model and catalog updates](weight_update.md) |
| Online catalog replacement | Supported on the asynchronous Beam path | [Model and catalog updates](weight_update.md) |

## How to read this matrix

- "Supported" describes a documented execution path. The linked guides record
  the validation scope, including model and graph configurations where applicable.
- A feature can be supported on a device and still be excluded for a specific
  model geometry or execution mode. Check the linked page before relying on a
  combination that is not listed here.
- Startup and request validation reject unsupported configurations. A valid
  request can still fall back to native dispatch when it misses a graph capture.

<a id="related"></a>

## See also

- [Platform backend contract](../design/module/platform_backend.md) for how GR matches the native
  vLLM platform to a built-in or independently installed GR implementation.
- [Supported models](../models/supported_models.md) for the model list.
