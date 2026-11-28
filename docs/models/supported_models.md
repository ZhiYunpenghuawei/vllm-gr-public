# Supported models

vLLM-GR runs OneRec recommendation workloads on CUDA. Use
full-vocabulary Beam search or supply a constraint table to limit results to
your catalog.

## Model list

| Checkpoint | Task and output | CUDA |
| --- | --- | --- |
| [OneRec-1.7B](https://huggingface.co/OpenOneRec/OneRec-1.7B) | Full-vocabulary Beam generation, ranked token sequences | [CUDA quickstart](../getting_started/quickstart.md#cuda) |
| OneRec-1.7B with a constraint table | Catalog-constrained recommendation, ranked semantic IDs (SIDs) | [Offline and HTTP examples](../examples/README.md) |

## Implementation and compatibility

The offline OneRec entry point is `GRLLM.beam_search()`. It uses the GR Beam
worker and scheduler. The [feature matrix](../user_guide/feature_compatibility.md)
covers differences in graph execution, candidate widths and constraint tables
across devices.
