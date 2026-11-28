# Constraint table layout and execution

The [configuration guide](../../configuration/constraint_tables.md) covers
building an artifact and choosing a Trie per prompt. The compiler, artifact
loader and device backends share a three-stage table representation. Each stage
restricts the next SID token to legal continuations of the prefix.

## Artifact layout

A single-Trie safetensors file (`constraint_table_v1`) contains six tensors. A
multi-Trie file (`constraint_table_v2`) adds `first_token_offsets` and records
`trie_count` in its metadata. All Tries share a model vocabulary and tokenizer.
Let `T` be the Trie count, `V` the model vocabulary size, and `a`, `b`, `c` the
three SID token IDs.

| Tensor in `constraint_table_v2` | Contents |
| --- | --- |
| `first_token_offsets` | `T + 1` offsets delimiting the first-token list for each Trie. |
| `first_token_ids` | Concatenated lists of legal first tokens (`a`). |
| `prefix1_offsets` | `T * V + 1` offsets. Row `trie_id * V + a` locates the legal second tokens. |
| `prefix1_values` | Concatenated lists of legal second tokens (`b`). |
| `prefix1_pair_keys` | Sorted keys of present pairs, encoded as `trie_id * V * V + a * V + b`. |
| `prefix2_value_offsets` | Boundaries of the third-token list for each stored pair. |
| `prefix2_values` | Concatenated lists of legal third tokens (`c`). |

Pair keys are `int64` and the other tensors are `int32`. Offsets address the
concatenated arrays. The Trie ID distinguishes identical token prefixes in
different catalogs. For a single-Trie file, the loader derives the root interval
`[0, len(first_token_ids)]`. The tensor payload is limited to 512 MiB, and the
device operators reject `max_top_k > 256`. See the
[table validation](../../../vllm_gr/constraints/table.py) and
[safetensors reader and writer](../../../vllm_gr/constraints/artifact.py).

## Request routing

At startup the engine obtains the loaded Trie count from the workers. The input
processor attaches one selector to each request item. EngineCore validates its
range before allocating a native request or worker session. The selection stays
with the request across all generation stages and parent-beam reordering. A
mixed-Trie batch can therefore share one model forward while applying different
catalog constraints to its rows. The selector identifies a Trie within the
table. [Online catalog reload](../feature/weight_update.md#update-lifecycle)
can change which table version serves a later lookup.

## Device execution

### CUDA

The constraint lookup addresses the selected root in the combined tables. With
one Trie it uses the single-root addressing path. When LM-head slicing is
active, the projection uses a continuous token-ID window spanning all stages of
all Tries. Slicing is not applied when tensor parallelism exceeds 1 and is
dropped when the computed window equals the full vocabulary. Legal candidate
selection uses the requested Trie, while log probabilities are normalized over
the computed window. Adding a catalog can widen that window and affect scores;
this is documented intent rather than a tested invariant. The
[CUDA backend](../../../vllm_gr/constraints/cuda_backend.py) implements
selection.

### CPU

The [CPU backend](../../../vllm_gr/constraints/cpu_backend.py) is a host-table
backend that follows the shared full-window scoring contract. Parity tests compare it with the torch fallback
routed through a CPU-resident operator.

## Reload and retention

A reload rejects a changed Trie count or a changed vocabulary projection.
Publishers must keep each version's artifact immutable. The management
endpoint returns success immediately for the committed version. A worker
loading a candidate checks the artifact digest against a same-named table
only while that table is retained. Old versions expire lazily on a later
table operation after `gr.trie_retention_seconds` (default 60). See the
[versioned table](../../../vllm_gr/constraints/versioned_table.py).

## Validation

| Test | Coverage |
| --- | --- |
| [Table tests](../../../tests/constraints/test_table.py) | Tensor layout, offsets, pair keys and payload limits. |
| [Artifact tests](../../../tests/constraints/test_artifact.py) | Safetensors round trips and multi-Trie metadata. |
| [CPU backend tests](../../../tests/constraints/test_cpu_backend.py) | Reference candidate selection and scoring. |
| [CUDA backend tests](../../../tests/constraints/test_cuda_backend.py) | CUDA constraint table execution and readiness. |

## Source map

| File | Responsibility |
| --- | --- |
| [Table validation](../../../vllm_gr/constraints/table.py) | Tensor shapes, offsets, pair keys and limits. |
| [Artifact I/O](../../../vllm_gr/constraints/artifact.py) | Reads and writes single- and multi-Trie safetensors. |
| [CUDA backend](../../../vllm_gr/constraints/cuda_backend.py) | CUDA lookup, slicing and selection. |
| [CPU backend](../../../vllm_gr/constraints/cpu_backend.py) | Host-table selection and the shared scoring contract. |
