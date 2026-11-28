# Offline Beam Search Demo

Run `quickstart.py` with a local OneRec model. The script calls
`GRLLM.beam_search` over the full vocabulary by default. Pass `--catalog`
to compile a temporary Constraint Table that restricts generation to your
catalog. The commands below run from the repository root in Bash.

## Run the V1 example

Install the [CUDA](../../../docs/getting_started/installation/gpu.md) runtime and set `ONEREC_MODEL`
to a local OneRec-1.7B snapshot with its tokenizer. See the
[model preparation steps](../../README.md#run-onerec-offline) to set the path
or download the snapshot. On CUDA, use the default backend.

```bash
python examples/offline_inference/beam_search/quickstart.py --model "$ONEREC_MODEL"
```

The script defaults to ModelRunner V2 on CUDA. See
[ModelRunner selection](../../README.md#choose-a-modelrunner) for the CUDA override.

The default prompt is `Recommend the next video.` The script prints
`request=0 returned_beams=128` for the first request, followed by the five
highest-ranked candidates with `beam_tokens` and `cumulative_logprob`. The
token list includes the OneRec boundary markers. `--beam-width` sets the
maximum number of beams retained at each step, and `--print-beams` limits the
number of rows shown per request.

Pass `--prompt` with a UTF-8 file to use a custom prompt at its natural token
length. The command's help lists the available options.

```bash
python examples/offline_inference/beam_search/quickstart.py --help
```

For paired cold/warm latency measurements with the OneRec dataset, use the
[offline A/B benchmark](../../../tools/AB_Tests/README.md).

## Run with a Constraint Table

Use the [two-SID sample catalog](../../online_serving/catalog/sample_catalog.json)
to check catalog membership without downloading the OneRec dataset. The
following commands write a prompt and convert the raw catalog into triples.

```bash
export ONEREC_WORKDIR="$PWD/results/onerec-example"
mkdir -p "$ONEREC_WORKDIR"
printf '%s' 'Recommend the next video.' > "$ONEREC_WORKDIR/prompt.txt"
cp examples/online_serving/catalog/sample_catalog.json "$ONEREC_WORKDIR/catalog.json"
python tools/constraints/build_constraint_triples.py \
  --catalog "$ONEREC_WORKDIR/catalog.json" \
  --output "$ONEREC_WORKDIR/constraint_triples.json" \
  --begin '<|sid_begin|>' --end '<|sid_end|>'
```

With `ONEREC_MODEL` set as above, run both commands with the same prompt.

```bash
python examples/offline_inference/beam_search/quickstart.py \
  --model "$ONEREC_MODEL" --prompt "$ONEREC_WORKDIR/prompt.txt"

python examples/offline_inference/beam_search/quickstart.py \
  --model "$ONEREC_MODEL" --prompt "$ONEREC_WORKDIR/prompt.txt" \
  --catalog "$ONEREC_WORKDIR/constraint_triples.json"
```

The converter's `--catalog` reads the raw `catalog.json`. The quickstart's
`--catalog` reads `constraint_triples.json` and compiles a temporary table with
the model's tokenizer. Both runs encode the prompt at its natural length.

The full-vocabulary run prints `beam_tokens` and can return candidates outside
the catalog. In the constrained run, check that each `sid_tokens` list matches
one of the two catalog entries, including the boundary markers. The small
catalog can return fewer than 128 candidates, and the model and prompt determine
their ranking.

The [HTTP quickstart](../../online_serving/beam_search/README.md#use-your-own-inputs)
accepts the same model, natural-length prompt and triples. It uses the two-SID
sample by default and checks the response against the original catalog.

To use your own catalog or the full OneRec mapping, start by
[setting the model and dataset paths](../../online_serving/catalog/README.md#set-the-model-and-dataset-paths).
When switching from the sample to the full dataset, set `ONEREC_DATA` and
choose a new `ONEREC_WORKDIR`. Generate `catalog.json` from the full mapping,
then regenerate `constraint_triples.json` from that catalog. Pass the new
triples file to the constrained command.
The same guide shows how to compile a persistent table for online serving.

## Beam Search V1

EngineCore schedules Beam Search V1 asynchronously. The profiling and A/B
scripts use the configurations below.

### Public entrypoint

Set the `gr` Beam configuration and call `GRLLM.beam_search()`.
`GRLLM.beam_search()` requires V1 and never falls through to upstream vLLM's
implementation. EngineCore owns scheduling for the V1 session.

The V1 flow is:

```text
beam_search()
  -> prepare and validate one request
  -> submit one EngineCore session
  -> keep Prefill, Decode, Beam state, and decisions on the Worker/GPU
  -> rank and pack the final result on the GPU
  -> copy one compact payload to the host
  -> create lazy public Beam outputs
  -> release the session
```

Execution mode is selected when the engine is initialized. Native non-Beam
inference remains available in ordinary engine configurations; do not mix it
with an active V1 Beam session.

### Current restrictions

Both modes require:

- CUDA.
- `async_scheduling=True`.
- Logical B=1..8 with shared generation parameters; online batch/capture policy
  defaults to `[1, 2, 4]` and must fit engine capacity.
- TP, PP, DP, DCP, and PCP sizes equal to 1.
- Set the request `beam_width` to the maximum value in `gr.beam_width`.
- Token/text prompts rather than embedding prompts.
- No speculative decoding, LoRA, quantization, hybrid model, encoder-decoder
  model, or multimodal model.

For catalog-constrained generation, pass `--catalog`, set
`catalog_strategy=None`, and generate exactly three tokens. Without a catalog,
the engine supports 1 through `gr.max_decode_steps` effective tokens. The
example CLI accepts 1 to 3.

### Constraint Table artifact

Start by [setting the model and dataset paths](../../online_serving/catalog/README.md#set-the-model-and-dataset-paths).
The guide shows how to generate constraint triples and save a reusable artifact.
The offline examples compile a temporary table when passed the triples.

The table compiler prints `artifact_digest` and `tokenizer_digest`. Both values
belong in the canonical configuration and prevent loading a table built from
different constraint data or a different tokenizer.

### Engine configuration

With `--catalog`, the script builds the CUDA configuration below in memory.
Save the same mapping as YAML for other applications.

```yaml
attention-config:
  backend: CUSTOM
gr:
  beam_graph_enabled: true
  worker_decision: true
  beam_width: [128, 128, 128]
  candidate_width: [128, 128, 128]
  max_decode_steps: 3
  constraint_mode: table
  constraint_table:
    backend: cuda
    max_top_k: 128
    path: /absolute/path/to/video_constraint_table.safetensors
    format: constraint_table_v1
    digests:
      artifact: sha256:REPLACE_WITH_ARTIFACT_DIGEST
      tokenizer: sha256:REPLACE_WITH_TOKENIZER_DIGEST
```

#### Beam configuration parameters

These values describe the example's CUDA configuration. The CLI flags below
belong to `offline_beam_search.py`. Library defaults can differ.

| Parameter | Example value | How the example sets it |
| --- | --- | --- |
| `beam_graph_enabled` | `true` | The Beam example preserves CUDA graph execution. |
| `worker_decision` | `true` | The script enables Beam selection on the Worker. |
| `beam_width` | `[128, 128, 128]` | `--beam-width` sets both the engine capacity and the request width. |
| `max_decode_steps` | `3` | `--generated-steps` sets the effective token count. The three-layer Constraint Table requires `3`. |

With `gr.constraint_mode: none` and `gr.constraint_table: null`, one engine can
accept per-request effective lengths from `1` through its configured
`gr.max_decode_steps`, up to the supported platform limit. Dynamic widths remain limited to three
steps.
Increasing capacity does not change the depth of an enabled
`constraint_table_v1`: constrained requests still require exactly three
effective tokens.

#### Constraint Table parameters

These values match the CUDA YAML above. With `--catalog`, the offline script
compiles a temporary artifact and fills in its path and digests.

| Parameter | Example value | How to set it |
| --- | --- | --- |
| `gr.constraint_mode` | `table` | The script enables the table when you pass `--catalog`. It selects `none` for full-vocabulary Beam Search. |
| `backend` | `cuda` | The example selects CUDA. |
| `max_top_k` | `128` | The script sets this capacity to the candidate width. In YAML, set it to at least that width. |
| `candidate_width` | Omitted, resolves to `128` | Use `--candidate-width` to set the candidates per Beam row independently of `--beam-width`. The accepted range is 1 through 256. Set it explicitly when the Beam width exceeds 256. |
| `path` | `/absolute/path/to/video_constraint_table.safetensors` | In YAML, use the absolute path to the compiled artifact. |
| `format` | `constraint_table_v1` | Use this format for the three-layer table. |
| `artifact_digest` | `sha256:REPLACE_WITH_ARTIFACT_DIGEST` | In YAML, substitute the artifact digest printed by `build_constraint_table.py`. |
| `tokenizer_digest` | `sha256:REPLACE_WITH_TOKENIZER_DIGEST` | In YAML, substitute the tokenizer digest printed by the same command. |

Use an absolute artifact path. Do not combine this canonical configuration with
legacy `vllm_gr_*` keys in `additional_config`.

The Python `vllm_gr_config` input and
`additional_config["vllm_gr"]` use the same canonical serialization. Engine
creation normalizes default fields in both forms and preserves unrelated
`additional_config` entries. Equivalent GR settings therefore contribute the
same configuration hash to vLLM's compilation cache, including when a disabled
Constraint Table omits optional fields.
Normalization can change the cache key of a previously sparse configuration,
so the first run after upgrading may compile again. Subsequent runs with the
same normalized settings reuse the new cache key.

#### Important `GRLLM` parameters

These values apply to the single-request CUDA command in
[Run the V1 example](#run-the-v1-example).

| Parameter | Example value | How the example sets it |
| --- | --- | --- |
| `vllm_gr_config` | In-memory mapping | The script builds the mapping from its arguments. The Python API also accepts a YAML path. |
| `async_scheduling` | `True` | EngineCore requires it for Beam Search V1 sessions. |
| `enable_prefix_caching` | `True` | Enables Prefill prefix reuse. Correctness is independent of prefix caching. |
| `enable_chunked_prefill` | `False` | The script processes the complete prompt in Prefill. |
| `max_model_len` | `8192` | Set with `--max-model-len`. It must cover the prompt and generation budget. |
| `max_num_batched_tokens` | `8192` | The script uses the larger of `max_model_len` and the batch's Prefill token budget. The full batch must fit when chunked Prefill is disabled. |
| `max_num_seqs` | `128` | The script uses the larger of `--beam-width` and `--batch-size`. |
| `max_logprobs` | `128` | The script uses `--candidate-width` when set and `--beam-width` otherwise. |
| `enforce_eager` | `False` | CUDA Beam execution uses graph capture when eligible. |

### Request parameters

The offline script builds `vllm_gr.sampling_params.BeamSearchParams` with
these values for the fixed-width example.

| Parameter | Example value | How the example sets it |
| --- | --- | --- |
| `beam_width` | `128` | Taken from `--beam-width`. Match the maximum value in `gr.beam_width`. |
| `max_tokens` | `5` | The script adds two boundary positions to `--generated-steps`. |
| `temperature` | `0.0` | Fixed by the script for Beam Search. |
| `ignore_eos` | `True` | Ignores stop and EOS tokens during the fixed-length search. |
| `include_stop_str_in_output` | `True` | Keeps the configured boundary token in the returned sequence. |
| `begin_token` | `<\|sid_begin\|>` | The script configures it as the marker appended to the prompt before Prefill. |
| `end_token` | `<\|sid_end\|>` | Marks a stop boundary or is appended when the result reaches its length limit. |
| `length_penalty` | `1.0` | Set with `--length-penalty`. It affects ranking for candidates of different lengths. |
| `catalog_strategy` | `None` | Keep this value when using the Constraint Table. |

The effective generation budget is:

```text
N = max_tokens - begin_token_position - end_token_position
```

With both OneRec boundary tokens, `max_tokens=5` represents three generated SID
tokens and an effective V1 budget of 3.

### Optional thinking before SID generation

The same `--prompt` file can be used in three modes.

| Options | Execution |
| --- | --- |
| Neither thinking flag | Direct Beam Search, with a Constraint Table if `--catalog` is provided |
| `--enable-thinking` | Ordinary full-vocabulary LLM generation only; no catalog or SID Beam phase |
| `--cot` | Ordinary thinking generation, followed by constrained SID Beam Search |

Choose either `--cot` or `--enable-thinking` with a checkpoint that
supports thinking. The combined mode also requires dedicated `<think>` and `</think>`
tokenizer tokens and a compatible SID catalog.

#### CUDA setup and commands

Run from the repository root in the prepared CUDA environment. Replace the
model and prompt paths.

```bash
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V2_MODEL_RUNNER=0
MODEL=/absolute/path/to/OneRec_Qwen3_1.7B
PROMPT=/absolute/path/to/recommendation_prompt.txt
BATCH_SIZE=1
mkdir -p results/cot
```

Set `BATCH_SIZE=4` to run four requests instead. This CLI repeats the supplied
input for each row, with independent row cache salts in the combined workflow.
It does not accept a separate thinking prompt.

Direct full-vocabulary Beam Search uses fixed-length input preparation.

```bash
python -u examples/offline_inference/beam_search/offline_beam_search.py \
  --model "$MODEL" --prompt "$PROMPT" \
  --batch-size "$BATCH_SIZE" --beam-width 128 \
  --input-len 128 --generated-steps 3 --print-beams 5
```

Ordinary thinking-only generation with constraints disabled. Plain text uses the
model's chat template; serialized Qwen chat retains its existing messages and
opens the terminal thinking prefix as described below. `--max-tokens` limits the
complete generated continuation, including any thinking and final answer:

```bash
python -u examples/offline_inference/beam_search/offline_beam_search.py \
  --model "$MODEL" --prompt "$PROMPT" \
  --batch-size "$BATCH_SIZE" --enable-thinking \
  --max-tokens 1024 --max-model-len 8192 \
  --temperature 0 --top-p 1 --top-k -1 \
  --output-json "results/cot/thinking_b${BATCH_SIZE}.json"
```

Thinking followed by constrained SID Beam Search requires triples built from
your own catalog using the [Constraint Table instructions](#constraint-table-artifact).

```bash
ONEREC_TRIPLES=/absolute/path/to/constraint_triples.json
python -u examples/offline_inference/beam_search/offline_beam_search.py \
  --model "$MODEL" --catalog "$ONEREC_TRIPLES" --prompt "$PROMPT" \
  --batch-size "$BATCH_SIZE" --beam-width 128 --generated-steps 3 \
  --cot --cot-max-tokens 1024 --max-model-len 8192 \
  --temperature 0 --top-p 1 --top-k -1 --print-beams 5 \
  --output-json "results/cot/beam_b${BATCH_SIZE}.json"
```

#### Input, budget and handoff contract

Both thinking modes retain the complete input and ignore `--input-len`.
Thinking-only mode also ignores `--catalog` and prints a notice when it is
supplied. The CoT mode uses the catalog for its SID generation stage. Plain
input uses the model's chat template. If that template emits a terminal empty
`<think>...</think>` span, both modes reopen it without changing earlier token
IDs. Thinking-only otherwise retains the template's output; the combined CoT
mode additionally opens a missing thinking tag and rejects nonempty, already
closed template reasoning rather than deleting it.

Qwen-style serialized ChatML is accepted either as text with real line breaks,
or as one complete JSON string whose decoded value is ChatML. JSON is decoded
exactly once, only for a string beginning with `<|im_start|>` (apart from leading
whitespace); other plain text, including quoted text, is kept as plain text.
Both modes preserve the existing messages, open the terminal empty assistant/
thinking prefix, and change only a terminal `/no_think` directive to `/think`.
This serialized path requires dedicated `<think>` and `</think>` tokenizer
tokens. Nonempty final assistant content is rejected rather than silently
removed. Input files are never modified.

Legacy bare single-line ChatML containing literal `\n` is rejected: there is no
safe way to distinguish escaped separators from body text such as `C:\new` or
an intentional literal `\n`. This includes the existing
`tests/resources/single_one_rec_prompt.txt` fixture when used directly in a
thinking mode. Supply the intended conversation with real line breaks, or
serialize that intended text with `json.dumps` instead of globally replacing
backslashes. For example:

```bash
python - <<'PY'
import json
from pathlib import Path

chatml = (
    "<|im_start|>user\n"
    r"Keep the path C:\new and the literal \n unchanged."
    "\n/no_think<|im_end|>\n<|im_start|>assistant\n"
)
Path("thinking_chatml.json").write_text(
    json.dumps(chatml, ensure_ascii=False), encoding="utf-8"
)
PY
```

Pass `--prompt thinking_chatml.json` in either thinking mode. JSON encodes a
structural newline as `\n` and a body backslash as `\\`, so the path and literal
escape above survive decoding unchanged. The direct SID Beam mode keeps its
existing fixed-length preparation and literal file-text encoding; it does not
apply this thinking-input JSON decoding or normalization.

`--cot-max-tokens` defaults to 1024 and applies only to the thinking phase.
`--generated-steps` remains the SID generation budget. The full input, thinking
budget, separator and SID boundary positions must fit `--max-model-len`; the
actual handed-off length is checked again before Beam dispatch.

The default `--cot-on-limit error` requires the model to emit `</think>` before
starting Beam. An unclosed span, including EOS without `</think>`, fails the
whole batch before the SID phase. To explicitly continue after the thinking
budget is exhausted, add `--cot-on-limit truncate` to the combined command.
Only an actual length-budget stop permits this policy: every generated token
is retained, a closing token is appended, and the report records
`thinking_truncated=true` and `forced_close=true`. This is forced closure,
not a naturally completed thought. Other incomplete stop conditions still fail.

The SID phase receives the original input token IDs, the thinking continuation
IDs and a separator. It does not decode and re-encode the continuation. Both
phases use one loaded engine, but have separate cache salts: this version
hands off token IDs and performs a fresh Beam Prefill, without cross-phase KV
reuse. The JSON output includes raw generated thinking IDs, effective handed-off
IDs, truncation status, per-row context lengths and final Beam results.

These are functional examples, with no performance threshold or recommendation
quality improvement claimed. Optional existing timing/profiling arguments are
diagnostics. CUDA hardware validation of the combined workflow is not included.

### Measure latency without profiling

Use `--input-len` with `offline_beam_search.py` to set a fixed prompt length
that includes the SID begin token. Its default is 1024 tokens.

Use `--repeat 20 --warmup-runs 3` to time 20 complete API calls after three
untimed warmup batches, without configuring a profiler. Each call submits
`batch-size` copies of the prompt. Prefix caching stays enabled, but each row
and run uses an independent cache salt, so Prefill is included in every sample.

The `beam_benchmark` JSON log reports batch latency in milliseconds (mean, P50,
P95, min/max), requests per second, actual returned Beams per second, and every
sample. Percentiles use linear interpolation. `--output-json` optionally saves
the same data and test conditions to a file. Only the last batch's Beams are
printed. The timer covers `beam_search` until output objects return; engine
startup, graph capture, warmup, pre-call prompt preparation, post-call result
inspection, statistics collection and printing are excluded. Work performed
inside the API is included. `--repeat` cannot be combined with `--profile-dir`.
Without either option, the script still runs one batch.

From the repository root, run the CUDA Beam benchmark for B=1/2/4. Each
invocation starts a full-vocabulary engine and captures a fixed `(B, W)` shape.

```bash
set -o pipefail
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
BENCH_RUN="$PWD/test-results/graph_latency_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BENCH_RUN"
for B in 1 2 4; do
  python -u examples/offline_inference/beam_search/offline_beam_search.py \
    --model /path/to/OneRec_Qwen3_1.7B \
    --prompt tests/resources/single_one_rec_prompt.txt \
      --batch-size "$B" --beam-width 128 --input-len 128 \
    --generated-steps 3 --print-beams 5 \
    --warmup-runs 3 --repeat 20 \
    --output-json "$BENCH_RUN/graph_b${B}_w128.json" \
    2>&1 | tee "$BENCH_RUN/graph_b${B}_w128.log" || exit 1
done
```

Keep the model, prompt, generation budget, memory setting and device identical
across B values. Request throughput is `B * repeat / total_timed_seconds`; batch
latency divided by B is only amortized cost, not individual request latency.

### Per-stage Beam widths through the Python API

CUDA supports a retained Beam width for each generated token, plus a
separate final result limit. CUDA can additionally set
a per-parent candidate count for each stage. Omitting it preserves the existing
candidate-width policy.

For an existing CUDA engine `llm` and a B=1, 2, or 4 batch `prompts`, use:

```python
from vllm_gr.sampling_params import BeamSearchParams

params = BeamSearchParams(
    beam_width=128,
    dynamic_beam_widths=[128, 64, 32],
    result_width=16,
    max_tokens=5,
    begin_token="<|sid_begin|>",
    end_token="<|sid_end|>",
    temperature=0.0,
    ignore_eos=True,
    include_stop_str_in_output=True,
    length_penalty=1.0,
)
outputs = llm.beam_search(prompts, params)
```

For CUDA, K and W can be configured independently after
the first stage:

```python
params = BeamSearchParams(
    beam_width=4,
    dynamic_beam_widths=[2, 4, 4],  # W1, W2, W3
    per_parent_beam_widths=[2, 2, 1],        # K1, K2, K3 per parent
    result_width=2,
    max_tokens=3,
    ignore_eos=True,
)
```

The first stage requires `K_1=W_1`. Each later stage must satisfy
`W_i <= W_(i-1) * K_i`. W supports up to 1024. Unconstrained CUDA supports K
up to 1024; current constraint-table operators cap K at 256.

An explicit `per_parent_beam_widths` overrides the startup
`constraint_table.candidate_width`, subject to backend `max_top_k` capacity.
Without it, existing behavior and startup configuration restrictions remain.
Each configured per-parent K is an upper bound: the shared execution policy uses
`min(K, W_out)` candidates per parent because no parent can supply more than
`W_out` winners to the retained set.
An explicit schedule can order tied candidates
differently and retain different candidates at a tied cutoff from native
selection, even when its widths happen to match.
HTTP clients can pass the schedule in `vllm_xargs` through Completion or Chat;
native non-GR Beam rejects an explicit schedule.

`beam_width` remains the engine capacity in `[1, 256]` and must equal
the maximum value in `gr.beam_width`.
The schedule has exactly one integer in `[1, beam_width]` per effective generated
step N, which must equal 3 when the constraint table is enabled. Without
constraints, N can range from 1 through `gr.max_decode_steps`, subject to the
backend's existing capacity limits. Begin/end boundary positions are excluded
from the schedule. With both boundary tokens above, `max_tokens=5` means N=3:
retain 128 SA candidates, then 64 SB candidates, then 32 SC candidates.
The API appends BEGIN before Prefill; do not append it again to the input prompt.
END is handled by the existing boundary/output rules, not an extra scheduled
model step.

Prefill processes the prompt tokens and selects the first-stage winners from
one final prompt position per request. It does not forward 128 Beam model rows
for that first selection. Subsequent Decode stages use the previous retained
width: for B=4, this example forwards 512 and 256 Decode rows. Increasing
schedules such as `[32, 64, 128]` are supported within the same capacity.
All prompts within one call share the same schedule and result limit.

`result_width` is a positive integer no larger than the final retained width.
It limits the number returned per request to at most R; a catalog with fewer
valid paths may produce fewer. If omitted, R is the final schedule width, or
`beam_width` when no schedule is supplied. Setting only `result_width` limits
returned results while retaining the fixed search widths.

Use the shared Beam execution constraints:
`async_scheduling=True`, `enable_chunked_prefill=False`, token/text prompts,
`ignore_eos=True`, and `catalog_strategy=None`. Existing TP/PP/DP/DCP/PCP=1,
FP16/BF16 and unsupported-model restrictions still apply. These options do not
add concurrent sessions or per-prompt schedules. Engine processes must use the
same code and schema revision across the frontend, EngineCore, and Workers.
Stop and restart them together when deploying this change. Mixed-revision
operation and rolling upgrades are unsupported for **all** Beam requests,
including fixed-width requests: the appended array-encoded schedule fields are
sent even when their values are `None`, and an older strict decoder rejects them.
Matching request-contract version numbers alone do not establish compatibility.

This path uses a fixed generated-token horizon: `ignore_eos=True` becomes
`ignore_stop_tokens=True` at engine admission. Surviving paths within a request
have the same final length, so truncating to `result_width` before length
normalization preserves score order. Variable-length early stopping is not
supported by this dynamic-width implementation.

See the [CUDA graph design](../../../docs/design/module/model_runner_v2.md)
for capture and fallback policy.

Narrower intermediate widths intentionally prune the search; later expansion
cannot recover an earlier discarded prefix. Results and low-precision model
scores can change as execution shapes change. Validate application Recall/NDCG
and latency for the selected schedule; catalog validity and implementation
parity alone do not establish unchanged recommendation quality.

### Profile inference

Add `--profile-dir /path/to/traces` to enable the vLLM Worker profiler. Collection
starts after engine initialization and unprofiled warmup, and stops before graph
statistics and Beam output printing. Without this option, no profiler is
configured or started; use `--repeat` for separate latency measurements.

- `--profile-warmup-runs` defaults to 1; use 0 to include the first inference.
- `--profile-runs` defaults to 1 and collects complete batches, each with one
  Prefill and `generated-steps - 1` Decode stages. Only the last result is printed.
- Prefix caching stays enabled, but separate cache salts for every row and run
  prevent warmup or previous runs from skipping Prefill.
- Stack, shape and memory recording are disabled to reduce profiling overhead.
  Traces cover the Worker rather than the frontend, and are not gzip compressed.

For CUDA B=2/W=128, run full-vocabulary Beam Search from the repository root.

```bash
set -o pipefail
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PROFILE_RUN="$PWD/test-results/bs2_bw128_profile_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PROFILE_RUN/logs"
python -u examples/offline_inference/beam_search/offline_beam_search.py \
  --model /path/to/OneRec_Qwen3_1.7B \
  --prompt tests/resources/single_one_rec_prompt.txt \
  --batch-size 2 --beam-width 128 --input-len 128 \
  --generated-steps 3 --print-beams 128 \
  --profile-dir "$PROFILE_RUN/profile/graph" \
  --profile-warmup-runs 1 --profile-runs 5 \
  2>&1 | tee "$PROFILE_RUN/logs/graph.log"
```

Confirm the trace
contains Worker model operations and actual graph replay, not just frontend
events. Use separate unprofiled repeated runs for latency/throughput measurements.

### Benchmark JSON and returned-result metadata

The `beam_benchmark` JSON log and `--output-json` file contain the same
measurements and configuration.

| Field | Meaning |
| --- | --- |
| `beam_width` | Engine capacity |
| `batch_size`, `input_len`, `generated_steps` | Input and generation conditions |
| `catalog` | Triples input path or `null` for full-vocabulary Beam Search |
| `repeat`, `warmup_runs` | Measured and untimed batch counts |
| `batch_latency_ms` | Mean, P50, P95, min and max batch API latency |
| `runs[].returned_beams` | Actual total valid Beams returned across the whole batch in that run |
| `requests_per_second` | Timed request throughput |
| `returned_beams_per_second` | Timed throughput based on actual returned counts, not assumed B times R |

The benchmark JSON contains conditions and timing/count metadata; it is not a
serialized list of result sequences. Standard output separately prints
`request=... returned_beams=...` for each request in the last batch, followed by
up to `--print-beams` JSON records containing `request`, `rank`,
`cumulative_logprob`, `finish_reason` and `stop_reason`. The records include
`beam_token_ids` and `beam_tokens` by default. With `--catalog`, those fields
are named `sid_token_ids` and `sid_tokens`. Both include the generated tokens
and OneRec boundaries. Printing fewer ranks leaves the return counts intact.

Compare latency only with matching model, prompt, constraint settings, device,
B, capacity, input length, return limit, Prefill setting and measurement setup.
