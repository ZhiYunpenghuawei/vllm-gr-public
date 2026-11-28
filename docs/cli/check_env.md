# vllm check-env

`vllm check-env --gr` checks the installed vLLM version and reports the
selected platform. An independent GR platform plugin can supply its own check.

## Usage

```bash
vllm check-env --gr          # Validate the selected platform
```

## What it checks

The command runs the environment check supplied by the selected GR platform.
The built-in CUDA implementation requires vLLM 0.30.0. A native platform
without a compatible GR plugin is not supported.

## Output and exit status

On success the command prints a header followed by one `name: value` line per
validated item:

```text
vLLM-GR environment check passed
  vllm: 0.30.0
  platform: cuda
```

A version mismatch, unsupported native platform or failed platform check raises
an error and the command exits non-zero.

## Scope

The check covers the runtime pair for the selected platform only. Unselected
vendor runtimes are not validated, and the command does not import the serving
command's optional dependencies, such as the gRPC extra.

For the package contract behind these versions, see
[installation](../getting_started/installation/README.md) and
[dependency policy](../contributing/dependency_policy.md).
