# Running tests locally

Select the test level and device for your change before running the suite locally.

## Prerequisites

Prepare the development environment for [CUDA](../../getting_started/installation/gpu.md).
Identify the devices
assigned to your job.

## Run the test suite

Use this command from the repository root to select device 0.

```bash
./tools/run_test.sh --device-id 0
```

For multiple assigned devices, pass a comma-separated list such as
`--device-id 0,2`. Add `--wheel` to run the wheel smoke test after the checks
succeed. For OneRec tests that use a local checkpoint, set
`VLLM_GR_TEST_MODEL` to the snapshot directory. See the
[OneRec CUDA example](../../user_guide/onerec_quickstart.md#5-validate-and-troubleshoot).

## CUDA checks

The local device runner runs `pre-commit run --all-files` on the selected
CUDA devices, including the test hook. For host-only checks, use the
[targeted validation commands](test_levels.md#running-by-level).

## Select devices and test levels

On shared machines, pass the IDs allocated to your job. The runner uses
`--device-id` to set `CUDA_VISIBLE_DEVICES`
for the tests, replacing any inherited value. Without that option, the runner
first runs resource probes, then waits for an idle device from the visible
set. A probe on multi-GPU hosts can temporarily select GPU 1. Automatic
selection gives no exclusive reservation for the suite.

The [optional resource helpers](../../../tools/devices/resource.sh) are for
single-host development. Use the scheduler or reservation mechanism to
allocate devices on shared hosts. Callers own cleanup of processes started
by the lock helpers.

An unset visibility mask allows all enumerated devices, and an empty mask
exposes none. Malformed masks stop selection. The helpers intersect valid
masks with the available devices and skip devices whose telemetry is missing
or invalid. CUDA locks verify the physical device UUID.

Buildkite runs L1 CPU and L2 CUDA tests in the Ready pipeline and L3 CUDA
tests in the Merge pipeline. The GitHub Actions device job uses `pr-device`
to select L2 and accelerator-marked L1 tests. On CUDA, the GitHub Actions
test jobs exclude the L2 and L3 GPU cases assigned to Buildkite.
The separate Pre-commit checks workflow runs static hooks on Ubuntu for
non-draft PRs without waiting for the `ready` label.
By default the local test hook runs the full pytest suite with `--run-slow`
and `--strict-test-levels`. Set
`VLLM_GR_TEST_LEVEL=pr` to select L1 and L2. For a CUDA level run, set
`VLLM_GR_TARGET_DEVICE=cuda` to select CPU and CUDA tests within those levels.
See
[running by level](test_levels.md#running-by-level) for other selections.

## CUDA process inspection and cleanup

Run the read-only process check on GPUs allocated to your job. From the
repository root, replace `0` with the numeric device index assigned to you.

```bash
(
  set -e
  export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
  CI_CUDA_MAPPING_FILE=$(mktemp)
  export CI_CUDA_MAPPING_FILE
  trap 'rm -f -- "$CI_CUDA_MAPPING_FILE"' EXIT
  python tools/ci/check_cuda_device_mapping.py --output "$CI_CUDA_MAPPING_FILE"
  python tools/ci/gpu_process_guard.py check
)
```

The [cleanup wrapper](../../../scripts/kill_vllm_on_device.sh) sends SIGTERM
to matching EngineCore processes and fails if compute processes remain.
Confirm job ownership and replace the final command with
`bash scripts/kill_vllm_on_device.sh --yes` only when you intend to clean those
devices. The wrapper requires `--yes` before querying devices or signaling processes.
