# Ascend installation

Prepare the Ascend runtime and operators, then verify the vLLM-GR installation on the NPU.

## Prerequisites

- An Ascend NPU host (verified on 910B/950) with the matching driver
- A container runtime (Docker or Podman)
- 32GB+ RAM (recommended)
- A vendor image matching the host hardware and operating system; see the
  image variants below

For the shared checkout, target-flag, and interpreter steps, see
[Installation](README.md).

### Vendor images

Select a tag for your Ascend generation and host operating system.

| Image tag | Hardware | Operating system |
| --- | --- | --- |
| `quay.io/ascend/vllm-ascend:v0.30.0rc2` | A2 (910B) | Ubuntu |
| `quay.io/ascend/vllm-ascend:v0.30.0rc2-openeuler` | A2 (910B) | openEuler |
| `quay.io/ascend/vllm-ascend:v0.30.0rc2-a5` | A5 (950DT) | Ubuntu |
| `quay.io/ascend/vllm-ascend:v0.30.0rc2-a5-openeuler` | A5 (950DT) | openEuler |

The validated stack includes Python 3.12, CANN 9.2.0, PyTorch / torch-npu
2.10.0, vLLM 0.30.0, vLLM-Ascend 0.30.0rc2 and Triton-Ascend 3.6.0.
Keep the NumPy and OpenCV builds supplied by the image.

## Setup

### Start the container

Follow the vendor's release-specific container instructions for
[A2](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0rc2/docs/source/getting_started/quick_start/ascend_image/atlas-a2.inc.md)
or [950DT / A5](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0rc2/docs/source/getting_started/quick_start/ascend_image/atlas-950dt.inc.md).
Use the image tag for your host from the table above and pass the assigned
NPU through to the container. Mount your GR checkout, or clone it inside the
container using the [shared checkout steps](README.md#shared-steps).

Inside the container, confirm that your assigned device appears, then change
to the GR repository root for the remaining steps.

```bash
npu-smi info
```

### Install xllm-ops

```bash
gr_checkout=$PWD
git clone https://github.com/xLLM-AI/xllm-ops.git

# Prepare the xllm-ops build
cd xllm-ops
git submodule init
git submodule update

# Load the CANN environment supplied by the base image
source "${ASCEND_TOOLKIT_HOME}/set_env.sh"

# Build the operator subset vllm-gr calls. CANN 9.2.0's op store ships a
# built-in CausalConv1d that collides with xllm-ops' custom operator of the
# same type name, so build only the required operators instead of the full
# set; the --no-build-* flags skip the unrelated third-party operator
# libraries.
bash build.sh \
  -n "beam_search_group;cache_unshared_kv;select_unshared_kv;x_attention;onerec_final_beam_select;rec_constrained_top_k" \
  --no-build-transformer --no-build-flash

# Load the freshly installed custom operator library
source "${ASCEND_HOME_PATH}/opp/vendors/custom_xllm_math/bin/set_env.bash"

# Install the xllm-ops library
cd test/python_test
bash build_and_run.sh
cd "$gr_checkout"
```

### Install vLLM-GR

=== "Standard"

    ```bash
    VLLM_GR_TARGET_DEVICE=npu python -m pip install -e . --no-build-isolation
    ```

=== "Development"

    ```bash
    VLLM_GR_TARGET_DEVICE=npu python -m pip install -e ".[dev]" --no-build-isolation
    ```

The vendor runtime comes from the container. Verify the installed packages
in its Python environment.

```bash
python -c 'from importlib.metadata import version; print("vllm", version("vllm")); print("vllm-ascend", version("vllm-ascend")); print("vllm-gr", version("vllm-gr"))'
```

### Verify the runtime

In the same shell with the CANN and custom operator environment loaded, check
the platform, NumPy/OpenCV import and the three fused Beam execution operators.

```bash
python -m vllm_gr.entrypoints.cli.main check-env --gr --npu
python -c "import numpy, cv2; print('numpy', numpy.__version__, 'cv2', cv2.__version__)"
python - <<'PYOPS'
import torch_npu
from vllm_gr.ops.npu import beam_ops

if not beam_ops.is_available():
    raise SystemExit(beam_ops.unavailable_reason())
print("Beam execution operators available")
PYOPS
```

The environment report should show `platform: npu`. The operator probe checks
that the binding exports `cache_unshared_kv`, `x_attention` and
`select_unshared_kv`. The quickstart below exercises inference on the device.

To use `vllm ... --gr` commands, apply the
[vLLM 0.30.0 CLI dispatch patch](../../cli/README.md#command-form-and-installation).
The vendor image installs vLLM in editable mode. Use the source root reported
by `python -m pip show vllm` as the patch target.

## Quick start

- [Run OneRec offline](../quickstart.md#ascend) to generate candidates on the NPU.
- [Serve OneRec over HTTP](../quickstart.md#online-serving) to check returned SIDs against a catalog.
- [Beam search on Ascend](../../user_guide/beam_search_npu.md) covers full-vocabulary generation. [Decode execution modes](../../design/feature/npu_decode_execution.md) describes xllm and cascade attention.

## Experimental owned ModelRunner V2

The instructions above remain the qualified legacy runtime. The GR-owned
NPU platform is an opt-in implementation candidate for independent vLLM
upgrades. It has not yet passed the NPU device, model, graph, accuracy or
performance acceptance gates. The existing vendor-image lane continues to
use its validated vLLM-Ascend package.

For the owned candidate, prepare a **separate** Python environment using
vLLM 0.30.0, Triton-Ascend 3.6.0, the matching Torch/torch-npu and CANN stack,
and the xllm-ops build for that stack. Retain the actual environment and
operator provenance. Do not install vLLM-Ascend, including editable source
checkouts. An unmodified legacy vendor image is not a clean owned environment.
No independent wheel/container combination is claimed qualified yet.

Select the independent runtime before importing vLLM, including in spawned
workers. From the prepared environment and the owned checkout:

```bash
export VLLM_GR_OWNED_NPU=1
export VLLM_GR_TARGET_DEVICE=npu
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python tools/ci/resolve_runtime.py resolve --device npu \
  --output /tmp/vllm-gr-owned-npu-runtime.json
python -m pip install -e ".[dev]" --no-build-isolation
python -m vllm_gr.entrypoints.cli.main check-env --gr --npu
```

The resolver records the selected mode and inherited package versions in the
cache identity. The environment check rejects any installed or importable
vLLM-Ascend package and requires the GR-owned native NPU platform. Successful
package checks are prerequisites only; they do not execute the model.

Start device validation with a supported dense FP16/BF16 text model,
CUSTOM attention, an explicit GR Beam configuration, async scheduling,
unchunked Prefill, and eager execution. Keep all parallel sizes at one and
leave KV transfer, speculative decoding, LoRA and quantization disabled.
Set `gr.beam_graph_enabled=false` and `gr.prefill_graph_enabled=false` explicitly;
the current candidate rejects those public graph modes until device qualification.
Do not change the V1 example flags alone and interpret its output as V2
qualification. Run the V2-specific regression and real device acceptance
workloads before switching any default or expanding the support matrix.

## Optional XAttention output binding

For direct writes to the caller's attention buffer, follow the
[optional extension build and verification steps](../../user_guide/beam_search_npu.md#optional-xattention-output-binding).
Build in the selected runtime mode and use a fresh output directory. Owned
receipts exclude vLLM-Ascend and cannot be used by legacy mode; old legacy
receipts are likewise declined by the owned runtime.

## Dependency checking on Ascend

For local validation, see the
[Ascend test runtime](../../contributing/ci/local_runner.md#ascend-test-runtime).
For the environment check's output and exit status, see [check-env](../../cli/check_env.md).

### Vendor package metadata

!!! note "Dependency metadata in vendor images"

    Keep the output of `python -m pip check` as a diagnostic. The validated
    image has the metadata conflicts listed below. Use the runtime checks
    above to confirm imports and operator availability.

- Declaration conflict: the image's opencv-python-headless 5.x wheel declares
  `numpy>=2` while triton-ascend 3.6.0 declares `numpy==1.26.4`. The
  successful runtime validation used the vendor-supplied cv2 build. Confirm
  its actual import with NumPy 1.26.4; a version declaration or vLLM's lazy
  import fallback is not evidence of binary compatibility.
- Metadata drift: the image's vllm-ascend metadata declares slightly older
  companions (triton-ascend 3.2.2, torch-npu 2.10.0.post4) than the image
  actually ships (3.6.0, post7).

Treat these recorded vendor-image discrepancies separately from new missing
or incompatible dependencies. Validate runtime versions, direct numpy/cv2
imports, custom operator loading, and inference inside the selected image.
Do not ignore unrelated `pip check` errors or replace vendor packages merely
to silence the known reports. Package availability on PyPI and consistency
of installed package metadata are separate questions.
