#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

set -euo pipefail

RUN_WHEEL=false

CLI_DEVICE_ID=""

usage() {
    echo "Usage: $0 [--wheel] [--device-id N]"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --wheel)
            RUN_WHEEL=true
            shift
            ;;
        --device-id)
            if [ $# -lt 2 ] || [ -z "$2" ]; then
                echo "--device-id requires a non-empty value" >&2
                usage >&2
                exit 1
            fi
            CLI_DEVICE_ID="$2"
            shift 2
            ;;
        --device-id=*)
            CLI_DEVICE_ID="${1#*=}"
            if [ -z "$CLI_DEVICE_ID" ]; then
                echo "--device-id requires a non-empty value" >&2
                usage >&2
                exit 1
            fi
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export PATH=${CONDA_PREFIX}/bin:${PATH:-}
fi

source "./tools/devices/resource.sh"
export VLLM_GR_TARGET_DEVICE=cuda
mem_th=18

# Verify device-index parsing honors the visibility variables. This is pure
# logic, so it runs before auto-detection on both the explicit-device and
# auto-detect paths, and does not need nvidia-smi to be present.
set +e
_parsed_ids=$(CUDA_VISIBLE_DEVICES="0,2" _cuda_device_indices 4)
rc=$?
set -e
if [ $rc -ne 0 ] || [ "$_parsed_ids" != "$(printf '0\n2')" ]; then
    echo "CUDA_VISIBLE_DEVICES is not honored by _cuda_device_indices" >&2
    exit 1
fi
if [ "$(unset CUDA_VISIBLE_DEVICES; _cuda_device_indices 3)" != "$(printf '0\n1\n2')" ]; then
    echo "unset CUDA_VISIBLE_DEVICES fallback is wrong" >&2
    exit 1
fi
if CUDA_VISIBLE_DEVICES='' _cuda_device_indices 3 >/dev/null 2>&1; then
    echo "empty CUDA_VISIBLE_DEVICES should expose no devices" >&2
    exit 1
fi
if [ "$(CUDA_VISIBLE_DEVICES="0,99" _cuda_device_indices 4)" != "0" ]; then
    echo "out-of-range CUDA_VISIBLE_DEVICES id is not intersected out" >&2
    exit 1
fi
if CUDA_VISIBLE_DEVICES="GPU-0" _cuda_device_indices 4 >/dev/null 2>&1; then
    echo "invalid CUDA_VISIBLE_DEVICES token was not rejected" >&2
    exit 1
fi

DEVICE_ID="$CLI_DEVICE_ID"

if [ -n "$DEVICE_ID" ]; then
    if ! [[ "$DEVICE_ID" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "Invalid device id '$DEVICE_ID': must be a comma-separated list of non-negative integers" >&2
        exit 1
    fi
    idle_gpu="$DEVICE_ID"
    echo "using externally provided device $idle_gpu (skip resource auto-detection)"
else
    # Test development suite before selecting an idle GPU.
    ./tests/development/test_resource.sh
    # test pre-commit
    set +e
    idle_gpu=$(get_gpu "$mem_th" 15)
    rc=$?
    set -e
    if [ $rc -ne 0 ] || [ -z "$idle_gpu" ]; then
        echo "No GPUs found that are 0% utilized, using < ${mem_th}MiB memory, and error-free. (exit code: $rc)"
        exit 1
    fi
fi

echo "running on device $idle_gpu"
NODEENV_SSL_NO_VERIFY=1 \
CUDA_VISIBLE_DEVICES=$idle_gpu \
pre-commit run --all-files

if [[ "$RUN_WHEEL" == "true" ]]; then
    VLLM_GR_TARGET_DEVICE=cuda \
    CUDA_VISIBLE_DEVICES=$idle_gpu \
    ./tools/packaging/run_wheel_smoke_test.sh --python python3
fi
