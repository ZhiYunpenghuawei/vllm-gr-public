# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Check the selected Ascend runtime without loading an unselected backend."""

import os
import sys
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

from ..compat import (
    SUPPORTED_TRITON_ASCEND_VERSION,
    SUPPORTED_VLLM_ASCEND_VERSION,
    check_vllm_runtime,
    installed_version,
    require_runtime_version,
)


def assert_no_ascend_runtime_loaded() -> None:
    """Reject old runtime modules and binary mappings in an owned worker."""
    if any(name == "vllm_ascend" or name.startswith("vllm_ascend.") for name in sys.modules):
        raise RuntimeError("vLLM-Ascend was loaded in the owned NPU process")
    maps = Path("/proc/self/maps")
    if maps.is_file():
        for line in maps.read_text(encoding="utf-8").splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and fields[5].startswith("/"):
                path = fields[5]
                if "/vllm_ascend/" in path or "_C_ascend" in Path(path).name:
                    raise RuntimeError("vLLM-Ascend binary mapped in the owned NPU process")


def check_ascend_environment(*, require_npu_platform: bool = False) -> dict[str, str]:
    """Validate the complete Ascend runtime used by vllm-gr.

    Checking the ``triton-ascend`` distribution alone is insufficient because
    the standard Triton wheel installs the same top-level ``triton`` package
    and can overwrite its files. Import the Ascend backend as the authoritative
    check of the module that Python will actually use.
    """
    report = check_vllm_runtime()
    owned = os.environ.get("VLLM_GR_OWNED_NPU") == "1"
    if owned:
        # Metadata alone does not distinguish a clean environment from a
        # source checkout or a previously imported editable plugin.
        if (
            installed_version("vllm-ascend") is not None
            or any(name == "vllm_ascend" or name.startswith("vllm_ascend.") for name in sys.modules)
            or find_spec("vllm_ascend") is not None
        ):
            raise RuntimeError("The owned NPU runtime requires an environment without vLLM-Ascend")
        version = installed_version("torch-npu")
        if version is None:
            raise RuntimeError("The owned NPU runtime requires torch-npu")
        report["torch-npu"] = version
        report["npu-runtime"] = "owned (experimental)"
    else:
        report["vllm-ascend"] = require_runtime_version(
            "vllm-ascend", SUPPORTED_VLLM_ASCEND_VERSION
        )

    report["triton-ascend"] = require_runtime_version(
        "triton-ascend", SUPPORTED_TRITON_ASCEND_VERSION
    )
    if owned:
        import_module("torch_npu")

    triton_module = None
    try:
        triton_module = import_module("triton")
        import_module("triton._C.libtriton.ascend")
    except Exception as exc:
        triton_origin = getattr(triton_module, "__file__", "unavailable")
        raise RuntimeError(
            "Triton-Ascend distribution is installed, but the active triton module "
            "does not provide triton._C.libtriton.ascend. This usually means the "
            "standard Triton wheel overwrote the shared triton package. "
            f"Active module: {triton_origin}. Reinstall triton-ascend "
            f"{SUPPORTED_TRITON_ASCEND_VERSION} after "
            + ("vllm." if owned else "vllm and vllm-ascend.")
        ) from exc

    report["triton-module"] = str(getattr(triton_module, "__file__", "unknown"))
    report["triton-ascend-backend"] = "available"

    report["triton"] = installed_version("triton") or "not installed"

    if require_npu_platform:
        from vllm.platforms import current_platform

        device_type = str(current_platform.device_type)
        report["platform"] = device_type
        if device_type != "npu":
            raise RuntimeError(
                "Ascend environment check requires the vLLM NPU platform, "
                f"but the detected platform device_type is {device_type!r}."
            )
        if owned:
            from .owned_platform import OwnedNPUPlatform

            if not isinstance(current_platform, OwnedNPUPlatform):
                raise RuntimeError("The owned runtime requires the GR-owned vLLM NPU platform")

    if owned:
        assert_no_ascend_runtime_loaded()

    return report
