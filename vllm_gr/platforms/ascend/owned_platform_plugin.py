# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Discover the experimental independent Ascend platform without importing vLLM."""

import os


def register() -> str | None:
    """Select the owned platform only when explicitly requested.

    Import torch-npu to register the ``npu`` device and HCCL backend. Avoid
    constructing tensors or setting a device in the frontend process.
    """
    if os.environ.get("VLLM_GR_OWNED_NPU") != "1":
        return None

    import torch
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("VLLM_GR_OWNED_NPU=1 requires an available NPU")
    # Native platform discovery precedes the GR general plugin. Force spawn
    # before constructing executors, including if probing initialized NPU.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    return "vllm_gr.platforms.ascend.owned_platform.OwnedNPUPlatform"
