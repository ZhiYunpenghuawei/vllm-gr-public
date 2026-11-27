# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Single-rank device communication adapter for the owned NPU platform."""

import torch
from torch.distributed import ProcessGroup
from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase


class NPUCommunicator(DeviceCommunicatorBase):
    """Keep the native communicator ABI; HCCL is supplied by torch-npu.

    vLLM normally skips communicator instantiation for a one-rank group.
    Reject multi-rank and expert communication until those execution models
    have their own implementation and acceptance coverage.
    """

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
        global_ranks: list[int] | None = None,
        global_world_size: int | None = None,
        use_all2all: bool = False,
    ) -> None:
        if use_all2all or cpu_group.size() != 1:
            raise ValueError("The owned NPU backend supports only single-rank communication")
        if device is None or device.type != "npu" or device_group is None:
            raise ValueError("An NPU device and its device group are required")
        super().__init__(
            cpu_group=cpu_group,
            device=device,
            device_group=device_group,
            unique_name=unique_name,
            global_ranks=global_ranks,
            global_world_size=global_world_size,
            use_all2all=False,
        )
