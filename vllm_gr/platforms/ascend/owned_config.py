# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Engine configuration for the owned NPU provider."""

from typing import Any

from vllm_gr.config import GRConfig
from vllm_gr.config.execution import validate_gr_execution_config

from .config import AscendConfigService


class OwnedNPUConfigService(AscendConfigService):
    def validate_engine(self, config: Any, gr: GRConfig) -> None:
        if gr.beam_max_width is None:
            raise ValueError(
                "Owned NPU V2 requires GR Beam configuration; ordinary generation is unsupported"
            )
        if not getattr(config, "use_v2_model_runner", False):
            raise ValueError("Owned NPU backend requires ModelRunner V2")
        # Shared validation is called by EngineArgs after publishing the GR snapshot.
        if getattr(getattr(config, "model_config", None), "quantization", None) is not None:
            raise ValueError("Owned NPU V2 does not support quantization")

    def finalize_native_config(self, config: Any, gr: GRConfig) -> None:
        del gr
        validate_gr_execution_config(config)
        from vllm_gr.v1.worker.npu.model_layers import validate_supported_model_config

        validate_supported_model_config(config)
        # Shared GR hooks already own EngineCore initialization. There is no
        # Ascend additional_config validator to sanitize in this backend.
