# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Read the runner-local GR configuration."""

from typing import Any

from vllm_gr.config import GRConfig


def _get_gr_config(runner: Any) -> GRConfig:
    return GRConfig.from_vllm_config(getattr(runner, "vllm_config", None))
