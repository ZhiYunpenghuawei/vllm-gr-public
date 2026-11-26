# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Platform-specific configuration and engine admission on the host."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .errors import UnsupportedBackendError

if TYPE_CHECKING:
    from vllm_gr.config import GRConfig


class BackendConfigService:
    """Host-side configuration policy for a GR engine."""

    # Passed to native EngineArgs only when the user leaves --worker-cls on auto.
    worker_cls: str | None = None
    default_worker_decision = False
    supports_constraint_vocab_slicing = False

    def apply_native_defaults(self, config: Any) -> None:
        pass

    def validate_engine(self, config: Any, gr: GRConfig) -> None:
        pass

    def finalize_native_config(self, config: Any, gr: GRConfig) -> None:
        pass


def engine_config_service() -> BackendConfigService:
    """Apply the selected platform's defaults before admitting the engine."""
    from .table import current_gr_platform

    try:
        integration = current_gr_platform()
    except UnsupportedBackendError:
        # Native EngineArgs can run with GR disabled on other platforms.
        # An enabled GR engine must resolve its integration at plan admission.
        return BackendConfigService()
    return integration.create_config_service()


def current_constraint_backend(fallback: str) -> str:
    """Choose the canonical table implementation declared by the selected GR platform."""
    from .table import current_gr_platform

    try:
        platform = current_gr_platform()
    except UnsupportedBackendError:
        return fallback
    return platform.canonical_constraint_backend
