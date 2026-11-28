# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Import-light probes for the GR platforms shipped with this package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .cuda import cuda_platform_plugin

if TYPE_CHECKING:
    from .interface import GRPlatform


BUILTIN_PROBES = {"cuda": cuda_platform_plugin}


def current_gr_platform() -> GRPlatform:
    from .discovery import selected_platform

    return selected_platform()
