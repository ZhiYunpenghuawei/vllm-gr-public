# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Discover a GR platform for the native device vLLM selected in this process."""

from __future__ import annotations

import os
from dataclasses import dataclass
from threading import RLock

from vllm.platforms.interface import Platform
from vllm.utils.import_utils import resolve_obj_by_qualname

from .errors import AmbiguousBackendError, UnsupportedBackendError
from .identity import qualified_class_name
from .interface import GRPlatform
from .table import BUILTIN_PROBES


@dataclass(frozen=True)
class _Selection:
    native: Platform
    platform: GRPlatform | None
    unavailable: str = ""


_lock = RLock()
_selection: _Selection | None = None
_resolving = False


def _after_fork() -> None:
    global _lock, _selection, _resolving
    # A different thread could have owned the lock in the parent.
    _lock = RLock()
    _selection = None
    _resolving = False


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def _resolve(native: Platform) -> _Selection:
    builtins = [(name, path) for name, probe in BUILTIN_PROBES.items() if (path := probe())]
    if not builtins:
        return _Selection(
            native,
            None,
            unavailable=(
                f"No GR integration for {qualified_class_name(type(native))} "
                f"({native._enum.name}, {native.device_type}); available={tuple(BUILTIN_PROBES)}"
            ),
        )
    if len(builtins) > 1:
        raise AmbiguousBackendError(f"Multiple built-in GR integrations match: {builtins}")
    origin, path = builtins[0]

    platform_cls = resolve_obj_by_qualname(path)
    if not isinstance(platform_cls, type) or not issubclass(platform_cls, GRPlatform):
        raise TypeError(f"{origin} selected {path}, expected a GRPlatform subclass")
    native_type = getattr(platform_cls, "native_platform_type", None)
    if not isinstance(native_type, type) or not issubclass(type(native), native_type):
        raise RuntimeError(
            f"{origin} selected {path} for incompatible native platform {qualified_class_name(type(native))}"
        )
    try:
        platform = platform_cls()
    except Exception as exc:
        raise RuntimeError(f"Could not initialize selected GR platform {origin}") from exc
    return _Selection(native, platform)


def selected_platform() -> GRPlatform:
    """Resolve, instantiate and cache the winning platform for this process."""
    from vllm.platforms import current_platform

    global _selection, _resolving
    with _lock:
        if _selection is None or _selection.native is not current_platform:
            if _resolving:
                raise RuntimeError("Recursive GR platform discovery")
            _resolving = True
            try:
                _selection = _resolve(current_platform)
            finally:
                _resolving = False
        if _selection.platform is None:
            raise UnsupportedBackendError(_selection.unavailable)
        return _selection.platform
