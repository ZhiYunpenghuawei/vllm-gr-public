# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Discover a GR platform for the native device vLLM selected in this process."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from importlib import metadata
from inspect import signature
from threading import RLock

from vllm.platforms.interface import Platform
from vllm.utils.import_utils import resolve_obj_by_qualname

from .errors import AmbiguousBackendError, UnsupportedBackendError
from .identity import qualified_class_name
from .interface import GRPlatform
from .table import BUILTIN_PROBES

logger = logging.getLogger(__name__)
PLATFORM_PLUGINS_GROUP = "vllm_gr.platform_plugins"


@dataclass(frozen=True)
class _Selection:
    native: Platform
    platform: GRPlatform | None
    versions: dict[str, str | int] = field(default_factory=dict)
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


def _origin(entry: metadata.EntryPoint) -> str:
    dist = entry.dist
    return f"{dist.name if dist is not None else '<unknown distribution>'}:{entry.name}"


def _resolve(native: Platform) -> _Selection:
    matches: list[tuple[metadata.EntryPoint, str]] = []
    for entry in metadata.entry_points(group=PLATFORM_PLUGINS_GROUP):
        try:
            probe = entry.load()
        except Exception:
            logger.warning("Could not load GR probe %s", _origin(entry), exc_info=True)
            continue
        # An old probe(identity) must not silently fall back to a built-in.
        try:
            signature(probe).bind()
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"GR platform probe {_origin(entry)} must accept zero arguments"
            ) from exc
        try:
            path = probe()
        except Exception:
            logger.warning("GR platform probe %s failed", _origin(entry), exc_info=True)
            continue
        if path is not None:
            if not isinstance(path, str) or not path:
                raise TypeError(
                    f"GR platform probe {_origin(entry)} must return a class path or None"
                )
            matches.append((entry, path))

    if len(matches) > 1:
        origins = ", ".join(sorted(_origin(entry) for entry, _ in matches))
        raise AmbiguousBackendError(f"Multiple external GR platforms match: {origins}")

    selected_entry: metadata.EntryPoint | None = None
    if matches:
        selected_entry, path = matches[0]
        origin = _origin(selected_entry)
    else:
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
    if (
        type(platform_cls.api_version) is not int
        or platform_cls.api_version != GRPlatform.api_version
    ):
        raise RuntimeError(
            f"GR platform {origin} requires platform API {platform_cls.api_version!r}; "
            f"this vllm-gr supports version {GRPlatform.api_version}"
        )
    native_type = getattr(platform_cls, "native_platform_type", None)
    if not isinstance(native_type, type) or not issubclass(type(native), native_type):
        raise RuntimeError(
            f"{origin} selected {path} for incompatible native platform {qualified_class_name(type(native))}"
        )
    if selected_entry is not None and (
        selected_entry.dist is None
        or not selected_entry.dist.name
        or not selected_entry.dist.version
    ):
        raise RuntimeError(
            f"Missing distribution metadata for GR platform {_origin(selected_entry)}"
        )
    try:
        platform = platform_cls()
    except Exception as exc:
        raise RuntimeError(f"Could not initialize selected GR platform {origin}") from exc
    if selected_entry is None:
        return _Selection(native, platform)
    assert selected_entry.dist is not None
    return _Selection(
        native,
        platform,
        versions={
            "platform_api": GRPlatform.api_version,
            "platform_plugin": selected_entry.name,
            "platform_distribution": selected_entry.dist.name,
            "platform_version": selected_entry.dist.version,
            "native_enum": native._enum.name,
            "native_class": qualified_class_name(type(native)),
            "native_placement": native.device_type,
        },
    )


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


def plugin_versions(platform: GRPlatform) -> dict[str, str | int]:
    """Record the installed source of the selected class in the admitted plan."""
    selection = _selection
    if selection is None or selection.platform is not platform:
        return {}
    return selection.versions.copy()
