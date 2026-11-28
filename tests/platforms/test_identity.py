# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Native platform identity and in-tree GR implementation selection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import vllm.platforms

from tests.helpers.native_platform import native_platform
from vllm_gr.platforms import table
from vllm_gr.platforms.errors import UnsupportedBackendError

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


@pytest.mark.parametrize(  # type: ignore[misc]
    "enum, device", [("ROCM", "cuda"), ("CPU", "cpu"), ("OOT", "foreign"), ("OOT", "vendor")]
)
def test_only_supported_native_platforms_match(
    monkeypatch: pytest.MonkeyPatch, enum: str, device: str
) -> None:
    monkeypatch.setattr(vllm.platforms, "current_platform", native_platform(enum, device))
    with pytest.raises(UnsupportedBackendError, match="No GR integration"):
        table.current_gr_platform()


def test_legacy_native_ascend_identity_has_no_builtin_gr_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Reproduce the identity the retired built-in probe matched, without the
    # vendor distribution or its import side effects.
    from tests.helpers.native_platform import VendorNative

    class NPUPlatform(VendorNative):
        device_type = "npu"

    NPUPlatform.__module__ = "vllm_ascend.platform"
    NPUPlatform.__qualname__ = "NPUPlatform"
    monkeypatch.setattr(vllm.platforms, "current_platform", NPUPlatform())
    with pytest.raises(UnsupportedBackendError, match="No GR integration"):
        table.current_gr_platform()


@pytest.mark.parametrize(  # type: ignore[misc]
    "enum, device, name", [("CUDA", "cuda", "cuda")]
)
def test_winner_inherits_native_platform(
    monkeypatch: pytest.MonkeyPatch, enum: str, device: str, name: str
) -> None:
    native = native_platform(enum, device)
    monkeypatch.setattr(vllm.platforms, "current_platform", native)
    selected = table.current_gr_platform()
    assert isinstance(selected, type(native))
    assert selected.backend_id == name
    assert table.current_gr_platform() is selected


def test_worker_rejects_choices_admitted_for_another_native_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.platforms

    from vllm_gr.config import GRConfig
    from vllm_gr.platforms.planning import admit_backend_plan
    from vllm_gr.v1.worker.runner_binding import bind_backend_services

    config = SimpleNamespace(additional_config={})
    GRConfig(beam_max_width=16).write_to_vllm_config(config)
    monkeypatch.setattr(vllm.platforms, "current_platform", native_platform("CUDA", "cuda"))
    plan = admit_backend_plan(config)
    assert plan is not None and plan.backend_id == "cuda"
    monkeypatch.setattr(vllm.platforms, "current_platform", native_platform("OOT", "vendor"))
    with pytest.raises(UnsupportedBackendError, match="No GR integration"):
        bind_backend_services(SimpleNamespace(vllm_config=config, device="vendor"))
