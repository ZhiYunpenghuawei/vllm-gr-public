# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Production plugin routing with synthetic platform identities and native hooks."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.helpers.native_platform import native_platform
from vllm_gr import patch, plugin
from vllm_gr.engine import startup
from vllm_gr.platforms import compat, table

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def test_plugin_routes_validation_and_patches_through_selected_integration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.platforms

    monkeypatch.setattr(vllm.platforms, "current_platform", native_platform("CUDA", "cuda"))
    events: list[str] = []

    def hook(name: str, result: object) -> Mock:
        def invoke(*args: object, **kwargs: object) -> object:
            events.append(name)
            return result

        return Mock(side_effect=invoke)

    monkeypatch.setattr(startup, "configure_model_runner_default", hook("defaults", None))
    cuda_check = hook("cuda_environment", {})
    monkeypatch.setattr(compat, "check_vllm_runtime", cuda_check)
    beam = hook("beam", 1)
    monkeypatch.setattr(patch, "_apply_gr_beam_patches", beam)
    monkeypatch.setattr(patch, "get_last_beam_patch_report", lambda: None)

    activate = Mock(wraps=table.current_gr_platform)
    monkeypatch.setattr(table, "current_gr_platform", activate)
    for attempt in range(2):
        events.clear()
        activate.reset_mock()
        state = plugin.initialize_runtime()
        activate.assert_called_once_with()
        assert events == ["defaults", "cuda_environment", "beam"]
        assert state == {
            "patch_count": 1,
            "patches": {},
            "platform_seam": attempt == 0,
        }
        beam.assert_called_with()
    cuda_check.assert_called_with()


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "platform_name,placement",
    [("ROCM", "cuda"), ("CPU", "cpu"), ("UNSPECIFIED", ""), ("OOT", "vendor")],
)
def test_ordinary_vllm_startup_on_platform_without_gr(
    monkeypatch: pytest.MonkeyPatch, platform_name: str, placement: str
) -> None:
    import vllm.platforms

    from vllm_gr.platforms.errors import UnsupportedBackendError

    monkeypatch.setattr(
        vllm.platforms,
        "current_platform",
        native_platform(platform_name, placement),
    )
    defaults = Mock()
    monkeypatch.setattr(startup, "configure_model_runner_default", defaults)
    plugin.register()
    with pytest.raises(UnsupportedBackendError):
        plugin.initialize_runtime()
    defaults.assert_not_called()


def test_environment_failure_prevents_runtime_patch_installation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.platforms

    monkeypatch.setattr(vllm.platforms, "current_platform", native_platform("CUDA", "cuda"))
    monkeypatch.setattr(startup, "configure_model_runner_default", lambda: None)
    monkeypatch.setattr(compat, "check_vllm_runtime", Mock(side_effect=RuntimeError("version")))
    install = Mock()
    monkeypatch.setattr(patch, "re_apply_patches", install)
    with pytest.raises(RuntimeError, match="version"):
        plugin.initialize_runtime()
    install.assert_not_called()


def test_partial_platform_failure_stops_common_plan_and_retains_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An external platform may fail during its own early patch installation.
    failure = RuntimeError("native worker hook failed")
    platform = SimpleNamespace(backend_id="vendor", apply_runtime_patches=Mock(side_effect=failure))
    monkeypatch.setattr(table, "current_gr_platform", lambda: platform)
    beam = Mock()
    monkeypatch.setattr(patch, "_apply_gr_beam_patches", beam)

    with pytest.raises(RuntimeError, match="backend 'vendor'.*partial.*restart") as raised:
        patch.re_apply_patches()
    assert raised.value.__cause__ is failure
    beam.assert_not_called()
