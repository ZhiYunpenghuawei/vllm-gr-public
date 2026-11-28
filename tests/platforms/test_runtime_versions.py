# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

from argparse import Namespace
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError
from unittest import mock

import pytest

from tests.helpers.native_platform import native_platform
from vllm_gr.platforms.compat import check_vllm_runtime, require_runtime_version
from vllm_gr.platforms.table import current_gr_platform


def test_environment_checks_only_vllm_on_cuda() -> None:
    from tests.helpers.cuda_platform import CudaGRPlatform

    platform = native_platform("CUDA", "cuda")
    with (
        mock.patch("vllm.platforms.current_platform", platform),
        mock.patch(
            "vllm_gr.platforms.compat.check_vllm_runtime", return_value={"vllm": "0.30.0"}
        ) as check_versions,
    ):
        assert current_gr_platform().check_environment() == {
            "vllm": "0.30.0",
            "platform": "cuda",
        }
        assert isinstance(current_gr_platform(), CudaGRPlatform)

    check_versions.assert_called_once_with()


@pytest.mark.l1  # type: ignore[misc]
def test_environment_rejects_unknown_platform_without_cuda_fallback() -> None:
    from vllm_gr.platforms.errors import UnsupportedBackendError

    with (
        mock.patch(
            "vllm.platforms.current_platform",
            native_platform("UNSPECIFIED", "unknown"),
        ),
        mock.patch("vllm_gr.platforms.compat.check_vllm_runtime") as check_versions,
        pytest.raises(UnsupportedBackendError, match="unknown"),
    ):
        current_gr_platform().check_environment()
    check_versions.assert_not_called()


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    ("versions", "check", "expected"),
    [
        (
            {"vendor-runtime": "1.0+local"},
            lambda: require_runtime_version("vendor-runtime", "1.0"),
            "1.0+local",
        ),
        ({"vllm": "0.30.0+cu130"}, check_vllm_runtime, {"vllm": "0.30.0+cu130"}),
    ],
    ids=["vendor-local-suffix", "core-local-suffix"],
)
def test_supported_runtime_version_keeps_the_local_suffix(
    versions: dict[str, str], check: Callable[[], object], expected: object
) -> None:
    with mock.patch(
        "vllm_gr.platforms.compat.metadata.version",
        side_effect=lambda name: versions[name],
    ):
        assert check() == expected


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    "installed", ["0.29.0", "0.30.0rc2", "0.30+vendor"], ids=["older", "rc", "shortened"]
)
def test_rejects_unsupported_vllm_versions(installed: str) -> None:
    with (
        mock.patch("vllm_gr.platforms.compat.metadata.version", return_value=installed),
        pytest.raises(RuntimeError, match="required 0.30.0"),
    ):
        check_vllm_runtime()


@pytest.mark.l1  # type: ignore[misc]
def test_check_env_accepts_external_platform() -> None:
    from vllm_gr.entrypoints.cli.check_env import CheckEnvSubcommand

    external = mock.Mock(backend_id="vendor")
    external.check_environment.return_value = {"platform": "vendor"}
    with (
        mock.patch("vllm.platforms.current_platform", native_platform("OOT", "vendor")),
        mock.patch("vllm_gr.platforms.table.current_gr_platform", return_value=external),
    ):
        CheckEnvSubcommand.cmd(Namespace())
    external.check_environment.assert_called_once_with()


@pytest.mark.l1  # type: ignore[misc]
def test_check_env_cli_uses_shared_runtime_checks(capsys: pytest.CaptureFixture[str]) -> None:
    from vllm_gr.entrypoints.cli.check_env import CheckEnvSubcommand

    versions = {"vllm": "0.30.0+local"}
    platform = native_platform("CUDA", "cuda")

    def version(name: str) -> str:
        if name not in versions:
            raise PackageNotFoundError(name)
        return versions[name]

    with (
        mock.patch("vllm.platforms.current_platform", platform),
        mock.patch("vllm_gr.platforms.compat.metadata.version", side_effect=version),
    ):
        CheckEnvSubcommand.cmd(Namespace())

    report = capsys.readouterr().out
    assert "environment check passed" in report
    assert "platform: cuda" in report
    for name, installed in versions.items():
        assert f"{name}: {installed}" in report


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize(  # type: ignore[misc]
    ("platform_enum", "placement", "message"),
    [
        ("ROCM", "cuda", "ROCM"),
        ("OOT", "vendor", "OOT"),
    ],
)
def test_check_env_cli_rejects_unsupported_native_identity(
    monkeypatch: pytest.MonkeyPatch,
    platform_enum: str,
    placement: str,
    message: str,
) -> None:
    import vllm.platforms

    from vllm_gr.entrypoints.cli.check_env import CheckEnvSubcommand

    native = native_platform(platform_enum, "foreign" if platform_enum == "OOT" else placement)
    native.device_type = placement
    monkeypatch.setattr(vllm.platforms, "current_platform", native)
    with pytest.raises(RuntimeError, match=message):
        CheckEnvSubcommand.cmd(Namespace())


pytestmark = pytest.mark.cpu_test
