# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Host-only integration for the third-platform lifecycle and admission test."""

from typing import Any

from vllm_gr.config.backend_plan import BackendPlan, BackendRequest, PlanData
from vllm_gr.platforms.interface import GRPlatform


class FakeGRPlatform(GRPlatform):  # type: ignore[misc]
    from tests.helpers.native_platform import VendorNative as native_platform_type

    backend_id = "fakeaccel"

    def resolve(self, request: BackendRequest) -> BackendPlan:
        return BackendPlan(
            self.backend_id,
            request,
            PlanData(
                {
                    "runner": "test.runner",
                    "stage": "test.stage",
                    "beam_limits": {
                        "device_type": self.backend_id,
                        "max_decode_rows": 4,
                        "retain_candidate_widths": False,
                        "constraint_backend": None,
                        "supports_unconstrained": True,
                        "runner_generation": "v1",
                        "local_kv_heads": None,
                        "head_size": None,
                    },
                }
            ),
        )

    def validate_worker(self, *, plan: BackendPlan, native_runner: Any) -> None:
        assert plan.backend_id == self.backend_id and native_runner.device == self.backend_id
