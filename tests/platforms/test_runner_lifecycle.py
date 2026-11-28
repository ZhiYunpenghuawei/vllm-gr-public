# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Native entry-point dispatch, completion ownership, and generation cleanup."""

import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from vllm_gr.config import GRConfig
from vllm_gr.config.backend_plan import BackendPlan, BackendRequest, PlanData
from vllm_gr.config.resolved_choices import ResolvedChoices
from vllm_gr.platforms import table
from vllm_gr.platforms.interface import GRPlatform
from vllm_gr.platforms.planning import admit_backend_plan
from vllm_gr.v1.engine.worker_readiness import worker_backend_readiness
from vllm_gr.v1.worker import runner_binding as binding
from vllm_gr.v1.worker.runner_binding import (
    BackendBinding,
    bind_after_model_load,
    close_backend_binding,
)
from vllm_gr.v1.worker.runner_lifecycle import (
    bind_native_runner,
    has_runner_lifecycle,
    install_runner_lifecycle,
)
from vllm_gr.v1.worker.stage_binding import StageBinding

pytestmark = [pytest.mark.cpu_test, pytest.mark.l1]


def _plan(implementation: str) -> BackendPlan:
    empty = PlanData({})
    request = BackendRequest(empty, empty, empty, empty)
    return BackendPlan("test", request, PlanData({"runner": implementation}))


def _mock_process_platform(monkeypatch: pytest.MonkeyPatch, **options: Any) -> Mock:
    import vllm.platforms

    implementation = Mock(spec=GRPlatform, **options)
    monkeypatch.setattr(vllm.platforms, "current_platform", implementation)
    return implementation


@pytest.mark.parametrize("prepare", ["_prepare_inputs", "prepare_inputs"])  # type: ignore[misc]
def test_native_calls_use_the_owning_adapter_and_keep_late_completion(prepare: str) -> None:
    events: list[tuple[str, str]] = []

    class Runner:
        _gr_backend_binding: BackendBinding

        def __init__(self, name: str) -> None:
            self.name = name
            self.receipt: Any = None

        def execute_model(self, receipt: Any) -> None:
            events.append((self.name, "execute"))
            self.receipt = receipt

        def sample_tokens(self, *, fail: bool = False) -> Any:
            events.append((self.name, "sample"))
            if fail:
                raise ValueError("sample failed")
            receipt, self.receipt = self.receipt, None
            return receipt

    def prepare_inputs(runner: Runner, *, fail: bool = False) -> int:
        events.append((runner.name, "prepare"))
        if fail:
            raise ValueError("prepare failed")
        return 7

    setattr(Runner, prepare, prepare_inputs)
    install_runner_lifecycle(Runner, "native", prepare=prepare)
    entry = Runner.execute_model
    install_runner_lifecycle(Runner, "native", prepare=prepare)
    assert Runner.execute_model is entry
    assert has_runner_lifecycle(Runner, "native")
    first, second = Runner("first"), Runner("second")
    receipt = object()
    first.execute_model(receipt)
    assert first.sample_tokens() is receipt  # unplanned native fallback

    plan = _plan("native")
    for runner in (first, second):
        adapter = bind_native_runner(plan, runner)
        runner._gr_backend_binding = BackendBinding(plan, adapter, None, None)
    probe = Mock(wraps=first._gr_backend_binding.runner_adapter.execute_model)
    first._gr_backend_binding.runner_adapter.execute_model = probe
    events.clear()
    assert getattr(first, prepare)() == 7
    first.execute_model(receipt)
    probe.assert_called_once_with(receipt)
    with pytest.raises(ValueError, match="prepare failed"):
        getattr(second, prepare)(fail=True)
    with pytest.raises(ValueError, match="sample failed"):
        first.sample_tokens(fail=True)
    assert first.receipt is receipt and second.receipt is None
    assert first.sample_tokens() is receipt
    assert events == [
        ("first", "prepare"),
        ("first", "execute"),
        ("second", "prepare"),
        ("first", "sample"),
        ("first", "sample"),
    ]
    old = first._gr_backend_binding.runner_adapter
    close_backend_binding(first)
    close_backend_binding(first)
    with pytest.raises(RuntimeError, match="closed"):
        old.execute_model(receipt)
    assert second._gr_backend_binding.runner_adapter.plan == plan
    with pytest.raises(RuntimeError, match="admitted backend plan"):
        bind_native_runner(_plan("other"), first)


def test_full_load_rebinds_the_same_plan_only_once_per_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    implementation = _mock_process_platform(
        monkeypatch,
        backend_id="test",
        vendor_distribution=None,
    )
    implementation.resolve.side_effect = lambda request: BackendPlan(
        "test", request, PlanData({"runner": "native"})
    )
    old, new = Mock(implementation_id="native"), Mock(implementation_id="native")
    implementation.create_model_runner_adapter.side_effect = [old, new]
    monkeypatch.setattr(table, "current_gr_platform", lambda: implementation)
    config = SimpleNamespace(additional_config={})
    GRConfig(beam_max_width=16).write_to_vllm_config(config)
    plan = admit_backend_plan(config)
    assert plan is not None
    runner = SimpleNamespace(vllm_config=config)
    worker = SimpleNamespace(model_runner=runner)

    def load_weights(self: Any, model: object) -> object:
        assert getattr(self, "_gr_backend_binding", None) is None
        self.model = model
        return model

    load = bind_after_model_load(load_weights)
    for adapter in (old, new):
        model = object()
        assert load(runner, model) is model
        assert runner._gr_backend_binding.runner_adapter is adapter
        assert worker_backend_readiness(worker)["fingerprint"] == plan.fingerprint
        assert binding.bind_backend_services(runner) is runner._gr_backend_binding
        adapter.close.assert_not_called()
    old.close.assert_called_once_with()
    assert implementation.create_model_runner_adapter.call_count == 2


@pytest.fixture  # type: ignore[misc]
def autofit_runner(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any, BackendPlan]:
    cleanup_module = ModuleType("vllm_gr.v1.worker.model_runner_patch")
    setattr(cleanup_module, "close_constraint_backend", Mock())
    monkeypatch.setitem(sys.modules, cleanup_module.__name__, cleanup_module)
    implementation = _mock_process_platform(
        monkeypatch,
        backend_id="test",
        vendor_distribution=None,
    )
    implementation.resolve.side_effect = lambda request: BackendPlan(
        "test",
        request,
        PlanData(
            {
                "runner": "native",
                "admitted_max_model_len": request.native.to_dict()["model_config"]["max_model_len"],
            }
        ),
    )
    monkeypatch.setattr(table, "current_gr_platform", lambda: implementation)
    config = SimpleNamespace(
        additional_config={}, model_config=SimpleNamespace(max_model_len=4096, dtype="bfloat16")
    )
    GRConfig(beam_max_width=16).write_to_vllm_config(config)
    plan = admit_backend_plan(config)
    assert plan is not None
    return SimpleNamespace(vllm_config=config), implementation, plan


def test_first_load_binds_admitted_plan_after_native_autofit(
    autofit_runner: tuple[Any, Any, BackendPlan],
) -> None:
    runner, implementation, plan = autofit_runner
    adapter = Mock(implementation_id="native")
    implementation.create_model_runner_adapter.return_value = adapter

    def load_weights(self: Any, model: object) -> object:
        assert getattr(self, "_gr_backend_binding", None) is None
        self.vllm_config.model_config.max_model_len = 2048
        self.model = model
        return model

    observed_lengths = []
    implementation.validate_worker.side_effect = lambda **kwargs: observed_lengths.append(
        kwargs["native_runner"].vllm_config.model_config.max_model_len
    )
    model = object()
    assert bind_after_model_load(load_weights)(runner, model) is model
    assert runner._gr_backend_binding.plan == plan
    assert runner._gr_backend_binding.runner_adapter is adapter
    assert runner.vllm_config.model_config.max_model_len == 2048
    assert (
        runner.vllm_config.additional_config["vllm_gr_choices"]
        == ResolvedChoices.from_plan(plan).to_wire()
    )
    assert observed_lengths == [2048]
    implementation.create_model_runner_adapter.assert_called_once_with(
        plan=plan, native_runner=runner
    )
    assert worker_backend_readiness(SimpleNamespace(model_runner=runner))["fingerprint"] == (
        plan.fingerprint
    )


def test_full_reload_rebinds_admitted_plan_after_native_autofit(
    autofit_runner: tuple[Any, Any, BackendPlan],
) -> None:
    runner, implementation, plan = autofit_runner
    old, new = Mock(implementation_id="native"), Mock(implementation_id="native")
    implementation.create_model_runner_adapter.side_effect = [old, new]

    def load_weights(self: Any, model: object) -> object:
        assert getattr(self, "_gr_backend_binding", None) is None
        self.model = model
        return model

    load = bind_after_model_load(load_weights)
    first = object()
    assert load(runner, first) is first
    previous = runner._gr_backend_binding
    runner.vllm_config.model_config.max_model_len = 2048
    report = worker_backend_readiness(SimpleNamespace(model_runner=runner))
    assert report["services"] == previous.report()["services"]
    assert report["fingerprint"] == previous.plan.fingerprint
    assert report["max_model_len"] == runner.vllm_config.model_config.max_model_len
    second = object()
    assert load(runner, second) is second
    assert runner._gr_backend_binding is not previous
    assert runner._gr_backend_binding.plan == plan
    assert runner._gr_backend_binding.runner_adapter is new
    assert runner.vllm_config.model_config.max_model_len == 2048
    assert (
        runner.vllm_config.additional_config["vllm_gr_choices"]
        == ResolvedChoices.from_plan(plan).to_wire()
    )
    assert implementation.create_model_runner_adapter.call_count == 2
    assert implementation.validate_worker.call_count == 2
    assert all(
        call.kwargs["plan"] == plan for call in implementation.validate_worker.call_args_list
    )
    old.close.assert_called_once_with()
    new.close.assert_not_called()
    assert worker_backend_readiness(SimpleNamespace(model_runner=runner))["fingerprint"] == (
        plan.fingerprint
    )


@pytest.mark.parametrize(  # type: ignore[misc]
    "drift",
    ["length-increase", "length-zero", "length-bool", "dtype", "beam", "runner", "integration"],
)
def test_autofit_does_not_hide_worker_plan_drift(
    autofit_runner: tuple[Any, Any, BackendPlan], drift: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner, implementation, _ = autofit_runner

    def load_weights(self: Any) -> None:
        self.vllm_config.model_config.max_model_len = 2048
        if drift.startswith("length-"):
            self.vllm_config.model_config.max_model_len = {
                "length-increase": 8192,
                "length-zero": 0,
                "length-bool": True,
            }[drift]
        elif drift == "dtype":
            self.vllm_config.model_config.dtype = "float32"
        elif drift == "beam":
            GRConfig(beam_max_width=32).write_to_vllm_config(self.vllm_config)
        elif drift == "runner":
            self.vllm_config.use_v2_model_runner = True
        else:

            class OtherPlatform(Mock):
                pass

            other = OtherPlatform(backend_id=implementation.backend_id, vendor_distribution=None)
            monkeypatch.setattr(table, "current_gr_platform", lambda: other)

    with pytest.raises(RuntimeError, match="Worker backend plan mismatch"):
        bind_after_model_load(load_weights)(runner)
    implementation.validate_worker.assert_not_called()
    implementation.create_model_runner_adapter.assert_not_called()
    assert getattr(runner, "_gr_backend_binding", None) is None


@pytest.mark.parametrize("cleanup_fails", [False, True])  # type: ignore[misc]
def test_reload_preserves_live_generation_and_releases_failed_replacement(
    monkeypatch: pytest.MonkeyPatch, cleanup_fails: bool
) -> None:
    _mock_process_platform(monkeypatch)
    events: list[str] = []
    plan = _plan("native")
    previous = Mock()
    runner = SimpleNamespace(
        _gr_backend_binding=BackendBinding(plan, previous, None, None), pending_receipt=object()
    )
    native_error = RuntimeError("pending native receipt")

    def release_runtime(self: Any) -> None:
        if self.pending_receipt is not None:
            raise native_error
        events.append("release runtime")
        if cleanup_fails and self._gr_backend_binding is None:
            raise RuntimeError("native cleanup failed")

    def native_load(self: Any) -> None:
        pytest.fail("an undrained receipt must prevent native model replacement")

    with pytest.raises(RuntimeError) as rejected:
        bind_after_model_load(native_load, close_runtime=release_runtime)(runner)
    assert rejected.value is native_error
    previous.close.assert_not_called()
    assert events == []
    runner.pending_receipt = None

    def replace(self: Any) -> None:
        assert self._gr_backend_binding is None
        events.append("native load")

    def fail_binding(self: Any) -> None:
        events.append("bind")
        raise ValueError("factory failed")

    def release_table(self: Any) -> None:
        events.append("release table")

    cleanup_module = ModuleType("vllm_gr.v1.worker.model_runner_patch")
    setattr(cleanup_module, "close_constraint_backend", release_table)
    monkeypatch.setitem(sys.modules, cleanup_module.__name__, cleanup_module)
    monkeypatch.setattr(binding, "bind_backend_services", fail_binding)
    with pytest.raises(ValueError, match="factory failed"):
        bind_after_model_load(replace, close_runtime=release_runtime)(runner)
    previous.close.assert_called_once_with()
    assert runner._gr_backend_binding is None
    assert events == ["release runtime", "native load", "bind", "release runtime", "release table"]


def test_first_load_cleans_partial_resources_before_a_binding_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_process_platform(monkeypatch)
    runner = SimpleNamespace(native_resource=None)
    native_resource = Mock()
    failure = ValueError("partial native load")

    def allocate_then_fail(self: Any) -> None:
        self.native_resource = native_resource
        raise failure

    def release_runtime(self: Any) -> None:
        if self.native_resource is not None:
            self.native_resource.close()
            self.native_resource = None

    close_table = Mock()
    cleanup_module = ModuleType("vllm_gr.v1.worker.model_runner_patch")
    setattr(cleanup_module, "close_constraint_backend", close_table)
    monkeypatch.setitem(sys.modules, cleanup_module.__name__, cleanup_module)
    with pytest.raises(ValueError) as raised:
        bind_after_model_load(allocate_then_fail, close_runtime=release_runtime)(runner)
    assert raised.value is failure
    native_resource.close.assert_called_once_with()
    close_table.assert_called_once_with(runner)
    assert runner.native_resource is None


def test_uninstalled_platform_seam_rejects_model_load(monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm.platforms

    from tests.helpers.native_platform import native_platform

    monkeypatch.setattr(vllm.platforms, "current_platform", native_platform("OOT", "foreign"))
    cleanup_module = ModuleType("vllm_gr.v1.worker.model_runner_patch")
    close_table = Mock()
    setattr(cleanup_module, "close_constraint_backend", close_table)
    monkeypatch.setitem(sys.modules, cleanup_module.__name__, cleanup_module)
    native_load = Mock()
    runner = SimpleNamespace()

    def load(self: Any) -> None:
        native_load(self)

    with pytest.raises(RuntimeError, match="requires the GR platform as current_platform"):
        bind_after_model_load(load)(runner)
    native_load.assert_not_called()
    close_table.assert_called_once_with(runner)


@pytest.mark.parametrize("failure", [False, True])  # type: ignore[misc]
def test_stage_dispatch_uses_the_admitted_runner_generation(
    monkeypatch: pytest.MonkeyPatch, failure: bool
) -> None:
    import vllm.platforms
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_gr.v1.beam.dispatch import GR_RETIRE_OUTPUT_ATTR, GR_STAGE_OUTPUT_ATTR
    from vllm_gr.v1.worker.model_runner_common import make_patched_execute_model

    native = Mock(return_value="ordinary")
    native_sample = Mock(return_value="native sample")
    stage = StageBinding("test.stage", Mock(), Mock(), Mock(), Mock(), Mock())

    class Runner:
        # Any attempt to rediscover the hardware during dispatch must fail.
        @property
        def device(self) -> str:
            if getattr(self, "_gr_backend_binding", None) is not None:
                raise AssertionError("bound dispatch inspected device")
            return "fakeaccel"

        @make_patched_execute_model  # type: ignore[misc]
        def execute_model(self, scheduled: Any) -> Any:
            return native(self, scheduled)

        def sample_tokens(self, grammar_output: Any) -> Any:
            return native_sample(self, grammar_output)

        def prepare_inputs(self) -> None:
            pass

    from tests.helpers.beam_requests import make_batch_request, make_multi_batch_request
    from tests.helpers.native_platform import native_platform
    from vllm_gr.platforms.beam_requests import (
        decode_batch_capacity,
        validate_decode_token_budget,
        validate_gr_device_config,
    )

    platform = native_platform("OOT", "fakeaccel")
    monkeypatch.setitem(
        table.BUILTIN_PROBES, "fakeaccel", lambda: "tests.platforms.fake_accel.FakeGRPlatform"
    )
    monkeypatch.setattr(vllm.platforms, "current_platform", platform)
    config = SimpleNamespace(additional_config={})
    GRConfig(beam_max_width=16).write_to_vllm_config(config)
    plan = admit_backend_plan(config)
    assert plan is not None and plan.backend_id == "fakeaccel"
    from vllm_gr.plugin import install_gr_platform

    assert install_gr_platform(table.current_gr_platform())
    request = make_batch_request()
    validate_gr_device_config(config, request)
    validate_decode_token_budget(request, config)
    assert decode_batch_capacity(config, beam_width=4) == 1
    with pytest.raises(ValueError, match="FAKEACCEL Beam Decode requires 8 physical tokens"):
        validate_decode_token_budget(make_multi_batch_request(), config)
    install_runner_lifecycle(Runner, "test.runner", prepare="prepare_inputs", stage=stage)
    runner: Any = Runner()
    runner.vllm_config = config
    ordinary = SchedulerOutput.make_empty()
    assert runner.execute_model(ordinary) == "ordinary"
    frame = SchedulerOutput.make_empty()
    setattr(frame, GR_STAGE_OUTPUT_ATTR, {"session": object()})
    with pytest.raises(RuntimeError, match="before the runner bound"):
        runner.execute_model(frame)

    load = bind_after_model_load(lambda self: None)
    with monkeypatch.context() as mismatch:
        from dataclasses import replace

        mismatch.setattr(Runner, "_gr_stage_binding", replace(stage, implementation_id="wrong"))
        with pytest.raises(RuntimeError, match="differs from the admitted backend plan"):
            load(runner)
    assert getattr(runner, "_gr_backend_binding", None) is None
    load(runner)
    original_binding = runner._gr_backend_binding
    assert original_binding.stage is stage
    from vllm_gr.v1.engine.worker_readiness import (
        validate_worker_readiness,
        worker_backend_readiness,
    )

    report = worker_backend_readiness(SimpleNamespace(model_runner=runner))
    assert report["services"]["stage"] == stage.implementation_id
    validate_worker_readiness(original_binding.plan, [report], expected_workers=1)
    report["services"]["stage"] = "different.stage"
    with pytest.raises(RuntimeError, match="stage implementation mismatch"):
        validate_worker_readiness(original_binding.plan, [report], expected_workers=1)
    error = ValueError("stage owner creation failed")
    execute = stage.execute
    assert isinstance(execute, Mock)
    execute.side_effect = error if failure else None
    execute.return_value = None
    if failure:
        with pytest.raises(ValueError) as raised:
            runner.execute_model(frame)
        assert raised.value is error
    else:
        assert runner.execute_model(frame) is None
    owned = execute.call_args.args[1]
    assert owned is not frame and owned.finished_req_ids is not frame.finished_req_ids
    with pytest.raises(RuntimeError, match="before model reload"):
        load(runner)
    assert runner._gr_backend_binding is original_binding
    if failure:
        with pytest.raises(ValueError) as sampled:
            runner.sample_tokens(None)
        assert sampled.value is error
    else:
        assert isinstance(stage.sample, Mock)
        assert runner.sample_tokens(None) is stage.sample.return_value
        stage.sample.assert_called_once_with(runner, None)
    native_sample.assert_not_called()

    retirement = SchedulerOutput.make_empty()
    setattr(retirement, GR_RETIRE_OUTPUT_ATTR, object())
    assert isinstance(stage.retire, Mock)
    assert runner.execute_model(retirement) is stage.retire.return_value
    assert stage.retire.call_args.args[1] is not retirement
    native.assert_called_once()
    load(runner)
    assert runner._gr_backend_binding is not original_binding
    ordinary.finished_req_ids.add("done")
    assert runner.execute_model(ordinary) == "ordinary"
    assert isinstance(stage.cancel_finished, Mock)
    stage.cancel_finished.assert_called_once_with(runner, {"done"})
    assert runner.sample_tokens(None) == "native sample"
    close_backend_binding(runner)
    with pytest.raises(RuntimeError, match="before the runner bound"):
        runner.execute_model(retirement)
