# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Device-free regression coverage for the offline CUDA Beam example."""

import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from examples.offline_inference.beam_search import offline_beam_search as example


def _cli_args(tmp_path: Path) -> list[str]:
    return [
        "example",
        "--model",
        str(tmp_path),
        "--catalog",
        str(tmp_path / "catalog.json"),
        "--prompt",
        str(tmp_path / "prompt.txt"),
        "--constraint-backend",
        "cuda",
        "--beam-width",
        "8",
    ]


@pytest.mark.parametrize(  # type: ignore[misc]
    "options,message",
    [
        (["--constraint-backend", "npu"], "invalid choice"),
        (["--dynamic-beam-widths", "8,4,2"], "unrecognized arguments"),
        (["--result-width", "2"], "unrecognized arguments"),
    ],
)
def test_removed_ascend_cli_options_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    options: list[str],
    message: str,
) -> None:
    monkeypatch.setattr(sys, "argv", _cli_args(tmp_path) + options)
    with pytest.raises(SystemExit) as exc:
        example.parse_args()
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("backend", ["cuda"])  # type: ignore[misc]
@pytest.mark.parametrize("steps", [1, 2])  # type: ignore[misc]
@pytest.mark.parametrize("mode", [[], ["--cot"]])  # type: ignore[misc]
def test_constrained_example_rejects_incomplete_sid_at_cli(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    backend: str,
    steps: int,
    mode: list[str],
) -> None:
    options = ["--constraint-backend", backend, "--generated-steps", str(steps)]
    monkeypatch.setattr(sys, "argv", _cli_args(tmp_path) + options + mode)
    with pytest.raises(SystemExit) as exc:
        example.parse_args()
    assert exc.value.code == 2
    assert "requires --generated-steps 3" in capsys.readouterr().err

    monkeypatch.setattr(sys, "argv", _cli_args(tmp_path) + options + ["--enable-thinking"])
    assert example.parse_args().generated_steps == steps


@pytest.mark.parametrize("with_catalog", [True, False])  # type: ignore[misc]
def test_cuda_beam_example_wires_graphs_and_benchmark(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    with_catalog: bool,
) -> None:
    output_path = tmp_path / "benchmark.json"
    cli_args = _cli_args(tmp_path)
    if with_catalog:
        (tmp_path / "prompt.txt").write_text("A custom prompt.", encoding="utf-8")
    else:
        index = cli_args.index("--catalog")
        del cli_args[index : index + 2]
        index = cli_args.index("--prompt")
        del cli_args[index : index + 2]
    monkeypatch.setattr(
        sys,
        "argv",
        cli_args
        + [
            "--batch-size",
            "2",
            "--repeat",
            "1",
            "--warmup-runs",
            "0",
            "--output-json",
            str(output_path),
        ],
    )
    args = example.parse_args()
    engine_kwargs: dict[str, Any] = {}
    params_kwargs: dict[str, Any] = {}
    sequence = SimpleNamespace(
        tokens=[1, 2, 3, 4, 5], cum_logprob=-1.0, finish_reason="length", stop_reason=None
    )
    outputs = [SimpleNamespace(sequences=[sequence, sequence]) for _ in range(2)]

    def beam_search(prompts: list[Any], params: Any, *, output_mode: str) -> list[Any]:
        assert output_mode == "full"
        assert len(prompts) == 2
        assert params.beam_width == 8
        return outputs

    def encode(text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        if text == "\n":
            return [1]
        assert text == ("A custom prompt." if with_catalog else "Recommend the next video.")
        return [11, 12]

    llm = SimpleNamespace(
        get_tokenizer=lambda: SimpleNamespace(encode=encode, convert_ids_to_tokens=lambda ids: ids),
        beam_search=beam_search,
    )

    def make_llm(**kwargs: Any) -> Any:
        engine_kwargs.update(kwargs)
        return nullcontext(llm)

    def make_params(**kwargs: Any) -> Any:
        params_kwargs.update(kwargs)
        return SimpleNamespace(**kwargs)

    modules: dict[str, dict[str, Any]] = {
        "vllm.inputs": {"tokens_input": lambda ids, **kw: dict(prompt_token_ids=ids, **kw)},
        "vllm_gr.entrypoints.gr": {"GRLLM": make_llm},
        "vllm_gr.sampling_params": {"BeamSearchParams": make_params},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    artifact_path = tmp_path / "unused.safetensors"
    artifact_builds = []

    def build_single_trie_artifact(*paths: Path) -> tuple[str, str]:
        artifact_builds.append(paths)
        return "artifact", "tokenizer"

    monkeypatch.setitem(
        sys.modules,
        "vllm_gr.constraints.artifact_builder",
        SimpleNamespace(build_single_trie_artifact=build_single_trie_artifact),
    )

    example.run(args, artifact_path if with_catalog else None)

    assert params_kwargs["beam_width"] == 8
    config = engine_kwargs["gr"]
    assert config["beam_width"] == config["candidate_width"] == [8, 8, 8]
    assert config["beam_batch_sizes"] == [1, 2]
    assert config["beam_graph_enabled"] is True
    report = json.loads(output_path.read_text())
    assert config["constraint_mode"] == ("table" if with_catalog else "none")
    if with_catalog:
        assert artifact_builds == [(tmp_path, tmp_path / "catalog.json", artifact_path)]
        assert report["catalog"] == str(tmp_path / "catalog.json")
        assert report["prompt"] == str(tmp_path / "prompt.txt")
    else:
        assert not artifact_builds
        assert config["constraint_table"] is None
        assert report["catalog"] is None
        assert report["prompt"] is None
    assert report["runs"][0]["returned_beams"] == 4


@pytest.mark.parametrize(  # type: ignore[misc]
    "backend,override,expected",
    [
        ("cuda", None, "1"),
        ("cuda", "", "1"),
        ("cuda", "0", "0"),
    ],
)
def test_offline_example_selects_model_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: str,
    override: str | None,
    expected: str,
) -> None:
    key = "VLLM_USE_V2_MODEL_RUNNER"
    monkeypatch.setenv(key, "0")
    if override is None:
        monkeypatch.delenv(key)
    else:
        monkeypatch.setenv(key, override)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "example",
            "--model",
            str(tmp_path),
            "--constraint-backend",
            backend,
        ],
    )
    monkeypatch.setattr(example, "run", lambda args: None)

    example.main()

    assert os.environ[key] == expected


@pytest.mark.l1  # type: ignore[misc]
@pytest.mark.parametrize("length", [2, 1100])  # type: ignore[misc]
def test_quickstart_submits_prompt_at_natural_length(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, length: int
) -> None:
    prompt = tmp_path / "prompt.txt"
    text = "A custom prompt."
    prompt.write_text(text, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["example", "--model", str(tmp_path), "--prompt", str(prompt)])
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    tokens = list(range(length))
    submitted: list[list[int]] = []

    def encode(value: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        assert value in (text, "\n")
        return tokens if value == text else [42]

    def beam_search(prompts: list[dict[str, Any]], *args: Any, **kwargs: Any) -> list[Any]:
        submitted.extend(prompt["prompt_token_ids"] for prompt in prompts)
        return [SimpleNamespace(sequences=[])]

    tokenizer = SimpleNamespace(encode=encode)
    llm = SimpleNamespace(get_tokenizer=lambda: tokenizer, beam_search=beam_search)
    modules: dict[str, dict[str, Any]] = {
        "transformers": {
            "AutoTokenizer": SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer)
        },
        "vllm.inputs": {"tokens_input": lambda ids, **kw: dict(prompt_token_ids=ids, **kw)},
        "vllm_gr.entrypoints.gr": {"GRLLM": lambda **kwargs: nullcontext(llm)},
        "vllm_gr.sampling_params": {"BeamSearchParams": SimpleNamespace},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    example.main(quickstart=True)
    example.main()

    assert submitted[0] == tokens
    assert submitted[1] == ([42] * max(0, 1023 - length) + tokens[-1023:])


pytestmark = pytest.mark.cpu_test
