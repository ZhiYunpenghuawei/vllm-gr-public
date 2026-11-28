# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""CUDA-owned logits projection preparation for a loaded model."""

from typing import Any

from vllm.logger import init_logger

from vllm_gr.config import constraint_backend_name
from vllm_gr.v1.worker.model_runner_patch import _runner_vocab_size

logger = init_logger(__name__)


def prepare_cuda_model(runner: Any) -> None:
    from vllm_gr.v1.worker.cuda.lm_head_cache import install_lm_head_cache_reload_hooks

    install_lm_head_cache_reload_hooks(runner.model)


def prepare_cuda_constraint_projection(runner: Any, backend: Any) -> None:
    """Configure the model's projection for the initial canonical table.

    Replacement tables must preserve the already-published vocabulary slice;
    do not change live model projection state during candidate preparation.
    """
    if getattr(runner, "constraint_backend", None) is not None:
        return
    kind = getattr(backend.info, "backend", None)
    if kind is None or constraint_backend_name(kind) != "cuda":
        return

    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    window = backend.vocab_slice
    parallel = getattr(getattr(runner, "vllm_config", None), "parallel_config", None)
    if getattr(parallel, "tensor_parallel_size", 1) != 1:
        # All ranks retain the original full-logits collective path.
        window = None
        backend.vocab_slice = None
        logger.info(
            "vllm-gr: canonical CUDA vocabulary slicing disabled for TP>1; "
            "using full-vocabulary projection"
        )
    if window == (0, _runner_vocab_size(runner)):
        window = None
    processors = tuple(
        module for module in runner.model.modules() if isinstance(module, LogitsProcessor)
    )
    for processor in processors:
        processor._gr_vocab_slice = window
    runner._gr_vocab_slice_processors = processors
