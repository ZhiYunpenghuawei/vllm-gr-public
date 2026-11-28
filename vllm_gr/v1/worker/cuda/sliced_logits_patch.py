# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""CUDA V2 sampler bindings for sliced-vocabulary logits."""

from functools import wraps
from importlib import import_module

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


def _make_clamped_v2_topk(original):
    # Late importers may already bind the patched defining-module function.
    # Reuse it rather than stacking another wrapper for every import site.
    if getattr(original, "_vllm_gr_clamped_topk", False):
        return original

    @wraps(original)
    def compute_topk_scores(logits, num_logprobs, *args, **kwargs):
        if logits is not None and torch.is_tensor(logits) and logits.numel() > 0:
            num_logprobs = min(num_logprobs, logits.size(-1))
        return original(logits, num_logprobs, *args, **kwargs)

    setattr(compute_topk_scores, "_vllm_gr_clamped_topk", True)
    return compute_topk_scores


def apply_cuda_v2_sliced_logits_patch() -> None:
    # V2 ModelRunner: sampling lives in the new worker-local module, whose
    # ``compute_topk_scores`` does the same torch.topk the V1
    # ``gather_logprobs`` clamp above protects. With a narrow catalog
    # vocabulary slice the logits have far fewer columns than the request's
    # num_logprobs (e.g. an impossible 3-token catalog slices to a single
    # column), and the unclamped topk raises "selected index k out of
    # range". The callers bind the function into their own module
    # namespace (``from ... import compute_topk_scores``), so the clamp
    # must be installed on every importer's binding, not just the defining
    # module.
    try:
        from vllm.v1.worker.gpu.sample import logprob as v2_logprob

        v2_logprob.compute_topk_scores = _make_clamped_v2_topk(v2_logprob.compute_topk_scores)
    except Exception as exc:
        logger.debug("vllm_gr: Could not patch V2 compute_topk_scores: %s", exc)

    for _v2_module_name in (
        "vllm.v1.worker.gpu.sample.sampler",
        "vllm.v1.worker.gpu.spec_decode.rejection_sampler",
    ):
        try:
            _v2_module = import_module(_v2_module_name)
            _bound = getattr(_v2_module, "compute_topk_scores", None)
            if callable(_bound):
                setattr(_v2_module, "compute_topk_scores", _make_clamped_v2_topk(_bound))
        except Exception as exc:
            logger.debug(
                "vllm_gr: Could not patch %s.compute_topk_scores: %s",
                _v2_module_name,
                exc,
            )
