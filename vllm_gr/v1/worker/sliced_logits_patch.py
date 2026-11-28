# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Runtime installer for sliced-vocabulary logits projection."""

from vllm.logger import init_logger

from vllm_gr.ops.sliced_logits import (
    compute_sliced_logits,
)

logger = init_logger(__name__)

_SLICED_LOGITS_PATCHED = False


def apply_sliced_logits_patch() -> bool:
    """Install sliced-logits hooks and report whether this call applied them."""
    global _SLICED_LOGITS_PATCHED

    if _SLICED_LOGITS_PATCHED:
        return False

    try:
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.v1.sample.sampler import Sampler

        original_get_logits = LogitsProcessor._get_logits

        def patched_get_logits(self, hidden_states, lm_head, embedding_bias, skip_gather=False):
            window = getattr(self, "_gr_vocab_slice", None)
            if window is None:
                return original_get_logits(
                    self, hidden_states, lm_head, embedding_bias, skip_gather
                )
            return compute_sliced_logits(
                hidden_states,
                lm_head,
                embedding_bias,
                offset=window[0],
                slice_size=window[1],
                logits_processor=self,
                orig_get_logits=original_get_logits,
                skip_gather=skip_gather,
            )

        LogitsProcessor._get_logits = patched_get_logits

        try:
            original_gather = Sampler.gather_logprobs

            def patched_gather_logprobs(logprobs, num_logprobs, token_ids, *args, **kwargs):
                if logprobs is not None and logprobs.numel() > 0:
                    num_logprobs = min(num_logprobs, logprobs.size(-1))
                return original_gather(logprobs, num_logprobs, token_ids, *args, **kwargs)

            Sampler.gather_logprobs = staticmethod(patched_gather_logprobs)
        except Exception as exc:
            logger.debug("vllm_gr: Could not patch Sampler.gather_logprobs: %s", exc)

        _SLICED_LOGITS_PATCHED = True
        logger.info("vllm_gr: Patched LogitsProcessor._get_logits for sliced vocabulary projection")
        return True
    except Exception as exc:
        logger.debug("vllm_gr: Skipping sliced-logits patch: %s", exc)
        return False
