# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""CUDA implementation of the legacy constrained Top-K execution protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm_gr.constraints.execution import _ConstrainedTopKExecutionBase
from vllm_gr.constraints.table import token_id_window

if TYPE_CHECKING:
    from vllm_gr.constraints.compiler import ConstraintTableResource


class CUDAConstrainedTopKExecution(_ConstrainedTopKExecutionBase):
    """CUDA constrained Top-K backed by the Triton lookup/gather kernel."""

    def load_tables(self, resource: "ConstraintTableResource") -> None:
        from vllm_gr.ops.cuda.rec_constrained_topk import validate_cuda_sort_capacity

        if self._digest == resource.digest:
            return
        validate_cuda_sort_capacity(
            int(resource.first_token_ids.size),
            resource.max_prefix1_degree,
            resource.max_prefix2_degree,
        )
        super().load_tables(resource)
        self._vocab_slice = token_id_window(
            resource.first_token_ids, resource.prefix1_values, resource.prefix2_values
        )

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def _default_device(self) -> torch.device:
        return torch.device("cuda")

    def _vocab_window(self, logits: torch.Tensor) -> tuple[int, int]:
        """Resolve ``(offset, full_vocab_size)`` for full or sliced logits."""

        if self._vocab_size is None:
            raise RuntimeError("loaded constraint tables are missing vocab_size")
        if logits.shape[1] == self._vocab_size:
            return 0, self._vocab_size

        offset, slice_size = self._vocab_slice
        if logits.shape[1] != slice_size:
            raise ValueError(
                f"CUDA constraint logits width {logits.shape[1]} must match the table "
                f"vocabulary {self._vocab_size} or its token window {slice_size}"
            )
        return offset, self._vocab_size

    def _expected_logits_vocab_size(self, logits: torch.Tensor) -> int:
        self._vocab_window(logits)
        return int(logits.shape[1])

    def _call_op(
        self,
        logits: torch.Tensor,
        sequence_group: torch.Tensor,
        temperatures: torch.Tensor,
        current_step: int,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm_gr.ops.cuda.rec_constrained_topk import rec_constrained_topk_cuda

        vocab_offset, table_vocab_size = self._vocab_window(logits)
        tokens, logprobs, _, _ = rec_constrained_topk_cuda(
            logits,
            sequence_group,
            self._tables["first_token_ids"],
            self._tables["prefix1_offsets"],
            self._tables["prefix1_values"],
            self._tables["prefix1_pair_keys"],
            self._tables["prefix2_value_offsets"],
            self._tables["prefix2_values"],
            temperatures,
            current_step,
            top_k,
            self._max_degrees["max_prefix1_degree"],
            self._max_degrees["max_prefix2_degree"],
            vocab_offset=vocab_offset,
            table_vocab_size=table_vocab_size,
        )
        return tokens, logprobs
