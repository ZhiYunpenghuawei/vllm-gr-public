# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""CUDA-owned Beam KV resources and fused reorder."""

import torch

from vllm_gr.v1.beam.attention_context import BeamAttentionContext


class CudaBeamAttentionContext(BeamAttentionContext):
    def __init__(
        self,
        max_batch: int,
        max_beams: int,
        num_kv_heads: int,
        max_decode_steps: int,
        head_dim: int,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype,
        enable_short_suffix_attention: bool = False,
    ):
        device = torch.device(device)
        if enable_short_suffix_attention:
            if device.type != "cuda":
                raise ValueError("Short suffix attention requires CUDA")
            if max_decode_steps < 1:
                raise ValueError("Short suffix attention requires positive pool capacity")
            from vllm_gr.v1.attention.backends.beam_attn_gpu import (
                short_suffix_head_size_supported,
            )

            if not short_suffix_head_size_supported(head_dim) or dtype not in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
            ):
                raise ValueError(
                    "Short suffix attention requires a FlashAttention-compatible "
                    "head size (multiple of 8, at most 256) and FP16/BF16/FP32"
                )
        super().__init__(
            max_batch,
            max_beams,
            num_kv_heads,
            max_decode_steps,
            head_dim,
            num_layers,
            device,
            dtype,
        )
        self.enable_short_suffix_attention = enable_short_suffix_attention
        # Stable pointer tables are consumed by the fused all-layer reorder.
        if device.type == "cuda":
            self.pool_key_ptrs = torch.tensor(
                [pool.unshared_key.data_ptr() for pool in self.pools],
                dtype=torch.int64,
                device=device,
            )
            self.pool_value_ptrs = torch.tensor(
                [pool.unshared_value.data_ptr() for pool in self.pools],
                dtype=torch.int64,
                device=device,
            )
        else:
            self.pool_key_ptrs = None
            self.pool_value_ptrs = None

    def reorder(
        self,
        parent: torch.Tensor,
        slot: int,
        decode_step: int,
        *,
        layout_prevalidated: bool = False,
    ) -> None:
        if self.pools and self.pools[0].unshared_key.is_cuda:
            from vllm_gr.ops.cuda.beam_kv_reorder import select_unshared_kv_fused

            if select_unshared_kv_fused(
                parent,
                slot,
                [pool.unshared_key for pool in self.pools],
                [pool.unshared_value for pool in self.pools],
                decode_step,
                aux_key=self.aux_key_buf,
                aux_value=self.aux_value_buf,
                pool_key_ptrs=self.pool_key_ptrs,
                pool_value_ptrs=self.pool_value_ptrs,
                validate_layout=not layout_prevalidated,
            ):
                return
        super().reorder(parent, slot, decode_step)
