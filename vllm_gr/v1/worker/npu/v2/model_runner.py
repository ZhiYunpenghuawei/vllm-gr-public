# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Owned eager NPU runner implementing the vLLM 0.30 V2 Worker protocol.

Native scheduler state and physical Beam rows have separate lifetimes. This
runner uses the native model loader, KV spec/allocation and forward context,
without constructing a CUDA runner or an Ascend runner.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from threading import Lock
from typing import Any

import torch

from vllm_gr.config.execution import validate_gr_execution_config
from vllm_gr.config.models import GRConfig, SUPPORTED_BEAM_BATCH_SIZES, resolve_candidate_width
from vllm_gr.v1.beam import BEAM_DATA_VAR
from vllm_gr.v1.beam.dispatch import GR_STAGE_OUTPUT_ATTR, GRStageKind
from vllm_gr.v1.beam.execution_batch import BeamBatchExecutionView
from vllm_gr.v1.worker.npu.v2.request_state import NativeRequestState


@dataclass(frozen=True)
class InputBatch:
    req_ids: tuple[str, ...]
    logits_indices: torch.Tensor
    starts: dict[str, int]
    counts: dict[str, int]


@dataclass(frozen=True)
class PreparedForward:
    input_ids: torch.Tensor
    positions: torch.Tensor
    attention_metadata: dict[str, Any]
    slot_mappings: dict[str, torch.Tensor]
    beam_info: dict[str, Any] | None = None
    input_batch: InputBatch | None = None


@dataclass(frozen=True)
class ExecuteModelState:
    hidden_states: torch.Tensor
    input_batch: InputBatch


def _ceildiv(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


class NPUModelRunnerV2:
    """Single-device uniform decoder runner; GR owns all sampling.

    The executor still calls execute_model and sample_tokens separately. GR
    consumes Prefill hidden states and invokes the physical Decode forward
    independently; no Beam expansion enters the native request state.
    """

    def __init__(self, vllm_config: Any, device: torch.device) -> None:
        validate_gr_execution_config(vllm_config)
        if device.type != "npu":
            raise ValueError("Owned NPU V2 requires an NPU device")
        model_config = vllm_config.model_config
        if model_config.runner_type != "generate":
            raise ValueError("Owned NPU V2 only supports text Beam generation")
        self.vllm_config = vllm_config
        self.model_config = model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.scheduler_config = vllm_config.scheduler_config
        self.compilation_config = vllm_config.compilation_config
        self.device = device
        self.use_async_scheduling = self.scheduler_config.async_scheduling
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.supports_mm_inputs = False
        self.speculative_config = None
        self.lora_config = None
        self.request_state = NativeRequestState(self.max_num_reqs, model_config.max_model_len)
        self.execute_model_state: ExecuteModelState | None = None
        self.model: Any = None
        self.model_memory_usage = 0
        self.constraint_backend: Any = None
        self.kv_connector_output = None
        self.kv_cache_config: Any = None
        self.kv_caches: list[torch.Tensor] = []
        self.attn_groups: list[list[Any]] = []
        self._metadata_builder: Any = None
        self._cache_spec: Any = None
        self._layer_names: tuple[str, ...] = ()
        self._owned_context: dict[str, Any] = {}
        self._decode_advance: tuple[dict[str, int], dict[str, int]] | None = None
        self._closed = False
        self._failed: BaseException | None = None
        self._weight_update_lock = Lock()
        self._weight_update_active = False
        self._owned_graph_registry: Any = None
        self._graph_generation = 0
        self._initial_free_memory, self._total_memory = torch.npu.mem_get_info(device)

    def _require_live(self) -> None:
        if self._closed:
            raise RuntimeError("Owned NPU V2 runner is closed")
        if getattr(self, "_weight_update_active", False):
            raise RuntimeError("Owned NPU V2 weight update is in progress")
        if self._failed is not None:
            raise RuntimeError(
                "Owned NPU V2 runner cannot continue after execution failure"
            ) from self._failed

    def get_supported_tasks(self) -> tuple[str, ...]:
        return ("generate",)

    def get_model(self) -> Any:
        self._require_live()
        if self.model is None:
            raise RuntimeError("Owned NPU V2 model is not loaded")
        return self.model

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader import get_model_loader

        self._require_live()
        if self.model is not None:
            raise RuntimeError("Owned NPU V2 full model reload requires a new runner")
        load_config = copy(self.vllm_config.load_config)
        if load_dummy_weights:
            load_config.load_format = "dummy"
        before = torch.npu.memory_allocated(self.device)
        context = self.compilation_config.static_forward_context
        existing = set(context)
        try:
            with set_current_vllm_config(self.vllm_config):
                loader = get_model_loader(load_config)
                self.model = loader.load_model(
                    vllm_config=self.vllm_config, model_config=self.model_config
                )
            self.model_memory_usage = max(0, torch.npu.memory_allocated(self.device) - before)
            self._init_metadata_builder()
        except BaseException as error:
            self._failed = error
            raise
        finally:
            self._owned_context.update(
                {key: layer for key, layer in context.items() if key not in existing}
            )

    def get_kv_cache_spec(self) -> dict[str, Any]:
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        self.get_model()
        result = {}
        for name, layer in self.compilation_config.static_forward_context.items():
            if not callable(getattr(layer, "get_kv_cache_spec", None)):
                continue
            if getattr(layer, "kv_sharing_target_layer_name", None):
                raise ValueError("Owned NPU V2 does not support KV sharing layers")
            spec = layer.get_kv_cache_spec(self.vllm_config)
            if (
                type(spec) is not FullAttentionSpec
                or spec.dtype not in (torch.float16, torch.bfloat16)
                or spec.head_size != spec.head_size_v
                or spec.page_size_bytes != spec.real_page_size_bytes
                or spec.sliding_window is not None
                or spec.attention_chunk_size is not None
                or spec.non_causal
            ):
                raise ValueError("Owned NPU V2 requires uniform unquantized full attention")
            if layer.get_attn_backend().get_name() != "CUSTOM":
                raise ValueError("Owned NPU V2 requires GR CUSTOM attention")
            result[name] = layer.get_attn_backend().customize_spec(spec)
        if not result or len(set(result.values())) != 1:
            raise ValueError("Owned NPU V2 requires one uniform KV cache specification")
        return result

    def _init_metadata_builder(self) -> None:
        from vllm_gr.v1.attention.backends.beam_attn_metadata import BeamAttentionMetadataBuilder

        specs = self.get_kv_cache_spec()
        self._layer_names = tuple(specs)
        self._cache_spec = next(iter(specs.values()))
        self._metadata_builder = BeamAttentionMetadataBuilder(
            self._cache_spec, list(self._layer_names), self.vllm_config, self.device
        )

    def update_max_model_len(self, max_model_len: int) -> None:
        if self.kv_cache_config is not None or self.request_state.requests:
            raise RuntimeError("Cannot resize an active owned NPU V2 runner")
        if not 0 < max_model_len <= self.model_config.max_model_len:
            raise ValueError("Invalid maximum model length")
        self.model_config.max_model_len = max_model_len
        self.request_state.max_model_len = max_model_len

    def _beam_memory_reservation(self) -> int:
        gr = GRConfig.from_vllm_config(self.vllm_config)
        width = gr.beam_max_width or 1
        batch = min(self.max_num_reqs, max(SUPPORTED_BEAM_BATCH_SIZES))
        rows = min(batch * width, self.max_num_tokens)
        steps = gr.beam_max_decode_steps
        heads = self.model_config.get_num_kv_heads(self.parallel_config)
        dim = self.model_config.get_head_size()
        layers = self.model_config.get_num_layers(self.parallel_config)
        # Reserve base suffix pools, per-layer reordering scratch and all
        # possible distinct dynamic-width banks. All selected model dtypes
        # occupy two bytes per element. Use the full configured capacity as
        # an upper bound even when a request chooses a narrower width.
        pool_rows = batch * width
        suffix = 4 * steps * heads * dim * (pool_rows * (layers * (steps + 1) + 1) + width * layers)
        # Candidate token/score/mask and history scratch. Full-vocabulary FP32
        # scoring is bounded by the admitted physical token budget.
        vocab = self.model_config.get_vocab_size()
        candidate_width = resolve_candidate_width(gr)
        decision = pool_rows * (candidate_width * 32 + steps * 32)
        logits = rows * vocab * 4 * 3
        return suffix + decision + logits

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import set_forward_context

        model = self.get_model()
        if self.kv_cache_config is not None:
            raise RuntimeError("KV memory profiling must precede cache initialization")
        torch.npu.synchronize(self.device)
        torch.npu.empty_cache()
        torch.npu.reset_peak_memory_stats(self.device)
        free_before, _ = torch.npu.mem_get_info(self.device)
        allocated_before = torch.npu.memory_allocated(self.device)
        # Exercise actual model layers at the configured token capacity. The
        # attention profiling path accepts None metadata before KV allocation.
        ids = torch.zeros(self.max_num_tokens, dtype=torch.int32, device=self.device)
        positions = torch.arange(self.max_num_tokens, dtype=torch.int64, device=self.device)
        positions.remainder_(self.model_config.max_model_len)
        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=self.max_num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        ):
            hidden = model(input_ids=ids, positions=positions)
        if not isinstance(hidden, torch.Tensor):
            raise RuntimeError("Owned NPU V2 profiling requires a tensor model output")
        del hidden, ids, positions
        torch.npu.synchronize(self.device)
        peak = max(0, torch.npu.max_memory_allocated(self.device) - allocated_before)
        torch.npu.empty_cache()
        free_after, _ = torch.npu.mem_get_info(self.device)
        # Account for non-PyTorch allocations using the actual device free
        # memory in addition to the allocator's transient peak.
        used_since_start = max(0, self._initial_free_memory - min(free_before, free_after))
        budget = int(self._total_memory * self.cache_config.gpu_memory_utilization)
        budget = min(budget, self._initial_free_memory)
        available = (
            min(free_after, budget - used_since_start) - peak - self._beam_memory_reservation()
        )
        explicit = self.cache_config.kv_cache_memory_bytes
        if explicit is not None:
            if explicit > available:
                raise RuntimeError("Explicit KV memory exceeds the profiled NPU budget")
            available = explicit
        if available <= 0:
            raise RuntimeError("Insufficient KV capacity after NPU model/Beam memory profiling")
        return int(available)

    def initialize_kv_cache(self, kv_cache_config: Any) -> None:
        from vllm.v1.kv_cache_layout import KVCacheLayout
        from vllm.v1.worker.utils import AttentionGroup, allocate_kv_cache, bind_kv_cache

        self.get_model()
        if self.kv_cache_config is not None:
            raise RuntimeError("Owned NPU V2 KV cache already initialized")
        groups = kv_cache_config.kv_cache_groups
        if (
            len(groups) != 1
            or groups[0].kv_cache_spec != self._cache_spec
            or set(groups[0].layer_names) != set(self._layer_names)
            or kv_cache_config.num_blocks < 2
            or any(tensor.host_resident for tensor in kv_cache_config.kv_cache_tensors)
        ):
            raise ValueError("Unsupported owned NPU V2 KV cache allocation plan")
        layout = self.cache_config.get_resolved_kv_cache_layout()
        if layout != KVCacheLayout.LBNHC:
            raise ValueError("Owned NPU V2 requires the LBNHC native KV layout")
        backend = self.compilation_config.static_forward_context[
            self._layer_names[0]
        ].get_attn_backend()
        group = AttentionGroup(backend, list(self._layer_names), self._cache_spec, 0)
        group.metadata_builders.append(self._metadata_builder)
        caches = allocate_kv_cache(
            kv_cache_config, self.device, layout, [self._cache_spec.block_size]
        )
        expected = (
            kv_cache_config.num_blocks,
            self._cache_spec.num_kv_heads,
            self._cache_spec.block_size,
            self._cache_spec.head_size * 2,
        )
        if set(caches) != set(self._layer_names) or any(
            tuple(cache.shape) != expected or cache.dtype != self._cache_spec.dtype
            for cache in caches.values()
        ):
            raise RuntimeError("Native KV allocation disagrees with the NPU attention contract")
        bind_kv_cache(
            caches,
            self.compilation_config.static_forward_context,
            self.kv_caches,
            kv_cache_groups=groups,
        )
        self.attn_groups = [[group]]
        self.kv_cache_config = kv_cache_config

    def _require_cache(self) -> None:
        self.get_model()
        if self.kv_cache_config is None:
            raise RuntimeError("Owned NPU V2 KV cache is not initialized")

    @torch.inference_mode()
    def consume_native_state(self, scheduler_output: Any) -> None:
        """Apply scheduler updates, with no physical Beam-token expansion."""
        self._require_cache()
        if self.execute_model_state is not None or self._decode_advance is not None:
            raise RuntimeError("Native V2 state still has an unconsumed dispatch")
        num_blocks = self.kv_cache_config.num_blocks
        candidate = self.request_state.prepare(
            scheduler_output, block_size=self._cache_spec.block_size, num_blocks=num_blocks
        )
        zero_ids = tuple(scheduler_output.new_block_ids_to_zero or ())
        copies = tuple(scheduler_output.kv_cache_block_copies or ())
        if any(type(i) is not int or not 0 <= i < num_blocks for i in zero_ids) or any(
            not 0 <= src < num_blocks or not 0 <= dst < num_blocks for src, dst in copies
        ):
            raise ValueError("Invalid scheduler KV zero/copy request")
        try:
            for cache in self.kv_caches:
                if zero_ids:
                    indices = torch.tensor(zero_ids, dtype=torch.long, device=self.device)
                    cache.index_fill_(0, indices, 0)
                if copies:
                    src, dst = zip(*copies, strict=True)
                    sources = torch.tensor(src, dtype=torch.long, device=self.device)
                    targets = torch.tensor(dst, dtype=torch.long, device=self.device)
                    cache.index_copy_(0, targets, cache.index_select(0, sources))
        except BaseException as error:
            self._failed = error
            raise
        self.request_state.publish(candidate)

    def _prepare_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_lengths: tuple[int, ...],
        sequence_lengths: tuple[int, ...],
        block_rows: tuple[tuple[int, ...], ...],
        slots: torch.Tensor,
        *,
        beam_info: dict[str, Any] | None = None,
        input_batch: InputBatch | None = None,
    ) -> PreparedForward:
        from vllm.v1.attention.backend import CommonAttentionMetadata

        rows, batch = input_ids.numel(), len(query_lengths)
        if (
            not 0 < batch <= self.max_num_reqs
            or not 0 < rows <= self.max_num_tokens
            or len(sequence_lengths) != batch
            or len(block_rows) != batch
            or sum(query_lengths) != rows
            or input_ids.ndim != 1
            or positions.shape != input_ids.shape
            or slots.shape != input_ids.shape
            or any(count <= 0 for count in query_lengths)
            or any(not 0 < length <= self.model_config.max_model_len for length in sequence_lengths)
        ):
            raise ValueError("Invalid owned NPU V2 forward geometry")
        starts = [0]
        for count in query_lengths:
            starts.append(starts[-1] + count)
        block_size = self._cache_spec.block_size
        # Include only blocks that the metadata can reference. All padding
        # block IDs are valid (reserved block zero); real lengths bound reads.
        max_blocks = max(len(blocks) for blocks in block_rows)
        if max_blocks > _ceildiv(self.model_config.max_model_len, block_size):
            raise ValueError("Forward block table exceeds native model capacity")
        if any(
            not blocks
            or any(
                type(block) is not int or not 0 <= block < self.kv_cache_config.num_blocks
                for block in blocks
            )
            for blocks in block_rows
        ):
            raise ValueError("Invalid owned NPU V2 block table")
        table = torch.tensor(
            [list(blocks) + [0] * (max_blocks - len(blocks)) for blocks in block_rows],
            dtype=torch.int32,
            device=self.device,
        )
        starts_cpu = torch.tensor(starts, dtype=torch.int32)
        lengths_cpu = torch.tensor(sequence_lengths, dtype=torch.int32)
        common = CommonAttentionMetadata(
            query_start_loc=starts_cpu.to(self.device),
            query_start_loc_cpu=starts_cpu,
            seq_lens=lengths_cpu.to(self.device),
            seq_lens_cpu_upper_bound=lengths_cpu,
            max_seq_len=max(sequence_lengths),
            num_reqs=batch,
            num_actual_tokens=rows,
            max_query_len=max(query_lengths),
            block_table_tensor=table,
            slot_mapping=slots,
            causal=True,
        )
        token = BEAM_DATA_VAR.set(beam_info)
        try:
            metadata = self._metadata_builder.build(0, common)
        finally:
            BEAM_DATA_VAR.reset(token)
        return PreparedForward(
            input_ids,
            positions,
            dict.fromkeys(self._layer_names, metadata),
            dict.fromkeys(self._layer_names, slots),
            beam_info,
            input_batch,
        )

    def prepare_inputs(self, scheduler_output: Any) -> PreparedForward:
        """Build actual Prefill inputs from logical scheduler rows."""
        req_ids = tuple(scheduler_output.num_scheduled_tokens)
        starts: dict[str, int] = {}
        counts = dict(scheduler_output.num_scheduled_tokens)
        tokens, positions, slots, logits = [], [], [], []
        tables = []
        lengths = []
        block_size = self._cache_spec.block_size
        for req_id in req_ids:
            state = self.request_state.requests[req_id]
            start = state.computed
            end = start + counts[req_id]
            if end != len(state.tokens):
                raise ValueError("Owned NPU V2 requires the complete remaining Prefill prompt")
            starts[req_id] = start
            tokens.extend(state.tokens[start:end])
            positions.extend(range(start, end))
            slots.extend(
                state.blocks[pos // block_size] * block_size + pos % block_size
                for pos in range(start, end)
            )
            logits.append(len(tokens) - 1)
            tables.append(state.blocks[: _ceildiv(end, block_size)])
            lengths.append(end)
        input_batch = InputBatch(
            req_ids, torch.tensor(logits, dtype=torch.long, device=self.device), starts, counts
        )
        return self._prepare_forward(
            torch.tensor(tokens, dtype=torch.int32, device=self.device),
            torch.tensor(positions, dtype=torch.long, device=self.device),
            tuple(counts.values()),
            tuple(lengths),
            tuple(tables),
            torch.tensor(slots, dtype=torch.int32, device=self.device),
            input_batch=input_batch,
        )

    def prepare_beam_decode(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        layout: BeamBatchExecutionView,
        kv_blocks: tuple[tuple[int, ...], ...],
        session: Any,
    ) -> PreparedForward:
        """Construct physical B*W rows without modifying native logical counters."""
        self._require_cache()
        step = layout.decode_step
        rows = layout.num_execution_rows
        if (
            layout.kind != "decode"
            or step is None
            or input_ids.ndim != 1
            or input_ids.numel() != rows
            or positions.shape != input_ids.shape
            or input_ids.device != self.device
            or positions.device != self.device
            or len(kv_blocks) != layout.batch_size
            or not session.is_active
        ):
            raise ValueError("Invalid owned NPU V2 Beam Decode binding")
        block_size = self._cache_spec.block_size
        tables = []
        lengths = []
        beam_data = {}
        for req_id, prompt_len, blocks in zip(
            layout.request_ids, layout.num_prompt_tokens_by_request, kv_blocks, strict=True
        ):
            state = self.request_state.requests[req_id]
            required = _ceildiv(prompt_len, block_size)
            if (
                state.computed != prompt_len + step
                or prompt_len != len(state.tokens)
                or len(blocks) < required
                or tuple(blocks[:required]) != state.blocks[:required]
            ):
                raise RuntimeError("Beam Decode prefix/step does not match native request state")
            tables.append(tuple(blocks[:required]))
            lengths.append(prompt_len + step + 1)
            beam_data[req_id] = {
                "prefix_len": prompt_len,
                "cache_len": prompt_len,
                "is_beam_decode": True,
                "beam_width": layout.input_width,
                "decode_step": step,
            }
        return self._prepare_forward(
            input_ids,
            positions,
            (layout.input_width,) * layout.batch_size,
            tuple(lengths),
            tuple(tables),
            torch.full((rows,), -1, dtype=torch.int32, device=self.device),
            beam_info={
                "beam_data": beam_data,
                "req_ids": list(layout.request_ids),
                "beam_session": session,
            },
        )

    @torch.inference_mode()
    def forward_prepared(self, prepared: PreparedForward, *, capture: bool = False) -> torch.Tensor:
        if capture:
            raise ValueError("Use the reserved startup capture API for owned NPU graphs")
        from vllm_gr.v1.worker.npu.owned_graph_runtime import forward_with_owned_graphs

        return forward_with_owned_graphs(self, prepared, self._forward_eager)

    @torch.inference_mode()
    def capture_prepared(self, prepared: PreparedForward, *, reserved: Any) -> bool:
        from vllm_gr.v1.worker.npu.owned_graph_runtime import capture_owned_prepared

        return capture_owned_prepared(self, prepared, self._forward_eager, reserved=reserved)

    @torch.inference_mode()
    def _forward_eager(self, prepared: PreparedForward) -> torch.Tensor:
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import BatchDescriptor, set_forward_context

        model = self.get_model()
        rows = prepared.input_ids.numel()
        token = BEAM_DATA_VAR.set(prepared.beam_info)
        try:
            with set_forward_context(
                prepared.attention_metadata,
                self.vllm_config,
                num_tokens=rows,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                batch_descriptor=BatchDescriptor(
                    num_tokens=rows, uniform=prepared.beam_info is not None
                ),
                slot_mapping=prepared.slot_mappings,
            ):
                result = model(input_ids=prepared.input_ids, positions=prepared.positions)
            if not isinstance(result, torch.Tensor) or result.shape[0] != rows:
                raise RuntimeError("Owned NPU V2 forward returned unexpected hidden states")
            return result
        finally:
            BEAM_DATA_VAR.reset(token)

    def forward_beam_decode(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        layout: BeamBatchExecutionView,
        kv_blocks: tuple[tuple[int, ...], ...],
        session: Any,
        *,
        capture: bool = False,
    ) -> torch.Tensor:
        prepared = self.prepare_beam_decode(input_ids, positions, layout, kv_blocks, session)
        return self.forward_prepared(prepared, capture=capture)

    def begin_decode(self, scheduler_output: Any) -> None:
        """Hold a logical Decode count receipt until GR finishes its stage."""
        counts = dict(scheduler_output.num_scheduled_tokens)
        if not counts or any(count != 1 for count in counts.values()):
            raise ValueError("Owned NPU V2 Decode schedules one logical token per request")
        self.consume_native_state(scheduler_output)
        self._decode_advance = (
            {req_id: self.request_state.requests[req_id].computed for req_id in counts},
            counts,
        )

    def complete_decode(self) -> None:
        if self._decode_advance is None:
            raise RuntimeError("No owned NPU V2 Decode dispatch is pending")
        self.request_state.advance(*self._decode_advance)
        self._decode_advance = None

    @torch.inference_mode()
    def execute_model(self, scheduler_output: Any) -> Any:
        from vllm.v1.outputs import ModelRunnerOutput

        self._require_live()
        stages = getattr(scheduler_output, GR_STAGE_OUTPUT_ATTR, None)
        if not stages and scheduler_output.total_num_scheduled_tokens == 0:
            self.consume_native_state(scheduler_output)
            return ModelRunnerOutput(req_ids=[], req_id_to_index={})
        if not stages or len(stages) != 1:
            raise ValueError("Owned NPU V2 only supports GR Beam sessions")
        stage = next(iter(stages.values()))
        if stage.kind is not GRStageKind.PREFILL:
            raise RuntimeError("GR must dispatch physical Decode through the NPU V2 adapter")
        try:
            self.consume_native_state(scheduler_output)
            prepared = self.prepare_inputs(scheduler_output)
            assert prepared.input_batch is not None
            self.execute_model_state = ExecuteModelState(
                self.forward_prepared(prepared), prepared.input_batch
            )
            return None
        except BaseException as error:
            self._failed = error
            self.execute_model_state = None
            raise

    @torch.inference_mode()
    def consume_prefill_logits(self, request_ids: tuple[str, ...]) -> torch.Tensor:
        """Consume pending V2 hidden state and commit one logical Prefill count."""
        state = self.execute_model_state
        if state is None:
            raise RuntimeError("No owned NPU V2 Prefill state is pending")
        batch = state.input_batch
        if len(request_ids) != len(set(request_ids)) or set(request_ids) != set(batch.req_ids):
            raise ValueError("GR Prefill request membership disagrees with native V2")
        try:
            indices = batch.logits_indices
            if request_ids != batch.req_ids:
                mapping = {req_id: index for index, req_id in enumerate(batch.req_ids)}
                order = torch.tensor(
                    [mapping[key] for key in request_ids], dtype=torch.long, device=self.device
                )
                indices = indices[order]
            logits = self.get_model().compute_logits(state.hidden_states[indices])
            if not isinstance(logits, torch.Tensor) or logits.shape[0] != len(request_ids):
                raise RuntimeError("Owned NPU V2 Prefill returned invalid logical logits rows")
            self.request_state.advance(batch.starts, batch.counts)
            return logits
        except BaseException as error:
            self._failed = error
            raise
        finally:
            self.execute_model_state = None

    def sample_tokens(self, grammar_output: Any) -> Any:
        del grammar_output
        raise RuntimeError("Owned NPU V2 sampling must be consumed by the GR Beam adapter")

    def abandon_pending_dispatch(self) -> None:
        """Called only when the external GR owner is drained and being retired."""
        self.execute_model_state = None
        self._decode_advance = None

    @torch.inference_mode()
    def compile_or_warm_up_model(self) -> Any:
        from vllm.v1.worker.worker_base import CompilationTimes

        self._require_cache()
        block_size = self._cache_spec.block_size
        # Exercise the real attention/KV path in an unleased block. No
        # scheduler request is installed; startup owns this temporary write.
        prepared = self._prepare_forward(
            torch.zeros(1, dtype=torch.int32, device=self.device),
            torch.zeros(1, dtype=torch.long, device=self.device),
            (1,),
            (1,),
            ((1,),),
            torch.tensor([block_size], dtype=torch.int32, device=self.device),
        )
        self.get_model().compute_logits(self.forward_prepared(prepared))
        torch.npu.synchronize(self.device)
        for cache in self.kv_caches:
            cache[1].zero_()
        torch.npu.synchronize(self.device)
        return CompilationTimes(language_model=0.0, encoder=0.0)

    def shutdown(self) -> None:
        from vllm.v1.worker.utils import clear_layer_kv_caches
        from vllm_gr.v1.worker.npu.v2.lifecycle import invalidate_owned_graphs

        if self._closed:
            return
        if not self._weight_update_lock.acquire(blocking=False):
            raise RuntimeError("Cannot close owned NPU V2 during a checkpoint update")
        try:
            torch.npu.synchronize(self.device)
            invalidate_owned_graphs(self)
            owner = getattr(self, "_gr_npu_stage_runner", None)
            if owner is not None and not owner.closed:
                owner._release()
            from vllm_gr.v1.worker.model_runner_patch import close_constraint_backend

            close_constraint_backend(self)
            self._gr_npu_stage_runner = None
            self._gr_npu_eager_context = None
            self._gr_npu_graph_context = None
            self._beam_context = None
            self._gr_npu_sid_stage_vocab = None
            self.abandon_pending_dispatch()
            self.request_state.requests.clear()
            if self.model is not None:
                clear_layer_kv_caches(self.model.modules())
            forward_context = self.compilation_config.static_forward_context
            for name, layer in self._owned_context.items():
                if forward_context.get(name) is layer:
                    del forward_context[name]
            self._owned_context.clear()
            self.kv_caches.clear()
            self.attn_groups.clear()
            self._metadata_builder = None
            self.kv_cache_config = None
            self.model = None
            torch.npu.empty_cache()
            self._closed = True
        except BaseException as error:
            self._failed = error
            raise
        finally:
            self._weight_update_lock.release()

    def update_config(self, overrides: dict[str, Any]) -> None:
        if overrides:
            raise ValueError("Owned NPU V2 runtime configuration mutation is unsupported")

    def reload_weights(self, *args: Any, **kwargs: Any) -> None:
        from vllm_gr.v1.worker.npu.v2.lifecycle import reload_owned_checkpoint

        reload_owned_checkpoint(self, *args, **kwargs)
