# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Eager V2 request state: logical scheduler rows, independent of Beam rows."""

import operator
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class RequestState:
    tokens: tuple[int, ...]
    blocks: tuple[int, ...]
    computed: int


class NativeRequestState:
    """Consume the native V2 scheduler protocol without CUDA request kernels.

    Scheduler block IDs and logical token counts remain authoritative. GR owns
    generated tokens independently; this state never stores physical Beam rows.
    """

    def __init__(self, max_reqs: int, max_model_len: int) -> None:
        self.max_reqs = max_reqs
        self.max_model_len = max_model_len
        self.requests: dict[str, RequestState] = {}

    @staticmethod
    def _blocks(groups: Any, num_blocks: int) -> tuple[int, ...]:
        if len(groups) != 1:
            raise ValueError("Owned NPU V2 requires one uniform KV group")
        blocks = tuple(groups[0])
        if any(type(block) is not int or not 0 <= block < num_blocks for block in blocks):
            raise ValueError("Scheduler block ID is outside allocated KV capacity")
        return blocks

    def prepare(self, output: Any, *, block_size: int, num_blocks: int) -> dict[str, RequestState]:
        """Validate a complete update before publishing any new state."""
        if output.preempted_req_ids or output.scheduled_cached_reqs.resumed_req_ids:
            raise ValueError("Owned NPU V2 Beam does not support request preemption or recompute")
        if output.scheduled_spec_decode_tokens or output.scheduled_encoder_inputs:
            raise ValueError("Owned NPU V2 supports text Beam without speculative decode")
        if (
            output.kv_connector_metadata is not None
            or output.ec_connector_metadata is not None
            or output.has_structured_output_requests
        ):
            raise ValueError("Owned NPU V2 does not support connectors or structured sampling")

        # Build a detached transaction; malformed metadata must not alter a
        # previous live owner's state.
        requests = {
            key: req for key, req in self.requests.items() if key not in output.finished_req_ids
        }
        for new in output.scheduled_new_reqs:
            if new.req_id in requests:
                raise ValueError("Duplicate native request registration")
            # GR's native token buffers may expose NumPy views. Do not test
            # an array's truth value or retain a mutable scheduler buffer.
            tokens = (
                tuple(map(operator.index, new.prompt_token_ids))
                if new.prompt_token_ids is not None
                else ()
            )
            prefill = (
                tuple(map(operator.index, new.prefill_token_ids))
                if new.prefill_token_ids is not None
                else ()
            )
            if (
                not tokens
                or prefill != tokens
                or new.mm_features
                or new.prompt_embeds is not None
                or new.lora_request is not None
                or new.pooling_params is not None
                or new.sampling_params is None
                or not 0 <= new.num_computed_tokens < len(tokens) <= self.max_model_len
                or any(token < 0 for token in tokens)
            ):
                raise ValueError("Unsupported owned NPU V2 Prefill request")
            blocks = self._blocks(new.block_ids, num_blocks)
            requests[new.req_id] = RequestState(tokens, blocks, new.num_computed_tokens)

        table_updates = output.block_table_updates or {}
        for req_id, groups in table_updates.items():
            if req_id in requests:
                requests[req_id] = replace(
                    requests[req_id], blocks=self._blocks(groups, num_blocks)
                )

        cached = output.scheduled_cached_reqs
        if len(set(cached.req_ids)) != len(cached.req_ids):
            raise ValueError("Duplicate cached request registration")
        for req_id, computed, new_blocks in zip(
            cached.req_ids, cached.num_computed_tokens, cached.new_block_ids, strict=True
        ):
            if req_id not in requests:
                raise RuntimeError("Cached request has no native V2 state")
            state = requests[req_id]
            if computed != state.computed:
                raise RuntimeError("Native V2 computed-token state disagrees with scheduler")
            blocks = state.blocks
            if new_blocks is not None and req_id not in table_updates:
                blocks += self._blocks(new_blocks, num_blocks)
            requests[req_id] = replace(state, blocks=blocks)

        if len(requests) > self.max_reqs:
            raise ValueError("Native V2 request capacity exceeded")
        if sum(output.num_scheduled_tokens.values()) != output.total_num_scheduled_tokens:
            raise ValueError("Native scheduler token count does not match total")
        for req_id, count in output.num_scheduled_tokens.items():
            if req_id not in requests or type(count) is not int or count < 1:
                raise ValueError("Invalid native request or scheduling count")
            req = requests[req_id]
            end = req.computed + count
            if end > self.max_model_len or len(req.blocks) * block_size < end:
                raise ValueError("Native request exceeds its model length or KV lease")
        return requests

    def publish(self, requests: dict[str, RequestState]) -> None:
        self.requests = requests

    def advance(self, starts: dict[str, int], counts: dict[str, int]) -> None:
        """Commit exactly one logical token-count update after model completion."""
        if starts.keys() != counts.keys():
            raise RuntimeError("Native V2 completion identity mismatch")
        updates = self.requests.copy()
        for key, count in counts.items():
            state = updates[key]
            if state.computed != starts[key] or count <= 0:
                raise RuntimeError("Stale or duplicate native V2 completion")
            updates[key] = replace(state, computed=state.computed + count)
        self.requests = updates
