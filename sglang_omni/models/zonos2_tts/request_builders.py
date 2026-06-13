# SPDX-License-Identifier: Apache-2.0
"""StagePayload ↔ SGLang Req adapters for ZONOS2 TTS."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.zonos2_tts.payload_types import Zonos2TtsState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData


@dataclass
class Zonos2SGLangRequestData(SGLangARRequestData):
    """Per-request state kept by the ZONOS2 tts_engine scheduler."""

    # 2-D prompt rows: (T_prompt, n_codebooks+1) as a list of rows
    prompt_rows: list[list[int]] = field(default_factory=list)
    # Speaker embedding vector (None = no voice cloning)
    speaker_embedding: list[float] | None = None
    # Token position where speaker embedding is injected
    speaker_token_position: int = -1

    # Sampling state accumulated across decode steps
    output_codes: list[list[int]] = field(default_factory=list)
    generation_done: bool = False

    # Repetition window: recent codes for penalty (n_codebooks wide)
    rep_window: list[list[int]] = field(default_factory=list)
    repetition_window: int = 50
    repetition_penalty: float = 1.2
    min_p: float = 0.18

    engine_start_s: float = 0.0
    stream_metadata: dict[str, Any] | None = None


def build_zonos2_request(
    state: Zonos2TtsState, *, request_id: str = "", vocab_size: int = 470
) -> Zonos2SGLangRequestData:
    """Convert Zonos2TtsState into a Zonos2SGLangRequestData."""
    # SGLang tracks token count via prompt length; use number of prompt rows
    prompt_rows = state.prompt_tokens  # list[list[int]], each row = n_cols ints
    n_tokens = len(prompt_rows)

    # Flatten to 1-D for SGLang's origin_input_ids (cb0 values for KV key space)
    flat_ids = [row[0] for row in prompt_rows]  # use cb0 column

    sp_kwargs: dict[str, Any] = {
        "max_new_tokens": int(state.max_tokens),
        "temperature": float(state.temperature),
        "top_p": max(float(state.top_p), 0.0),
    }
    if state.top_k > 0:
        sp_kwargs["top_k"] = int(state.top_k)
    if state.seed is not None:
        sp_kwargs["sampling_seed"] = int(state.seed)
    sampling_params = SamplingParams(**sp_kwargs)
    sampling_params.normalize(tokenizer=None)

    req = Req(
        rid=request_id,
        origin_input_text="",
        origin_input_ids=flat_ids,
        sampling_params=sampling_params,
        vocab_size=vocab_size,
        extra_key=None,
    )
    req._codec_suppress_tokens = None
    req._input_embeds_are_projected = False

    return Zonos2SGLangRequestData(
        input_ids=torch.tensor(flat_ids, dtype=torch.long),
        req=req,
        prompt_rows=prompt_rows,
        speaker_embedding=state.speaker_embedding,
        speaker_token_position=int(state.speaker_token_position),
        max_new_tokens=int(state.max_tokens),
        temperature=float(state.temperature),
        top_p=float(state.top_p),
        top_k=int(state.top_k) if state.top_k > 0 else -1,
        repetition_window=int(state.repetition_window),
        repetition_penalty=float(state.repetition_penalty),
        min_p=float(state.min_p),
    )


def make_zonos2_scheduler_adapters(
    model: Any,
    *,
    max_new_tokens_cap: int | None = None,
) -> tuple[Callable, Callable]:
    """Build (request_builder, result_adapter) closures for the ZONOS2 OmniScheduler."""

    def request_builder(payload: StagePayload) -> Zonos2SGLangRequestData:
        state = Zonos2TtsState.from_dict(payload.data)
        if max_new_tokens_cap is not None:
            state.max_tokens = min(int(state.max_tokens), int(max_new_tokens_cap))
        # vocab_size: text_vocab+1 so SGLang's sampler range is valid
        vocab_size = getattr(model.config, "text_vocab", 469) + 1
        data = build_zonos2_request(
            state, request_id=payload.request_id, vocab_size=vocab_size
        )
        data.engine_start_s = time.perf_counter()
        data.stage_payload = payload
        return data

    def result_adapter(data: Zonos2SGLangRequestData) -> StagePayload:
        payload = data.stage_payload
        state = Zonos2TtsState.from_dict(payload.data)
        if data.output_codes:
            state.output_codes = [list(row) for row in data.output_codes]
            state.completion_tokens = len(data.output_codes)
        state.prompt_tokens_count = len(data.prompt_rows)
        state.finish_reason = "stop"
        model.reset_request(payload.request_id)
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=state.to_dict(),
        )

    return request_builder, result_adapter


__all__ = [
    "Zonos2SGLangRequestData",
    "build_zonos2_request",
    "make_zonos2_scheduler_adapters",
]
