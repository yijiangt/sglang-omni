# SPDX-License-Identifier: Apache-2.0
"""Per-request state threaded through ZONOS2 TTS pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Zonos2TtsState:
    """Per-request state: preprocessing → speaker_encoder → tts_engine → vocoder.

    Fields populate lazily so a deserialised state is valid at any stage boundary.
    """

    # ── input ──
    text: str = ""

    # ── after preprocessing ──
    # 2-D prompt: list of rows, each row is [cb0..cb8, text_tok]
    prompt_tokens: list[list[int]] = field(default_factory=list)
    # Position in prompt where speaker embedding is injected (-1 = no speaker)
    speaker_token_position: int = -1

    # ── after speaker_encoder ──
    speaker_audio_path: str | None = None
    speaker_embedding: list[float] | None = None  # raw embedding vector

    # ── conditioning ──
    speaking_rate_bucket: int | None = None
    quality_buckets: list[int | None] | None = None
    accurate_mode: bool = True
    clean_speaker_background: bool = False

    # ── sampling params ──
    temperature: float = 1.15
    top_k: int = 106
    top_p: float = 0.0
    min_p: float = 0.18
    max_tokens: int = 2048
    repetition_penalty: float = 1.2
    repetition_window: int = 50
    seed: int | None = None

    # ── after tts_engine ──
    output_codes: list[list[int]] | None = None  # (T_out, 9)

    # ── after vocoder ──
    audio_data: list[float] | None = None
    sample_rate: int = 44100

    # ── usage ──
    prompt_tokens_count: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "text": self.text,
            "prompt_tokens": self.prompt_tokens,
            "speaker_token_position": self.speaker_token_position,
            "accurate_mode": self.accurate_mode,
            "clean_speaker_background": self.clean_speaker_background,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "min_p": self.min_p,
            "max_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
            "repetition_window": self.repetition_window,
            "sample_rate": self.sample_rate,
            "prompt_tokens_count": self.prompt_tokens_count,
            "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason,
        }
        if self.speaker_audio_path is not None:
            data["speaker_audio_path"] = self.speaker_audio_path
        if self.speaker_embedding is not None:
            data["speaker_embedding"] = self.speaker_embedding
        if self.speaking_rate_bucket is not None:
            data["speaking_rate_bucket"] = self.speaking_rate_bucket
        if self.quality_buckets is not None:
            data["quality_buckets"] = self.quality_buckets
        if self.seed is not None:
            data["seed"] = self.seed
        if self.output_codes is not None:
            data["output_codes"] = self.output_codes
        if self.audio_data is not None:
            data["audio_data"] = self.audio_data
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Zonos2TtsState":
        return cls(
            text=data.get("text", ""),
            prompt_tokens=data.get("prompt_tokens", []),
            speaker_token_position=int(data.get("speaker_token_position", -1)),
            speaker_audio_path=data.get("speaker_audio_path"),
            speaker_embedding=data.get("speaker_embedding"),
            speaking_rate_bucket=data.get("speaking_rate_bucket"),
            quality_buckets=data.get("quality_buckets"),
            accurate_mode=bool(data.get("accurate_mode", True)),
            clean_speaker_background=bool(data.get("clean_speaker_background", False)),
            temperature=float(data.get("temperature", 1.15)),
            top_k=int(data.get("top_k", 106)),
            top_p=float(data.get("top_p", 0.0)),
            min_p=float(data.get("min_p", 0.18)),
            max_tokens=int(data.get("max_tokens", 2048)),
            repetition_penalty=float(data.get("repetition_penalty", 1.2)),
            repetition_window=int(data.get("repetition_window", 50)),
            seed=data.get("seed"),
            output_codes=data.get("output_codes"),
            audio_data=data.get("audio_data"),
            sample_rate=int(data.get("sample_rate", 44100)),
            prompt_tokens_count=int(data.get("prompt_tokens_count", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            finish_reason=str(data.get("finish_reason", "stop")),
        )


__all__ = ["Zonos2TtsState"]
