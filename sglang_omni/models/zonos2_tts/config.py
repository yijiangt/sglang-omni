# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for ZONOS2 TTS."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.zonos2_tts"

DEFAULT_MAX_CONCURRENCY = 16


class Zonos2TtsPipelineConfig(PipelineConfig):
    """4-stage TTS pipeline: preprocessing → speaker_encoder → tts_engine → vocoder.

    - preprocessing: text normalization, byte tokenization, prompt frame assembly
    - speaker_encoder: extract speaker embedding from reference audio (voice cloning)
    - tts_engine: AR decoding with ZONOS2 sparse MoE model via sglang
    - vocoder: DAC 44kHz decode to waveform
    """

    architecture: ClassVar[str] = "zonos2"

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            next="speaker_encoder",
        ),
        StageConfig(
            name="speaker_encoder",
            process="pipeline",
            factory=f"{_PKG}.stages.create_speaker_encoder_executor",
            factory_args={"device": "cuda", "max_concurrency": DEFAULT_MAX_CONCURRENCY},
            gpu=0,
            next="tts_engine",
        ),
        StageConfig(
            name="tts_engine",
            process="pipeline",
            factory=f"{_PKG}.stages.create_tts_engine_executor",
            factory_args={"device": "cuda"},
            gpu=0,
            next="vocoder",
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory=f"{_PKG}.stages.create_vocoder_executor",
            factory_args={"device": "cuda"},
            gpu=0,
            terminal=True,
        ),
    ]


EntryClass = Zonos2TtsPipelineConfig
