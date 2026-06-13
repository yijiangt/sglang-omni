# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the ZONOS2 TTS pipeline.

Pipeline shape:
    preprocessing → speaker_encoder → tts_engine → vocoder

- ``create_preprocessing_executor``: text normalization + byte tokenization +
  2-D prompt frame assembly (silence prefix, conditioning tokens, speaker slot).
  Returns a :class:`ThreadedSimpleScheduler`.
- ``create_speaker_encoder_executor``: encodes reference audio → speaker
  embedding vector. Cached per audio path. Returns a :class:`SimpleScheduler`.
- ``create_tts_engine_executor``: AR generation via ZONOS2 sparse MoE under
  sglang.  Creates a staging directory with config.json + dummy safetensors so
  SGLang can initialise ModelConfig; actual weights are loaded from the .pth
  file. Returns an :class:`OmniScheduler`.
- ``create_vocoder_executor``: DAC 44kHz decode + shear_up.  Returns a
  :class:`SimpleScheduler`.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any

from sglang_omni.models.zonos2_tts.model_runner import Zonos2ModelRunner
from sglang_omni.models.zonos2_tts.payload_types import Zonos2TtsState
from sglang_omni.models.zonos2_tts.request_builders import (
    make_zonos2_scheduler_adapters,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend import (
    SGLangOutputProcessor,
    build_sglang_server_args,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONCURRENCY = 16

_MAX_REF_AUDIO_SEC = 30

# Module-level list keeps staging directory paths alive for the process lifetime.
# mkdtemp does not register a finalizer, so this prevents accidental cleanup if
# the caller ever switches to TemporaryDirectory.
_zonos2_staging_dirs: list[str] = []


# ─────────────────────────────────────────────────────────────────────────────
# 1. Preprocessing
# ─────────────────────────────────────────────────────────────────────────────


def create_preprocessing_executor(
    model_path: str,
    *,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
):
    """CPU stage: text normalisation + byte tokenisation + 2-D prompt assembly."""
    from zonos2.tts.prompt import TTSPromptBuilder, TTSPromptConfig
    from zonos2.utils import cached_load_checkpoint_config, resolve_model_path

    resolved = resolve_model_path(model_path)
    zonos_cfg = cached_load_checkpoint_config(resolved)

    quality_bucket_counts: tuple[int, ...] = ()
    if getattr(zonos_cfg, "quality_buckets", None):
        quality_bucket_counts = tuple(
            len(vs) for vs in zonos_cfg.quality_buckets.values()
        )

    prompt_config = TTSPromptConfig(
        n_codebooks=int(getattr(zonos_cfg, "n_codebooks", 9)),
        audio_pad_id=int(getattr(zonos_cfg, "audio_pad_id", 1025)),
        text_vocab=int(getattr(zonos_cfg, "text_vocab") or 469),
        speaking_rate_num_buckets=int(
            getattr(zonos_cfg, "speaking_rate_num_buckets", 0) or 0
        ),
        quality_bucket_counts=quality_bucket_counts,
        speaker_background_num_buckets=(
            2 if getattr(zonos_cfg, "speaker_background_token_enabled", False) else 0
        ),
        accurate_mode_num_buckets=(
            1 if getattr(zonos_cfg, "accurate_mode_token_enabled", False) else 0
        ),
        prepend_silence=True,
    )
    builder = TTSPromptBuilder(prompt_config)
    _speaker_enabled = bool(getattr(zonos_cfg, "speaker_enabled", False))

    def _preprocess(payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs or {}
        params = payload.request.params or {}
        if isinstance(inputs, str):
            inputs = {"text": inputs}

        text = inputs.get("input") or inputs.get("text") or ""
        references = inputs.get("references") or []
        speaker_audio_path: str | None = None
        if references and isinstance(references, list):
            first = references[0]
            if isinstance(first, dict):
                speaker_audio_path = first.get("audio_path") or first.get("path")
            elif isinstance(first, str):
                speaker_audio_path = first

        speaking_rate_bucket = params.get("speaking_rate_bucket")
        quality_buckets = params.get("quality_buckets")
        accurate_mode = bool(params.get("accurate_mode", True))
        clean_speaker_background = bool(params.get("clean_speaker_background", False))

        # Try text normalisation (optional; falls back gracefully)
        try:
            from zonos2.tokenizer.textnorm import normalize_text

            text = normalize_text(text)
        except Exception:
            pass

        # Build conditioning-annotated text prompt
        prompt_tensor = builder.build(
            text,
            speaking_rate_bucket=speaking_rate_bucket,
            quality_buckets=quality_buckets,
        )
        prompt_rows: list[list[int]] = prompt_tensor.tolist()

        speaker_token_position = -1
        if _speaker_enabled and speaker_audio_path:
            # Prepend speaker slot row; speaker embedding injected later
            slot = builder.speaker_slot()  # (1, n_cols)
            slot_row = slot[0].tolist()
            prompt_rows = [slot_row] + prompt_rows
            speaker_token_position = 0

        state = Zonos2TtsState(
            text=text,
            prompt_tokens=prompt_rows,
            speaker_token_position=speaker_token_position,
            speaker_audio_path=speaker_audio_path,
            speaking_rate_bucket=speaking_rate_bucket,
            quality_buckets=quality_buckets,
            accurate_mode=accurate_mode,
            clean_speaker_background=clean_speaker_background,
            temperature=float(params.get("temperature", 1.15)),
            top_k=int(params.get("top_k", 106)),
            top_p=float(params.get("top_p", 0.0)),
            min_p=float(params.get("min_p", 0.18)),
            max_tokens=int(
                params.get("max_tokens", params.get("max_new_tokens", 2048))
            ),
            repetition_penalty=float(params.get("repetition_penalty", 1.2)),
            repetition_window=int(params.get("repetition_window", 50)),
            seed=params.get("seed"),
        )
        payload.data = state.to_dict()
        return payload

    return ThreadedSimpleScheduler(_preprocess, max_concurrency=max_concurrency)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Speaker encoder
# ─────────────────────────────────────────────────────────────────────────────


def create_speaker_encoder_executor(
    model_path: str,
    *,
    device: str = "cuda",
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
):
    """GPU stage: extract speaker embedding from reference audio.

    Uses Qwen3-Voice-Embedding (2048-D) or ECAPA-TDNN (128-D) depending on
    the checkpoint's ``speaker_embedding_dim``.  Results are cached by file path.
    """
    import threading

    from zonos2.utils import cached_load_checkpoint_config, resolve_model_path

    resolved = resolve_model_path(model_path)
    zonos_cfg = cached_load_checkpoint_config(resolved)
    spk_enabled = bool(getattr(zonos_cfg, "speaker_enabled", False))
    spk_dim = int(getattr(zonos_cfg, "speaker_embedding_dim", 128))

    if not spk_enabled:
        # No speaker conditioning — just pass through
        def _passthrough(payload: StagePayload) -> StagePayload:
            return payload

        return ThreadedSimpleScheduler(_passthrough, max_concurrency=max_concurrency)

    _speaker_model = None
    _lock = threading.Lock()
    _cache: dict[str, list[float]] = {}

    def _get_speaker_model():
        nonlocal _speaker_model
        if _speaker_model is not None:
            return _speaker_model
        with _lock:
            if _speaker_model is not None:
                return _speaker_model
            if spk_dim >= 512:
                from zonos2.models.speaker_cloning import Qwen3SpeakerEmbedding

                _speaker_model = Qwen3SpeakerEmbedding(device=device)
            else:
                # ECAPA-TDNN path — use speechbrain if available
                try:
                    from speechbrain.pretrained import EncoderClassifier

                    _speaker_model = EncoderClassifier.from_hparams(
                        source="speechbrain/spkrec-ecapa-voxceleb",
                        run_opts={"device": device},
                    )
                except ImportError:
                    logger.warning(
                        "speechbrain not available; using Qwen3SpeakerEmbedding fallback"
                    )
                    from zonos2.models.speaker_cloning import Qwen3SpeakerEmbedding

                    _speaker_model = Qwen3SpeakerEmbedding(device=device)
        return _speaker_model

    def _encode_speaker(payload: StagePayload) -> StagePayload:
        state = Zonos2TtsState.from_dict(payload.data)
        audio_path = state.speaker_audio_path
        if not audio_path or state.speaker_token_position < 0:
            payload.data = state.to_dict()
            return payload

        if audio_path in _cache:
            state.speaker_embedding = _cache[audio_path]
            payload.data = state.to_dict()
            return payload

        import torch
        import torchaudio

        waveform, sr = torchaudio.load(audio_path)
        max_samples = int(_MAX_REF_AUDIO_SEC * sr)
        if waveform.shape[-1] > max_samples:
            waveform = waveform[..., :max_samples]
        spk_model = _get_speaker_model()

        with torch.no_grad():
            from zonos2.models.speaker_cloning import Qwen3SpeakerEmbedding

            if isinstance(spk_model, Qwen3SpeakerEmbedding):
                emb = spk_model(waveform, sr)
                # Pool along time dimension
                emb = emb.mean(dim=1).squeeze(0)  # (dim,)
            else:
                # SpeechBrain ECAPA-TDNN
                emb = spk_model.encode_batch(waveform.to(device), normalize=False)
                emb = emb.squeeze()  # (dim,)

        emb_list = emb.float().cpu().tolist()
        _cache[audio_path] = emb_list
        state.speaker_embedding = emb_list
        payload.data = state.to_dict()
        return payload

    return SimpleScheduler(
        _encode_speaker,
        max_batch_size=max_concurrency,
        max_batch_wait_ms=0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. TTS engine
# ─────────────────────────────────────────────────────────────────────────────


def _create_zonos2_staging_dir(model_path: str) -> str:
    """Create a temp directory with config.json + dummy safetensors.

    SGLang's ModelConfig.from_pretrained() requires a config.json; ZONOS2
    checkpoints only ship params.json. We write the HF-format config to a
    temp dir and pass that as the server_args model path. Actual weights are
    loaded by Zonos2SGLangModel.load_weights from the real checkpoint path
    stored in config._zonos2_model_path.
    """
    from zonos2.utils import cached_load_checkpoint_config, resolve_model_path

    from sglang_omni.models.zonos2_tts.hf_config import Zonos2HFConfig

    resolved = resolve_model_path(model_path)
    zonos_cfg = cached_load_checkpoint_config(resolved)
    hf_cfg = Zonos2HFConfig.from_zonos2_config(zonos_cfg, model_path=str(resolved))

    staging_dir = tempfile.mkdtemp(prefix="zonos2_staging_")

    # Write config.json
    cfg_dict = hf_cfg.to_dict()
    cfg_dict["architectures"] = ["Zonos2SGLangModel"]
    cfg_dict["model_type"] = "zonos2"
    # Persist the real checkpoint path in the HF config
    cfg_dict["_zonos2_model_path"] = str(resolved)
    with open(os.path.join(staging_dir, "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=2)

    # Write an empty safetensors file so SGLang's weight iterator returns nothing
    try:
        from safetensors.torch import save_file

        save_file({}, os.path.join(staging_dir, "model.safetensors"))
    except ImportError:
        # Create a minimal valid safetensors file manually (8-byte header)
        with open(os.path.join(staging_dir, "model.safetensors"), "wb") as f:
            # safetensors magic: 8 bytes for header length (little endian) + JSON header
            header = b"{}"
            import struct

            f.write(struct.pack("<Q", len(header)))
            f.write(header)

    logger.info("ZONOS2 staging dir: %s (real checkpoint: %s)", staging_dir, resolved)
    return staging_dir


def create_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda",
    max_new_tokens: int | None = 2048,
    server_args_overrides: dict[str, Any] | None = None,
    enable_async_decode: bool = False,
):
    """sglang-backed AR engine for ZONOS2 TTS."""
    from sglang_omni.models.zonos2_tts import (
        Zonos2HFConfig,  # triggers AutoConfig.register
    )

    staging_dir = _create_zonos2_staging_dir(model_path)
    _zonos2_staging_dirs.append(staging_dir)
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0

    overrides: dict[str, Any] = {
        "disable_cuda_graph": False,
        "cuda_graph_max_bs": DEFAULT_MAX_CONCURRENCY,
        "mem_fraction_static": 0.80,
        "max_running_requests": DEFAULT_MAX_CONCURRENCY,
        "chunked_prefill_size": 8192,
        "dtype": "bfloat16",
        "trust_remote_code": True,
    }
    if server_args_overrides:
        overrides.update(server_args_overrides)

    server_args = build_sglang_server_args(
        staging_dir,
        context_length=4096,
        **overrides,
    )
    server_args.disable_overlap_schedule = True

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = create_sglang_infrastructure(
        server_args,
        gpu_id,
        model_arch_override="Zonos2SGLangModel",
    )

    model = model_worker.model_runner.model

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model,
    )
    model_runner = Zonos2ModelRunner(model_worker, output_proc)
    request_builder, result_adapter = make_zonos2_scheduler_adapters(
        model,
        max_new_tokens_cap=max_new_tokens,
    )

    scheduler = OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        model_runner=model_runner,
        request_builder=request_builder,
        result_adapter=result_adapter,
        abort_callback=model.reset_request,
        enable_async_decode=enable_async_decode,
    )
    return scheduler


# ─────────────────────────────────────────────────────────────────────────────
# 4. Vocoder
# ─────────────────────────────────────────────────────────────────────────────


def create_vocoder_executor(
    model_path: str,
    *,
    device: str = "cuda",
    max_batch_size: int = DEFAULT_MAX_CONCURRENCY,
    max_batch_wait_ms: int = 10,
):
    """DAC 44kHz decode stage: output_codes → audio waveform.

    Applies shear_up to reverse the diagonal delay pattern before decoding.
    """
    import torch
    from zonos2.utils import cached_load_checkpoint_config, resolve_model_path

    resolved = resolve_model_path(model_path)
    zonos_cfg = cached_load_checkpoint_config(resolved)
    audio_pad_id = int(getattr(zonos_cfg, "audio_pad_id", 1025))
    codebook_size = int(getattr(zonos_cfg, "codebook_size", 1024))

    _dac_model: Any = None

    def _get_dac():
        nonlocal _dac_model
        if _dac_model is None:
            import dac as _dac_module

            _dac_model = (
                _dac_module.DAC.load(_dac_module.utils.download(model_type="44khz"))
                .eval()
                .to(device)
            )
        return _dac_model

    def _vocoder(payload: StagePayload) -> StagePayload:
        from zonos2.tokenizer.vocoder import shear_up

        dac_model = _get_dac()
        state = Zonos2TtsState.from_dict(payload.data)
        if not state.output_codes:
            state.audio_data = []
            state.sample_rate = 44100
            payload.data = state.to_dict()
            return payload

        try:
            codes = torch.tensor(
                state.output_codes, dtype=torch.int64, device=device
            )  # (T, n_codebooks)

            # Reverse diagonal delay pattern
            codes_sheared = shear_up(codes, pad_id=audio_pad_id)
            codes_sheared = codes_sheared.clamp(0, codebook_size - 1)

            # DAC expects (batch, codebooks, seq)
            codes_dac = codes_sheared.unsqueeze(0).permute(0, 2, 1)

            with torch.no_grad(), torch.inference_mode():
                z = dac_model.quantizer.from_codes(codes_dac)[0]
                audio = dac_model.decode(z).float().squeeze().cpu()

            state.audio_data = audio.numpy().tolist()
            state.sample_rate = 44100

        except Exception as exc:
            logger.error("ZONOS2 vocoder error: %s", exc, exc_info=True)
            state.audio_data = []
            state.sample_rate = 44100

        payload.data = state.to_dict()
        return payload

    return SimpleScheduler(
        _vocoder,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )


__all__ = [
    "DEFAULT_MAX_CONCURRENCY",
    "create_preprocessing_executor",
    "create_speaker_encoder_executor",
    "create_tts_engine_executor",
    "create_vocoder_executor",
]
