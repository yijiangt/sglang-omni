# SPDX-License-Identifier: Apache-2.0
"""HuggingFace-compatible config wrapper for ZONOS2 TTS.

ZONOS2 checkpoints use params.json (not config.json). This PretrainedConfig
subclass provides the HF interface that SGLang's ModelConfig expects, and is
populated from the ZONOS2-native params loaded by
``zonos2.utils.cached_load_checkpoint_config``.
"""

from __future__ import annotations

from typing import Any

import transformers


class Zonos2HFConfig(transformers.PretrainedConfig):
    """ZONOS2 TTS model config in HuggingFace format.

    All architecture parameters are stored as flat attributes so SGLang can
    inspect ``config.hidden_size``, ``config.num_attention_heads``, etc. without
    having to understand the ZONOS2-native ModelConfig dataclass.

    The extra attribute ``_zonos2_model_path`` carries the resolved checkpoint
    path so ``Zonos2SGLangModel.load_weights`` can open the .pth file directly
    (SGLang feeds a safetensors iterator that we ignore for custom checkpoints).
    """

    model_type = "zonos2"

    def __init__(
        self,
        # Standard transformer dims
        hidden_size: int = 4096,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        num_hidden_layers: int = 32,
        intermediate_size: int = 14336,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 500000.0,
        max_position_embeddings: int = 4096,
        # ZONOS2 audio specifics
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        text_vocab: int = 469,
        eoa_id: int = 1024,
        audio_pad_id: int = 1025,
        loss_softcap: float = 15.0,
        # MoE
        moe_n_experts: int = 1,
        moe_router_dim: int = 256,
        moe_start_from_layer: int = 0,
        moe_end_from_layer: int = 0,
        num_experts_per_tok: int = 1,
        special_topk_layers: dict[str, Any] | None = None,
        norm_topk_prob: bool = False,
        moe_balancing_strategy: str = "legacy",
        moe_intermediate_size: int = 0,
        # Speaker conditioning
        speaker_enabled: bool = False,
        speaker_embedding_dim: int = 128,
        speaker_lda_dim: int | None = None,
        speaker_background_token_enabled: bool = False,
        accurate_mode_token_enabled: bool = False,
        # Conditioning
        speaking_rate_num_buckets: int = 0,
        quality_num_buckets: int = 0,
        quality_features: list[str] | None = None,
        quality_buckets: dict[str, list[str]] | None = None,
        # Internal: resolved checkpoint path for custom weight loading
        _zonos2_model_path: str = "",
        **kwargs: Any,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.text_vocab = text_vocab
        self.eoa_id = eoa_id
        self.audio_pad_id = audio_pad_id
        self.loss_softcap = loss_softcap

        self.moe_n_experts = moe_n_experts
        self.moe_router_dim = moe_router_dim
        self.moe_start_from_layer = moe_start_from_layer
        self.moe_end_from_layer = moe_end_from_layer
        self.num_experts_per_tok = num_experts_per_tok
        self.special_topk_layers = special_topk_layers or {}
        self.norm_topk_prob = norm_topk_prob
        self.moe_balancing_strategy = moe_balancing_strategy
        self.moe_intermediate_size = (
            moe_intermediate_size if moe_intermediate_size > 0 else intermediate_size
        )

        self.speaker_enabled = speaker_enabled
        self.speaker_embedding_dim = speaker_embedding_dim
        self.speaker_lda_dim = speaker_lda_dim
        self.speaker_background_token_enabled = speaker_background_token_enabled
        self.accurate_mode_token_enabled = accurate_mode_token_enabled

        self.speaking_rate_num_buckets = speaking_rate_num_buckets
        self.quality_num_buckets = quality_num_buckets
        self.quality_features = quality_features or []
        self.quality_buckets = quality_buckets or {}

        self._zonos2_model_path = _zonos2_model_path

        # vocab_size for SGLang: needs to be at least text_vocab+1 so the
        # scheduler can allocate token-id space. We set it to the text_vocab+1.
        vocab_size = kwargs.pop("vocab_size", text_vocab + 1)
        super().__init__(vocab_size=vocab_size, **kwargs)

    @classmethod
    def from_zonos2_config(
        cls, zonos_cfg: Any, *, model_path: str = ""
    ) -> "Zonos2HFConfig":
        """Build a Zonos2HFConfig from a ZONOS2-native ModelConfig (or checkpoint config)."""

        # Handle both ModelConfig dataclass and raw checkpoint config objects
        def _get(obj: Any, name: str, default: Any = None) -> Any:
            return getattr(obj, name, default)

        head_dim = _get(zonos_cfg, "head_dim", 128)
        dim = _get(zonos_cfg, "dim") or _get(zonos_cfg, "hidden_size", 4096)
        n_heads = (
            _get(zonos_cfg, "n_heads")
            or _get(zonos_cfg, "num_qo_heads")
            or (dim // head_dim)
        )
        n_kv_heads = (
            _get(zonos_cfg, "n_kv_heads") or _get(zonos_cfg, "num_kv_heads") or n_heads
        )
        n_layers = _get(zonos_cfg, "n_layers") or _get(zonos_cfg, "num_layers", 32)

        ffn_mul = _get(zonos_cfg, "ffn_dim_multiplier")
        if ffn_mul is not None:
            ffn_dim = int(ffn_mul * dim)
            multiple_of = _get(zonos_cfg, "multiple_of", 256)
            intermediate_size = multiple_of * (
                (ffn_dim + multiple_of - 1) // multiple_of
            )
        else:
            intermediate_size = _get(zonos_cfg, "intermediate_size", 14336)

        rope_theta = _get(zonos_cfg, "rope_theta", 500000.0)
        max_seqlen = _get(zonos_cfg, "max_seqlen") or _get(
            zonos_cfg, "max_position_embeddings", 4096
        )
        norm_eps = _get(zonos_cfg, "norm_eps") or _get(zonos_cfg, "rms_norm_eps", 1e-5)

        moe_n_experts = int(_get(zonos_cfg, "moe_n_experts", 1) or 1)
        moe_router_topk = int(_get(zonos_cfg, "moe_router_topk", 1) or 1)
        moe_router_dim = int(_get(zonos_cfg, "moe_router_dim", 256) or 256)
        moe_start = int(_get(zonos_cfg, "moe_start_from_layer", 0) or 0)
        moe_end = int(_get(zonos_cfg, "moe_end_from_layer", 0) or 0)
        moe_balancing = str(
            _get(zonos_cfg, "moe_balancing_strategy", "legacy") or "legacy"
        )

        quality_features_raw = _get(zonos_cfg, "quality_features") or []
        quality_features = list(quality_features_raw)
        quality_buckets_raw = _get(zonos_cfg, "quality_buckets") or {}
        quality_buckets = (
            {str(k): [str(v) for v in vs] for k, vs in quality_buckets_raw.items()}
            if quality_buckets_raw
            else {}
        )

        return cls(
            hidden_size=int(dim),
            num_attention_heads=int(n_heads),
            num_key_value_heads=int(n_kv_heads),
            head_dim=int(head_dim),
            num_hidden_layers=int(n_layers),
            intermediate_size=int(intermediate_size),
            rms_norm_eps=float(norm_eps),
            rope_theta=float(rope_theta),
            max_position_embeddings=int(max_seqlen),
            n_codebooks=int(_get(zonos_cfg, "n_codebooks", 9)),
            codebook_size=int(_get(zonos_cfg, "codebook_size", 1024)),
            text_vocab=int(_get(zonos_cfg, "text_vocab") or 469),
            eoa_id=int(_get(zonos_cfg, "eoa_id", 1024)),
            audio_pad_id=int(_get(zonos_cfg, "audio_pad_id", 1025)),
            loss_softcap=float(_get(zonos_cfg, "loss_softcap", 15.0)),
            moe_n_experts=moe_n_experts,
            moe_router_dim=moe_router_dim,
            moe_start_from_layer=moe_start,
            moe_end_from_layer=moe_end,
            num_experts_per_tok=moe_router_topk if moe_n_experts > 1 else 1,
            norm_topk_prob=bool(_get(zonos_cfg, "norm_topk_prob", False)),
            moe_balancing_strategy=moe_balancing,
            moe_intermediate_size=int(intermediate_size) if moe_n_experts > 1 else 0,
            special_topk_layers=dict(_get(zonos_cfg, "special_topk_layers") or {}),
            speaker_enabled=bool(_get(zonos_cfg, "speaker_enabled", False)),
            speaker_embedding_dim=int(_get(zonos_cfg, "speaker_embedding_dim", 128)),
            speaker_lda_dim=_get(zonos_cfg, "speaker_lda_dim"),
            speaker_background_token_enabled=bool(
                _get(zonos_cfg, "speaker_background_token_enabled", False)
            ),
            accurate_mode_token_enabled=bool(
                _get(zonos_cfg, "accurate_mode_token_enabled", False)
            ),
            speaking_rate_num_buckets=int(
                _get(zonos_cfg, "speaking_rate_num_buckets", 0) or 0
            ),
            quality_num_buckets=int(_get(zonos_cfg, "quality_num_buckets", 0) or 0),
            quality_features=quality_features,
            quality_buckets=quality_buckets,
            _zonos2_model_path=model_path,
        )

    def get_layer_num_experts(self, layer_id: int) -> int:
        """Return the top-k experts for this layer (accounts for special_topk_layers)."""
        if self.moe_n_experts <= 1:
            return 0
        if self.special_topk_layers and str(layer_id) in self.special_topk_layers:
            return int(self.special_topk_layers[str(layer_id)])
        return self.num_experts_per_tok

    def is_moe_layer(self, layer_id: int) -> bool:
        """Return True if this layer should be a MoE layer."""
        if self.moe_n_experts <= 1:
            return False
        if layer_id < self.moe_start_from_layer:
            return False
        if (self.num_hidden_layers - layer_id) <= self.moe_end_from_layer:
            return False
        return True
