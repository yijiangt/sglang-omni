# SPDX-License-Identifier: Apache-2.0
"""SGLang-native ZONOS2 TTS model.

Architecture: sparse MoE decoder-only transformer with multi-codebook I/O.

Input per token: (n_codebooks + 1) integers — 9 audio codebooks + 1 text column.
Output per token: (n_codebooks, audio_vocab) logits — 9 codebooks sampled independently.

Attention: separate wq / wk / wv projections (not fused QKV), learnable per-head
temperature and headwise sigmoid gating.

FFN: dense (ChunkedLinear SwiGLU) for early/late layers, MoE with EDA routing for
the middle block. EDA threads router_states across consecutive MoE layers.

Weight format: flat .pth state dict with no "model." prefix; loaded via
``zonos2.models.weight.load_checkpoint_weight``.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sglang.srt.layers.logits_processor import LogitsProcessorOutput

from sglang_omni.vendor.sglang.core import ForwardBatch
from sglang_omni.vendor.sglang.layers import (
    MergedColumnParallelLinear,
    RMSNorm,
    RowParallelLinear,
    StandardTopKOutput,
    VocabParallelEmbedding,
    get_moe_impl_class,
    get_rope,
)

from .hf_config import Zonos2HFConfig

logger = logging.getLogger(__name__)

_DEFAULT_MAX_BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Multi-embedding: sum over n_codebooks + 1 parallel tables
# ---------------------------------------------------------------------------


class Zonos2MultiEmbedding(nn.Module):
    """Summed multi-codebook embedding.

    Checkpoint keys: ``multi_embedder.embedders.{i}.weight``
    """

    def __init__(self, config: Zonos2HFConfig) -> None:
        super().__init__()
        n_cols = config.n_codebooks + (1 if config.text_vocab is not None else 0)
        audio_vocab = config.codebook_size + 2  # +eoa +pad

        embedders = []
        for _ in range(config.n_codebooks):
            embedders.append(
                VocabParallelEmbedding(
                    num_embeddings=audio_vocab,
                    embedding_dim=config.hidden_size,
                )
            )
        if config.text_vocab is not None:
            embedders.append(
                VocabParallelEmbedding(
                    num_embeddings=config.text_vocab + 1,
                    embedding_dim=config.hidden_size,
                )
            )
        self.embedders = nn.ModuleList(embedders)
        self._n_cols = n_cols

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """Sum embeddings over all columns.

        Args:
            codes: (total_tokens, n_cols) int64
        Returns:
            (total_tokens, hidden_size)
        """
        out = self.embedders[0](codes[:, 0].contiguous())
        for i in range(1, self._n_cols):
            out = out + self.embedders[i](codes[:, i].contiguous())
        return out


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class Zonos2Attention(nn.Module):
    """Multi-head attention with separate Q/KV projections, QK-norm, temperature,
    and per-head sigmoid gating.

    Checkpoint keys:
    - ``layers.{N}.attention.wq.weight``         [local_q_dim, hidden]
    - ``layers.{N}.attention.wkv.weight``        [2, local_kv_dim, hidden]  (3-D!)
    - ``layers.{N}.attention.wo.weight``         [hidden, local_q_dim]
    - ``layers.{N}.attention.temp``              [1, local_heads, 1]
    - ``layers.{N}.attention.gater.weight``      [local_heads, hidden]
    """

    def __init__(
        self,
        config: Zonos2HFConfig,
        layer_id: int,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        from sglang_omni.vendor.sglang.layers import RadixAttention

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        # Separate Q, K, V projections matching checkpoint naming.
        # wq/wk/wv are plain nn.Linear; ZONOS2 TTS is typically served at TP=1.
        self.wq = nn.Linear(config.hidden_size, self.q_size, bias=False)
        self.wk = nn.Linear(config.hidden_size, self.kv_size, bias=False)
        self.wv = nn.Linear(config.hidden_size, self.kv_size, bias=False)
        self.wo = RowParallelLinear(self.q_size, config.hidden_size, bias=False)

        # Learnable QK temperature: [1, num_heads, 1]
        self.temp = nn.Parameter(torch.ones(1, self.num_heads, 1, dtype=torch.bfloat16))

        # Headwise sigmoid gate applied to attention output before wo
        self.gater = nn.Linear(config.hidden_size, self.num_heads, bias=False)

        # RoPE (interleaved = NOT neox style, is_neox_style=False)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            is_neox_style=False,
        )

        # Paged attention
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        gate = torch.sigmoid(self.gater(x))  # (T, num_heads)

        q = self.wq(x)  # (T, q_size)
        k = self.wk(x)  # (T, kv_size)
        v = self.wv(x)  # (T, kv_size)

        T = q.shape[0]
        q = q.view(T, self.num_heads, self.head_dim)
        k = k.view(T, self.num_kv_heads, self.head_dim)

        # QK norm with learnable temperature (reference: F.rms_norm + temp.abs())
        q = F.rms_norm(q, (self.head_dim,), eps=1e-6) * self.temp.abs().to(q.dtype)
        k = F.rms_norm(k, (self.head_dim,), eps=1e-6)

        # RoPE (interleaved / non-neox format)
        q, k = self.rotary_emb(positions, q.flatten(-2), k.flatten(-2))
        q = q.view(T, self.num_heads, self.head_dim)

        # Paged attention
        o = self.attn(
            q, k, v.view(T, self.num_kv_heads, self.head_dim), forward_batch
        )  # (T, num_heads*head_dim)

        # Headwise gating
        o = o.view(T, self.num_heads, self.head_dim)
        o = o * gate.unsqueeze(-1)
        o = o.view(T, self.q_size)

        out = self.wo(o)
        return out[0] if isinstance(out, tuple) else out


# ---------------------------------------------------------------------------
# Dense feed-forward
# ---------------------------------------------------------------------------


class Zonos2FeedForward(nn.Module):
    """Dense SwiGLU feed-forward.

    Checkpoint keys:
    - ``layers.{N}.feed_forward.w_in.weight``   [2, inter, hidden]  (3-D!)
    - ``layers.{N}.feed_forward.w_out.weight``  [hidden, inter]

    The 3-D w_in encodes [up_weights, gate_weights] along dim-0 (ChunkedLinear
    convention where first chunk = up, second chunk = gate).  The forward is:
        out = up_proj(x) * silu(gate_proj(x))
    """

    def __init__(
        self, config: Zonos2HFConfig, quant_config=None, prefix: str = ""
    ) -> None:
        super().__init__()
        # MergedColumnParallelLinear fuses gate and up projections.
        # We load up into shard-0 and gate into shard-1 to match the checkpoint
        # ordering, then slice first-half=up, second-half=gate in forward().
        self.w_in = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            bias=False,
        )
        self.w_out = RowParallelLinear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.w_in(x)
        inter = gate_up.shape[-1] // 2
        # Checkpoint w_in[0] → up, w_in[1] → gate (ChunkedLinear ordering).
        # We load w_in[0] into shard-0, w_in[1] into shard-1, so:
        #   first half = up projection, second half = gate projection.
        up = gate_up[..., :inter]
        gate = gate_up[..., inter:]
        h = up * F.silu(gate)
        out, _ = self.w_out(h)
        return out


# ---------------------------------------------------------------------------
# MoE Router with EDA (Expert Distribution Awareness)
# ---------------------------------------------------------------------------


class Zonos2Router(nn.Module):
    """EDA Router for ZONOS2 MoE layers.

    Checkpoint keys under ``layers.{N}.feed_forward.router.*``:
    - ``down_proj.weight``, ``down_proj.bias``
    - ``router_mlp.0.weight``, ``router_mlp.0.bias``
    - ``router_mlp.2.weight``, ``router_mlp.2.bias``
    - ``router_mlp.4.weight``
    - ``rmsnorm_eda.weight``
    - ``router_states_scale``   (EDA layers only)
    - ``balancing_biases``

    Plain nn.Linear is used throughout (no wrapper) so parameter names match
    checkpoint keys after the nn.Sequential index → named-attribute remap.
    """

    def __init__(self, config: Zonos2HFConfig, layer_id: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.router_dim = config.moe_router_dim
        self.num_experts = config.moe_n_experts
        self.top_k = config.get_layer_num_experts(layer_id)
        self._layer_id = layer_id
        self._use_legacy = config.moe_balancing_strategy in (
            "legacy",
            "old",
            "aux",
            "aux_loss",
        )

        # Plain nn.Linear — parameter paths are e.g. router.down_proj.weight
        self.down_proj = nn.Linear(self.hidden_size, self.router_dim, bias=True)

        # Names mirror checkpoint indices (0, 2, 4) after remap in _remap_weight_key
        self.router_mlp_0 = nn.Linear(self.router_dim, self.router_dim, bias=True)
        self.router_mlp_2 = nn.Linear(self.router_dim, self.router_dim, bias=True)
        self.router_mlp_4 = nn.Linear(self.router_dim, self.num_experts, bias=False)

        # RMSNorm for EDA input
        self.rmsnorm_eda = RMSNorm(self.router_dim, eps=config.rms_norm_eps)

        # EDA: all layers except the first MoE layer use router_states from prev
        self._use_eda = layer_id != config.moe_start_from_layer
        if self._use_eda:
            self.router_states_scale = nn.Parameter(torch.ones(self.router_dim))

        # Balancing biases (non-trainable)
        self.balancing_biases = nn.Parameter(
            torch.zeros(self.num_experts), requires_grad=False
        )

    def forward(
        self,
        hidden: torch.Tensor,
        router_states: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (topk_weights, topk_ids, router_states_next)."""
        h = self.down_proj(hidden)

        if self._use_eda and router_states is not None:
            h = h + router_states * self.router_states_scale

        h_next = h.clone()

        # SGLang RMSNorm without residual returns a plain tensor
        h_normed = self.rmsnorm_eda(h)
        h_normed = F.gelu(self.router_mlp_0(h_normed))
        h_normed = F.gelu(self.router_mlp_2(h_normed))
        logits = self.router_mlp_4(h_normed).float()

        expert_prob = torch.softmax(logits, dim=-1)

        with torch.no_grad():
            bias = self.balancing_biases.float()
            routing_scores = (
                expert_prob + bias if self._use_legacy else expert_prob - bias
            )
            _, expert_choice = torch.topk(routing_scores, self.top_k, dim=-1)

        topk_weights = torch.gather(expert_prob, dim=-1, index=expert_choice)
        topk_ids = expert_choice.to(torch.int32)

        return topk_weights, topk_ids, h_next


# ---------------------------------------------------------------------------
# MoE Feed-Forward
# ---------------------------------------------------------------------------


class Zonos2MoEFeedForward(nn.Module):
    """MoE feed-forward with EDA routing.

    Checkpoint keys under ``layers.{N}.feed_forward.*``:
    - ``router.*`` → Zonos2Router
    - ``experts.gate_up_proj``  [n_experts, 2*inter, hidden]  (pre-fused)
    - ``experts.down_proj``     [n_experts, hidden, inter]
    (or unfused w1/w3/w2, or SonicMoE w13/w2 — converted in load_weights)
    """

    def __init__(
        self,
        config: Zonos2HFConfig,
        layer_id: int,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.top_k = config.get_layer_num_experts(layer_id)
        self.router = Zonos2Router(config, layer_id)

        inter = (
            config.moe_intermediate_size
            if config.moe_intermediate_size > 0
            else config.intermediate_size
        )
        FusedMoECls = get_moe_impl_class(quant_config)
        self.experts = FusedMoECls(
            num_experts=config.moe_n_experts,
            top_k=self.top_k,
            hidden_size=config.hidden_size,
            intermediate_size=inter,
            layer_id=layer_id,
            quant_config=quant_config,
            reduce_results=True,
            prefix=f"{prefix}.experts" if prefix else "experts",
        )

    def forward(
        self,
        x: torch.Tensor,
        router_states: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        topk_weights, topk_ids, router_states_next = self.router(x, router_states)

        topk_output = StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=None,
        )
        out = self.experts(x, topk_output)
        return out, router_states_next


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------


class Zonos2TransformerBlock(nn.Module):
    """Single ZONOS2 transformer layer (attention + FFN).

    Checkpoint keys:
    - ``layers.{N}.attention_norm.weight``
    - ``layers.{N}.attention.*``
    - ``layers.{N}.ffn_norm.weight``
    - ``layers.{N}.feed_forward.*``
    """

    def __init__(
        self,
        config: Zonos2HFConfig,
        layer_id: int,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self._layer_id = layer_id
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = Zonos2Attention(config, layer_id, quant_config=quant_config)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self._is_moe = config.is_moe_layer(layer_id)
        if self._is_moe:
            self.feed_forward = Zonos2MoEFeedForward(
                config, layer_id, quant_config=quant_config
            )
        else:
            self.feed_forward = Zonos2FeedForward(config, quant_config=quant_config)

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor],
        router_states: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # SGLang's RMSNorm(x, residual) returns (normed, updated_residual) when
        # residual is not None, but just the normed tensor when residual is None.
        # First-layer (residual=None): set residual = x first, then norm plain x.
        # This matches RMSNormFused(x, None) semantics from the ZONOS2 reference.
        if residual is None:
            residual = x
            h = self.attention_norm(x)
        else:
            h, residual = self.attention_norm(x, residual)

        h = self.attention(h, positions, forward_batch)

        # ffn_norm: residual is always non-None here
        h, residual = self.ffn_norm(h, residual)

        if self._is_moe:
            h, router_states = self.feed_forward(h, router_states)
        else:
            h = self.feed_forward(h)
            router_states = None

        return h, residual, router_states


# ---------------------------------------------------------------------------
# ZONOS2 SGLang Model
# ---------------------------------------------------------------------------


class Zonos2SGLangModel(nn.Module):
    """ZONOS2 sparse MoE TTS model for sglang inference.

    Registered in ModelRegistry as ``"Zonos2SGLangModel"``.

    Input: 2D codes tensor (total_tokens, n_codebooks+1) stored in
    ``_cg_active_last_codes`` buffer for decode, or via ``forward_batch.input_embeds``
    for prefill (pre-computed by the model runner).

    Output: ``LogitsProcessorOutput`` with ``next_token_logits`` shape
    ``(total_tokens, n_codebooks * audio_vocab)`` — the model runner samples
    per-codebook and stores codes; ``next_token_ids`` is set to cb0.
    """

    def __init__(
        self,
        config: Zonos2HFConfig,
        quant_config=None,
        prefix: str = "",
        max_batch_size: int = _DEFAULT_MAX_BATCH_SIZE,
    ) -> None:
        super().__init__()
        self.config = config
        self.n_codebooks = config.n_codebooks
        self.audio_vocab = config.codebook_size + 2
        self._max_batch_size = int(max_batch_size)

        self.multi_embedder = Zonos2MultiEmbedding(config)

        # Optional speaker conditioning
        self.speaker_lda_projection: Optional[nn.Linear] = None
        self.speaker_projection: Optional[nn.Linear] = None
        if config.speaker_enabled:
            lda_dim = int(config.speaker_lda_dim) if config.speaker_lda_dim else None
            if lda_dim:
                self.speaker_lda_projection = nn.Linear(
                    config.speaker_embedding_dim, lda_dim, bias=True
                )
                proj_in = lda_dim
            else:
                proj_in = config.speaker_embedding_dim
            self.speaker_projection = nn.Linear(proj_in, config.hidden_size, bias=True)

        self.layers = nn.ModuleList(
            [
                Zonos2TransformerBlock(config, i, quant_config=quant_config)
                for i in range(config.num_hidden_layers)
            ]
        )

        self.out_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Multi-output head: (hidden) → (n_codebooks * audio_vocab)
        self.multi_output = nn.Linear(
            config.hidden_size,
            config.n_codebooks * self.audio_vocab,
            bias=False,
        )

        # Decode-step codes buffer: stores (n_codebooks+1) codes per slot
        n_cols = config.n_codebooks + (1 if config.text_vocab is not None else 0)
        pool_size = self._max_batch_size + 1
        self._cg_active_last_codes = torch.zeros(pool_size, n_cols, dtype=torch.long)
        self._cg_active_last_codes[:, : config.n_codebooks] = config.audio_pad_id
        if config.text_vocab is not None:
            self._cg_active_last_codes[:, config.n_codebooks] = config.text_vocab

        self._cg_output_codes = torch.zeros(
            pool_size, config.n_codebooks, dtype=torch.long
        )
        self._cg_generation_done = torch.zeros(pool_size, dtype=torch.bool)

        # Per-request slot management
        self._rid_to_row: dict[str, int] = {}
        self._free_rows: list[int] = list(range(self._max_batch_size))
        self._padding_row = self._max_batch_size

    # ------------------------------------------------------------------
    # Row management
    # ------------------------------------------------------------------

    def acquire_row(self, req_id: str) -> int:
        row = self._rid_to_row.get(req_id)
        if row is not None:
            return row
        if not self._free_rows:
            raise RuntimeError(
                f"Zonos2SGLangModel slot pool exhausted (max_batch_size="
                f"{self._max_batch_size})"
            )
        row = self._free_rows.pop()
        self._rid_to_row[req_id] = row
        self._reset_row(row)
        return row

    def release_row(self, req_id: str) -> None:
        row = self._rid_to_row.pop(req_id, None)
        if row is not None:
            self._free_rows.append(row)

    def reset_request(self, req_id: str) -> None:
        self.release_row(req_id)

    def _reset_row(self, row: int) -> None:
        self._cg_active_last_codes[row, : self.n_codebooks] = self.config.audio_pad_id
        if self.config.text_vocab is not None:
            self._cg_active_last_codes[row, self.n_codebooks] = self.config.text_vocab
        self._cg_generation_done[row] = False

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def get_input_embeddings(self) -> nn.Module:
        return self.multi_embedder.embedders[0]

    @staticmethod
    def _is_decode_step(forward_batch: ForwardBatch) -> bool:
        forward_mode = getattr(forward_batch, "forward_mode", None)
        return forward_mode is not None and forward_mode.is_decode()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> LogitsProcessorOutput:
        is_decode = self._is_decode_step(forward_batch)

        if is_decode:
            bs = input_ids.shape[0]
            codes = self._cg_active_last_codes[:bs].to(input_ids.device)
            x = self.multi_embedder(codes)
        elif input_embeds is not None:
            x = input_embeds
        else:
            x = self.multi_embedder.embedders[-1](input_ids)

        # Speaker injection happens BEFORE emb_norm (matches reference ordering).
        if self.speaker_projection is not None:
            speaker_emb_values: Optional[torch.Tensor] = getattr(
                forward_batch, "speaker_emb_values", None
            )
            speaker_token_positions: Optional[torch.Tensor] = getattr(
                forward_batch, "speaker_token_positions", None
            )
            if (
                speaker_emb_values is not None
                and speaker_token_positions is not None
                and speaker_emb_values.numel() > 0
                and speaker_token_positions.numel() > 0
            ):
                spk = speaker_emb_values.to(x.dtype)
                if self.speaker_lda_projection is not None:
                    spk = self.speaker_lda_projection(spk)
                spk = self.speaker_projection(spk)
                x = x.index_copy(0, speaker_token_positions, spk.to(x.dtype))

        # emb_norm: no learnable weight (elementwise_affine=False in reference)
        x = F.rms_norm(x, (x.shape[-1],), eps=self.config.rms_norm_eps)

        # Transformer layers with EDA router_states threading
        residual: Optional[torch.Tensor] = None
        router_states: Optional[torch.Tensor] = None
        for layer in self.layers:
            x, residual, router_states = layer(
                x, residual, router_states, positions, forward_batch
            )

        # Final norm: residual is always non-None after ≥1 transformer blocks
        x, _ = self.out_norm(x, residual)

        logits = self.multi_output(x)  # (total_tokens, n_codebooks * audio_vocab)

        if self.config.loss_softcap > 0:
            cap = self.config.loss_softcap
            logits = cap * torch.tanh(logits / cap)

        return LogitsProcessorOutput(
            next_token_logits=logits,
            hidden_states=x,
        )

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """Load ZONOS2 .pth checkpoint.

        The SGLang-supplied safetensors iterator is drained but ignored — the
        staging directory has an empty model.safetensors so the iterator yields
        nothing.  Actual weights come from the .pth file via
        ``zonos2.models.weight.load_checkpoint_weight``.
        """
        for _ in weights:
            pass

        model_path = getattr(self.config, "_zonos2_model_path", "")
        if not model_path:
            raise ValueError(
                "Zonos2SGLangModel: config._zonos2_model_path is not set. "
                "Ensure the staging directory config.json includes '_zonos2_model_path'."
            )

        try:
            from zonos2.distributed.info import set_tp_info, try_get_tp_info
            from zonos2.models.weight import load_checkpoint_weight
        except ImportError as exc:
            raise ImportError(
                "zonos2 package not found. Install via: pip install -e /path/to/ZONOS2/python"
            ) from exc

        if try_get_tp_info() is None:
            set_tp_info(rank=0, size=1)

        logger.info("Loading ZONOS2 weights from %s", model_path)
        state_dict = load_checkpoint_weight(model_path, device=torch.device("cpu"))

        # Pre-process expert weights to handle all checkpoint formats:
        #   1. Pre-fused: gate_up_proj / down_proj present → load directly
        #   2. Unfused:   w1.weight + w3.weight → cat([w1, w3], dim=1) = gate_up_proj
        #                 w2.weight             → down_proj
        #   3. SonicMoE:  w13 (interleaved gate/up) → de-interleave to gate_up_proj
        #                 w2                         → down_proj
        self._preprocess_moe_expert_weights(state_dict)

        params = dict(self.named_parameters(remove_duplicate=False))
        loaded: set[str] = set()
        skipped: list[str] = []

        for ckpt_key, tensor in state_dict.items():
            mapped = self._remap_weight_key(ckpt_key, tensor, params, loaded)
            if not mapped:
                skipped.append(ckpt_key)

        if skipped:
            logger.debug(
                "ZONOS2 skipped %d checkpoint keys: %s…", len(skipped), skipped[:5]
            )
        logger.info("ZONOS2 loaded %d parameter tensors", len(loaded))
        return loaded

    @staticmethod
    def _preprocess_moe_expert_weights(state_dict: dict) -> None:
        """Fuse / rename expert weight keys in-place.

        Handles three formats per FusedGroupedExperts.load_state_dict reference:
        - unfused w1/w3 → cat to gate_up_proj
        - SonicMoE w13 → de-interleave to gate_up_proj, w2 → down_proj
        - pre-fused gate_up_proj / down_proj: no-op
        """
        # Collect all layer prefixes that have MoE experts
        expert_prefixes: set[str] = set()
        for key in list(state_dict.keys()):
            if ".feed_forward.experts." in key:
                pfx = key.split(".feed_forward.experts.")[0] + ".feed_forward.experts"
                expert_prefixes.add(pfx)

        for pfx in expert_prefixes:
            gate_up_key = f"{pfx}.gate_up_proj"
            down_key = f"{pfx}.down_proj"
            w1_key = f"{pfx}.w1.weight"
            w3_key = f"{pfx}.w3.weight"
            w2_key = f"{pfx}.w2.weight"
            sonic_w13_key = f"{pfx}.w13"
            sonic_w2_key = f"{pfx}.w2"

            # SonicMoE interleaved format
            if gate_up_key not in state_dict and sonic_w13_key in state_dict:
                w13 = state_dict.pop(sonic_w13_key)
                gate = w13[:, 0::2, :]
                up = w13[:, 1::2, :]
                state_dict[gate_up_key] = torch.cat([gate, up], dim=1)

            # Unfused w1/w3 format
            elif (
                gate_up_key not in state_dict
                and w1_key in state_dict
                and w3_key in state_dict
            ):
                w1 = state_dict.pop(w1_key)
                w3 = state_dict.pop(w3_key)
                # w1=gate, w3=up; gate_up_proj = [gate, up] per FusedMoE convention
                state_dict[gate_up_key] = torch.cat([w1, w3], dim=1)

            # down_proj rename
            if down_key not in state_dict and sonic_w2_key in state_dict:
                state_dict[down_key] = state_dict.pop(sonic_w2_key)
            elif down_key not in state_dict and w2_key in state_dict:
                state_dict[down_key] = state_dict.pop(w2_key)

    def _remap_weight_key(
        self,
        name: str,
        tensor: torch.Tensor,
        params: dict,
        loaded: set,
    ) -> bool:
        """Map a checkpoint key to model parameter(s). Returns True if handled."""
        from sglang.srt.model_loader.weight_utils import default_weight_loader

        # ── wkv: 3-D [2, kv_dim, hidden] → split into wk and wv ──
        if ".attention.wkv.weight" in name:
            layer_pfx = name.split(".attention.wkv.weight")[0]
            for suffix, idx in [
                (".attention.wk.weight", 0),
                (".attention.wv.weight", 1),
            ]:
                key = layer_pfx + suffix
                if key in params:
                    t = tensor[idx] if tensor.dim() == 3 else tensor
                    p = params[key]
                    default_weight_loader(p, t.to(p.dtype))
                    loaded.add(key)
            return True

        # ── w_in: 3-D [2, inter, hidden] → gate+up via MergedColumnParallelLinear.
        #    Checkpoint convention: w_in[0]=up weights, w_in[1]=gate weights.
        #    We load up into shard-0 and gate into shard-1 so the MergedColumn
        #    output layout is [up_proj | gate_proj], matching our forward slicing.
        if ".feed_forward.w_in.weight" in name:
            key = name
            if key in params:
                p = params[key]
                if tensor.dim() == 3 and hasattr(p, "weight_loader"):
                    p.weight_loader(p, tensor[0], 0)  # shard-0 ← up weights
                    p.weight_loader(p, tensor[1], 1)  # shard-1 ← gate weights
                else:
                    t = (
                        tensor.reshape(-1, tensor.shape[-1])
                        if tensor.dim() == 3
                        else tensor
                    )
                    default_weight_loader(p, t.to(p.dtype))
                loaded.add(key)
            return True

        # ── attention.temp: direct match ──
        if ".attention.temp" in name:
            if name in params:
                p = params[name]
                default_weight_loader(p, tensor.to(p.dtype))
                loaded.add(name)
            return True

        # ── MoE router_mlp: checkpoint uses nn.Sequential indices (0, 2, 4).
        #    Our model stores them as router_mlp_0, router_mlp_2, router_mlp_4
        #    (plain nn.Linear, no .linear. infix).
        for old_sfx, new_sfx in [
            (".router.router_mlp.0.", ".router.router_mlp_0."),
            (".router.router_mlp.2.", ".router.router_mlp_2."),
            (".router.router_mlp.4.", ".router.router_mlp_4."),
        ]:
            if old_sfx in name:
                new_name = name.replace(old_sfx, new_sfx)
                if new_name in params:
                    p = params[new_name]
                    default_weight_loader(p, tensor.to(p.dtype))
                    loaded.add(new_name)
                return True

        # ── FusedMoE experts: pre-fused gate_up_proj / down_proj ──
        if ".feed_forward.experts." in name:
            if name in params:
                p = params[name]
                weight_loader = getattr(p, "weight_loader", default_weight_loader)
                weight_loader(p, tensor.to(p.dtype))
                loaded.add(name)
                return True

        # ── speaker projections: direct match (nn.Linear, no infix needed) ──
        for key_suffix in [
            "speaker_lda_projection.weight",
            "speaker_lda_projection.bias",
            "speaker_projection.weight",
            "speaker_projection.bias",
        ]:
            if name.endswith(key_suffix) and name in params:
                p = params[name]
                default_weight_loader(p, tensor.to(p.dtype))
                loaded.add(name)
                return True

        # ── Default: direct key match ──
        if name in params:
            p = params[name]
            weight_loader = getattr(p, "weight_loader", default_weight_loader)
            weight_loader(p, tensor.to(p.dtype))
            loaded.add(name)
            return True

        return False


__all__ = ["Zonos2SGLangModel"]
