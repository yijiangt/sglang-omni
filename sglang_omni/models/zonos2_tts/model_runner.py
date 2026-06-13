# SPDX-License-Identifier: Apache-2.0
"""ZONOS2 TTS model runner — handles 2-D codes tensor and per-codebook sampling."""

from __future__ import annotations

import logging
from typing import Any

import torch
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN

from sglang_omni.model_runner.base import ModelRunner

logger = logging.getLogger(__name__)

_EOA_ID = 1024  # End-of-audio token in codebook 0


class Zonos2ModelRunner(ModelRunner):
    """ModelRunner for :class:`Zonos2SGLangModel`.

    Each decode step:
    1. ``before_decode``: writes last codes into ``model._cg_active_last_codes``
    2. Model forward computes multi-codebook logits (T, n_codebooks * audio_vocab)
    3. ``post_decode``: samples per-codebook, detects EOS (cb0 == eoa_id), emits
       cb0 as ``result.next_token_ids`` for SGLang's scheduler bookkeeping.
    """

    def __init__(self, tp_worker: Any, output_processor: Any) -> None:
        super().__init__(tp_worker, output_processor)
        self._eoa_id = _EOA_ID

    # ─────────────────────────────────────────────────────────────
    # Prefill
    # ─────────────────────────────────────────────────────────────

    def before_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del schedule_batch
        forward_batch.req_ids = [req.request_id for req in requests]
        input_embeds = self._build_prefill_embeds(forward_batch, requests)
        if input_embeds is not None:
            forward_batch.input_embeds = input_embeds

        # Inject speaker embeddings into forward_batch metadata
        self._inject_speaker_metadata(forward_batch, requests)

    def post_prefill(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del schedule_batch
        self._collect_step_output(result, forward_batch, requests, is_prefill=True)

    # ─────────────────────────────────────────────────────────────
    # Decode
    # ─────────────────────────────────────────────────────────────

    def before_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
        *,
        is_lookahead: bool = False,
    ) -> None:
        del schedule_batch, is_lookahead
        forward_batch.req_ids = [req.request_id for req in requests]
        model = self.model

        # Scatter each request's persistent pool slot into its batch position.
        # The model forward reads _cg_active_last_codes[:bs] by batch index, so
        # row_idx (pool) → b (batch position) must be explicit here.
        for b, sched_req in enumerate(requests):
            row_idx = model.acquire_row(sched_req.request_id)
            model._cg_active_last_codes[b] = model._cg_active_last_codes[row_idx]

    def post_decode(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del schedule_batch
        self._collect_step_output(result, forward_batch, requests, is_prefill=False)

    # ─────────────────────────────────────────────────────────────
    # Core helpers
    # ─────────────────────────────────────────────────────────────

    def _build_prefill_embeds(
        self, forward_batch: Any, requests: list
    ) -> torch.Tensor | None:
        """Build summed multi-codebook embeddings for all prefill tokens.

        Each request has a 2-D prompt (T_prompt, n_cols). We concatenate all
        prompts along the token dimension and compute MultiEmbedding.
        """
        model = self.model
        device = next(model.parameters()).device

        all_rows: list[list[int]] = []
        for sched_req in requests:
            data = sched_req.data
            prompt_rows: list[list[int]] = getattr(data, "prompt_rows", [])
            all_rows.extend(prompt_rows)

        if not all_rows:
            return None

        codes = torch.tensor(all_rows, dtype=torch.long, device=device)
        with torch.no_grad():
            return model.multi_embedder(codes)

    def _inject_speaker_metadata(self, forward_batch: Any, requests: list) -> None:
        """Attach speaker embeddings to forward_batch for the model's speaker injection."""
        model = self.model
        if not model.config.speaker_enabled:
            return

        device = next(model.parameters()).device
        spk_values: list[torch.Tensor] = []
        spk_positions: list[int] = []
        token_offset = 0

        for sched_req in requests:
            data = sched_req.data
            prompt_rows = getattr(data, "prompt_rows", [])
            spk_emb: list[float] | None = getattr(data, "speaker_embedding", None)
            spk_pos: int = getattr(data, "speaker_token_position", -1)

            if spk_emb is not None and spk_pos >= 0:
                abs_pos = token_offset + spk_pos
                spk_positions.append(abs_pos)
                spk_values.append(torch.tensor(spk_emb, dtype=torch.float32))

            token_offset += len(prompt_rows)

        if spk_values:
            forward_batch.speaker_emb_values = torch.stack(spk_values, dim=0).to(device)
            forward_batch.speaker_token_positions = torch.tensor(
                spk_positions, dtype=torch.long, device=device
            )

    def _collect_step_output(
        self,
        result: Any,
        forward_batch: Any,
        requests: list,
        *,
        is_prefill: bool,
    ) -> None:
        """Sample per-codebook logits and update result.next_token_ids."""
        if not requests:
            return

        model = self.model
        n_codebooks = model.n_codebooks
        audio_vocab = model.audio_vocab
        n_cols = n_codebooks + (1 if model.config.text_vocab is not None else 0)

        # Get logits from result (total_tokens, n_codebooks * audio_vocab)
        logits_flat = result.logits_output.next_token_logits

        if is_prefill:
            # During prefill, only the last token of each request matters
            extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
            if extend_seq_lens is not None:
                last_indices = torch.cumsum(extend_seq_lens, dim=0) - 1
                logits_flat = logits_flat[last_indices]
            else:
                logits_flat = logits_flat[-len(requests) :]

        # Shape: (B, n_codebooks, audio_vocab)
        B = len(requests)
        logits_BNV = logits_flat.view(B, n_codebooks, audio_vocab).float()

        # Sample per codebook
        sampled_BN = self._sample_codebooks(
            logits_BNV, requests
        )  # (B, n_codebooks) on device

        # Build next input codes (sampled audio codes + text_vocab pad for text col)
        device = sampled_BN.device
        text_vocab = model.config.text_vocab or 469

        cb0_list: list[int] = []
        for b, sched_req in enumerate(requests):
            data = sched_req.data
            req = data.req

            if getattr(req, "is_chunked", 0) > 0:
                cb0_list.append(0)
                continue

            codes_N = sampled_BN[b].to(torch.long)  # (n_codebooks,)
            cb0 = int(codes_N[0].item())

            # Detect EOS
            is_eos = cb0 == self._eoa_id
            if is_eos and not getattr(data, "generation_done", False):
                data.generation_done = True
                if req.finished_reason is None:
                    req.finished_reason = FINISH_MATCHED_TOKEN(self._eoa_id)

            was_done = getattr(data, "generation_done", False)
            if not was_done or is_eos:
                # Append codes to output
                if not hasattr(data, "output_codes") or data.output_codes is None:
                    data.output_codes = []
                data.output_codes.append(codes_N.cpu().tolist())

            # Build next-step codes row: [cb0..cb8, text_pad]
            next_row = torch.zeros(n_cols, dtype=torch.long, device=device)
            next_row[:n_codebooks] = codes_N
            next_row[n_codebooks] = text_vocab  # text column = pad
            # Write to persistent pool slot; before_decode scatters to batch pos.
            row_idx = model.acquire_row(sched_req.request_id)
            model._cg_active_last_codes[row_idx] = next_row

            # Update repetition window
            self._update_rep_window(data, codes_N.cpu().tolist())

            cb0_list.append(cb0)

        result.next_token_ids = torch.tensor(cb0_list, dtype=torch.long, device=device)

    def _sample_codebooks(
        self,
        logits_BNV: torch.Tensor,
        requests: list,
    ) -> torch.Tensor:
        """Sample independently per codebook using per-request params.

        Returns: (B, n_codebooks) int64 on the same device as logits.
        """
        from zonos2.tts.sampler import sample_tts

        B, N, V = logits_BNV.shape
        device = logits_BNV.device

        # Build per-request sampling tensors
        temps = []
        top_ks = []
        top_ps = []
        min_ps = []
        rep_penalties = []
        rep_token_ids_list: list[list[list[int]]] = []

        for sched_req in requests:
            data = sched_req.data
            temps.append(float(getattr(data, "temperature", 1.15)))
            top_ks.append(
                int(getattr(data, "top_k", 106))
                if getattr(data, "top_k", -1) > 0
                else V
            )
            top_ps.append(float(getattr(data, "top_p", 0.0)))
            min_ps.append(float(getattr(data, "min_p", 0.18)))
            rep_penalties.append(float(getattr(data, "repetition_penalty", 1.2)))

            rep_window: list[list[int]] = getattr(data, "rep_window", [])
            rep_token_ids_list.append(
                rep_window[-getattr(data, "repetition_window", 50) :]
                if rep_window
                else []
            )

        temperatures = torch.tensor(temps, dtype=torch.float32, device=device)
        top_ks_t = torch.tensor(top_ks, dtype=torch.int64, device=device)
        top_ps_t = torch.tensor(top_ps, dtype=torch.float32, device=device)
        min_ps_t = torch.tensor(min_ps, dtype=torch.float32, device=device)
        rep_penalties_t = torch.tensor(
            rep_penalties, dtype=torch.float32, device=device
        )

        # Build repetition_token_ids: (B, N, window)
        win = (
            max((len(w) for w in rep_token_ids_list), default=0)
            if rep_token_ids_list
            else 0
        )
        rep_token_ids = None
        if win > 0:
            rep_arr = torch.full((B, N, win), -1, dtype=torch.long, device=device)
            for b, rows in enumerate(rep_token_ids_list):
                for t, row in enumerate(rows[-win:]):
                    row_t = min(len(row), N)
                    rep_arr[b, :row_t, t] = torch.tensor(row[:row_t], dtype=torch.long)
            rep_token_ids = rep_arr

        text_vocab = getattr(self.model.config, "text_vocab", 469) or 469

        sampled_rows = sample_tts(
            logits_BNV,
            temperatures=temperatures,
            top_ks=top_ks_t,
            top_ps=top_ps_t,
            min_ps=min_ps_t,
            repetition_token_ids=rep_token_ids,
            repetition_penalties=rep_penalties_t,
            text_vocab=text_vocab,
        )
        # sampled_rows: list[list[int]], each entry is [cb0..cb8, text_pad]
        # We only want the first N (audio) tokens
        codes_list = [row[:N] for row in sampled_rows]
        return torch.tensor(codes_list, dtype=torch.long, device=device)

    def _update_rep_window(self, data: Any, new_codes: list[int]) -> None:
        """Append new codebook codes to the sliding repetition window."""
        if not hasattr(data, "rep_window") or data.rep_window is None:
            data.rep_window = []
        win_size = int(getattr(data, "repetition_window", 50))
        data.rep_window.append(new_codes)
        if len(data.rep_window) > win_size:
            data.rep_window = data.rep_window[-win_size:]


__all__ = ["Zonos2ModelRunner"]
