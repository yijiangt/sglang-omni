# SPDX-License-Identifier: Apache-2.0
"""Validate CUDA graph batch coverage for SGLang-backed generation stages.

Background
----------
SGLang-backed generation stages have three batch-size facts that must stay
consistent but are configured in different places:

1. **Serving config** -- ``max_running_requests`` and ``cuda_graph_max_bs`` on
   the engine ``server_args``.
2. **Captured CUDA graph batch sizes** -- the list SGLang actually captured
   graphs for, computed by ``get_batch_sizes_to_capture`` and stored on the
   runner's ``graph_runner.capture_bs``.
3. **Model-side buffers** -- per-request pools/staging tensors a model
   allocates itself (e.g. Higgs ``pool_size``, Qwen3-TTS ``_feedback_buffer``),
   typically sized from the live ``max_running_requests``.

#756 fixed a case where (3) drifted from (1): Higgs sized its sampler/CUDA
graph buffers from a stale constant, so capturing at a larger ``cuda_graph_max_bs``
overran them and crashed. This module reports the three facts for a running
stage and flags disagreements, so that class of bug is caught at startup
instead of at capture time (and so #721/#781 tuning has the numbers it needs).

Layout
------
The verdict logic (:func:`evaluate_cuda_graph_batch_sizing`) is pure and has no
SGLang or GPU dependency -- it is unit tested on CPU. The introspection layer
(:func:`inspect_model_runner`) reads a *live* runner after capture and only
yields meaningful values on a GPU server; it is defensive about upstream
attribute names so a SGLang upgrade degrades to a warning rather than a crash.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# Upstream SGLang attribute names this module depends on, kept in one place so a
# version bump that renames them is easy to find and patch.
#   model_runner.graph_runner                  -> the CudaGraphRunner instance
#   graph_runner.capture_bs                    -> list[int] of captured sizes
#   model_runner.req_to_token_pool.size        -> the request-slot count that
#                                                 capture_bs is clamped to
_RUNNER_GRAPH_ATTR = "graph_runner"
_CAPTURE_BS_ATTR = "capture_bs"


@dataclass(frozen=True)
class CudaGraphBatchReport:
    """Outcome of validating one stage's CUDA graph batch sizing."""

    stage: str
    # (1) serving config
    max_running_requests: int | None
    cuda_graph_max_bs: int | None
    # (2) actually-captured graph batch sizes
    captured_bs: list[int] | None
    # request-slot capacity capture_bs is clamped to (req_to_token_pool.size)
    request_slots: int | None
    # (3) model-side buffer capacity, when the model exposes it (else None)
    buffer_capacity: int | None
    ok: bool
    findings: list[str] = field(default_factory=list)

    @property
    def max_captured_bs(self) -> int | None:
        return max(self.captured_bs) if self.captured_bs else None

    def format(self) -> str:
        """Render a human-readable multi-line report."""
        lines = [f"Stage: {self.stage}"]
        lines.append(
            "  serving config:     "
            f"max_running_requests={self.max_running_requests}, "
            f"cuda_graph_max_bs={self.cuda_graph_max_bs}"
        )
        lines.append(f"  captured graph bs:  {self.captured_bs} "
                     f"(max captured = {self.max_captured_bs})")
        lines.append(f"  request slots:      {self.request_slots}")
        lines.append(f"  model-side buffers: {self.buffer_capacity}")
        verdict = "OK" if self.ok else "MISMATCH"
        lines.append(f"  VERDICT: {verdict}")
        for f in self.findings:
            lines.append(f"    - {f}")
        return "\n".join(lines)


def evaluate_cuda_graph_batch_sizing(
    *,
    stage: str,
    max_running_requests: int | None,
    cuda_graph_max_bs: int | None,
    captured_bs: list[int] | None,
    request_slots: int | None,
    buffer_capacity: int | None,
) -> CudaGraphBatchReport:
    """Compare the three batch-size facts and produce a verdict.

    Pure function -- no SGLang/GPU dependency. ``buffer_capacity`` may be
    ``None`` when the model does not expose a probe; the check then validates
    only what is observable (config vs. captured sizes) and notes the gap.
    """
    findings: list[str] = []
    ok = True

    captured = [b for b in (captured_bs or []) if b is not None]
    max_captured = max(captured) if captured else None

    # The load-bearing check, and the exact #756 failure mode: graphs captured
    # for a batch larger than the model's per-request buffers will overrun them.
    if buffer_capacity is not None and max_captured is not None:
        if max_captured > buffer_capacity:
            ok = False
            findings.append(
                f"graphs captured up to bs={max_captured} but model-side "
                f"buffers are sized for {buffer_capacity}; capture/replay above "
                f"{buffer_capacity} will overrun model buffers (the #756 bug class)."
            )
    elif buffer_capacity is None:
        findings.append(
            "model exposes no buffer-capacity probe; validated serving config "
            "vs. captured sizes only (model-side sizing not checked)."
        )

    # Captured sizes are clamped to the request-slot count upstream; a max
    # captured size below the configured cap means the cap never takes effect.
    if (
        cuda_graph_max_bs is not None
        and max_captured is not None
        and max_captured < cuda_graph_max_bs
    ):
        findings.append(
            f"max captured bs ({max_captured}) is below cuda_graph_max_bs "
            f"({cuda_graph_max_bs}); the configured cap is clamped, likely by "
            f"request slots ({request_slots}) / max_running_requests "
            f"({max_running_requests})."
        )

    # Buffers sized below the admission limit cannot serve peak concurrency.
    if (
        buffer_capacity is not None
        and max_running_requests is not None
        and buffer_capacity < max_running_requests
    ):
        ok = False
        findings.append(
            f"model-side buffers ({buffer_capacity}) are smaller than "
            f"max_running_requests ({max_running_requests}); peak concurrency "
            f"cannot be served."
        )

    if ok and not findings:
        findings.append("captured sizes and model-side buffers track the serving config.")

    return CudaGraphBatchReport(
        stage=stage,
        max_running_requests=max_running_requests,
        cuda_graph_max_bs=cuda_graph_max_bs,
        captured_bs=captured or None,
        request_slots=request_slots,
        buffer_capacity=buffer_capacity,
        ok=ok,
        findings=findings,
    )


def read_captured_bs(model_runner: object) -> list[int] | None:
    """Read the actually-captured CUDA graph batch sizes off a live runner.

    Returns ``None`` (with a warning) if CUDA graphs were not captured or the
    upstream attribute layout changed. GPU/live-engine only.
    """
    graph_runner = getattr(model_runner, _RUNNER_GRAPH_ATTR, None)
    if graph_runner is None:
        logger.warning(
            "cuda_graph_batch_validator: model_runner has no '%s' "
            "(CUDA graphs disabled, or upstream SGLang renamed it).",
            _RUNNER_GRAPH_ATTR,
        )
        return None
    captured = getattr(graph_runner, _CAPTURE_BS_ATTR, None)
    if captured is None:
        logger.warning(
            "cuda_graph_batch_validator: %s has no '%s' "
            "(upstream SGLang attribute layout may have changed).",
            type(graph_runner).__name__,
            _CAPTURE_BS_ATTR,
        )
        return None
    try:
        return sorted(int(b) for b in captured)
    except TypeError:
        logger.warning(
            "cuda_graph_batch_validator: '%s' is not iterable of ints (%r).",
            _CAPTURE_BS_ATTR,
            captured,
        )
        return None


def inspect_model_runner(
    model_runner: object,
    *,
    stage: str,
    buffer_capacity: int | None = None,
) -> CudaGraphBatchReport:
    """Validate a live SGLang ``ModelRunner`` after CUDA graph capture.

    Reads the serving config, captured batch sizes, and request-slot count from
    the running engine, then defers to :func:`evaluate_cuda_graph_batch_sizing`.
    ``buffer_capacity`` is the model-side per-request buffer size when the model
    can report it (e.g. Higgs ``pool_size``); pass ``None`` to skip that check.
    GPU/live-engine only.
    """
    server_args = getattr(model_runner, "server_args", None)
    max_running_requests = getattr(server_args, "max_running_requests", None)
    cuda_graph_max_bs = getattr(server_args, "cuda_graph_max_bs", None)

    req_pool = getattr(model_runner, "req_to_token_pool", None)
    request_slots = getattr(req_pool, "size", None)

    captured_bs = read_captured_bs(model_runner)

    return evaluate_cuda_graph_batch_sizing(
        stage=stage,
        max_running_requests=max_running_requests,
        cuda_graph_max_bs=cuda_graph_max_bs,
        captured_bs=captured_bs,
        request_slots=request_slots,
        buffer_capacity=buffer_capacity,
    )
