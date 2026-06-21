# SPDX-License-Identifier: Apache-2.0
"""Validate CUDA graph batch coverage for SGLang-backed generation stages.

Background
----------
A SGLang-backed generation stage has three batch-size facts that must stay
consistent but are configured in different places:

1. **Serving config** -- ``max_running_requests`` and ``cuda_graph_max_bs`` on
   the engine ``server_args``.
2. **Captured CUDA graph batch sizes** -- the list SGLang actually captured
   graphs for, computed by ``get_batch_sizes_to_capture`` and stored on the
   runner's ``graph_runner.capture_bs``.
3. **Model-side buffers** -- the per-request pool/staging tensors a model
   allocates itself (e.g. the Higgs sampler pool, the Qwen3-TTS feedback
   buffer), sized from ``max_running_requests`` at construction.

The class of bug this catches: a model sizes (3) from a stale constant instead
of the live serving config, so a CUDA graph captured for a batch larger than
the buffer overruns it -- the capture either crashes or silently corrupts. This
module reads all three facts from a running stage and flags disagreement, so
the mismatch surfaces at startup instead of at capture time (and so CUDA graph
tuning sweeps have the numbers they need).

Unit of validation
------------------
The unit is a **stage**, not a model. A model is a pipeline of stages, and only
some capture CUDA graphs. A single pipeline can host more than one SGLang stage
-- e.g. Qwen3-Omni speech runs both a ``thinker`` and a ``talker_ar`` engine,
each in its own process -- so validation is keyed on the stage and each is
validated independently and named explicitly. :func:`sglang_stage_names`
enumerates a pipeline config's SGLang stages, and :func:`validate_stage_scheduler`
is the uniform per-stage entry point: every stage worker may call it, and it
returns the right thing for that stage (or ``None`` to skip).

Scope across stage types (from a repo-wide audit):

* **SGLang-backed generation stages** -- the primary target. The three-way check
  (serving config / captured batch sizes / model-side buffer) applies. Covers
  every model's TTS/thinker/talker AR stage.
* **Custom-graph stages** -- only the MOSS-TTS-Local streaming vocoder captures
  its own CUDA graphs. It captures over per-step frame count ``T`` against a
  fixed slot pool (not request batch vs ``max_running_requests``), and is
  overrun-guarded, so it gets a distinct :class:`CustomGraphReport` (coverage
  only), not the SGLang batch-overrun check.
* **All other stages** -- preprocessing, encoders, aggregation, plain vocoders,
  code2wav -- capture no CUDA graphs and allocate no fixed-batch buffers.

Every stage is reported, never silently skipped: a stage with no CUDA graph (a
plain stage, or an SGLang stage run with graphs disabled) gets an explicit
:class:`NoGraphReport` ("no CUDA graph enabled; nothing to flag"). So a full
model scan via :func:`validate_stages` yields one independently-flagged report
per stage regardless of stage type.

Layout
------
The verdict logic (:func:`evaluate_cuda_graph_batch_sizing`) is pure -- no
SGLang or GPU dependency -- and is unit tested on CPU. The introspection layer
(:func:`read_captured_bs`, :func:`read_model_buffer_capacity`,
:func:`validate_stage`) reads a *live* runner after capture and only yields
meaningful values on a GPU server. Every read is defensive: a missing or
renamed attribute degrades to a warning and a partial report rather than a
crash. :func:`sglang_stage_names` is config-side (no GPU) and duck-typed (no
config-package import), so it too is CPU-testable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# Upstream SGLang attribute names this module depends on, kept in one place so a
# version bump that renames them is easy to find and patch.
#   model_runner.graph_runner            -> the CudaGraphRunner instance
#   graph_runner.capture_bs              -> list[int] of captured batch sizes
#   model_runner.req_to_token_pool.size  -> request-slot count capture_bs is
#                                           clamped to
#   model_runner.model                   -> the model nn.Module
_RUNNER_GRAPH_ATTR = "graph_runner"
_CAPTURE_BS_ATTR = "capture_bs"


@dataclass(frozen=True)
class _BufferProbe:
    """Recipe for reading one model's allocated per-request buffer first dim.

    ``tensor_paths`` are dotted attribute paths, evaluated from the model
    nn.Module, each resolving to a live tensor whose ``shape[0]`` is the
    allocated capacity. The first path that resolves wins. Reading the real
    tensor (rather than recomputing from config) is what lets the validator
    catch a buffer hard-coded below the serving config.
    """

    tensor_paths: tuple[str, ...]
    note: str = ""


# Per-model buffer recipes, keyed by ``type(model).__name__``. Each model sizes
# its per-request buffer from ``max_running_requests`` at construction; the
# allocated first dim equals max_running_requests, except Higgs which adds one
# reserved padding row (pool_size = max_running_requests + 1). An unregistered
# model yields no buffer reading, and the report falls back to config-vs-
# captured validation only.
_BUFFER_PROBES: dict[str, _BufferProbe] = {
    "HiggsTTSModel": _BufferProbe(
        ("_sampler_pool.seeds", "_cg_codes_BN", "_cg_active_last_codes"),
        note="sampler pool = max_running_requests + 1 (one reserved padding row)",
    ),
    "Qwen3TTSTalker": _BufferProbe(("_feedback_buffer",)),
    "MossTTSDelaySGLangModel": _BufferProbe(("_decode_input_embedding.weight",)),
    "MossTTSLocalSGLangModel": _BufferProbe(("_decode_input_embedding.weight",)),
    "S2ProSGLangTextModel": _BufferProbe(
        ("_vq_codes",),
        note="allocated only after setup_vq_decode()",
    ),
    "VoxtralSGLangTTSModel": _BufferProbe(("_decode_input_embed_buffer",)),
    "Qwen3OmniTalker": _BufferProbe(("_feedback_buffer",)),
}


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
    # (3) model-side buffer capacity (first dim of the allocated buffer)
    buffer_capacity: int | None
    # where (3) was read from, or why it could not be read
    buffer_source: str | None
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
        lines.append(
            f"  captured graph bs:  {self.captured_bs} "
            f"(max captured = {self.max_captured_bs})"
        )
        lines.append(f"  request slots:      {self.request_slots}")
        lines.append(
            f"  model-side buffers: {self.buffer_capacity} "
            f"[{self.buffer_source}]"
        )
        lines.append(f"  VERDICT: {'OK' if self.ok else 'MISMATCH'}")
        for finding in self.findings:
            lines.append(f"    - {finding}")
        return "\n".join(lines)


def evaluate_cuda_graph_batch_sizing(
    *,
    stage: str,
    max_running_requests: int | None,
    cuda_graph_max_bs: int | None,
    captured_bs: list[int] | None,
    request_slots: int | None,
    buffer_capacity: int | None,
    buffer_source: str | None = None,
) -> CudaGraphBatchReport:
    """Compare the three batch-size facts and produce a verdict.

    Pure function -- no SGLang/GPU dependency. ``buffer_capacity`` may be
    ``None`` when the model-side buffer could not be read; the verdict then
    validates only what is observable (config vs. captured sizes) and notes the
    gap.
    """
    findings: list[str] = []
    ok = True

    captured = [b for b in (captured_bs or []) if b is not None]
    max_captured = max(captured) if captured else None

    # The load-bearing check: graphs captured for a batch larger than the
    # model's per-request buffer will overrun it on replay.
    if buffer_capacity is not None and max_captured is not None:
        if max_captured > buffer_capacity:
            ok = False
            findings.append(
                f"graphs captured up to bs={max_captured} but model-side "
                f"buffer holds {buffer_capacity}; capture/replay above "
                f"{buffer_capacity} overruns the model buffer."
            )
    elif buffer_capacity is None:
        findings.append(
            f"model-side buffer not read ({buffer_source}); validated serving "
            f"config vs. captured sizes only."
        )

    # A model buffer smaller than the admission limit cannot serve peak
    # concurrency, regardless of capture coverage.
    if (
        buffer_capacity is not None
        and max_running_requests is not None
        and buffer_capacity < max_running_requests
    ):
        ok = False
        findings.append(
            f"model-side buffer ({buffer_capacity}) is smaller than "
            f"max_running_requests ({max_running_requests}); peak concurrency "
            f"cannot be served."
        )

    if ok and not findings:
        findings.append(
            "captured sizes and model-side buffer track the serving config."
        )

    return CudaGraphBatchReport(
        stage=stage,
        max_running_requests=max_running_requests,
        cuda_graph_max_bs=cuda_graph_max_bs,
        captured_bs=captured or None,
        request_slots=request_slots,
        buffer_capacity=buffer_capacity,
        buffer_source=buffer_source,
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
    except (TypeError, ValueError):
        logger.warning(
            "cuda_graph_batch_validator: '%s' is not an iterable of ints (%r).",
            _CAPTURE_BS_ATTR,
            captured,
        )
        return None


def _resolve_tensor_first_dim(obj: object, path: str) -> int | None:
    """Walk a dotted attribute ``path`` from ``obj`` to a tensor's first dim."""
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    shape = getattr(cur, "shape", None)
    if shape is None or len(shape) == 0:
        return None
    try:
        return int(shape[0])
    except (TypeError, ValueError):
        return None


def read_model_buffer_capacity(model: object) -> tuple[int | None, str]:
    """Read the model's allocated per-request buffer first dim.

    Returns ``(capacity, source)`` where ``source`` names the attribute path
    that matched, or explains why none did. Model-agnostic via the per-class
    :data:`_BUFFER_PROBES` registry: an unregistered model yields
    ``(None, ...)`` so validation degrades to config-vs-captured only.
    GPU/live-engine only.
    """
    if model is None:
        return None, "no model object on runner"

    cls = type(model).__name__
    probe = _BUFFER_PROBES.get(cls)
    if probe is None:
        return None, f"no buffer probe registered for model class {cls!r}"

    # Some models alias their buffer up to the top-level module; others keep it
    # on the inner ``.model`` submodule. Try both.
    candidates = [model]
    inner = getattr(model, "model", None)
    if inner is not None and inner is not model:
        candidates.append(inner)

    for obj in candidates:
        for path in probe.tensor_paths:
            dim = _resolve_tensor_first_dim(obj, path)
            if dim is not None:
                where = "model" if obj is model else "model.model"
                source = f"{where}.{path}.shape[0]"
                if probe.note:
                    source += f" ({probe.note})"
                return dim, source

    return None, (
        f"model class {cls!r} registered but none of its buffer paths "
        f"resolved ({', '.join(probe.tensor_paths)}); buffer may not be "
        f"allocated yet"
    )


def validate_stage(
    stage_name: str,
    model_runner: object,
    *,
    buffer_capacity: int | None = None,
) -> CudaGraphBatchReport:
    """Validate one SGLang-backed stage after its CUDA graph capture.

    ``stage_name`` is the pipeline stage this runner belongs to (e.g.
    ``"tts_engine"``, ``"thinker"``, ``"talker_ar"``); it names the report so a
    multi-stage pipeline produces one clearly-labelled report per stage. Reads
    all three batch-size facts from the running engine -- serving config,
    actually-captured batch sizes, and the model-side buffer capacity (auto-read
    via the probe registry; pass ``buffer_capacity`` to override) -- then defers
    to :func:`evaluate_cuda_graph_batch_sizing`. GPU/live-engine only.
    """
    server_args = getattr(model_runner, "server_args", None)
    max_running_requests = getattr(server_args, "max_running_requests", None)
    cuda_graph_max_bs = getattr(server_args, "cuda_graph_max_bs", None)

    req_pool = getattr(model_runner, "req_to_token_pool", None)
    request_slots = getattr(req_pool, "size", None)

    captured_bs = read_captured_bs(model_runner)

    model = getattr(model_runner, "model", None)
    if buffer_capacity is None:
        buffer_capacity, buffer_source = read_model_buffer_capacity(model)
    else:
        buffer_source = "caller-provided"

    model_cls = type(model).__name__ if model is not None else "unknown-model"

    return evaluate_cuda_graph_batch_sizing(
        stage=f"{stage_name} ({model_cls})",
        max_running_requests=max_running_requests,
        cuda_graph_max_bs=cuda_graph_max_bs,
        captured_bs=captured_bs,
        request_slots=request_slots,
        buffer_capacity=buffer_capacity,
        buffer_source=buffer_source,
    )


# Factory dotted-path fragments that mark a stage as SGLang-backed (each such
# factory calls scheduling.bootstrap.create_sglang_infrastructure). Used to
# enumerate the SGLang stages of a pipeline without importing the config
# package -- the role maps alone miss the thinker/ASR stages, which have no
# generation/talker role entry.
_SGLANG_FACTORY_MARKERS = (
    "create_sglang",
    "_thinker_executor_from_config",
    "_talker_ar_executor_from_config",
    "create_generation_executor",
)


def sglang_stage_names(pipeline_config: object) -> list[str]:
    """Enumerate the SGLang-backed stage names of a pipeline config.

    Duck-typed: ``pipeline_config`` need only expose a ``stages`` iterable whose
    items have ``.name`` and ``.factory`` (so this stays CPU-testable and avoids
    importing the config package). A stage is SGLang-backed when its factory
    dotted-path contains one of :data:`_SGLANG_FACTORY_MARKERS`. This catches
    every SGLang stage -- including the ones with no generation/talker role map
    (thinker, ASR) and the two distinct stages of the Qwen3-Omni speech pipeline
    (``thinker`` and ``talker_ar``) -- so callers validate each rather than a
    single guessed "generation" stage.
    """
    stages = getattr(pipeline_config, "stages", None) or []
    names: list[str] = []
    for stage in stages:
        name = getattr(stage, "name", None)
        factory = getattr(stage, "factory", "") or ""
        if name and any(marker in factory for marker in _SGLANG_FACTORY_MARKERS):
            names.append(name)
    return names


@dataclass(frozen=True)
class CustomGraphReport:
    """CUDA graph coverage for a non-SGLang stage that captures its own graphs.

    Such a stage (only the MOSS-TTS-Local streaming vocoder today) captures
    graphs over a different axis than SGLang request batch -- per-step frame
    count ``T`` against a fixed slot pool -- so it gets its own report shape
    rather than the SGLang three-way comparison. This is a coverage/visibility
    report: it states what was captured and the fixed slot capacity, so the
    stage is not silently invisible to CUDA graph tuning.
    """

    stage: str
    captured_frames: list[int] | None
    slot_capacity: int | None
    ok: bool = True  # coverage report; no failure condition defined for this axis
    findings: list[str] = field(default_factory=list)

    def format(self) -> str:
        lines = [f"Stage: {self.stage} (custom graph)"]
        lines.append(f"  captured frames (T): {self.captured_frames}")
        lines.append(f"  slot capacity:       {self.slot_capacity}")
        lines.append("  VERDICT: OK (coverage report)")
        for finding in self.findings:
            lines.append(f"    - {finding}")
        return "\n".join(lines)


@dataclass(frozen=True)
class NoGraphReport:
    """A stage that has no CUDA graph to validate.

    Emitted -- rather than silently skipping -- so every stage of every model
    produces an explicit, independently-flagged output: either it was validated,
    or it is on record as having no CUDA graph enabled (nothing to flag). Covers
    plain stages (preprocessing, encoders, plain vocoders) and SGLang stages run
    with CUDA graphs disabled.
    """

    stage: str
    reason: str = ""
    ok: bool = True  # nothing to flag

    def format(self) -> str:
        lines = [f"Stage: {self.stage} (no CUDA graph)"]
        lines.append("  VERDICT: OK (no CUDA graph enabled; nothing to flag)")
        if self.reason:
            lines.append(f"    - {self.reason}")
        return "\n".join(lines)


def _read_custom_graph_report(stage_name: str, scheduler: object) -> CustomGraphReport | None:
    """Read CUDA graph coverage from a custom-graph (non-SGLang) scheduler.

    Currently recognises the MOSS-TTS-Local streaming vocoder, whose
    ``scheduler._session`` exposes ``has_cuda_graph_runner()`` /
    ``captured_frames()`` and a fixed slot pool. Returns ``None`` if the
    scheduler is not a recognised custom-graph stage or captured no graphs.
    """
    session = getattr(scheduler, "_session", None)
    has_runner = getattr(session, "has_cuda_graph_runner", None)
    if session is None or not callable(has_runner):
        return None
    if not has_runner():
        return None

    captured_fn = getattr(session, "captured_frames", None)
    captured = sorted(captured_fn()) if callable(captured_fn) else None
    slot_capacity = getattr(session, "_batch_size", None)

    findings: list[str] = []
    if captured:
        findings.append(
            f"captured {len(captured)} frame-size graph(s) up to T={max(captured)}; "
            f"uncaptured frame sizes fall back to eager."
        )
    findings.append(
        f"graphs are keyed by frame count T against a fixed slot pool "
        f"({slot_capacity}); batch is bounded by the slot allocator, not by a "
        f"serving max_running_requests, so the SGLang batch-overrun check does "
        f"not apply here."
    )
    return CustomGraphReport(
        stage=f"{stage_name} ({type(scheduler).__name__})",
        captured_frames=captured,
        slot_capacity=slot_capacity,
        findings=findings,
    )


StageReport = "CudaGraphBatchReport | CustomGraphReport | NoGraphReport"


def _resolve_sglang_model_runner(scheduler: object) -> object | None:
    """Find the live SGLang ``ModelRunner`` from a stage scheduler.

    Stage schedulers expose the SGLang runner via different shapes:

    * ``OmniScheduler`` / ``QwenTalkerScheduler`` / ``DllmScheduler`` set
      ``tp_worker`` (the ``ModelWorker``), so the runner is
      ``scheduler.tp_worker.model_runner``.
    * ``FishScheduler`` has no ``tp_worker``; it holds a model-specific runner
      at ``scheduler._model_runner`` whose ``.tp_worker.model_runner`` is the
      SGLang runner.

    Returns the SGLang ``ModelRunner`` (the object carrying ``graph_runner`` /
    ``server_args`` / ``model``), or ``None`` for a non-SGLang scheduler.
    """
    # Direct: scheduler.tp_worker.model_runner (OmniScheduler family).
    tp_worker = getattr(scheduler, "tp_worker", None)
    runner = getattr(tp_worker, "model_runner", None)
    if runner is not None:
        return runner

    # Wrapped: scheduler._model_runner.tp_worker.model_runner (FishScheduler).
    inner = getattr(scheduler, "_model_runner", None)
    inner_tp = getattr(inner, "tp_worker", None)
    runner = getattr(inner_tp, "model_runner", None)
    if runner is not None:
        return runner

    return None


def validate_stage_scheduler(
    stage_name: str,
    scheduler: object,
):
    """Validate one stage from its scheduler; the per-stage entry point.

    Always returns a report -- never ``None`` -- so every stage of every model
    produces an explicit, independently-flagged output. Each stage worker calls
    this with its scheduler, uniformly:

    * **SGLang stage with CUDA graphs** (``tp_worker.model_runner`` present and
      a graph was captured): full :class:`CudaGraphBatchReport` three-way check.
    * **SGLang stage with CUDA graphs disabled** (runner present but no captured
      graph): :class:`NoGraphReport` -- nothing to flag.
    * **Custom-graph stage** (the MOSS-TTS-Local streaming vocoder): a
      :class:`CustomGraphReport` of its frame-coverage and slot capacity.
    * **Any other stage** (preprocessing, encoders, plain vocoders): a
      :class:`NoGraphReport` -- on record as having no CUDA graph.

    GPU/live-engine only.
    """
    model_runner = _resolve_sglang_model_runner(scheduler)
    if model_runner is not None:
        # SGLang-backed stage. Distinguish graphs-captured from graphs-disabled
        # so a disabled stage is recorded rather than reported as a mismatch.
        if read_captured_bs(model_runner) is None:
            model = getattr(model_runner, "model", None)
            model_cls = type(model).__name__ if model is not None else "unknown-model"
            return NoGraphReport(
                stage=f"{stage_name} ({model_cls})",
                reason="SGLang stage with no captured CUDA graph "
                "(cuda_graph disabled or capture skipped).",
            )
        return validate_stage(stage_name, model_runner)

    custom = _read_custom_graph_report(stage_name, scheduler)
    if custom is not None:
        return custom

    return NoGraphReport(
        stage=f"{stage_name} ({type(scheduler).__name__})",
        reason="stage captures no CUDA graphs.",
    )


def validate_stages(stage_schedulers):
    """Validate every stage of a model and return one report per stage.

    ``stage_schedulers`` is an iterable of ``(stage_name, scheduler)`` pairs --
    all the stages discovered for a given model. Returns a list of reports, one
    per stage, in order. Every stage is represented: SGLang stages get the
    three-way check, the custom-graph vocoder gets a coverage report, and any
    stage with no CUDA graph gets a :class:`NoGraphReport` -- so the result is a
    comprehensive, independently-flagged scan of the whole model.

    This is the model-level view. At production runtime a model's stages run in
    separate processes, so each process validates its own stage via
    :func:`validate_stage_scheduler` and the logs aggregate to this same
    per-stage coverage; this driver is for contexts (offline tools, tests) where
    the stages are reachable together.
    """
    return [
        validate_stage_scheduler(stage_name, scheduler)
        for stage_name, scheduler in stage_schedulers
    ]


class CudaGraphBatchMismatch(RuntimeError):
    """Raised pre-capture when the captured batch sizes would overrun a buffer.

    Surfaces the overrun as a clear, actionable error BEFORE CUDA graph capture
    runs, instead of the cryptic shape crash capture would otherwise produce.
    """


def _predicted_cuda_graph_max_bs(model_runner: object) -> int | None:
    """Predict the largest batch size CUDA graph capture will attempt.

    Reads the configured graph batch sizes (defensive across SGLang versions:
    ``server_args.cuda_graph_bs`` or ``server_args.cuda_graph_config.decode.bs``)
    and clamps to the request-slot count, mirroring upstream
    ``get_batch_sizes_to_capture``. Returns ``None`` if it cannot be predicted.
    """
    server_args = getattr(model_runner, "server_args", None)
    if server_args is None:
        return None

    bs_list = getattr(server_args, "cuda_graph_bs", None)
    if bs_list is None:
        cg_cfg = getattr(server_args, "cuda_graph_config", None)
        decode = getattr(cg_cfg, "decode", None)
        bs_list = getattr(decode, "bs", None)
    cap = getattr(server_args, "cuda_graph_max_bs", None)

    candidates = []
    if bs_list:
        try:
            candidates.extend(int(b) for b in bs_list)
        except (TypeError, ValueError):
            pass
    if cap is not None:
        try:
            candidates.append(int(cap))
        except (TypeError, ValueError):
            pass
    if not candidates:
        return None

    predicted = max(candidates)

    # Upstream clamps capture sizes to the request-slot count.
    req_pool = getattr(model_runner, "req_to_token_pool", None)
    slots = getattr(req_pool, "size", None)
    if isinstance(slots, int) and slots > 0:
        predicted = min(predicted, slots)
    return predicted


def precapture_guard(model_runner: object, *, stage_name: str = "") -> None:
    """Fail fast BEFORE CUDA graph capture if it would overrun the model buffer.

    Compares the predicted max capture size against the model-side buffer
    (read via the probe registry) and raises :class:`CudaGraphBatchMismatch`
    when capture would overrun -- turning the 756-class capture crash into a
    clear startup error. No-ops (returns) when it cannot predict either side,
    so it never blocks a legitimate capture. Call this just before the real
    capture (e.g. in a ``ModelRunner.init_device_graphs`` override, before
    ``super()``). GPU/live-engine only.
    """
    predicted = _predicted_cuda_graph_max_bs(model_runner)
    if predicted is None:
        return

    model = getattr(model_runner, "model", None)
    buffer_capacity, buffer_source = read_model_buffer_capacity(model)
    if buffer_capacity is None:
        return  # model-side buffer not allocated yet / not registered

    if predicted > buffer_capacity:
        label = stage_name or (type(model).__name__ if model is not None else "stage")
        raise CudaGraphBatchMismatch(
            f"[{label}] CUDA graph capture would attempt batch size "
            f"{predicted}, but the model-side buffer holds {buffer_capacity} "
            f"({buffer_source}). Capture above {buffer_capacity} overruns the "
            f"buffer. Lower cuda_graph_max_bs to <= {buffer_capacity}, or size "
            f"the model buffer from max_running_requests."
        )
