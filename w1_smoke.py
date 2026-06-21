# SPDX-License-Identifier: Apache-2.0
"""TEMP: all-stage CUDA graph batch validation smoke driver.

Given a model_path, enumerates EVERY stage of the model's pipeline and prints
one report per stage:

* SGLang-backed stage with CUDA graphs -> the three-way validation report
  (serving config / captured batch sizes / model-side buffer).
* SGLang-backed stage with CUDA graphs disabled -> a "no CUDA graph" note.
* Custom-graph stage (MOSS-TTS-Local vocoder) -> a coverage report.
* Any other stage (preprocessing, encoders, plain vocoders) -> a "no CUDA
  graph" note.

Two SGLang stages cannot be constructed in one process (upstream initializes a
process-global tensor-parallel group, so the second trips
"tensor model parallel group is already initialized"). So each stage is built
in its OWN child subprocess: the driver enumerates stages and re-invokes itself
once per stage with --stage NAME; the child constructs just that stage,
validates it, prints, and exits. One stage's failure never stops the rest.

Usage (lab machine):
    python w1_smoke.py --model higgs           # alias: higgs | moss | moss_local
    python w1_smoke.py --model moss_local
    python w1_smoke.py --model-path <full/hf-id>            # any other checkpoint
    python w1_smoke.py --model higgs --stage tts_engine    # single stage (child mode)

Only higgs / moss / moss_local have public, dependency-complete aliases.
Other models (qwen3_tts needs the qwen-tts package; fishaudio / voxtral need
fine-tuned omni checkpoints) must be passed via --model-path.

REMOVE this file before opening the PR.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import traceback


def _load_config(model_path: str | None, config_path: str | None):
    """Load a PipelineConfig from a yaml config file or a model path.

    A config file (``config_cls`` + ``model_path``) is preferred when available
    because it resolves by class name and avoids HF arch resolution.
    """
    from sglang_omni.config.manager import ConfigManager

    if config_path:
        return ConfigManager.from_file(config_path).config
    return ConfigManager.from_model_path(model_path).config


def _enumerate_stages(model_path: str | None, config_path: str | None):
    """Return (cfg, [(stage_name, factory, is_sglang_or_customgraph)])."""
    from sglang_omni.utils.cuda_graph_batch_validator import _SGLANG_FACTORY_MARKERS

    cfg = _load_config(model_path, config_path)
    stages = []
    for stage in cfg.stages:
        factory = stage.factory or ""
        likely_graph = any(m in factory for m in _SGLANG_FACTORY_MARKERS) or (
            "vocoder" in stage.name and "moss_tts_local" in factory
        )
        stages.append((stage.name, factory, likely_graph))
    return cfg, stages


def _run_one_stage(model_path: str | None, config_path: str | None,
                   stage_name: str) -> int:
    """Child mode: construct ONE stage, validate it, print its report."""
    from sglang_omni.config.runtime import resolve_stage_factory_args
    from sglang_omni.utils.cuda_graph_batch_validator import validate_stage_scheduler
    from sglang_omni.utils.imports import import_string

    cfg = _load_config(model_path, config_path)
    stage = next((s for s in cfg.stages if s.name == stage_name), None)
    if stage is None:
        print(f"[{stage_name}] ERROR: stage not found in pipeline", flush=True)
        return 1

    # The real runner sets the CUDA device before constructing a GPU stage
    # (stage_workers.py: torch.cuda.set_device). Replicate so GPU stages build
    # on gpu 0 in this single-GPU diagnostic.
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_device(0)
    except Exception:
        pass

    print(f"\n######## constructing stage '{stage_name}' "
          f"(factory={stage.factory}) ########", flush=True)
    kwargs = resolve_stage_factory_args(stage, cfg, gpu_id=0)
    factory = import_string(stage.factory)
    scheduler = factory(**kwargs)

    report = validate_stage_scheduler(stage_name, scheduler)
    print(f"\n===== W1 STAGE REPORT [{stage_name}] =====")
    print(report.format())
    print(f"===== END [{stage_name}] =====\n", flush=True)
    return 0


def _run_all_stages(model_path: str | None, config_path: str | None) -> int:
    """Driver mode: enumerate stages, run each in its own child subprocess."""
    source = config_path or model_path
    try:
        cfg, stages = _enumerate_stages(model_path, config_path)
    except (ValueError, FileNotFoundError) as exc:
        print(
            f"\nCould not load a pipeline config for {source!r}: {exc}\n"
            f"Pass a checkpoint whose config.json declares a known omni "
            f"architecture (--model-path), or point at a config file "
            f"(--config examples/configs/<model>.yaml).",
            flush=True,
        )
        return 1

    # Child-mode args to reproduce this exact config in each subprocess.
    src_args = (["--config", config_path] if config_path
                else ["--model-path", model_path])

    print(f"\n######## model {source} ########")
    print(f"pipeline: {type(cfg).__name__}  ({len(stages)} stages)")
    for name, factory, likely_graph in stages:
        tag = "graph?" if likely_graph else "no-graph"
        print(f"  - {name:20s} [{tag}]  {factory}")
    print("######## validating each stage in its own process ########\n",
          flush=True)

    results = {}
    for name, _factory, _lg in stages:
        # Each stage in a fresh process: avoids the SGLang TP-group singleton
        # and isolates failures.
        proc = subprocess.run(
            [sys.executable, __file__, *src_args, "--stage", name],
        )
        results[name] = "ok" if proc.returncode == 0 else f"exit {proc.returncode}"

    print("\n======== ALL-STAGE SUMMARY ========")
    print(f"model: {source}  ({type(cfg).__name__})")
    for name, status in results.items():
        print(f"  {name:20s}: {status}")
    return 0


# Short aliases -> full model path (the real checkpoints the repo's
# examples/configs use). qwen3_tts additionally needs the `qwen-tts` package
# installed; fishaudio/voxtral need their TTS checkpoints present. For models
# whose arch does not resolve from the HF id alone, prefer --config with the
# matching examples/configs/*.yaml (loads via config_cls, no arch resolution).
_MODEL_ALIASES = {
    "higgs": "boson-sglang/higgs-audio-v3-TTS-4B-grpo05200410999",
    "moss_local": "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
    "moss": "OpenMOSS-Team/MOSS-TTS-v1.5",
    "qwen3_tts": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "fishaudio": "fishaudio/s2-pro",
    "voxtral": "mistralai/Voxtral-4B-TTS-2603",
}

# Aliases -> the repo example config (loaded via ConfigManager.from_file, which
# resolves by config_cls and so works even when the bare HF id would not).
_CONFIG_ALIASES = {
    "fishaudio": "examples/configs/s2pro_tts.yaml",
    "voxtral": "examples/configs/voxtral_tts.yaml",
    "qwen3_tts": "examples/configs/qwen3_tts_0_6b.yaml",
}


def _resolve_source(model, model_path, config):
    """Resolve CLI inputs to (model_path, config_path); config wins if given.

    For an alias that has a known example config (fishaudio/voxtral/qwen3_tts),
    prefer the config file -- it loads by config_cls and avoids HF arch
    resolution that the bare id can fail.
    """
    if config:
        return None, config
    if model_path:
        return model_path, None
    if model:
        if model in _CONFIG_ALIASES:
            return None, _CONFIG_ALIASES[model]
        return _MODEL_ALIASES.get(model, model), None
    raise SystemExit(
        "provide --model <alias> (" + ", ".join(_MODEL_ALIASES) + "), "
        "--model-path <path>, or --config <yaml>"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default=None,
        help="Short alias (" + ", ".join(_MODEL_ALIASES) + ") or a literal path/HF id.",
    )
    ap.add_argument(
        "--model-path",
        default=None,
        help="Full model path/HF id.",
    )
    ap.add_argument(
        "--config",
        default=None,
        help="Path to a pipeline config yaml (e.g. examples/configs/voxtral_tts.yaml).",
    )
    ap.add_argument(
        "--stage",
        default=None,
        help="Child mode: construct and validate only this stage.",
    )
    args = ap.parse_args()
    model_path, config_path = _resolve_source(args.model, args.model_path, args.config)

    try:
        if args.stage is not None:
            return _run_one_stage(model_path, config_path, args.stage)
        return _run_all_stages(model_path, config_path)
    except Exception as exc:
        print(f"\n!!!! failed: {exc!r}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
