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
    python w1_smoke.py --model-path boson-sglang/higgs-audio-v3-TTS-4B-grpo05200410999
    python w1_smoke.py --model-path <id> --stage tts_engine   # single stage (child mode)

REMOVE this file before opening the PR.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import traceback


def _enumerate_stages(model_path: str):
    """Return the model's [(stage_name, factory, is_sglang_or_customgraph)] list."""
    from sglang_omni.config.manager import ConfigManager
    from sglang_omni.utils.cuda_graph_batch_validator import _SGLANG_FACTORY_MARKERS

    cfg = ConfigManager.from_model_path(model_path).config
    stages = []
    for stage in cfg.stages:
        factory = stage.factory or ""
        likely_graph = any(m in factory for m in _SGLANG_FACTORY_MARKERS) or (
            "vocoder" in stage.name and "moss_tts_local" in factory
        )
        stages.append((stage.name, factory, likely_graph))
    return cfg, stages


def _run_one_stage(model_path: str, stage_name: str) -> int:
    """Child mode: construct ONE stage, validate it, print its report."""
    from sglang_omni.config.manager import ConfigManager
    from sglang_omni.config.runtime import resolve_stage_factory_args
    from sglang_omni.utils.cuda_graph_batch_validator import validate_stage_scheduler
    from sglang_omni.utils.imports import import_string

    cfg = ConfigManager.from_model_path(model_path).config
    stage = next((s for s in cfg.stages if s.name == stage_name), None)
    if stage is None:
        print(f"[{stage_name}] ERROR: stage not found in pipeline", flush=True)
        return 1

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


def _run_all_stages(model_path: str) -> int:
    """Driver mode: enumerate stages, run each in its own child subprocess."""
    cfg, stages = _enumerate_stages(model_path)
    print(f"\n######## model {model_path} ########")
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
            [sys.executable, __file__, "--model-path", model_path, "--stage", name],
        )
        results[name] = "ok" if proc.returncode == 0 else f"exit {proc.returncode}"

    print("\n======== ALL-STAGE SUMMARY ========")
    print(f"model: {model_path}  ({type(cfg).__name__})")
    for name, status in results.items():
        print(f"  {name:20s}: {status}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument(
        "--stage",
        default=None,
        help="Child mode: construct and validate only this stage.",
    )
    args = ap.parse_args()

    try:
        if args.stage is not None:
            return _run_one_stage(args.model_path, args.stage)
        return _run_all_stages(args.model_path)
    except Exception as exc:
        print(f"\n!!!! failed: {exc!r}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
