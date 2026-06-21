# SPDX-License-Identifier: Apache-2.0
"""TEMP: single-process per-model CUDA graph batch validator.

For each requested model, boots its real SGLang generation stage IN THIS
PROCESS (no router, no spawned workers, no separate filesystem namespace) by
calling the model's own stage factory, captures the live ``model_runner`` via a
monkeypatch on ``create_sglang_infrastructure``, runs the validator, and prints
the report + raw upstream fields to stdout.

This proves the validator is model-agnostic: each model boots exactly as in
production, and the same introspection path is exercised for all of them.

Usage (lab machine, in the venv that has sglang):
    .venv/bin/python3 w1_smoke.py --models higgs qwen3_tts moss fishaudio voxtral
    .venv/bin/python3 w1_smoke.py --models higgs --model-path <override>

REMOVE this file before opening the W1 PR.
"""

from __future__ import annotations

import argparse
import sys
import traceback

# Each entry: factory module, factory function name, default HF model path.
# All generation factories take model_path as the first positional arg.
MODELS = {
    "higgs": (
        "sglang_omni.models.higgs_tts.stages",
        "create_sglang_tts_engine_executor",
        "boson-sglang/higgs-audio-v3-TTS-4B-grpo05200410999",
    ),
    "qwen3_tts": (
        "sglang_omni.models.qwen3_tts.stages",
        "create_sglang_tts_engine_executor",
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",  # override with --model-path as needed
    ),
    "moss": (
        "sglang_omni.models.moss_tts.stages",
        "create_sglang_tts_engine_executor",
        "OpenMOSS-Team/MOSS-TTS-v1.5",
    ),
    "moss_local": (
        "sglang_omni.models.moss_tts_local.stages",
        "create_sglang_tts_engine_executor",
        "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
    ),
    "fishaudio": (
        "sglang_omni.models.fishaudio_s2_pro.stages",
        "create_sglang_tts_engine_executor",
        "fishaudio/fish-speech-1.5",  # override with --model-path as needed
    ),
    "voxtral": (
        "sglang_omni.models.voxtral_tts.pipeline.stages",
        "create_generation_executor",
        "mistralai/Voxtral-Mini-3B-2507",  # override with --model-path as needed
    ),
}


def _run_one(name: str, model_path: str) -> None:
    import importlib

    import sglang_omni.scheduling.bootstrap as bootstrap
    from sglang_omni.utils.cuda_graph_batch_validator import (
        read_model_buffer_capacity,
        validate_stage,
    )

    mod_name, fn_name, _default = MODELS[name]
    factory = getattr(importlib.import_module(mod_name), fn_name)

    # Monkeypatch create_sglang_infrastructure to capture the model_worker,
    # regardless of which model factory calls it or how it wraps the result.
    captured = {}
    orig = bootstrap.create_sglang_infrastructure

    def _patched(*args, **kwargs):
        result = orig(*args, **kwargs)
        captured["model_worker"] = result[0]
        return result

    bootstrap.create_sglang_infrastructure = _patched
    # Some factories import the symbol into their own module namespace.
    factory_mod = importlib.import_module(mod_name)
    had_local = hasattr(factory_mod, "create_sglang_infrastructure")
    if had_local:
        factory_mod.create_sglang_infrastructure = _patched

    print(f"\n######## booting {name} ({model_path}) ########", flush=True)
    try:
        factory(model_path)
    finally:
        bootstrap.create_sglang_infrastructure = orig
        if had_local:
            factory_mod.create_sglang_infrastructure = orig

    mw = captured.get("model_worker")
    if mw is None:
        print(f"[{name}] ERROR: model_worker was never captured "
              f"(factory did not call create_sglang_infrastructure).", flush=True)
        return

    # buffer_capacity omitted -> the validator auto-reads the model-side buffer
    # via its per-model probe registry (the path being validated here).
    report = validate_stage(f"w1-smoke:{name}", mw.model_runner)
    print(f"\n===== W1 VALIDATOR [{name}] =====")
    print(report.format())
    mr = mw.model_runner
    gr = getattr(mr, "graph_runner", "<no graph_runner attr>")
    print("raw model class:", type(getattr(mr, "model", None)).__name__)
    print("raw graph_runner type:", type(gr).__name__)
    print("raw capture_bs:", getattr(gr, "capture_bs", "<no capture_bs attr>"))
    print("raw req_to_token_pool.size:",
          getattr(getattr(mr, "req_to_token_pool", None), "size", "<none>"))
    cap, src = read_model_buffer_capacity(getattr(mr, "model", None))
    print(f"raw model-side buffer: {cap}  [{src}]")
    print(f"===== END W1 VALIDATOR [{name}] =====\n", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        nargs="+",
        default=["higgs"],
        choices=list(MODELS),
        help="Which models to boot and validate.",
    )
    ap.add_argument(
        "--model-path",
        default=None,
        help="Override the HF model path (only valid with a single --models).",
    )
    args = ap.parse_args()

    if args.model_path and len(args.models) != 1:
        ap.error("--model-path requires exactly one --models entry")

    results = {}
    for name in args.models:
        path = args.model_path or MODELS[name][2]
        try:
            _run_one(name, path)
            results[name] = "ran"
        except Exception as exc:  # keep going to the next model
            results[name] = f"FAILED: {exc!r}"
            print(f"\n!!!! {name} boot failed !!!!", flush=True)
            traceback.print_exc()

    print("\n======== SUMMARY ========")
    for name, status in results.items():
        print(f"  {name}: {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
