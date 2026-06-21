# SPDX-License-Identifier: Apache-2.0
"""TEMP repro: does the pre-capture guard catch the pre-756 buffer overrun?

Runs on the PRE-756 branch, where the Higgs sampler buffer is hard-coded to
_DEFAULT_MAX_BATCH_SIZE = 64 (so the allocated pool is 65) regardless of the
serving cuda_graph_max_bs. We boot the Higgs AR stage with cuda_graph_max_bs
pushed to 128. Expected: SGLModelRunner.init_device_graphs() runs
precapture_guard BEFORE capture, sees predicted capture 128 > buffer 65, and
raises CudaGraphBatchMismatch -- a clear startup error instead of the cryptic
capture-time shape crash the original 756 bug produced.

Usage:
    python pre756_repro.py --model-path boson-sglang/higgs-audio-v3-TTS-4B-grpo05200410999
    python pre756_repro.py --model-path <id> --cuda-graph-max-bs 128
    python pre756_repro.py --model-path <id> --no-guard   # disable guard -> see the raw crash

DELETE this file and this branch after the experiment.
"""

from __future__ import annotations

import argparse
import sys
import traceback


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-path",
        default="boson-sglang/higgs-audio-v3-TTS-4B-grpo05200410999",
    )
    ap.add_argument("--cuda-graph-max-bs", type=int, default=128)
    ap.add_argument(
        "--no-guard",
        action="store_true",
        help="Monkeypatch the guard to a no-op, to observe the raw capture crash.",
    )
    args = ap.parse_args()

    if args.no_guard:
        import sglang_omni.utils.cuda_graph_batch_validator as v

        v.precapture_guard = lambda *a, **k: None
        print(">>> guard DISABLED: expecting the raw capture-time crash <<<\n",
              flush=True)
    else:
        print(">>> guard ENABLED: expecting CudaGraphBatchMismatch before capture <<<\n",
              flush=True)

    from sglang_omni.models.higgs_tts.stages import (
        create_sglang_tts_engine_executor,
    )
    from sglang_omni.utils.cuda_graph_batch_validator import CudaGraphBatchMismatch

    print(f"booting Higgs tts_engine with cuda_graph_max_bs="
          f"{args.cuda_graph_max_bs} (buffer is hard-coded 64 -> pool 65) ...",
          flush=True)
    try:
        create_sglang_tts_engine_executor(
            args.model_path,
            server_args_overrides={"cuda_graph_max_bs": args.cuda_graph_max_bs},
        )
    except CudaGraphBatchMismatch as exc:
        print("\n========================================")
        print("CAUGHT pre-capture, as intended:")
        print(exc)
        print("========================================")
        print("\nRESULT: the validator FLAGGED the 756 overrun before capture.")
        return 0
    except Exception as exc:
        print("\n========================================")
        print(f"Boot failed with a different error: {exc!r}")
        print("========================================")
        traceback.print_exc()
        return 2

    print("\nRESULT: no mismatch raised (capture fit the buffer).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
