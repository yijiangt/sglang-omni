# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the CUDA graph batch validator.

These exercise the pure verdict logic and the introspection layer against a
mocked SGLang ``ModelRunner`` (``SimpleNamespace``), so the whole suite runs
without a GPU or a real sglang install. The live-GPU path is exercised on the
lab machine separately.
"""

from __future__ import annotations

from types import SimpleNamespace

from sglang_omni.utils.cuda_graph_batch_validator import (
    evaluate_cuda_graph_batch_sizing,
    inspect_model_runner,
    read_captured_bs,
)


def _fake_runner(
    *,
    max_running_requests=64,
    cuda_graph_max_bs=64,
    capture_bs=None,
    request_slots=64,
    has_graph_runner=True,
):
    graph_runner = SimpleNamespace(capture_bs=capture_bs) if has_graph_runner else None
    return SimpleNamespace(
        server_args=SimpleNamespace(
            max_running_requests=max_running_requests,
            cuda_graph_max_bs=cuda_graph_max_bs,
        ),
        req_to_token_pool=SimpleNamespace(size=request_slots),
        graph_runner=graph_runner,
    )


# --- pure verdict logic ---------------------------------------------------


def test_consistent_sizing_is_ok():
    report = evaluate_cuda_graph_batch_sizing(
        stage="tts_engine",
        max_running_requests=64,
        cuda_graph_max_bs=64,
        captured_bs=[1, 2, 4, 8, 16, 32, 64],
        request_slots=64,
        buffer_capacity=65,  # pool_size = mrr + 1
    )
    assert report.ok
    assert report.max_captured_bs == 64


def test_captured_exceeds_buffers_is_the_756_bug():
    # graphs captured up to 128 but model buffers sized for 64 -> overrun
    report = evaluate_cuda_graph_batch_sizing(
        stage="tts_engine",
        max_running_requests=64,
        cuda_graph_max_bs=128,
        captured_bs=[1, 16, 64, 128],
        request_slots=128,
        buffer_capacity=65,
    )
    assert not report.ok
    assert any("overrun model buffers" in f for f in report.findings)


def test_buffers_below_admission_limit_flagged():
    report = evaluate_cuda_graph_batch_sizing(
        stage="tts_engine",
        max_running_requests=64,
        cuda_graph_max_bs=64,
        captured_bs=[1, 16, 32],
        request_slots=64,
        buffer_capacity=32,  # smaller than mrr
    )
    assert not report.ok
    assert any("cannot be served" in f for f in report.findings)


def test_clamped_cap_is_noted_not_failed():
    # configured cap 64 but capture clamped to 16 by request slots -> note, not failure
    report = evaluate_cuda_graph_batch_sizing(
        stage="talker_ar",
        max_running_requests=16,
        cuda_graph_max_bs=64,
        captured_bs=[1, 2, 4, 8, 16],
        request_slots=16,
        buffer_capacity=17,
    )
    assert report.ok
    assert any("clamped" in f for f in report.findings)


def test_missing_buffer_probe_validates_partial():
    report = evaluate_cuda_graph_batch_sizing(
        stage="voxtral_tts",
        max_running_requests=16,
        cuda_graph_max_bs=16,
        captured_bs=[1, 8, 16],
        request_slots=16,
        buffer_capacity=None,
    )
    assert report.ok
    assert any("no buffer-capacity probe" in f for f in report.findings)


# --- introspection layer (mocked runner) ----------------------------------


def test_inspect_reads_live_runner_fields():
    runner = _fake_runner(capture_bs=[1, 2, 4, 8, 16, 32, 64])
    report = inspect_model_runner(runner, stage="tts_engine", buffer_capacity=65)
    assert report.ok
    assert report.captured_bs == [1, 2, 4, 8, 16, 32, 64]
    assert report.request_slots == 64
    assert report.max_running_requests == 64


def test_read_captured_bs_sorts_and_ints():
    runner = _fake_runner(capture_bs=[64, 1, 16, 4])
    assert read_captured_bs(runner) == [1, 4, 16, 64]


def test_read_captured_bs_missing_graph_runner_returns_none():
    runner = _fake_runner(has_graph_runner=True, capture_bs=None)
    # graph_runner present but capture not populated
    assert read_captured_bs(runner) is None


def test_read_captured_bs_no_graph_runner_returns_none():
    runner = _fake_runner(has_graph_runner=False)
    assert read_captured_bs(runner) is None


def test_format_is_multiline_and_labeled():
    runner = _fake_runner(capture_bs=[1, 16, 64])
    out = inspect_model_runner(runner, stage="tts_engine", buffer_capacity=65).format()
    assert "Stage: tts_engine" in out
    assert "VERDICT:" in out
