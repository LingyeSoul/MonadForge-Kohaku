"""Tests for the stdlib-only MOSS training report CLI.

Builds tiny synthetic progress streams + trainer logs in tmp_path, so the
suite touches no torch, no GPU and no real run.
"""

from __future__ import annotations

import json
import math
import struct
import time
from itertools import pairwise

import pytest

from scripts.moss import train_report as tr


def _write_jsonl(path, events):
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(rec) + "\n" for rec in events)


def _linear_events(n=12, total=20, rate=0.05, ts0=10.0, pid=1):
    """A plain descending-loss run: loss_i = 1.0 - rate*i, ts_i = ts0*(i+1)."""
    events = [
        {
            "ev": "run_start",
            "ts": 0.0,
            "run": "test_run",
            "method": "moss",
            "preset": None,
            "total_steps": total,
            "total_epochs": 2,
            "pid": pid,
            "sampling_enabled": False,
        }
    ]
    for i in range(n):
        events.append(
            {
                "ev": "step",
                "ts": ts0 * (i + 1),
                "loss": 1.0 - rate * i,
                "sigma": 0.5,
                "lr": 1e-5,
                "samples": 1,
                "grad_norm": 0.5,
                "global_step": i + 1,
                "epoch": 1,
            }
        )
    return events


def test_trailing_mean_includes_partial_warmup():
    assert tr.trailing_mean([1.0, 2.0, 3.0], 2) == [1.0, 1.5, 2.5]
    assert tr.trailing_mean([1.0, 2.0, 3.0], 3) == [1.0, 1.5, 2.0]
    with pytest.raises(ValueError):
        tr.trailing_mean([1.0], 0)


def test_deciles_are_a_contiguous_partition():
    points = [(i + 1, 1.0 - 0.01 * i) for i in range(12)]
    blocks = tr.loss_deciles(points, 10)
    assert len(blocks) == 10
    assert sum(b["count"] for b in blocks) == 12
    assert blocks[0]["step_from"] == 1
    assert blocks[-1]["step_to"] == 12
    for prev, cur in pairwise(blocks):
        assert cur["step_from"] == prev["step_to"] + 1
    for block in blocks:
        chunk = [
            loss
            for step, loss in points
            if block["step_from"] <= step <= block["step_to"]
        ]
        assert block["mean"] == pytest.approx(sum(chunk) / len(chunk))
        assert block["min"] == pytest.approx(min(chunk))
        assert block["max"] == pytest.approx(max(chunk))
    assert tr.loss_deciles([], 10) == []


def test_read_progress_skips_torn_and_garbage_lines(tmp_path):
    path = tmp_path / "progress.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"ev": "run_start", "ts": 0.0, "total_steps": 4}) + "\n")
        fh.write('{"ev": "step", "loss": 0.5')  # torn last line
        fh.write("\n")
        fh.write("not json at all" + "\n")
        fh.write("[1, 2, 3]" + "\n")  # valid JSON, wrong shape
        fh.write(json.dumps({"ev": "step", "ts": 1.0, "loss": 0.4}) + "\n")
    events, skipped = tr.read_progress(path)
    assert len(events) == 2
    assert skipped == 3


def test_build_report_core_numbers(tmp_path):
    progress = tmp_path / "progress.jsonl"
    sample_png = tmp_path / "preview_0001.png"
    sample_png.write_bytes(b"\x89PNG")
    events = _linear_events()
    events.append(
        {
            "ev": "ckpt",
            "ts": 65.0,
            "global_step": 6,
            "path": str(tmp_path / "test-e1.safetensors"),
        }
    )
    events.append(
        {
            "ev": "sample",
            "ts": 70.0,
            "global_step": 6,
            "epoch": 1,
            "path": str(sample_png),
            "prompt": "x",
        }
    )
    events.append({"ev": "run_end", "ts": 125.0, "status": "ok", "final_step": 12})
    _write_jsonl(progress, events)

    report = tr.build_report(progress, None, ma_window=4, check_pid=False)
    assert report["status"] == "ok"
    assert report["run"] == "test_run"
    assert report["samples_done"] == 12
    assert report["total_steps"] == 20
    assert report["pct"] == pytest.approx(60.0)
    assert report["epoch"] == 1
    assert report["total_epochs"] == 2
    assert report["loss"]["last"] == pytest.approx(0.45)
    assert report["loss"]["min"] == pytest.approx(0.45)
    assert report["loss"]["max"] == pytest.approx(1.0)
    assert report["loss"]["ma"] == pytest.approx(0.525)
    assert report["loss"]["ma_window"] == 4
    assert report["wall_clock_s"] == pytest.approx(125.0)
    assert report["eta_s"] is None
    assert len(report["deciles"]) == 10
    assert [c["step"] for c in report["checkpoints"]] == [6]
    assert report["checkpoints"][0]["name"] == "test-e1.safetensors"
    assert report["checkpoints"][0]["exists"] is False
    assert len(report["samples"]) == 1
    assert report["samples"][0]["exists"] is True


def test_report_markdown_has_expected_sections(tmp_path):
    progress = tmp_path / "progress.jsonl"
    events = _linear_events()
    events.append({"ev": "run_end", "ts": 125.0, "status": "ok", "final_step": 12})
    _write_jsonl(progress, events)
    report = tr.build_report(progress, None, ma_window=4, check_pid=False)
    text = tr.render_markdown(report)
    for needle in (
        "## moss train report - test_run [moss] ok",
        "| samples | 12 / 20 (60.0%) |",
        "| epoch | 1 / 2 |",
        "MA4",
        "### loss deciles",
        "### checkpoints (0)",
        "### samples",
        "### health",
        "0 - sampling disabled",
    ):
        assert needle in text


def test_daemon_relaunch_uses_current_status_and_clock(tmp_path, monkeypatch):
    path = tmp_path / "progress.jsonl"
    previous = _linear_events(n=2, total=100, ts0=1000)
    previous += [
        {"ev": "ckpt", "ts": 2001, "global_step": 2, "path": "old.safetensors"},
        {"ev": "run_end", "ts": 2002, "status": "error", "error": "old failure"},
    ]
    current = _linear_events(n=2, total=100, ts0=10, pid=424242)
    for step in current[1:]:
        step["global_step"] += 2
    _write_jsonl(path, previous + current)
    monkeypatch.setattr(tr, "_pid_alive", lambda pid: pid == 424242)

    report = tr.build_report(path)
    assert report["status"] == "running"
    assert report["error"] is None
    assert report["samples_done"] == 4
    assert report["loss"]["count"] == 2
    assert report["throughput"]["s_per_sample_avg"] == pytest.approx(10)
    assert report["eta_s"] == pytest.approx(960)
    assert report["stalls"] == []
    assert len(report["checkpoints"]) == 1
    assert report["wall_clock_s"] == pytest.approx(2022)

    monkeypatch.setattr(tr, "_pid_alive", lambda pid: False)
    assert tr.build_report(path)["status"] == "dead"


def test_resumed_packed_progress_keeps_absolute_count_and_local_rate(tmp_path):
    path = tmp_path / "progress.jsonl"
    events = _linear_events(n=2, total=100, ts0=4)
    for step, count in zip(events[1:], (54, 58)):
        step.update(global_step=count, samples=4)
    _write_jsonl(path, events)

    report = tr.build_report(path, check_pid=False)
    assert report["samples_done"] == report["global_step"] == 58
    assert report["pct"] == pytest.approx(58)
    assert report["throughput"]["s_per_sample_avg"] == pytest.approx(1)
    assert report["eta_s"] == pytest.approx(42)


@pytest.mark.parametrize("final_step", [0, 60])
def test_final_step_is_authoritative_even_without_step_events(tmp_path, final_step):
    path = tmp_path / "progress.jsonl"
    events = _linear_events(n=0, total=100)
    events.append({"ev": "run_end", "ts": 1, "status": "ok", "final_step": final_step})
    _write_jsonl(path, events)
    report = tr.build_report(path, check_pid=False)
    assert report["samples_done"] == report["global_step"] == final_step
    assert report["pct"] == pytest.approx(final_step)


def test_nonfinite_loss_is_flagged_as_error(tmp_path):
    progress = tmp_path / "progress.jsonl"
    events = _linear_events(n=6, total=12)
    events[3]["loss"] = float("nan")
    _write_jsonl(progress, events)
    report = tr.build_report(progress, None, ma_window=2, check_pid=False)
    assert report["loss"]["nonfinite"] == [3]
    assert report["loss"]["count"] == 5
    kinds = {item["kind"]: item for item in report["anomalies"]}
    assert kinds["nonfinite_loss"]["severity"] == "error"
    assert "step 3" in kinds["nonfinite_loss"]["detail"]
    assert math.isnan(events[3]["loss"])  # fixture sanity


def test_loss_rising_is_flagged(tmp_path):
    progress = tmp_path / "progress.jsonl"
    events = _linear_events(n=40, total=40, rate=0.0, ts0=5.0)
    for i, rec in enumerate(events[1:]):
        rec["loss"] = 0.5 + 0.01 * i  # steady climb: last quarter >> prior quarter
    _write_jsonl(progress, events)
    report = tr.build_report(progress, None, ma_window=5, check_pid=False)
    kinds = {item["kind"]: item for item in report["anomalies"]}
    assert "loss_rising" in kinds
    assert kinds["loss_rising"]["severity"] == "warn"


def test_loss_wobble_within_noise_is_not_flagged(tmp_path):
    progress = tmp_path / "progress.jsonl"
    events = _linear_events(n=60, total=60, ts0=5.0)
    for i, rec in enumerate(events[1:]):
        base = 0.57 if i >= 45 else 0.55  # +3% last quarter, inside sigma noise
        rec["loss"] = base + (0.12 if i % 2 else 0.0)
    _write_jsonl(progress, events)
    report = tr.build_report(progress, None, ma_window=5, check_pid=False)
    assert all(item["kind"] != "loss_rising" for item in report["anomalies"])


def test_stall_gap_is_flagged(tmp_path):
    progress = tmp_path / "progress.jsonl"
    events = _linear_events(n=8, total=16, ts0=10.0)
    for rec in events[5:]:
        rec["ts"] += 400.0  # one 400s hole between step 4 and step 5
    _write_jsonl(progress, events)
    report = tr.build_report(progress, None, stall_s=180.0, check_pid=False)
    assert report["stalls"] == [{"after_step": 4, "gap_s": 410.0}]
    assert any(item["kind"] == "stall" for item in report["anomalies"])


def test_dead_pid_without_run_end(tmp_path, monkeypatch):
    progress = tmp_path / "progress.jsonl"
    events = _linear_events(n=3, total=6, pid=424242)
    _write_jsonl(progress, events)
    monkeypatch.setattr(tr, "_pid_alive", lambda pid: False)
    report = tr.build_report(progress, None, check_pid=True)
    assert report["status"] == "dead"
    kinds = {item["kind"]: item for item in report["anomalies"]}
    assert kinds["dead"]["severity"] == "error"


def test_stale_live_stream_is_flagged(tmp_path):
    progress = tmp_path / "progress.jsonl"
    events = _linear_events(n=3, total=6)
    _write_jsonl(progress, events)
    now = time.time() + 1000.0
    report = tr.build_report(progress, None, check_pid=False, now=now)
    assert report["status"] == "running"
    assert any(item["kind"] == "stale_stream" for item in report["anomalies"])


def test_parse_log_collects_throughput_peak_and_oom(tmp_path):
    log = tmp_path / "train.log"
    log.write_text(
        """\
105 samples, image 4104-4104 tokens
  progress 2/20 epoch 1/2 loss 0.9000 eta 0:01:40
  step 2/20 loss 0.9000 |g| 0.500 sigma 0.750 lr 1.00e-05 batch 1 peak 10.75 GB 6.500s/sample
epoch 1/2: loss 0.8000 120s peak 10.75 GB
  saved output/ckpt/x-e1.safetensors
  resume state output/ckpt/x-state.toml
CUDA out of memory (retry 1)
""",
        encoding="utf-8",
    )
    facts = tr.parse_log(log)
    assert facts["total_steps"] == 20
    assert facts["eta_s"] == 100
    assert facts["s_per_sample"] == pytest.approx(6.5)
    assert facts["peak_vram_gb"] == pytest.approx(10.75)
    assert facts["saved"] == ["output/ckpt/x-e1.safetensors"]
    assert facts["resume_states"] == ["output/ckpt/x-state.toml"]
    assert facts["epochs"][0]["loss"] == pytest.approx(0.8)
    assert facts["oom_hits"] == 1
    assert len(facts["anomaly_lines"]) == 1


def test_log_facts_feed_the_report(tmp_path):
    progress = tmp_path / "progress.jsonl"
    log = tmp_path / "train.log"
    _write_jsonl(progress, _linear_events(n=6, total=20))
    log.write_text(
        "  progress 6/20 epoch 1/2 loss 0.7500 eta 0:02:20\n"
        "  step 6/20 loss 0.7500 |g| 0.500 sigma 0.750 lr 1.00e-05 "
        "batch 1 peak 10.95 GB 6.400s/sample\n",
        encoding="utf-8",
    )
    report = tr.build_report(progress, log, ma_window=4, check_pid=False)
    assert report["throughput"]["source"] == "log"
    assert report["throughput"]["s_per_sample"] == pytest.approx(6.4)
    assert report["peak_vram_gb"] == pytest.approx(10.95)
    assert report["eta_s"] == 140
    assert report["log"]["steps"] == 1


def test_missing_log_is_tolerated(tmp_path):
    progress = tmp_path / "progress.jsonl"
    _write_jsonl(progress, _linear_events(n=3, total=6))
    report = tr.build_report(
        progress, str(tmp_path / "does-not-exist.log"), check_pid=False
    )
    assert report["log"] is None
    assert report["peak_vram_gb"] is None
    assert report["throughput"]["source"] == "stream"


def test_cli_markdown_and_json(tmp_path, capsys):
    progress = tmp_path / "progress.jsonl"
    _write_jsonl(progress, _linear_events(n=4, total=8))
    assert tr.main(["--progress", str(progress), "--no-pid-check"]) == 0
    out = capsys.readouterr().out
    assert "### health" in out
    assert "| samples | 4 / 8 (50.0%) |" in out

    assert tr.main(["--progress", str(progress), "--json", "--no-pid-check"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["samples_done"] == 4
    assert payload["total_steps"] == 8


def test_cli_reports_missing_or_empty_stream(tmp_path, capsys):
    assert tr.main(["--progress", str(tmp_path / "nope.jsonl")]) == 2
    assert "no progress stream" in capsys.readouterr().err

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert tr.main(["--progress", str(empty)]) == 2
    assert "no progress events" in capsys.readouterr().err


# --- checkpoint validation (header-only) ------------------------------------


def _lora_tensors(blocks=(0, 1, 2), rank=4, dtype="F16", skip_alpha=None):
    tensors = {}
    for block in blocks:
        for name in ("q", "k"):
            base = f"lora_unet_layers.{block}.self_attn.{name}"
            tensors[f"{base}.lora_down.weight"] = (dtype, (rank, 8))
            tensors[f"{base}.lora_up.weight"] = (dtype, (8, rank))
            if base != skip_alpha:
                tensors[f"{base}.alpha"] = (dtype, (1,))
    return tensors


def _lora_meta(blocks=3, rank=4, alpha=8, lora_dtype="fp16"):
    return {
        "base_model": "MOSS-Image",
        "algo": "lora",
        "rank": str(rank),
        "alpha": str(alpha),
        "ss_num_blocks": str(blocks),
        "lora_dtype": lora_dtype,
    }


def _write_safetensors(path, tensors, metadata=None):
    header = {}
    if metadata is not None:
        header["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    for name, (dtype, shape) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [0, 0]}
    raw = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)))
        fh.write(raw)


def test_checkpoint_report_accepts_a_consistent_lora(tmp_path):
    path = tmp_path / "moss_lora.safetensors"
    _write_safetensors(path, _lora_tensors(), _lora_meta())

    section = tr.checkpoint_report(path)

    assert section["ok"] is True
    assert section["dtype"] == "F16"
    assert section["modules"] == 6
    assert section["blocks"] == {"count": 3, "first": 0, "last": 2, "expected": 3}
    assert {check["name"] for check in section["checks"]} == {
        "metadata",
        "base model",
        "rank",
        "alpha",
        "dtype",
        "blocks",
    }
    assert all(check["ok"] for check in section["checks"])


def test_checkpoint_report_flags_a_missing_alpha(tmp_path):
    path = tmp_path / "bad.safetensors"
    missing = "lora_unet_layers.1.self_attn.k"
    _write_safetensors(path, _lora_tensors(skip_alpha=missing), _lora_meta())

    section = tr.checkpoint_report(path)

    alpha = next(c for c in section["checks"] if c["name"] == "alpha")
    assert section["ok"] is False
    assert alpha["ok"] is False
    assert missing in alpha["detail"]


def test_checkpoint_report_flags_an_off_lane_dtype(tmp_path):
    path = tmp_path / "dtype.safetensors"
    _write_safetensors(path, _lora_tensors(), _lora_meta(lora_dtype="bf16"))

    section = tr.checkpoint_report(path)

    dtype = next(c for c in section["checks"] if c["name"] == "dtype")
    assert dtype["ok"] is False and "BF16" in dtype["detail"]


def test_checkpoint_report_flags_a_block_gap(tmp_path):
    path = tmp_path / "gap.safetensors"
    _write_safetensors(path, _lora_tensors(blocks=(0, 1, 3)), _lora_meta(blocks=4))

    section = tr.checkpoint_report(path)

    blocks = next(c for c in section["checks"] if c["name"] == "blocks")
    assert blocks["ok"] is False
    assert "missing [2]" in blocks["detail"]


def test_checkpoint_report_rejects_a_non_safetensors_file(tmp_path):
    path = tmp_path / "resume.state.pt"
    path.write_bytes(b"not a safetensors file")

    with pytest.raises(ValueError):
        tr.checkpoint_report(path)


def test_checkpoint_report_rejects_a_non_object_header(tmp_path):
    path = tmp_path / "list.safetensors"
    raw = b"[1, 2, 3]"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)

    with pytest.raises(TypeError):
        tr.checkpoint_report(path)


def test_cli_emits_the_checkpoint_section(tmp_path, capsys):
    progress = tmp_path / "progress.jsonl"
    _write_jsonl(progress, _linear_events())
    ckpt = tmp_path / "moss_lora.safetensors"
    _write_safetensors(ckpt, _lora_tensors(), _lora_meta())

    code = tr.main(
        [
            "--progress",
            str(progress),
            "--no-pid-check",
            "--checkpoint",
            str(ckpt),
        ]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "## Checkpoint" in out
    assert "checks 6/6 passed" in out


def test_cli_json_carries_the_checkpoint_report(tmp_path, capsys):
    progress = tmp_path / "progress.jsonl"
    _write_jsonl(progress, _linear_events())
    ckpt = tmp_path / "moss_lora.safetensors"
    _write_safetensors(ckpt, _lora_tensors(), _lora_meta())

    code = tr.main(
        [
            "--progress",
            str(progress),
            "--no-pid-check",
            "--json",
            "--checkpoint",
            str(ckpt),
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["checkpoint"]["ok"] is True
    assert report["checkpoint"]["rank"] == 4


def test_cli_exits_2_when_the_checkpoint_is_unreadable(tmp_path, capsys):
    progress = tmp_path / "progress.jsonl"
    _write_jsonl(progress, _linear_events())
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"nope")

    code = tr.main(
        ["--progress", str(progress), "--no-pid-check", "--checkpoint", str(bad)]
    )

    err = capsys.readouterr().err
    assert code == 2
    assert str(bad) in err


@pytest.mark.parametrize("json_output", [False, True])
def test_cli_exits_2_when_checkpoint_checks_fail(tmp_path, capsys, json_output):
    progress = tmp_path / "progress.jsonl"
    _write_jsonl(progress, _linear_events())
    checkpoint = tmp_path / "missing_blocks.safetensors"
    _write_safetensors(checkpoint, _lora_tensors(blocks=(0,)), _lora_meta())
    argv = [
        "--progress",
        str(progress),
        "--checkpoint",
        str(checkpoint),
        "--no-pid-check",
    ]
    if json_output:
        argv.append("--json")
    assert tr.main(argv) == 2
    output = capsys.readouterr().out
    if json_output:
        assert json.loads(output)["checkpoint"]["ok"] is False
    else:
        assert "**FAILED**" in output


def test_checkpoint_report_checks_the_target_matrix(tmp_path):
    path = tmp_path / "targets.safetensors"
    meta = _lora_meta() | {"targets": r"layers\.\d+\.proj\.(q|k)"}
    _write_safetensors(path, _lora_tensors(), meta)

    section = tr.checkpoint_report(path)

    targets = next(c for c in section["checks"] if c["name"] == "targets")
    assert targets["ok"] is True
    assert "6 modules, expected 6 (3 x 2)" in targets["detail"]
    assert section["ok"] is True


def test_checkpoint_report_flags_a_missing_target(tmp_path):
    path = tmp_path / "missing_target.safetensors"
    meta = _lora_meta() | {"targets": r"layers\.\d+\.proj\.(q|k)"}
    tensors = _lora_tensors()
    base = "lora_unet_layers.1.self_attn.k"
    for suffix in (".lora_down.weight", ".lora_up.weight", ".alpha"):
        tensors.pop(f"{base}{suffix}")

    _write_safetensors(path, tensors, meta)
    section = tr.checkpoint_report(path)

    targets = next(c for c in section["checks"] if c["name"] == "targets")
    assert targets["ok"] is False
    assert "5 modules, expected 6" in targets["detail"]
    assert section["ok"] is False


def _comfy_style_tensors(blocks=(0, 1, 2), rank=4, dtype="F16"):
    """Real trainer shape: ComfyUI-style underscore keys, ffn targets included."""
    tensors = {}
    for block in blocks:
        for name in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down"):
            base = f"lora_unet_layers_{block}_proj_{name}"
            tensors[f"{base}.lora_down.weight"] = (dtype, (rank, 8))
            tensors[f"{base}.lora_up.weight"] = (dtype, (8, rank))
            tensors[f"{base}.alpha"] = (dtype, (1,))
    return tensors


def test_checkpoint_report_accepts_comfy_style_underscore_keys(tmp_path):
    path = tmp_path / "moss_lora.safetensors"
    meta = _lora_meta() | {
        "alpha": "128.0",
        "targets": r"layers\.\d+\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)",
    }
    _write_safetensors(path, _comfy_style_tensors(), meta)

    section = tr.checkpoint_report(path)

    assert section["ok"] is True
    assert section["alpha"] == 128
    assert section["blocks"] == {"count": 3, "first": 0, "last": 2, "expected": 3}
    targets = next(c for c in section["checks"] if c["name"] == "targets")
    assert targets["ok"] is True
    assert "18 modules, expected 18 (3 x 6)" in targets["detail"]
    assert section["modules"] == 18


def test_checkpoint_report_flags_an_unparsable_module_name(tmp_path):
    path = tmp_path / "unparsed.safetensors"
    tensors = _comfy_style_tensors()
    stray = "lora_unet_layers_1_proj_rotary"
    tensors[f"{stray}.lora_down.weight"] = ("F16", (4, 8))
    tensors[f"{stray}.lora_up.weight"] = ("F16", (8, 4))
    tensors[f"{stray}.alpha"] = ("F16", (1,))
    meta = _lora_meta() | {
        "targets": r"layers\.\d+\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)"
    }

    _write_safetensors(path, tensors, meta)

    section = tr.checkpoint_report(path)

    targets = next(c for c in section["checks"] if c["name"] == "targets")
    assert section["ok"] is False
    assert targets["ok"] is False
    assert "unparsed" in targets["detail"]
    assert "19 modules, expected 18" in targets["detail"]


def test_to_int_tolerates_float_metadata_but_not_fractions():
    assert tr._to_int("128.0") == 128
    assert tr._to_int(" 42 ") == 42
    assert tr._to_int("0.5") is None
    assert tr._to_int(None) is None
    assert tr._to_int("not a number") is None


def test_module_target_prefers_the_longest_declared_suffix():
    suffixes = {"q", "k", "v", "o", "ffn_up_gate", "ffn_down"}
    assert tr._module_target("lora_unet_layers_3_proj_ffn_up_gate", suffixes) == (
        "ffn_up_gate"
    )
    assert tr._module_target("lora_unet_layers_3_proj_q", suffixes) == "q"
    assert tr._module_target("lora_unet_layers.3.proj.ffn_down", suffixes) == "ffn_down"
    assert tr._module_target("lora_unet_layers_3_proj_rotary", suffixes) is None
