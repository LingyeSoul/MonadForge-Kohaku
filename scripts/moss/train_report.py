"""Compact health report for a MOSS-Image LoRA training run.

Reads the structured progress stream written by library.training.progress and,
optionally, the trainer's stdout log (live s/sample + peak VRAM only exist
there). Stdlib only, so it is safe to point at a *live* run from any
interpreter -- no torch, no repo import.

Usage::

    .venv/bin/python -m scripts.moss.train_report --progress output/logs/moss_lora.progress.jsonl --log output/logs/moss_train_20261025-1731.log
    .venv/bin/python -m scripts.moss.train_report --progress <stream> --json

The summary carries the run header (samples done / total, epoch, status),
throughput (s/sample from the log's own window, stream average as a fallback),
wall clock, loss last / trailing MA / min / max / deciles, checkpoint events,
preview images written, and an anomaly scan (non-finite loss, loss trending
up, stall gaps, OOM / NaN lines in the log, a dead pid).
"""

from __future__ import annotations

import argparse
import json
import errno
import math
import os
import re
import struct
import sys
import time
from collections import Counter
from itertools import pairwise
from pathlib import Path

DEFAULT_MA_WINDOW = 50
DEFAULT_DECILES = 10
DEFAULT_STALL_S = 180.0

#: lora_dtype metadata value -> safetensors dtype tag.
SAFETENSORS_DTYPES = {"fp16": "F16", "bf16": "BF16", "fp32": "F32"}

# ---------------------------------------------------------------------------
# stdout log parsing (train.py: "  progress ...", "  step ...", "epoch ...")
# ---------------------------------------------------------------------------

_LOG_PROGRESS_RE = re.compile(
    r"^\s*progress (\d+)/(\d+) epoch (\d+)/(\d+) loss (\S+) "
    r"eta (\d+):(\d{2}):(\d{2})"
)
_LOG_STEP_RE = re.compile(
    r"^\s*step (\d+)/(\d+) loss (\S+) \|g\| (\S+) sigma (\S+) lr (\S+) "
    r"batch (\d+) peak (\S+) GB(?: ([\d.]+)s/sample)?(?:\s+repa (\S+))?"
)
_LOG_EPOCH_RE = re.compile(
    r"^epoch (\d+)/(\d+): loss (\S+)(?: repa (\S+))? (\d+)s peak (\S+) GB"
)
_LOG_SAVED_RE = re.compile(r"^\s*saved (\S.*?)\s*$")
_LOG_RESUME_RE = re.compile(r"^\s*resume state (\S.*?)\s*$")
_LOG_ANOMALY_RE = re.compile(
    r"out of memory|\bOOM\b|CUDA error|Traceback|RuntimeError|"
    r"non-finite|\bNaN\b",
    re.IGNORECASE,
)
_LOG_OOM_RE = re.compile(r"out of memory|\bOOM\b", re.IGNORECASE)
_LOG_NAN_RE = re.compile(r"non-finite|\bNaN\b", re.IGNORECASE)


def _to_float(value):
    """Best-effort float, NaN/Inf preserved (callers decide what to skip)."""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _finite(value):
    return value is not None and math.isfinite(value)


def read_progress(path):
    """Load an append-only progress stream; torn / malformed lines are skipped."""
    events = []
    skipped = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                skipped += 1
                continue
            if isinstance(rec, dict):
                events.append(rec)
            else:
                skipped += 1
    return events, skipped


def _last(events, ev):
    for rec in reversed(events):
        if rec.get("ev") == ev:
            return rec
    return None


def _step_records(events):
    """Normalise step events to dicts with numeric fields where available."""
    out = []
    for rec in events:
        if rec.get("ev") != "step":
            continue
        out.append(
            {
                "step": int(rec.get("global_step") or 0),
                "epoch": rec.get("epoch"),
                "loss": _to_float(rec.get("loss")),
                "samples": int(rec.get("samples") or 1),
                "grad_norm": _to_float(rec.get("grad_norm")),
                "lr": _to_float(rec.get("lr")),
                "ts": _to_float(rec.get("ts")),
            }
        )
    return out


def trailing_mean(values, window):
    """Trailing moving average; the leading partial windows are included.

    Index i gets the mean of values[max(0, i-window+1) : i+1], so the list has
    the same length as the input and the last entry is the current MA.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    out = []
    total = 0.0
    for i, value in enumerate(values):
        total += value
        if i >= window:
            total -= values[i - window]
        out.append(total / min(i + 1, window))
    return out


def _mean_and_se(values):
    """Sample mean and standard error of the mean (0 se for a constant run)."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var / n)


def loss_deciles(points, count=DEFAULT_DECILES):
    """Split (step, loss) points into up to count contiguous blocks.

    "Deciles" here means equal-sized *runs of the step axis* (segments of the
    loss curve), not quantiles of the loss distribution: the sequence is in
    step order, so block i covers a contiguous stretch of training. Extra
    points go to the earliest blocks so the blocks stay contiguous.
    """
    n = len(points)
    if n == 0:
        return []
    blocks = min(int(count), n)
    base, rem = divmod(n, blocks)
    out = []
    start = 0
    for i in range(blocks):
        size = base + (1 if i < rem else 0)
        chunk = points[start : start + size]
        losses = [loss for _, loss in chunk]
        out.append(
            {
                "index": i + 1,
                "step_from": chunk[0][0],
                "step_to": chunk[-1][0],
                "count": len(chunk),
                "mean": sum(losses) / len(losses),
                "min": min(losses),
                "max": max(losses),
            }
        )
        start += size
    return out


def _wall_clock(events):
    """Sum of the ts spans of each run segment (restarts reset the clock)."""
    spans = []
    current = []
    for rec in events:
        ts = _to_float(rec.get("ts"))
        if rec.get("ev") == "run_start" and current:
            spans.append(current)
            current = []
        if ts is not None:
            current.append(ts)
    if current:
        spans.append(current)
    return float(sum(max(seg) - min(seg) for seg in spans)) if spans else None


def _pid_alive(pid):
    """True/False if the pid is/isn't running, None if unknowable."""
    if not pid:
        return None
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        # Windows reports a vanished pid as ERROR_INVALID_PARAMETER (87) rather
        # than ESRCH, so a dead trainer must be recognised from `winerror` too.
        if exc.errno == errno.ESRCH or getattr(exc, "winerror", None) == 87:
            return False
        return None
    return True


def parse_log(path):
    """Digest the trainer stdout log into the facts it alone carries."""
    facts = {
        "path": str(path),
        "steps": [],
        "progress": [],
        "epochs": [],
        "saved": [],
        "resume_states": [],
        "anomaly_lines": [],
        "oom_hits": 0,
        "nan_hits": 0,
        "total_steps": None,
        "peak_vram_gb": None,
        "s_per_sample": None,
        "eta_s": None,
        "mtime": None,
    }
    try:
        facts["mtime"] = os.path.getmtime(path)
    except OSError:
        pass
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            match = _LOG_STEP_RE.match(line)
            if match:
                peak = _to_float(match.group(8))
                per_sample = _to_float(match.group(9))
                repa = _to_float(match.group(10))
                facts["steps"].append(
                    {
                        "step": int(match.group(1)),
                        "loss": _to_float(match.group(3)),
                        "grad_norm": _to_float(match.group(4)),
                        "sigma": _to_float(match.group(5)),
                        "lr": _to_float(match.group(6)),
                        "batch": int(match.group(7)),
                        "peak_vram_gb": peak,
                        "s_per_sample": per_sample,
                        "repa": repa,
                    }
                )
                if peak is not None:
                    facts["peak_vram_gb"] = max(facts["peak_vram_gb"] or 0.0, peak)
                if per_sample is not None:
                    facts["s_per_sample"] = per_sample
                continue
            match = _LOG_PROGRESS_RE.match(line)
            if match:
                eta = (
                    int(match.group(6)) * 3600
                    + int(match.group(7)) * 60
                    + int(match.group(8))
                )
                facts["progress"].append(
                    {
                        "step": int(match.group(1)),
                        "total": int(match.group(2)),
                        "epoch": int(match.group(3)),
                        "loss": _to_float(match.group(5)),
                        "eta_s": eta,
                    }
                )
                facts["total_steps"] = int(match.group(2))
                facts["eta_s"] = eta
                continue
            match = _LOG_EPOCH_RE.match(line)
            if match:
                peak = _to_float(match.group(6))
                facts["epochs"].append(
                    {
                        "epoch": int(match.group(1)),
                        "loss": _to_float(match.group(3)),
                        "repa": _to_float(match.group(4)),
                        "seconds": float(match.group(5)),
                        "peak_vram_gb": peak,
                    }
                )
                if peak is not None:
                    facts["peak_vram_gb"] = max(facts["peak_vram_gb"] or 0.0, peak)
                continue
            match = _LOG_SAVED_RE.match(line)
            if match:
                facts["saved"].append(match.group(1))
                continue
            match = _LOG_RESUME_RE.match(line)
            if match:
                facts["resume_states"].append(match.group(1))
                continue
            if _LOG_ANOMALY_RE.search(line):
                facts["anomaly_lines"].append(line.strip())
                if _LOG_OOM_RE.search(line):
                    facts["oom_hits"] += 1
                if _LOG_NAN_RE.search(line):
                    facts["nan_hits"] += 1
    return facts


def _latest_segment(events):
    """Daemon relaunches append a new run_start and reset relative timestamps."""
    for index in range(len(events) - 1, -1, -1):
        if events[index].get("ev") == "run_start":
            return events[index:]
    return events


def _resolve_status(events, check_pid):
    events = _latest_segment(events)
    end = _last(events, "run_end")
    if end is not None:
        return end.get("status") or "unknown"
    start = _last(events, "run_start") or {}
    alive = _pid_alive(start.get("pid")) if check_pid else None
    if alive is False:
        return "dead"
    return "running"


def build_report(
    progress_path,
    log_path=None,
    *,
    ma_window=DEFAULT_MA_WINDOW,
    deciles=DEFAULT_DECILES,
    stall_s=DEFAULT_STALL_S,
    check_pid=True,
    now=None,
):
    """Digest progress (+ optional log) into one JSON-able report dict."""
    events, torn = read_progress(progress_path)
    if not events:
        raise ValueError(f"no progress events in {progress_path}")
    now = time.time() if now is None else now
    current = _latest_segment(events)
    start = _last(current, "run_start") or {}
    end = _last(current, "run_end")
    status = _resolve_status(current, check_pid=check_pid)

    # Current loss/rate/stalls must not cross a restart's clock or replayed steps.
    # Keep historical checkpoint/sample events and total active wall time below.
    steps = _step_records(current)
    finite = [(rec["step"], rec["loss"]) for rec in steps if _finite(rec["loss"])]
    nonfinite = [
        rec["step"]
        for rec in steps
        if rec["loss"] is not None and not _finite(rec["loss"])
    ]
    losses = [loss for _, loss in finite]
    ma_series = trailing_mean(losses, ma_window) if losses else []

    samples_done = (end or {}).get("final_step")
    if samples_done is None:
        samples_done = steps[-1]["step"] if steps else 0
    total_steps = start.get("total_steps")
    pct = (100.0 * samples_done / total_steps) if total_steps else None
    rate = None
    if len(steps) >= 2:
        first, last = steps[0], steps[-1]
        if first["ts"] is not None and last["ts"] is not None:
            d_samples = sum(rec["samples"] for rec in steps[1:])
            d_ts = last["ts"] - first["ts"]
            if d_samples > 0 and d_ts > 0:
                rate = d_samples / d_ts

    log_facts = None
    if log_path:
        try:
            log_facts = parse_log(log_path)
        except OSError:
            log_facts = None
    if total_steps is None and log_facts is not None:
        total_steps = log_facts["total_steps"]
        pct = (100.0 * samples_done / total_steps) if total_steps else None

    s_per_sample = log_facts.get("s_per_sample") if log_facts else None
    throughput = {
        "s_per_sample": (
            s_per_sample if s_per_sample is not None else (1.0 / rate if rate else None)
        ),
        "s_per_sample_avg": (1.0 / rate) if rate else None,
        "samples_per_s": rate,
        "source": "log" if s_per_sample is not None else "stream",
    }
    eta_s = None
    if status == "running":
        if log_facts and log_facts.get("eta_s") is not None:
            eta_s = log_facts["eta_s"]
        elif rate and total_steps:
            eta_s = max(0.0, (total_steps - samples_done) / rate)

    # ---- anomaly scan -----------------------------------------------------
    anomalies = []
    if nonfinite:
        anomalies.append(
            {
                "kind": "nonfinite_loss",
                "severity": "error",
                "detail": f"{len(nonfinite)} non-finite loss value(s), first at step {nonfinite[0]}",
            }
        )
    if status == "dead":
        anomalies.append(
            {
                "kind": "dead",
                "severity": "error",
                "detail": "no run_end event and pid {} is gone".format(
                    start.get("pid")
                ),
            }
        )
    recent_window = max(5, min(100, len(losses) // 4))
    if len(losses) >= 2 * recent_window:
        recent, recent_se = _mean_and_se(losses[-recent_window:])
        prior, prior_se = _mean_and_se(losses[-2 * recent_window : -recent_window])
        se = math.sqrt(recent_se * recent_se + prior_se * prior_se)
        shift = recent - prior
        z = shift / se if se > 0 else math.inf
        # The flow-matching loss scales with the sampled sigma, so raw window
        # means swing several percent on their own; demand a shift that clears
        # both a 2% floor and 3 standard errors of the mean difference.
        if prior > 0 and shift > 0.02 * prior and z > 3.0:
            anomalies.append(
                {
                    "kind": "loss_rising",
                    "severity": "warn",
                    "detail": (
                        f"last {recent_window} steps mean {recent:.4f} vs {prior:.4f} before "
                        f"(+{100.0 * shift / prior:.1f}%, {z:.1f} sigma)"
                    ),
                }
            )
    stalls = []
    for prev, cur in pairwise(steps):
        if prev["ts"] is None or cur["ts"] is None:
            continue
        gap = cur["ts"] - prev["ts"]
        if gap > stall_s:
            stalls.append({"after_step": prev["step"], "gap_s": round(gap, 1)})
    if stalls:
        worst = max(stalls, key=lambda s: s["gap_s"])
        anomalies.append(
            {
                "kind": "stall",
                "severity": "warn",
                "detail": "{} gap(s) > {:.0f}s, largest {:.0f}s after step {}".format(
                    len(stalls), stall_s, worst["gap_s"], worst["after_step"]
                ),
            }
        )
    last_write_age = None
    try:
        last_write_age = max(0.0, now - os.path.getmtime(progress_path))
    except OSError:
        pass
    if status == "running" and last_write_age is not None and last_write_age > stall_s:
        anomalies.append(
            {
                "kind": "stale_stream",
                "severity": "error",
                "detail": f"progress.jsonl not written for {last_write_age:.0f}s",
            }
        )
    if log_facts:
        if log_facts["oom_hits"]:
            anomalies.append(
                {
                    "kind": "oom",
                    "severity": "warn",
                    "detail": "{} OOM-related log line(s)".format(
                        log_facts["oom_hits"]
                    ),
                }
            )
        if log_facts["nan_hits"]:
            anomalies.append(
                {
                    "kind": "nan_in_log",
                    "severity": "error",
                    "detail": "{} NaN / non-finite log line(s)".format(
                        log_facts["nan_hits"]
                    ),
                }
            )
    grad_norms = [rec["grad_norm"] for rec in steps if _finite(rec["grad_norm"])]
    grad_norm_max = max(grad_norms) if grad_norms else None
    if grad_norms and grad_norms[-1] > 10.0:
        anomalies.append(
            {
                "kind": "grad_spike",
                "severity": "warn",
                "detail": f"last grad_norm {grad_norms[-1]:.2f} (max {grad_norm_max:.2f})",
            }
        )

    checkpoints = []
    for rec in events:
        if rec.get("ev") != "ckpt":
            continue
        path = rec.get("path") or ""
        checkpoints.append(
            {
                "step": rec.get("global_step"),
                "path": path,
                "name": Path(path).name,
                "exists": bool(path) and os.path.exists(path),
            }
        )
    samples = []
    for rec in events:
        if rec.get("ev") != "sample":
            continue
        path = rec.get("path") or ""
        samples.append(
            {
                "step": rec.get("global_step"),
                "epoch": rec.get("epoch"),
                "path": path,
                "name": Path(path).name,
                "exists": bool(path) and os.path.exists(path),
            }
        )

    epoch = None
    for rec in reversed(steps):
        if rec["epoch"] is not None:
            epoch = rec["epoch"]
            break

    report = {
        "progress_path": str(progress_path),
        "log_path": str(log_path) if log_path else None,
        "run": start.get("run"),
        "method": start.get("method"),
        "preset": start.get("preset"),
        "pid": start.get("pid"),
        "status": status,
        "error": (end or {}).get("error"),
        "events": len(events),
        "torn_lines": torn,
        "samples_done": samples_done,
        "total_steps": total_steps,
        "pct": pct,
        "global_step": samples_done if steps or end is not None else None,
        "epoch": epoch,
        "total_epochs": start.get("total_epochs"),
        "sampling_enabled": bool(start.get("sampling_enabled")),
        "wall_clock_s": _wall_clock(events),
        "throughput": throughput,
        "peak_vram_gb": log_facts.get("peak_vram_gb") if log_facts else None,
        "eta_s": eta_s,
        "loss": {
            "last": losses[-1] if losses else None,
            "first": losses[0] if losses else None,
            "mean": (sum(losses) / len(losses)) if losses else None,
            "min": min(losses) if losses else None,
            "max": max(losses) if losses else None,
            "ma_window": ma_window,
            "ma": ma_series[-1] if ma_series else None,
            "ma_min": min(ma_series) if ma_series else None,
            "count": len(losses),
            "nonfinite": nonfinite,
        },
        "deciles": loss_deciles(finite, deciles),
        "checkpoints": checkpoints,
        "samples": samples,
        "stalls": stalls,
        "anomalies": anomalies,
        "grad_norm_max": grad_norm_max,
        "last_write_age_s": last_write_age,
        "log": (
            {
                "steps": len(log_facts["steps"]),
                "progress": len(log_facts["progress"]),
                "epochs": log_facts["epochs"],
                "saved": log_facts["saved"],
                "resume_states": log_facts["resume_states"],
                "eta_s": log_facts["eta_s"],
                "s_per_sample": log_facts["s_per_sample"],
                "peak_vram_gb": log_facts["peak_vram_gb"],
            }
            if log_facts
            else None
        ),
    }
    return report


def _fmt_dur(seconds):
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "?"
    seconds = round(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def _fmt(value, spec=".5f"):
    if value is None:
        return "-"
    return format(value, spec)


def render_markdown(report, *, max_checkpoints=25, max_samples=10):
    """Render the report dict as a compact markdown block."""
    name = report.get("run") or Path(report["progress_path"]).name
    head = "moss train report - {} [{}] {}".format(
        name, report.get("method") or "?", report.get("status") or "?"
    )
    lines = ["## " + head, ""]
    total = report.get("total_steps")
    pct = report.get("pct")
    samples = report.get("samples_done") or 0
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    pct_cell = f" ({pct:.1f}%)" if pct is not None else ""
    lines.append(
        "| samples | {} / {}{} |".format(
            samples, total if total is not None else "?", pct_cell
        )
    )
    epochs = report.get("total_epochs")
    lines.append(
        "| epoch | {} / {} |".format(
            report.get("epoch") if report.get("epoch") is not None else "?",
            epochs if epochs is not None else "?",
        )
    )
    loss = report.get("loss") or {}
    if loss.get("last") is not None:
        loss_cell = "{} (MA{} {})".format(
            _fmt(loss.get("last")), loss.get("ma_window"), _fmt(loss.get("ma"))
        )
    else:
        loss_cell = "-"
    lines.append(f"| loss (last) | {loss_cell} |")
    lines.append(
        "| loss min / max | {} / {} |".format(
            _fmt(loss.get("min")), _fmt(loss.get("max"))
        )
    )
    tp = report.get("throughput") or {}
    s_per_sample = tp.get("s_per_sample")
    avg = tp.get("s_per_sample_avg")
    if s_per_sample is None:
        tp_cell = "-"
    elif avg is not None and abs(avg - s_per_sample) > 0.005:
        tp_cell = "{:.3f} s/sample (avg {:.3f}, {})".format(
            s_per_sample, avg, tp.get("source")
        )
    else:
        tp_cell = "{:.3f} s/sample ({})".format(s_per_sample, tp.get("source"))
    lines.append(f"| throughput | {tp_cell} |")
    lines.append(
        "| peak VRAM | {} GB |".format(_fmt(report.get("peak_vram_gb"), ".2f"))
    )
    lines.append("| wall clock | {} |".format(_fmt_dur(report.get("wall_clock_s"))))
    lines.append("| ETA | {} |".format(_fmt_dur(report.get("eta_s"))))
    if report.get("error"):
        lines.append("| error | {} |".format(report["error"]))

    deciles = report.get("deciles") or []
    if deciles:
        lines += [
            "",
            "### loss deciles",
            "",
            "| # | steps | mean | min | max |",
            "| --- | --- | --- | --- | --- |",
        ]
        for block in deciles:
            if block["step_from"] == block["step_to"]:
                span = str(block["step_from"])
            else:
                span = "{}..{}".format(block["step_from"], block["step_to"])
            lines.append(
                "| {} | {} | {:.4f} | {:.4f} | {:.4f} |".format(
                    block["index"], span, block["mean"], block["min"], block["max"]
                )
            )

    ckpts = report.get("checkpoints") or []
    lines += ["", f"### checkpoints ({len(ckpts)})", ""]
    if ckpts:
        lines.append("| step | file |")
        lines.append("| --- | --- |")
        for ckpt in ckpts[:max_checkpoints]:
            lines.append("| {} | {} |".format(ckpt.get("step"), ckpt["name"]))
        if len(ckpts) > max_checkpoints:
            lines.append(f"| ... | {len(ckpts) - max_checkpoints} more |")
    else:
        lines.append("none yet")

    samples_written = report.get("samples") or []
    lines += ["", "### samples", ""]
    if samples_written:
        present = sum(1 for s in samples_written if s["exists"])
        lines.append(
            f"{len(samples_written)} preview image(s) written ({present} on disk):"
        )
        for sample in samples_written[-max_samples:]:
            lines.append(
                "- " + sample["name"] + " (step {})".format(sample.get("step"))
            )
    elif report.get("sampling_enabled"):
        lines.append("sampling enabled but no preview written yet")
    else:
        lines.append("0 - sampling disabled for this run (sampling_enabled=false)")

    anomalies = report.get("anomalies") or []
    lines += ["", "### health", ""]
    if anomalies:
        for item in anomalies:
            lines.append(
                "- **{}** ({}) {}".format(
                    item["kind"], item["severity"], item["detail"]
                )
            )
    else:
        age = report.get("last_write_age_s")
        if age is not None:
            lines.append(f"no anomalies detected; last progress write {age:.0f}s ago")
        else:
            lines.append("no anomalies detected")
    if report.get("torn_lines"):
        lines.append(
            "- {} torn / unparsable progress line(s) skipped".format(
                report["torn_lines"]
            )
        )
    if report.get("grad_norm_max") is not None:
        lines.append("- max grad norm {:.2f}".format(report["grad_norm_max"]))
    return "\n".join(lines)


# --- checkpoint inspection (safetensors header only, no torch) ------------------

#: Block index of a LoRA module name. The trainer writes ComfyUI-style
#: underscore keys ("lora_unet_layers_0_proj_q"), while the older Anima-shaped
#: fixtures use dots ("lora_unet_layers.0.proj.q") -- accept either separator.
_BLOCK_RE = re.compile(r"layers[._](\d+)(?:[._]|$)")
#: LoRA state key suffixes written by library/lines/adapters.py:520.
_DOWN_SUFFIX = ".lora_down.weight"


def _to_int(value):
    """Integer metadata value, tolerating the float spelling the trainer stamps.

    The "alpha" header field lands as "128.0" (a rank could do the same), so a
    plain int() parse would drop it to None. Only integral floats are accepted --
    "0.5" is not an integer setting.
    """
    if value is None:
        return None
    text = str(value).strip()
    try:
        return int(text)
    except (TypeError, ValueError):
        pass
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else None


def _read_safetensors_header(path):
    """Return (tensors, metadata) from a .safetensors header; never reads weights."""
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError("not a safetensors file (no 8-byte header length)")
        (size,) = struct.unpack("<Q", raw)
        if size < 2 or size > 256 * 1024 * 1024:
            raise ValueError(f"implausible safetensors header length {size}")
        body = fh.read(size)
    try:
        header = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable safetensors header: {exc}") from exc
    if not isinstance(header, dict):
        raise TypeError("safetensors header is not an object")
    meta = header.pop("__metadata__", None) or {}
    tensors = {key: value for key, value in header.items() if isinstance(value, dict)}
    return tensors, {str(k): str(v) for k, v in meta.items()}


def _target_suffixes(pattern):
    """Alternatives of the trailing group of a targets regex, e.g. (q|k|v|o).

    library/moss/requests.py:410 keeps the trainer's default as a regex, so the
    module names a file should carry are its parenthesised alternatives.
    """
    if not pattern:
        return []
    match = re.search(r"\(([^()]+)\)", pattern)
    if match is None:
        return []
    return [part.strip() for part in match.group(1).split("|") if part.strip()]


def _module_target(module, suffixes):
    """Leaf target of a LoRA module name, e.g. "q" in "..._proj_q" or "...proj.q".

    MOSS stamps ComfyUI-style underscore keys, so the separator before the target
    is "." on the old Anima-shaped fixtures and "_" on a real trainer file. A
    target may itself contain underscores ("ffn_up_gate"), which rules out a
    rsplit -- nothing longer than the known targets can be carved off safely, so
    match against the declared target set, longest first.
    """
    for suffix in sorted(suffixes, key=len, reverse=True):
        if module.endswith((f".{suffix}", f"_{suffix}")):
            return suffix
    return None


def checkpoint_report(path, *, expected_dtype=None):
    """Structural check of a saved MOSS LoRA, from the safetensors header alone.

    Verifies what the trainer is supposed to stamp (base model, algo, rank,
    per-module alpha, dtype lane, block coverage) without loading a single
    weight, so it runs on a finished file with no GPU and no torch.
    """
    tensors, meta = _read_safetensors_header(path)
    if not tensors:
        raise ValueError(f"{path}: safetensors header lists no tensors")
    dtypes = dict(Counter(str(info.get("dtype")) for info in tensors.values()))
    algo = meta.get("algo")
    if not algo:
        algo = "lora" if any(k.endswith(".lora_down.weight") for k in tensors) else None
    rank_meta = _to_int(meta.get("rank"))
    alpha_meta = _to_int(meta.get("alpha"))
    down = {
        key: info for key, info in tensors.items() if key.endswith(".lora_down.weight")
    }
    alpha_keys = {key for key in tensors if key.endswith(".alpha")}
    block_ids = sorted(
        {
            int(match.group(1))
            for key in tensors
            if (match := _BLOCK_RE.search(key)) is not None
        }
    )
    ranks = sorted(
        {
            (tuple(info.get("shape") or []) or (None,))[0]
            for info in down.values()
            if (info.get("shape") or [])
        }
    )
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    base = meta.get("base_model")
    add(
        "metadata",
        bool(meta),
        f"{len(meta)} fields" if meta else "no __metadata__ block",
    )
    add("base model", base == "MOSS-Image", f"base_model={base!r}")
    if algo == "lora":
        rank_ok = bool(down) and len(ranks) == 1
        if rank_ok and rank_meta is not None:
            rank_ok = ranks[0] == rank_meta
        add(
            "rank",
            rank_ok,
            f"file rank {ranks if len(ranks) != 1 else ranks[0]}, metadata {rank_meta}",
        )
        missing = sorted(
            module
            for module in (
                key[: -len(_DOWN_SUFFIX)] for key in down if key.endswith(_DOWN_SUFFIX)
            )
            if f"{module}.alpha" not in alpha_keys
        )
        add(
            "alpha",
            not missing,
            f"{len(alpha_keys)} alpha tensors for {len(down)} modules"
            if not missing
            else f"missing alpha for {missing[:3]} ({len(missing)})",
        )
    want = SAFETENSORS_DTYPES.get(str(meta.get("lora_dtype") or "")) or expected_dtype
    if want:
        off = {dtype: n for dtype, n in dtypes.items() if dtype != want}
        add("dtype", not off, f"{dtypes} (expected {want})")
    blocks_meta = _to_int(meta.get("ss_num_blocks"))
    if blocks_meta:
        expect = set(range(blocks_meta))
        got = set(block_ids)
        missing = sorted(expect - got)
        extra = sorted(got - expect)
        detail = f"{len(got & expect)} of {blocks_meta} blocks present"
        if missing:
            detail += f", missing {missing[:5]}"
        if extra:
            detail += f", unexpected {extra[:5]}"
        add("blocks", got == expect, detail)
        suffixes = _target_suffixes(meta.get("targets"))
        if algo == "lora" and suffixes and blocks_meta:
            expected_suffixes = set(suffixes)
            by_block = {}
            unresolved = []
            for module in (
                key[: -len(_DOWN_SUFFIX)] for key in down if key.endswith(_DOWN_SUFFIX)
            ):
                match = _BLOCK_RE.search(module)
                target = _module_target(module, expected_suffixes)
                if match is None or target is None:
                    unresolved.append(module)
                    continue
                by_block.setdefault(int(match.group(1)), set()).add(target)
            incomplete = sorted(
                f"{block}: missing {sorted(expected_suffixes - got)[:3]}"
                for block, got in by_block.items()
                if expected_suffixes - got
            )
            expected_count = blocks_meta * len(expected_suffixes)
            detail = (
                f"{len(down)} modules, expected {expected_count} "
                f"({blocks_meta} x {len(expected_suffixes)})"
            )
            if incomplete:
                detail += f"; incomplete {incomplete[:3]}"
            if unresolved:
                detail += f"; unparsed {unresolved[:3]} ({len(unresolved)})"
            add(
                "targets",
                len(down) == expected_count
                and set(by_block) == expect
                and not incomplete
                and not unresolved,
                detail,
            )
    return {
        "path": str(path),
        "bytes": os.path.getsize(path),
        "tensors": len(tensors),
        "dtypes": dtypes,
        "metadata": meta,
        "algo": algo,
        "base_model": base,
        "rank": rank_meta
        if rank_meta is not None
        else (ranks[0] if len(ranks) == 1 else None),
        "alpha": alpha_meta,
        "dtype": next(iter(dtypes)) if len(dtypes) == 1 else None,
        "modules": len(down),
        "blocks": {
            "count": len(block_ids),
            "first": block_ids[0] if block_ids else None,
            "last": block_ids[-1] if block_ids else None,
            "expected": blocks_meta,
        },
        "checks": checks,
        "ok": all(check["ok"] for check in checks),
    }


def render_checkpoint(section):
    """Markdown block for a :func:`checkpoint_report` dict."""
    checks = section.get("checks") or []
    failed = [check for check in checks if not check["ok"]]
    blocks = section.get("blocks") or {}
    span = ""
    if blocks.get("count"):
        span = f" · blocks {blocks['first']}-{blocks['last']} ({blocks['count']}"
        if blocks.get("expected"):
            span += f"/{blocks['expected']}"
        span += ")"
    lines = [
        "## Checkpoint",
        "",
        (
            f"`{section['path']}` — {_fmt_bytes(section['bytes'])} · "
            f"{section['tensors']} tensors · {section['dtype'] or 'mixed dtypes'}"
        ),
        "",
        f"algo {section['algo'] or 'unknown'} · rank {section['rank']} / "
        f"alpha {section['alpha']} · {section['base_model'] or 'no base_model'}"
        f"{span}"
        + (f" · {section['modules']} modules" if section.get("modules") else ""),
        "",
        f"checks {len(checks) - len(failed)}/{len(checks)} passed"
        + ("" if section.get("ok") else " — **FAILED**"),
    ]
    for check in failed:
        lines.append(f"- {check['name']}: {check['detail']}")
    return "\n".join(lines)


def _fmt_bytes(size):
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit in ("B", "KB") else f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} GB"


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="scripts.moss.train_report",
        description="Compact markdown health report for a MOSS training run.",
    )
    parser.add_argument(
        "--progress",
        default="output/logs/moss_lora.progress.jsonl",
        help="progress.jsonl written by library.training.progress",
    )
    parser.add_argument(
        "--log", default=None, help="trainer stdout log (s/sample, peak VRAM)"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_MA_WINDOW,
        help="moving-average window (default 50)",
    )
    parser.add_argument(
        "--deciles",
        type=int,
        default=DEFAULT_DECILES,
        help="loss curve blocks (default 10)",
    )
    parser.add_argument(
        "--stall-s",
        type=float,
        default=DEFAULT_STALL_S,
        help="stall threshold in seconds (default 180)",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit the report dict as JSON"
    )
    parser.add_argument(
        "--no-pid-check",
        action="store_true",
        help="do not probe the trainer pid for liveness",
    )
    parser.add_argument("--max-checkpoints", type=int, default=25)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="also validate a saved LoRA (.safetensors) from its header",
    )
    args = parser.parse_args(argv)

    if not os.path.isfile(args.progress):
        print(f"no progress stream at {args.progress}", file=sys.stderr)
        return 2
    try:
        report = build_report(
            args.progress,
            args.log,
            ma_window=args.window,
            deciles=args.deciles,
            stall_s=args.stall_s,
            check_pid=not args.no_pid_check,
        )
    except (OSError, ValueError) as exc:
        print(f"{args.progress}: {exc}", file=sys.stderr)
        return 2
    checkpoint = None
    if args.checkpoint:
        try:
            checkpoint = checkpoint_report(args.checkpoint)
        except (OSError, ValueError, TypeError) as exc:
            print(f"{args.checkpoint}: {exc}", file=sys.stderr)
            return 2
        report["checkpoint"] = checkpoint
    if args.json:
        json.dump(report, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        text = render_markdown(
            report,
            max_checkpoints=args.max_checkpoints,
            max_samples=args.max_samples,
        )
        if checkpoint is not None:
            text += "\n\n" + render_checkpoint(checkpoint)
        print(text)
    return 2 if checkpoint is not None and not checkpoint["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
