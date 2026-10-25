"""Warm the MOSS kernel tuning tables before a training run.

The vendored op stack tunes every (kernel, problem key) on first use and caches the
winner under ~/.cache/library.moss.vendor.kohakumoss_kernels/tuning/<card>/<tag>/
<kernel>.json.  A fresh run therefore pays the autotune cost inside its first steps
(and, on a busy card, may pick a worse config because the timing runs under whatever
else is resident).  This script replays the op surface the training forward +
backward dispatches on scratch tensors, at the token lengths the dataset actually
feeds the DiT, so the first live step is a table lookup.

Usage::

    .venv/bin/python -m scripts.moss.prewarm_kernels \
        --cache post_image_dataset/moss \
        --dit path/to/moss_image_base_fp16_fixed.safetensors \
        --precision bf16 --batch-size 8

--dry-run prints the plan -- every (kernel, key) problem -- without touching CUDA,
and ends with a JSON summary (--json-only for just that).  A real run refuses to
start unless nvidia-smi reports at least --min-free-mib (default 3000) free, so it
cannot disturb a job that is already training; --allow-busy overrides.

What the trainer dispatches matters, so the summary says it:

* the four block ops (rms_norm_mod, qk_norm_rope, situ_glu, gated_residual) run
  only when library.moss.ops.fuse accepts the model.  train.py passes
  fp32_adapters=True on the fp16 lane and fuse declines that lane outright, so an
  fp16 job keeps the reference branch and reads none of the block-op tables.  The
  bf16 lane (sm80+) is what the block tuning is for.
* attention_varlen is reached by the packed path only (--batch-size > 1); at batch
  1 the blocks call torch SDPA.  On sm70 the Triton attention kernel is not
  available at all and the op falls back to eager, so the plan lists it but a real
  run skips it.

The lengths come from the cache headers (*.moss_latent.safetensors and
*.moss_te.safetensors), never the tensors: image tokens are the latent's patch
grid, text tokens the longest of embeds / embeds_plain / a variant's shared prefix
plus its tail.  Geometry comes from the DiT header
(library.moss.checkpoint.arch_config); without --dit the MOSS-Image-Base numbers
(42 x 3072, 24 heads x 128, ffn 8192) are assumed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from library.env import resolve_under_home
from library.moss.scan import (
    REF_SUFFIX,
    TE_SUFFIX,
    latent_files,
)

# --------------------------------------------------------------------------- #
# Pure helpers mirrored from library/moss/vendor/.../tuning/buckets.py.  They
# live here so --dry-run never imports torch; tests pin them against the vendor.
# --------------------------------------------------------------------------- #
LENGTH_FLOOR = 64
SEGMENT_CAP = 8
ROW_FLOOR = 1024

DTYPE_NAMES = {"fp16": "torch.float16", "bf16": "torch.bfloat16"}

BLOCK_OPS = ("rms_norm_mod", "qk_norm_rope", "situ_glu", "gated_residual")

ATTENTION_FWD_TABLES = ("attn_varlen_fwd_triton",)
ATTENTION_BWD_TABLES = (
    "attn_varlen_dkdv_triton",
    "attn_varlen_dq_triton",
    "attn_varlen_fused_triton",
    "attn_varlen_bwd_strategy_triton",
)

_VARIANT_RE = re.compile(r"embeds_v[1-9][0-9]*")


def _pow2(n: int, floor: int) -> int:
    """Smallest power of two >= max(n, floor) (floor must be a power of two)."""
    b = floor
    while b < n:
        b *= 2
    return b


def _segment_bucket(segments: int) -> int:
    """Bucket of the segment count: 1, 2, 4, then SEGMENT_CAP and above."""
    return min(_pow2(segments, 1), SEGMENT_CAP)


def _pow2_buckets(top: int, floor: int) -> list[int]:
    """One count per power-of-two bucket from floor up to top's, top last."""
    out, n = [], floor
    while True:
        out.append(min(n, top))
        if n >= top:
            return out
        n *= 2


# --------------------------------------------------------------------------- #
# Cache headers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Sample:
    """One cache pair: the lengths the trainer packs into the DiT stream."""

    stem: str
    width: int
    height: int
    image_tokens: int
    text_tokens: int
    ref_tokens: int = 0

    def stream_tokens(self, include_ref: bool) -> int:
        return (
            self.image_tokens
            + self.text_tokens
            + (self.ref_tokens if include_ref else 0)
        )


def read_header(
    path: Path,
) -> tuple[dict[str, tuple[str, tuple[int, ...]]], dict[str, str]]:
    """A safetensors header, without reading any tensor data."""
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError("shorter than the 8-byte header length")
            (size,) = struct.unpack("<Q", prefix)
            if size > 256 * 1024 * 1024:
                raise ValueError(f"implausible header length {size}")
            raw = handle.read(size)
    except OSError as exc:
        raise ValueError(f"{path.name}: cannot read the header ({exc})") from exc
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name}: header is not JSON ({exc})") from exc
    metadata = {
        str(key): str(value)
        for key, value in (document.pop("__metadata__", None) or {}).items()
    }
    tensors = {
        str(name): (str(entry["dtype"]), tuple(int(dim) for dim in entry["shape"]))
        for name, entry in document.items()
    }
    return tensors, metadata


def _tensor_rows(
    tensors: dict[str, tuple[str, tuple[int, ...]]], name: str, path: Path
) -> int:
    if name not in tensors:
        raise ValueError(f"{path.name}: missing the {name!r} tensor")
    shape = tensors[name][1]
    if not shape:
        raise ValueError(f"{path.name}: {name!r} is scalar, expected 2-D")
    return shape[0]


def _text_tokens(
    tensors: dict[str, tuple[str, tuple[int, ...]]],
    metadata: dict[str, str],
    path: Path,
) -> int:
    """Longest text embedding the trainer could load from a TE file."""
    n_text = _tensor_rows(tensors, "embeds", path)
    n_plain = (
        _tensor_rows(tensors, "embeds_plain", path) if "embeds_plain" in tensors else 0
    )
    tails = [
        shape[0]
        for name, (dtype, shape) in tensors.items()
        if _VARIANT_RE.fullmatch(name)
    ]
    try:
        prefix = int(metadata.get("variant_prefix", 0))
    except ValueError as exc:
        raise ValueError(
            f"{path.name}: variant_prefix={metadata.get('variant_prefix')!r} "
            "is not an int"
        ) from exc
    variants = prefix + max(tails) if tails else 0
    return max(n_text, n_plain, variants)


def _ref_tokens(path: Path) -> int:
    tensors, _ = read_header(path)
    if "latent" not in tensors:
        raise ValueError(f"{path.name}: missing the 'latent' tensor")
    shape = tensors["latent"][1]
    if len(shape) != 3:
        raise ValueError(f"{path.name}: latent is {len(shape)}-D, expected (C, H, W)")
    return shape[1] * shape[2]


def read_cache_samples(cache_dir: Path) -> tuple[list[Sample], list[str]]:
    """The cache pairs as (image, text, reference) token counts, plus warnings."""
    if not cache_dir.is_dir():
        raise ValueError(f"cache directory does not exist: {cache_dir}")
    newest: dict[str, tuple[float, int, int, Path]] = {}
    for stem, width, height, path in latent_files(cache_dir):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        previous = newest.get(stem)
        if previous is None or mtime > previous[0]:
            newest[stem] = (mtime, width, height, path)
    samples, warnings = [], []
    for stem in sorted(newest):
        _, width, height, latent_path = newest[stem]
        te_path = cache_dir / f"{stem}{TE_SUFFIX}"
        if not te_path.is_file():
            warnings.append(f"{stem}: no {TE_SUFFIX}; skipped")
            continue
        latent_header, _ = read_header(latent_path)
        if "latent" not in latent_header:
            raise ValueError(f"{latent_path.name}: missing the 'latent' tensor")
        shape = latent_header["latent"][1]
        if len(shape) != 3:
            raise ValueError(
                f"{latent_path.name}: latent is {len(shape)}-D, expected (C, H, W)"
            )
        te_header, te_meta = read_header(te_path)
        ref_path = cache_dir / f"{stem}{REF_SUFFIX}"
        samples.append(
            Sample(
                stem=stem,
                width=width,
                height=height,
                image_tokens=shape[1] * shape[2],
                text_tokens=_text_tokens(te_header, te_meta, te_path),
                ref_tokens=_ref_tokens(ref_path) if ref_path.is_file() else 0,
            )
        )
    return samples, warnings


# --------------------------------------------------------------------------- #
# Plan: geometry, lengths, problems, runs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Geometry:
    dim: int
    layers: int
    heads: int
    head_dim: int
    ffn_dim: int


MOSS_IMAGE_GEOMETRY = Geometry(
    dim=3072, layers=42, heads=24, head_dim=128, ffn_dim=8192
)


@dataclass(frozen=True)
class Problem:
    """One (kernel, key) row the warm-up would write to the tuning table."""

    op: str
    mode: str  # "fwd" | "bwd"
    kernel: str
    key: str
    tokens: int
    segments: int = 1


@dataclass(frozen=True)
class AttentionRun:
    op: str
    max_len: int
    segments: int
    sizes: tuple[int, ...]


@dataclass(frozen=True)
class BlockRun:
    op: str
    rows: int
    segments: int


@dataclass(frozen=True)
class Plan:
    cache_dir: Path
    samples: tuple[Sample, ...]
    skipped: tuple[tuple[str, int], ...]
    geometry: Geometry
    geometry_source: str
    precision: str
    dtype: str
    batch_size: int
    from_cache: bool
    lengths: tuple[int, ...]
    problems: tuple[Problem, ...]
    runs: tuple[object, ...]
    notes: tuple[str, ...]


def _attention_key(dtype: str, geometry: Geometry, max_len: int, segments: int) -> str:
    """Mirror of triton/attention/varlen.py::_key."""
    return (
        f"{dtype}-h{geometry.heads}-d{geometry.head_dim}"
        f"-L{_pow2(max_len, LENGTH_FLOOR)}-S{_segment_bucket(segments)}"
    )


def _block_keys(
    op: str, geometry: Geometry, dtype: str, rows: int
) -> tuple[tuple[str, str], tuple[str, str]]:
    """((fwd kernel, key), (bwd kernel, key)) for one block op.

    Keys mirror the Triton block kernels for the trainer's shape: the modulation
    weights, per-sample shift/scale and gates are frozen, so every backward key
    carries needs_* = False.  rows is the stream's row count (the backward row
    bucket is pow2(rows, 1024)).
    """
    n_bucket = _pow2(rows, ROW_FLOOR)
    if op == "rms_norm_mod":
        return (
            ("triton_rms_norm_fwd", f"d{geometry.dim}-{dtype}-TrueTrueTrue"),
            (
                "triton_rms_norm_bwd",
                f"d{geometry.dim}-{dtype}-TrueTrueTrue-False-n{n_bucket}",
            ),
        )
    if op == "qk_norm_rope":
        return (
            (
                "triton_qk_rope_fwd",
                f"dh{geometry.head_dim}-h{geometry.heads}-{dtype}",
            ),
            (
                "triton_qk_rope_bwd",
                f"dh{geometry.head_dim}-h{geometry.heads}-{dtype}-False",
            ),
        )
    if op == "situ_glu":
        return (
            ("triton_situ_fwd", f"f{geometry.ffn_dim}-{dtype}"),
            ("triton_situ_bwd", f"f{geometry.ffn_dim}-{dtype}"),
        )
    if op == "gated_residual":
        return (
            ("triton_gated_residual_fwd", f"d{geometry.dim}-{dtype}-True"),
            (
                "triton_gated_residual_bwd",
                f"d{geometry.dim}-{dtype}-False-n{n_bucket}",
            ),
        )
    raise ValueError(f"unknown block op {op!r}")


def _lane_notes(precision: str, batch_size: int, has_refs: bool) -> tuple[str, ...]:
    notes = []
    if precision == "fp16":
        notes.append(
            "fp16 lane: train.py passes fp32_adapters=True, ops.fuse() declines, "
            "so the four block ops stay on the reference branch"
        )
    if batch_size < 2:
        notes.append(
            "batch 1: blocks use torch SDPA, so attention_varlen is not called "
            "(the packed path needs --batch-size > 1)"
        )
    if has_refs and batch_size > 1:
        notes.append(
            "reference latents are ignored with --batch-size > 1 "
            "(packed training rejects them)"
        )
    return tuple(notes)


def build_plan(
    cache_dir: Path,
    samples: Sequence[Sample],
    geometry: Geometry,
    geometry_source: str,
    precision: str,
    batch_size: int,
    from_cache: bool,
    max_len: int = 0,
) -> Plan:
    """The problems and runs for the active samples (no CUDA, no tensor math)."""
    if precision not in DTYPE_NAMES:
        raise ValueError(f"precision must be one of {sorted(DTYPE_NAMES)}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    include_ref = batch_size == 1
    kept: list[Sample] = []
    skipped: list[tuple[str, int]] = []
    for sample in samples:
        joint = sample.stream_tokens(include_ref)
        if max_len and joint > max_len:
            skipped.append((sample.stem, joint))
        else:
            kept.append(sample)
    if not kept:
        raise ValueError(
            f"no sample fits --max-len ({max_len}); {len(skipped)} longer than it"
        )
    dtype = DTYPE_NAMES[precision]
    joints = sorted({sample.stream_tokens(include_ref) for sample in kept})
    has_refs = any(sample.ref_tokens for sample in kept)

    problems: list[Problem] = []
    runs: list[object] = []

    # Attention: one run per length bucket (the vendor's power-of-two cascade
    # unless --from-cache keeps only lengths the dataset actually feeds).
    candidates = (
        list(joints) if from_cache else _pow2_buckets(max(joints), LENGTH_FLOOR)
    )
    seen_attention: set[str] = set()
    used_lengths: list[int] = []
    for length in candidates:
        bucket = _pow2(length, LENGTH_FLOOR)
        key = _attention_key(dtype, geometry, length, batch_size)
        if key in seen_attention:
            continue
        seen_attention.add(key)
        in_bucket = [j for j in joints if _pow2(j, LENGTH_FLOOR) == bucket]
        rep = max(in_bucket) if in_bucket else length
        runs.append(
            AttentionRun("attention_varlen", rep, batch_size, (rep,) * batch_size)
        )
        used_lengths.append(rep)
        problems.extend(
            Problem("attention_varlen", "fwd", table, key, rep, batch_size)
            for table in ATTENTION_FWD_TABLES
        )
        problems.extend(
            Problem("attention_varlen", "bwd", table, key, rep, batch_size)
            for table in ATTENTION_BWD_TABLES
        )

    # Blocks: one run per (op, row bucket); the representative is the largest
    # observed joint in the bucket.
    row_buckets: dict[int, int] = {}
    for joint in joints:
        rows = joint * batch_size
        bucket = _pow2(rows, ROW_FLOOR)
        row_buckets[bucket] = max(row_buckets.get(bucket, 0), joint)
    for op in BLOCK_OPS:
        for bucket in sorted(row_buckets):
            rows = row_buckets[bucket] * batch_size
            runs.append(BlockRun(op, rows, batch_size))
            fwd, bwd = _block_keys(op, geometry, dtype, rows)
            problems.append(Problem(op, "fwd", fwd[0], fwd[1], rows, batch_size))
            problems.append(Problem(op, "bwd", bwd[0], bwd[1], rows, batch_size))

    return Plan(
        cache_dir=cache_dir,
        samples=tuple(kept),
        skipped=tuple(skipped),
        geometry=geometry,
        geometry_source=geometry_source,
        precision=precision,
        dtype=dtype,
        batch_size=batch_size,
        from_cache=from_cache,
        lengths=tuple(used_lengths),
        problems=tuple(problems),
        runs=tuple(runs),
        notes=_lane_notes(precision, batch_size, has_refs),
    )


# --------------------------------------------------------------------------- #
# Tuning-cache lookup (header-only; no torch)
# --------------------------------------------------------------------------- #
def tuning_root() -> Path:
    """<cache root>/tuning exactly as the vendored stack lays it out."""
    override = os.environ.get("KOHAKUMOSS_KERNEL_CACHE")
    root = (
        Path(override)
        if override
        else Path.home() / ".cache" / "library.moss.vendor.kohakumoss_kernels"
    )
    return root / "tuning"


def cached_problems(
    root: Path, problems: Sequence[Problem]
) -> tuple[dict[tuple[str, str], str], list[str]]:
    """{(kernel, key): "<card>/<tag>"} already on disk, plus the card dirs."""
    wanted = {(problem.kernel, problem.key) for problem in problems}
    kernels = {problem.kernel for problem in problems}
    hits: dict[tuple[str, str], str] = {}
    cards = []
    if not root.is_dir():
        return hits, cards
    for card in sorted(path for path in root.iterdir() if path.is_dir()):
        cards.append(card.name)
        for tag in sorted(path for path in card.iterdir() if path.is_dir()):
            for kernel in kernels:
                table = tag / f"{kernel}.json"
                if not table.is_file():
                    continue
                try:
                    rows = json.loads(table.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                for key in rows:
                    if (kernel, key) in wanted:
                        hits[(kernel, key)] = f"{card.name}/{tag.name}"
    return hits, cards


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _length_summary(plan: Plan) -> dict:
    include_ref = plan.batch_size == 1
    joints = sorted(sample.stream_tokens(include_ref) for sample in plan.samples)
    return {
        "image": sorted({sample.image_tokens for sample in plan.samples}),
        "text": sorted({sample.text_tokens for sample in plan.samples}),
        "ref": sorted(
            {sample.ref_tokens for sample in plan.samples if sample.ref_tokens}
        ),
        "joint_min": joints[0],
        "joint_max": joints[-1],
        "joint_unique": len(set(joints)),
    }


def summarize(
    plan: Plan,
    cached: dict[tuple[str, str], str],
    cards: Sequence[str],
    *,
    dry_run: bool,
    device: str | None = None,
    backends: dict[str, str] | None = None,
    run_seconds: Sequence[float] | None = None,
    failures: Sequence[dict] | None = None,
    skipped_ops: Sequence[str] = (),
) -> dict:
    problems = []
    for problem in plan.problems:
        hit = (problem.kernel, problem.key) in cached
        if any(problem.op == op for op in skipped_ops):
            state = "skipped"
        elif hit:
            state = "cached"
        elif dry_run:
            state = "would tune"
        else:
            state = "pending"
        problems.append(
            {
                "op": problem.op,
                "mode": problem.mode,
                "kernel": problem.kernel,
                "key": problem.key,
                "tokens": problem.tokens,
                "segments": problem.segments,
                "state": state,
            }
        )
    counts = {
        "runs": len(plan.runs),
        "problems": len(problems),
        "cached": sum(1 for item in problems if item["state"] == "cached"),
        "would_tune": sum(1 for item in problems if item["state"] == "would tune"),
        "tuned": sum(1 for item in problems if item["state"] == "tuned"),
        "pending": sum(1 for item in problems if item["state"] == "pending"),
        "skipped": sum(1 for item in problems if item["state"] == "skipped"),
    }
    return {
        "dry_run": dry_run,
        "cache": str(plan.cache_dir),
        "samples": len(plan.samples),
        "lengths": _length_summary(plan),
        "geometry": {
            "dim": plan.geometry.dim,
            "layers": plan.geometry.layers,
            "heads": plan.geometry.heads,
            "head_dim": plan.geometry.head_dim,
            "ffn_dim": plan.geometry.ffn_dim,
            "source": plan.geometry_source,
        },
        "precision": plan.precision,
        "dtype": plan.dtype,
        "batch_size": plan.batch_size,
        "mode": "from-cache" if plan.from_cache else "buckets",
        "attention_lengths": list(plan.lengths),
        "device": device,
        "backends": backends,
        "tuning_cache": {"path": str(tuning_root()), "cards": list(cards)},
        "problems": problems,
        "counts": counts,
        "notes": list(plan.notes),
        "skipped_samples": [
            {"stem": stem, "joint_tokens": joint} for stem, joint in plan.skipped
        ],
        "run_seconds": list(run_seconds or []),
        "failures": list(failures or []),
    }


def render_plan(
    plan: Plan, cached: dict[tuple[str, str], str], cards: Sequence[str]
) -> str:
    lengths = _length_summary(plan)
    ref = f", ref {lengths['ref'][0]}..{lengths['ref'][-1]}" if lengths["ref"] else ""
    mode = "observed only" if plan.from_cache else "power-of-two; --from-cache"
    lines = [
        f"cache       {plan.cache_dir}  ({len(plan.samples)} samples)",
        (
            f"lengths     image {lengths['image']}, "
            f"text {lengths['text']}{ref}, "
            f"joint {lengths['joint_min']}..{lengths['joint_max']}"
        ),
        (
            f"geometry    {plan.geometry.layers} x {plan.geometry.dim}, "
            f"{plan.geometry.heads} heads x {plan.geometry.head_dim}, "
            f"ffn {plan.geometry.ffn_dim}  ({plan.geometry_source})"
        ),
        f"dtype       {plan.dtype}   batch {plan.batch_size}",
        f"attention   lengths {plan.lengths}  ({mode})",
        f"tuning      {tuning_root()}  (cards: {', '.join(cards) or 'none yet'})",
        "",
    ]
    width_op = max([len(problem.op) for problem in plan.problems] + [10])
    width_kernel = max([len(problem.kernel) for problem in plan.problems] + [20])
    for problem in plan.problems:
        mark = " [cached]" if (problem.kernel, problem.key) in cached else ""
        lines.append(
            f"{problem.op:<{width_op}}  {problem.mode:<3}  "
            f"{problem.kernel:<{width_kernel}}  {problem.key}{mark}"
        )
    lines.append("")
    to_tune = sum(
        1 for problem in plan.problems if (problem.kernel, problem.key) not in cached
    )
    lines.append(
        f"{len(plan.problems)} problem(s) over {len(plan.runs)} run(s): "
        f"{to_tune} to tune, {len(plan.problems) - to_tune} already cached"
    )
    for note in plan.notes:
        lines.append(f"note  {note}")
    if plan.skipped:
        listed = ", ".join(f"{stem} ({joint})" for stem, joint in plan.skipped[:5])
        tail = " ..." if len(plan.skipped) > 5 else ""
        lines.append(
            f"skipped {len(plan.skipped)} sample(s) over --max-len: {listed}{tail}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Busy guard
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GpuMemory:
    used_mib: int
    total_mib: int

    @property
    def free_mib(self) -> int:
        return self.total_mib - self.used_mib


def query_gpu_memory(runner=None) -> GpuMemory:
    """nvidia-smi's used/total MiB; refuse to guess when it does not answer."""
    runner = runner or subprocess.run
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = runner(command, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"cannot query the GPU with nvidia-smi: {exc}") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip() or f"exit {proc.returncode}"
        raise ValueError(f"nvidia-smi failed: {detail}")
    lines = [line for line in (proc.stdout or "").strip().splitlines() if line.strip()]
    if not lines:
        raise ValueError("nvidia-smi printed no GPU rows")
    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 2:
        raise ValueError(
            f"cannot parse nvidia-smi output {lines[0]!r} (expected 'used, total')"
        )
    try:
        used, total = int(fields[0]), int(fields[1])
    except ValueError as exc:
        raise ValueError(f"cannot parse nvidia-smi output {lines[0]!r} as MiB") from exc
    return GpuMemory(used_mib=used, total_mib=total)


def check_gpu_free(min_free_mib: int, allow_busy: bool, runner=None) -> GpuMemory:
    """Refuse a real warm-up unless the card has min_free_mib free."""
    memory = query_gpu_memory(runner)
    if memory.free_mib < min_free_mib and not allow_busy:
        raise SystemExit(
            f"GPU is busy: {memory.used_mib} MiB used / {memory.total_mib} MiB "
            f"total, {memory.free_mib} MiB free < {min_free_mib} MiB required. "
            "Stop the other job or pass --allow-busy."
        )
    return memory


# --------------------------------------------------------------------------- #
# Real warm-up
# --------------------------------------------------------------------------- #
def _runtime_dtype(precision: str, device, torch):
    """The model's compute dtype, mirroring loader.resolve_compute_dtype."""
    if precision == "fp16":
        return torch.float16
    if torch.cuda.get_device_capability(device)[0] < 8:
        raise SystemExit(
            "bf16 needs sm80+; this card is below it. Use --precision fp16, or "
            "warm the kernels on the bf16 machine the training job runs on."
        )
    return torch.bfloat16


def _attention_runner(plan: Plan, run: AttentionRun, device, dtype, torch):
    def body(bound):
        n = sum(run.sizes)
        q, k, v = (
            torch.randn(
                n,
                plan.geometry.heads,
                plan.geometry.head_dim,
                device=device,
                dtype=dtype,
            )
            for _ in range(3)
        )
        cu = torch.tensor([0, *run.sizes], device=device, dtype=torch.int64).cumsum(0)
        cu = cu.to(torch.int32)
        leaves = [tensor.detach().requires_grad_() for tensor in (q, k, v)]
        scale = plan.geometry.head_dim**-0.5
        out = bound(*leaves, cu, run.sizes[0], scale)
        out.sum().backward()

    return body


def _block_runner(plan: Plan, run: BlockRun, device, dtype, torch, ops):
    geometry = plan.geometry
    rows, segments = run.rows, run.segments
    if rows % segments:
        raise ValueError(f"{rows} rows do not split into {segments} segments")

    def body(bound):
        if segments == 1:
            seg = torch.zeros(rows, dtype=torch.int32, device=device)
        else:
            seg = ops.segments(segments, rows // segments, device)
        if run.op == "rms_norm_mod":
            x = torch.randn(rows, geometry.dim, device=device, dtype=dtype)
            weight = torch.ones(geometry.dim, device=device, dtype=dtype)
            shift = torch.zeros(segments, geometry.dim, device=device, dtype=dtype)
            scale = torch.ones(segments, geometry.dim, device=device, dtype=dtype)
            y = bound(x.requires_grad_(), weight, shift, scale, seg, 1e-6)
            y.sum().backward()
        elif run.op == "qk_norm_rope":
            q = torch.randn(
                rows, geometry.heads, geometry.head_dim, device=device, dtype=dtype
            ).requires_grad_()
            k = torch.randn(
                rows, geometry.heads, geometry.head_dim, device=device, dtype=dtype
            ).requires_grad_()
            q_weight = torch.ones(geometry.head_dim, device=device, dtype=dtype)
            k_weight = torch.ones(geometry.head_dim, device=device, dtype=dtype)
            pos = torch.zeros(rows, 4, device=device, dtype=torch.float32)
            num_bands = geometry.head_dim // 8
            exponent = (
                torch.arange(num_bands, device=device, dtype=torch.float32) / num_bands
            )
            bands = torch.pow(torch.tensor(10000.0, device=device), -exponent).to(
                torch.float32
            )
            q_out, k_out = bound(q, k, q_weight, k_weight, pos, bands, 1e-6)
            (q_out.sum() + k_out.sum()).backward()
        elif run.op == "situ_glu":
            x = torch.randn(
                rows, 2 * geometry.ffn_dim, device=device, dtype=dtype
            ).requires_grad_()
            bound(x).sum().backward()
        elif run.op == "gated_residual":
            x = torch.randn(rows, geometry.dim, device=device, dtype=dtype)
            o = torch.randn(rows, geometry.dim, device=device, dtype=dtype)
            gate = torch.ones(segments, geometry.dim, device=device, dtype=dtype)
            out = bound(x.requires_grad_(), o.requires_grad_(), gate, seg, 1.0)
            out.sum().backward()
        else:
            raise ValueError(f"unknown block op {run.op!r}")

    return body


def run_plan(
    plan: Plan,
    memory: GpuMemory,
    cached: dict[tuple[str, str], str],
    quiet: bool = False,
) -> dict:
    """Execute the plan on the current CUDA device and report per problem."""
    import torch

    from library.moss import ops
    from library.moss.vendor.kohakumoss_kernels.registry import Bound, exercise

    def say(message: str) -> None:
        if not quiet:
            print(message, flush=True)

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device is visible; --dry-run needs no GPU")
    device = torch.device("cuda")
    dtype = _runtime_dtype(plan.precision, device, torch)
    name = torch.cuda.get_device_name(device)
    say(
        f"GPU         {name}  ({memory.used_mib} MiB used / "
        f"{memory.total_mib} MiB total, {memory.free_mib} MiB free)"
    )

    backends = {}
    skipped_ops = []
    for op in ("attention_varlen", *BLOCK_OPS):
        bound = ops.bind(op, device, dtype)
        backend = getattr(bound, "backend", type(bound).__name__)
        backends[op] = backend
        if backend == "eager":
            skipped_ops.append(op)
    for op, backend in backends.items():
        say(f"backend     {op}: {backend}")

    seconds: list[float] = []
    failures: list[dict] = []
    for run in plan.runs:
        if run.op in skipped_ops:
            where = getattr(run, "max_len", None) or getattr(run, "rows", None)
            say(f"skip        {run.op} ({where}): eager fallback, no table")
            continue
        bound = ops.bind(run.op, device, dtype)
        if isinstance(run, AttentionRun):
            body = _attention_runner(plan, run, device, dtype, torch)
        else:
            body = _block_runner(plan, run, device, dtype, torch, ops)
        started = time.perf_counter()
        try:
            if isinstance(bound, Bound):
                exercise(bound, body)
            else:
                body(bound)
        except Exception as exc:  # noqa: BLE001 - keep tuning the rest
            failures.append(
                {
                    "op": run.op,
                    "tokens": getattr(run, "max_len", None)
                    or getattr(run, "rows", None),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            say(f"FAILED      {run.op}: {type(exc).__name__}: {exc}")
            continue
        elapsed = time.perf_counter() - started
        seconds.append(elapsed)
        if isinstance(run, AttentionRun):
            where = f"{run.max_len} tokens x {run.segments} seg"
        else:
            where = f"{run.rows} rows x {run.segments} seg"
        say(f"tuned       {run.op:<16} {where:<28} {elapsed:6.2f}s")
        torch.cuda.empty_cache()

    after, _ = cached_problems(tuning_root(), plan.problems)
    merged = dict(cached)
    merged.update(after)
    return {
        "device": name,
        "backends": backends,
        "skipped_ops": skipped_ops,
        "seconds": seconds,
        "failures": failures,
        "cached_after": merged,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _geometry_from_dit(path: Path) -> tuple[Geometry, str]:
    from library.moss.checkpoint import arch_config

    try:
        config = arch_config(str(path))
    except Exception as exc:
        raise SystemExit(f"cannot read the DiT geometry from {path}: {exc}") from exc
    geometry = Geometry(
        dim=config["dim"],
        layers=config["num_layers"],
        heads=config["num_heads"],
        head_dim=config["dim"] // config["num_heads"],
        ffn_dim=config["ffn_dim"],
    )
    return geometry, f"DiT header: {path.name}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Warm the MOSS kernel tuning tables from the dataset's lengths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cache",
        default="post_image_dataset/moss",
        help="MOSS cache directory (the trainer's --cache)",
    )
    parser.add_argument(
        "--dit",
        default=None,
        help="single-file DiT safetensors; geometry is read from its header",
    )
    parser.add_argument(
        "--precision",
        default="fp16",
        choices=sorted(DTYPE_NAMES),
        help="compute dtype the warm-up binds (match the training run)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="samples per batch; >1 also warms packed attention segments",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=0,
        help="skip samples whose joint stream is longer than this (0 = no cap)",
    )
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="tune only the lengths the dataset feeds, not the power-of-two cascade",
    )
    parser.add_argument(
        "--min-free-mib",
        type=int,
        default=3000,
        help="free MiB a real run requires (see the busy guard)",
    )
    parser.add_argument(
        "--allow-busy",
        action="store_true",
        help="skip the busy guard (the warm-up shares the card with the job)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and exit without touching CUDA",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="print only the JSON summary",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    quiet = args.json_only
    if not args.dry_run:
        memory = check_gpu_free(args.min_free_mib, args.allow_busy)
    cache_dir = resolve_under_home(args.cache)
    try:
        samples, warnings = read_cache_samples(cache_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr, flush=True)
    if not samples:
        raise SystemExit(f"no complete (latent + TE) pairs under {cache_dir}")
    if args.dit:
        geometry, source = _geometry_from_dit(resolve_under_home(args.dit))
    else:
        geometry, source = MOSS_IMAGE_GEOMETRY, "built-in MOSS-Image-Base geometry"
    try:
        plan = build_plan(
            cache_dir,
            samples,
            geometry,
            source,
            args.precision,
            args.batch_size,
            args.from_cache,
            args.max_len,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    cached, cards = cached_problems(tuning_root(), plan.problems)
    if not quiet:
        print(render_plan(plan, cached, cards), flush=True)
    if args.dry_run:
        summary = summarize(plan, cached, cards, dry_run=True)
        if not quiet:
            print()
            print("-- json summary --", flush=True)
        print(json.dumps(summary, indent=2), flush=True)
        return 0

    result = run_plan(plan, memory, cached, quiet=quiet)
    summary = summarize(
        plan,
        result["cached_after"],
        cards,
        dry_run=False,
        device=result["device"],
        backends=result["backends"],
        run_seconds=result["seconds"],
        failures=result["failures"],
        skipped_ops=result["skipped_ops"],
    )
    print(json.dumps(summary, indent=2), flush=True)
    if not quiet:
        print(
            f"done: {summary['counts']['tuned']} tuned, "
            f"{summary['counts']['cached']} already cached, "
            f"{summary['counts']['skipped']} skipped, "
            f"{len(result['failures'])} failed",
            flush=True,
        )
    return 0 if not result["failures"] else 2


if __name__ == "__main__":
    sys.exit(main())
