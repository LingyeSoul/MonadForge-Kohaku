"""MOSS fp16 lanes bench: stream ratio and step time per ``fp16_fix``.

The MOSS-Image DiT's 42-block joint residual stream peaks near 6.2e4 on the real
weights (upstream's activation probe), inside fp16's 65504 by only ~6 %. Two
guards keep it finite, and they cost differently:

- ``online`` folds a constant ``c`` into the stream (``enable_stream_scale``): the
  weights are untouched, so the fused block ops still run;
- ``fp32_residual`` carries the stream in fp32 (``enable_fp32_residual``): the
  fused residual GEMM degrades to ``linear + residual kernel``.

**What this bench can and cannot show.** A synthetic transformer does not grow
the stream the way a trained one does: the AdaLN gates carry the signal and are
zero at init, so the absolute peak here is a fixture number, not the model's. The
measurement that transfers is the **ratio** between an unfolded run and a folded
one — that ratio is ``c``, whatever the weights — and the per-step cost of each
lane. For the real headroom, run the upstream activation probe on real weights
(``scripts/probe/activations.py`` there) or read the value in
``docs/moss/README.md``.

Usage::

    .venv/Scripts/python.exe -m bench.moss_fp16_lanes.run_bench [--blocks N] [--c 0.0625]

Drops ``result.json`` into ``bench/moss_fp16_lanes/results/<ts>[-<label>]/``.
"""

from __future__ import annotations

import argparse
import time

import torch

from bench._common import make_run_dir, write_result

FP16_MAX = torch.finfo(torch.float16).max


def _build(blocks: int):
    from library.moss.model import MOSSImageTransformer

    torch.manual_seed(0)
    return MOSSImageTransformer(
        dim=128,
        ffn_dim=192,
        num_heads=4,
        num_layers=blocks,
        text_dim=64,
        latent_channels=8,
    ).eval()


def _inputs():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, 8, 4, 6, generator=g)
    e = torch.randn(1, 5, 64, generator=g)
    mask = torch.ones(1, 5, dtype=torch.bool)
    return x, e, mask


def _peaks(model) -> list[float]:
    """max |x| entering each block, in order.

    On a synthetic transformer the stream does not grow the way a trained one
    does (the AdaLN gates carry the signal, and at init they are zero), so the
    absolute peak here is not the model's. The *ratio* between an unfolded run and
    a folded one is the measurement this bench is after — that ratio is ``c``
    independent of the weights.
    """
    seen = []

    def hook(_index, x, _n_img):
        seen.append(float(x.abs().max()))
        return x

    model.layer_hook = hook
    x, e, mask = _inputs()
    with torch.no_grad():
        model(x, 500.0, e, mask)
    model.layer_hook = None
    return seen


def _steps(model, n_steps: int, dtype=torch.float32) -> dict:
    x, e, mask = _inputs()
    with torch.no_grad():
        for _ in range(3):
            model(x, 500.0, e, mask)
        t0 = time.perf_counter()
        out = None
        for _ in range(n_steps):
            out = model(x, 500.0, e, mask)
        elapsed = time.perf_counter() - t0
    return {
        "finite": bool(torch.isfinite(out.sample).all()),
        "ms_per_step": round(elapsed / n_steps * 1000.0, 3),
    }


def run(blocks: int, c: float, n_steps: int) -> dict:
    base = _build(blocks)
    unfolded = _peaks(base)
    peak = max(unfolded)

    folded = _build(blocks)
    folded.load_state_dict(base.state_dict())
    folded.enable_stream_scale(c)
    folded_peaks = _peaks(folded)

    fp32 = _build(blocks)
    fp32.load_state_dict(base.state_dict())
    fp32.enable_fp32_residual()
    fp32_peaks = _peaks(fp32)

    ratios = [f / u for f, u in zip(folded_peaks, unfolded)]
    return {
        "blocks": blocks,
        "c": c,
        "stream": {
            "unfolded_peak": peak,
            "folded_peak": max(folded_peaks),
            "fp32_peak": max(fp32_peaks),
            "headroom_unfolded": round(FP16_MAX / peak, 3) if peak else None,
            "headroom_folded": round(FP16_MAX / max(folded_peaks), 3)
            if max(folded_peaks)
            else None,
            "fold_ratio_min": round(min(ratios), 6),
            "fold_ratio_max": round(max(ratios), 6),
        },
        "timing": {
            "unguarded": _steps(base, n_steps),
            "online": _steps(folded, n_steps),
            "fp32_residual": _steps(fp32, n_steps),
        },
        "peak_step_bytes": {
            "fp32_residual": "stream held at 4 B/element",
            "online": "stream held at the compute dtype (2 B/element in fp16)",
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blocks", type=int, default=8, help="block count")
    ap.add_argument("--c", type=float, default=0.0625, help="online stream scale")
    ap.add_argument("--steps", type=int, default=10, help="timed forward steps")
    ap.add_argument("--label", default="")
    args = ap.parse_args(argv)

    report = run(args.blocks, args.c, args.steps)
    run_dir = make_run_dir("moss_fp16_lanes", label=args.label or None)
    metrics = {"stream": report["stream"], "timing": report["timing"]}
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label or None,
        device="cpu",
        extra={"peak_step_bytes": report["peak_step_bytes"]},
    )
    print(f"=== moss_fp16_lanes -> {run_dir} ===")
    stream = report["stream"]
    print(
        f"stream peak {stream['unfolded_peak']:.1f} -> {stream['folded_peak']:.1f} "
        f"(c={args.c:g}), headroom {stream['headroom_unfolded']}x -> "
        f"{stream['headroom_folded']}x"
    )
    for lane, timing in report["timing"].items():
        print(
            f"{lane:<14} {timing['ms_per_step']:8.3f} ms/step  "
            f"finite={timing['finite']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
