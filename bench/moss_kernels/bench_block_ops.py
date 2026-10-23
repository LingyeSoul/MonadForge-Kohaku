#!/usr/bin/env python3
"""MOSS block ops: fused kernels against the eager reference and ``torch.compile``.

Measures the two things ``library/moss/ops.py`` claims: the four block ops cost less
than their eager composition, and a whole DiT block built from them costs less than the
reference block, forward + backward, at MOSS shapes.

The fused path is the vendored KohakuMoss kernels (``library.moss.vendor``); the
reference path is ``model._Block.forward`` itself; ``compile`` wraps that same
reference block the way ``loader.compile_blocks`` does in training.

    python -m bench.moss_kernels.bench_block_ops --label rtx4060
    python -m bench.moss_kernels.bench_block_ops --compile --label rtx4060-compile

Writes ``result.json`` (the standard envelope) into ``bench/moss_kernels/results/``;
paste its ``metrics`` into the PR.
"""

from __future__ import annotations

import argparse

import torch

from bench._common import make_run_dir, write_result
from library.moss import ops
from library.moss.model import _Block, _Fused, _RoPE
from library.moss.vendor.kohakumoss_kernels import registry

DIM, FFN, HEADS = 3072, 8192, 24
SIZES = {"512": 1024 + 192, "1mp": 4096 + 192, "2mp": 8192 + 192}


def _time(fn, warmup: int, rounds: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(rounds):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end))
    return best


def _op_rows(device, dtype, rounds: int):
    """``(op, tokens, eager ms, fused ms)``, forward + backward, MOSS DiT shapes."""
    torch.manual_seed(1)
    n = 4096
    x = torch.randn(n, DIM, device=device, dtype=dtype)
    weight = torch.randn(DIM, device=device, dtype=torch.float32)
    shift = torch.randn(1, DIM, device=device, dtype=torch.float32)
    scale = torch.randn(1, DIM, device=device, dtype=torch.float32)
    seg = ops.segments(1, n, device)
    up = torch.randn(n, 2 * FFN, device=device, dtype=dtype)
    o = torch.randn(n, DIM, device=device, dtype=dtype)
    gate = torch.randn(1, DIM, device=device, dtype=torch.float32)
    head_dim = DIM // HEADS
    q = torch.randn(n, HEADS, head_dim, device=device, dtype=dtype)
    k = torch.randn(n, HEADS, head_dim, device=device, dtype=dtype)
    qw = torch.randn(head_dim, device=device, dtype=torch.float32)
    kw = torch.randn(head_dim, device=device, dtype=torch.float32)
    pos = torch.rand(n, 4, device=device) * 64
    bands = torch.rand(head_dim // 8, device=device) * 0.5 + 0.5
    for tensor in (x, weight, up, o, gate, q, k, qw, kw):
        tensor.requires_grad_(True)

    cases = (
        (
            "rms_norm_mod",
            lambda fn: fn(x, weight, shift, scale, seg, 1e-6),
            (x, weight),
        ),
        ("situ_glu", lambda fn: fn(up), (up,)),
        ("gated_residual", lambda fn: fn(x, o, gate, seg, 1.0), (x, o, gate)),
        ("qk_norm_rope", lambda fn: fn(q, k, qw, kw, pos, bands, 1e-6), (q, k, qw, kw)),
    )
    rows = []
    for name, call, tensors in cases:

        def step(fn):
            for t in tensors:
                if t.is_floating_point():
                    t.grad = None
            out = call(fn)
            out = out[0] if isinstance(out, tuple) else out
            out.float().square().mean().backward()

        eager = registry.resolve(name, device, dtype, backend="eager")
        fused = ops.bind(name, device, dtype)
        rows.append(
            (
                name,
                n,
                _time(lambda: step(eager), 3, rounds),
                _time(lambda: step(fused), 3, rounds),
            )
        )
    return rows


def _block(device, dtype, fused: bool):
    torch.manual_seed(0)
    block = _Block(DIM, FFN, HEADS).to(device=device, dtype=dtype).train()
    for norm in (block.norm1, block.norm2):
        norm.weight.data.normal_()
    block.fused = fused
    return block


def _block_step(block, x, rope, mod, ctx):
    def step():
        for parameter in block.parameters():
            parameter.grad = None
        block(x, rope, mod, ctx=ctx).square().mean().backward()

    return step


def _block_row(device, dtype, size: str, rounds: int, compile: bool):
    tokens = SIZES[size]
    torch.manual_seed(0)
    x = torch.randn(1, tokens, DIM, device=device, dtype=dtype)
    mod = torch.randn(1, 6, DIM, device=device)
    pos = torch.rand(tokens, 4, device=device) * 64
    rope_mod = _RoPE(DIM // HEADS, 10000.0).to(device)
    rope = rope_mod.embed(pos).to(dtype)
    ctx = _Fused(
        pos=pos.unsqueeze(0),
        bands=rope_mod.bands,
        seg=ops.segments(1, tokens, device),
    )

    reference = _block(device, dtype, fused=False)
    reference_ms = _time(_block_step(reference, x, rope, mod, None), 2, rounds)
    del reference
    torch.cuda.empty_cache()

    block = _block(device, dtype, fused=True)
    fused_ms = _time(_block_step(block, x, rope, mod, ctx), 2, rounds)
    del block
    torch.cuda.empty_cache()

    compiled_ms = None
    compiled_fused_ms = None
    if compile:
        block = _block(device, dtype, fused=False)
        block.forward = torch.compile(block.forward, dynamic=True)
        compiled_ms = _time(_block_step(block, x, rope, mod, None), 2, rounds)
        del block
        torch.cuda.empty_cache()

        block = _block(device, dtype, fused=True)
        block.forward = torch.compile(block.forward, dynamic=True)
        compiled_fused_ms = _time(_block_step(block, x, rope, mod, ctx), 2, rounds)
        del block
        torch.cuda.empty_cache()
    return {
        "size": size,
        "tokens": tokens,
        "reference": reference_ms,
        "fused": fused_ms,
        "compile": compiled_ms,
        "fused_compile": compiled_fused_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--compile", action="store_true", help="also time torch.compile"
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--sizes", default="512,1mp", help="comma list of 512,1mp,2mp")
    args = parser.parse_args()

    device, dtype = torch.device(args.device), torch.bfloat16
    if not ops.available(device):
        raise SystemExit(f"fused kernels unavailable: {ops.describe(device, dtype)}")
    torch.cuda.reset_peak_memory_stats()

    op_rows = _op_rows(device, dtype, args.rounds)
    block_rows = [
        _block_row(device, dtype, size, args.rounds, args.compile)
        for size in args.sizes.split(",")
    ]

    print("| op | tokens | eager | fused | x |")
    print("| --- | --- | --- | --- | --- |")
    for name, n, eager_ms, fused_ms in op_rows:
        print(
            f"| {name} | {n} | {eager_ms:.3f} | {fused_ms:.3f} | {eager_ms / fused_ms:.2f} |"
        )
    print()
    if args.compile:
        print("| size | tokens | reference | compile | fused | fused+compile |")
        print("| --- | --- | --- | --- | --- | --- |")
    else:
        print("| size | tokens | reference | fused | x |")
        print("| --- | --- | --- | --- | --- |")
    for row in block_rows:
        if args.compile:
            print(
                f"| {row['size']} | {row['tokens']} | {row['reference']:.2f} | "
                f"{row['compile']:.2f} | {row['fused']:.2f} | "
                f"{row['fused_compile']:.2f} |"
            )
        else:
            ratio = row["reference"] / row["fused"]
            print(
                f"| {row['size']} | {row['tokens']} | {row['reference']:.2f} | "
                f"{row['fused']:.2f} | {ratio:.2f} |"
            )

    run_dir = make_run_dir("moss_kernels", args.label)
    write_result(
        run_dir,
        script=__file__,
        args=vars(args),
        metrics={
            "backends": ops.describe(device, dtype),
            "op_ms": [
                {"op": n, "tokens": t, "eager": round(e, 3), "fused": round(f, 3)}
                for n, t, e, f in op_rows
            ],
            "block_ms": [
                {
                    k: (round(v, 2) if isinstance(v, float) else v)
                    for k, v in row.items()
                }
                for row in block_rows
            ],
            "block_speedup": round(
                min(row["reference"] / row["fused"] for row in block_rows), 3
            ),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
        },
        label=args.label,
        device=device,
    )
    print(f"\nwrote {run_dir}")


if __name__ == "__main__":
    main()
