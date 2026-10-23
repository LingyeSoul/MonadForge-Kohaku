# moss_kernels

What `library/moss/ops.py` claims: the four fused block ops beat their eager composition,
and a DiT block built from them beats the reference block forward + backward at MOSS
shapes. Method and the design behind the kernels: `docs/moss/kernels.md`.

## Run

```bash
python -m bench.moss_kernels.bench_block_ops --label rtx4060
python -m bench.moss_kernels.bench_block_ops --compile --sizes 512,1mp,2mp --label rtx4060-compile
```

Needs a CUDA device with Triton; exits with the op reasons when the stack is unavailable.
`--compile` also times `torch.compile(block.forward)` and the compiled fused block, which
is the comparison that matters when `--compile` is on (it is the training default).

## Output

`results/<YYYYMMDD-HHMM>[-<label>]/result.json`: `backends` (`{op: bound backend}`),
`op_ms`, `block_ms`, `block_speedup` (the worst size), `peak_vram_gb`. The same tables are
printed to stdout.

## Reading it

- `reference` is `model._Block.forward`; `fused` is the same block after
  `library.moss.ops.fuse`; `compile` wraps the reference block the way
  `loader.compile_blocks` does.
- The op rows are forward + backward with weight gradients, on one 4096-token sample.
- A block is GEMM-bound, so expect the block ratio to be far below the op ratios, and
  expect `compile` to sit between `reference` and `fused` on cards where Inductor works.
- Numbers from one card do not transfer: the vendored registry binds per arch, dtype and
  shape, so a new card needs its own run.
