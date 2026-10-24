# bench/moss_fp16_lanes — the two fp16 guards, measured

## Why this exists

The MOSS-Image DiT's 42-block joint residual stream peaks near **6.2e4** on the
real weights — inside fp16's 65504 by about 6 % (upstream KohakuMossTrainer's
activation probe, `docs/concepts/numerics.md` upstream). A trained LoRA moves that
number around, so fp16 needs a guard. This repo ships the two upstream guards:

| `fp16_fix` | mechanism | what it costs |
|---|---|---|
| `online` (default) | fold a constant `c` into the joint and text streams: stream `×c`, each block's two AdaLN gate rows `×c`, every stream-reading RMSNorm's eps `×c²`. Exact reparameterisation — the function is unchanged. | one multiply per writer; the stream stays 16-bit, so the fused block ops still run |
| `fp32_residual` | carry the stream in fp32, keep the matmuls fp16 | 2× the stream's bytes; the fused residual GEMM degrades to `linear + residual kernel` |

`c = 1/16` (the default, `fp16_fix_c`) takes the base's ~6.2e4 peak to ~3.9e3,
leaving ~16× headroom for a trained adapter.

## `run_bench.py` — ratio and step time

Builds a tiny synthetic MOSS transformer (CPU, no real weights) and measures, per
lane:

- `stream.fold_ratio_min` / `fold_ratio_max` — max |x| entering each block, folded
  over unfolded. **This is the measurement that transfers**: it is `c`, whatever
  the weights are.
- `timing.{unguarded,online,fp32_residual}.ms_per_step` — per-step wall time.
- `timing.*.finite` — the output stays finite.

```bash
.venv/Scripts/python.exe -m bench.moss_fp16_lanes.run_bench [--blocks N] [--c 0.0625] [--steps N]
```

Result lands in `bench/moss_fp16_lanes/results/<ts>[-<label>]/result.json`.

## Two caveats

**Timing.** On the CPU the three lanes land within run-to-run noise of each other
(5.9–8.4 ms/step on the same 6-block fixture, `online` measuring both "slower" and
"faster" than `fp32_residual` across runs). CPU autocast dispatch does not
represent tensor-core throughput, and the fold's cost is one elementwise multiply
per writer. Run on the V100/T4 target for numbers that mean anything; the ratio
and the finiteness check are what the CI run is for.

**The absolute peak.** A synthetic transformer's AdaLN gates are zero at init, so
its stream does not accumulate the way a trained 42-block stack does — the
fixture's peak is a fixture number (a few units), and the headroom columns derived
from it are not the model's headroom. Reproducing the real overflow needs real
weights: upstream's `scripts/probe/activations.py` on the checkpoint, fp32 model,
prompt × sigma × resolution grid.

`tests/test_moss_fp16.py` pins the mechanism instead: the fold is exact
(`test_stream_scale_is_an_exact_reparameterisation`), every block's stream is
exactly `c` times the unfolded one
(`test_stream_scale_shrinks_the_stream_by_exactly_c`), and the guard is inert at
`c = 1` (`test_stream_scale_one_is_a_no_op`).

## Related

- `tests/test_moss_fp16.py` — the lane's contracts (dtype resolution, both
  guards, adapter math, `fuse` on both lanes).
- `tests/test_moss_kernels.py` — fused-vs-reference parity, including the fp32
  stream and the gathered per-sample skip.
- `docs/moss/README.md` — the user-facing table of `precision` / `fp16_fix`.
