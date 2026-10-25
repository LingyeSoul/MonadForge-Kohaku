# Packed batches (varlen training)

`moss-train --batch_size N` runs one DiT pass for `N` samples instead of the
per-sample loop. Image and text tokens from every sample go into one flat
token-major stream; `attention_varlen` reads `cu_seqlens` and never lets a query
see another sample's keys. No shape grouping, no padding, no resize.

This is the first port of KohakuMossTrainer's packed-training lever; the
per-sample path remains the default and the reference.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--batch_size N` | 1 | Samples per packed forward. `1` keeps the old loop. |
| `--token_batch T` | 0 | Soft cap on joint image+text tokens per batch. `0` uses `batch_size` alone. Needs `batch_size >= 2`; a single larger sample still gets its own batch. |

Both are `TrainRequest` fields, so the WebUI and `--config` see them with no
extra wiring. `grad_accum` still counts **batches** per optimizer update.

```bash
make daemon-run ARGS="--stall-timeout 900 scripts/moss/train.py \
    --cache post_image_dataset/moss/<set> --batch_size 4 --token_batch 12000"
```

## What runs packed

- Text-to-image training: latents, text embeddings, per-sample sigma and the
  LoRA all run once per batch.
- `channel_scaling`, `dc_isolate`, SVD-down init, LoRA / LoKr / LoHa: the
  adapter math is token-major already.
- The four fused block ops and the vendored varlen attention: one kernel pass
  over the packed stream, with `seg` preserving each sample's modulation.

## What stays on the batch-1 path

| Feature | Why |
|---|---|
| Reference latents (i2i) | `ref_span` / `ref_isolate` are per-sample token spans; the packed path would need a per-segment key mask. `--batch_size` > 1 raises instead of silently changing the arithmetic. |
| REPA (`--repa_weight > 0`) | it installs a block `layer_hook` that expects a per-sample image-token view. |
| `skip_layers` | the trainer never uses it; the packed block path rejects it rather than guessing. |

The check is explicit (`SystemExit`), not a fallback. Run i2i and REPA with
`--batch_size 1`.

## Memory and compile behaviour

- The VRAM reserve uses the **worst-case batch** (`batch_size` largest samples,
  capped by `token_batch`) instead of one sample. `--fit_report` still runs the
  largest sample alone first, so its number stays a real lower bound.
- `cu_seqlens` is padded with zero-length segments to `batch_size + 1`, and
  `max_seqlen` is rounded up to a 128-token bucket. The compiled block sees a
  stable small-tensor shape across a short final batch; expect one extra
  compile if `token_batch` lets the total token count move far outside the
  first batch's band. `--token_batch` is the knob that narrows that band.
- `--blocks_to_swap`, `--activation_memory_budget`, `--grad_checkpointing` and
  `--compile` apply unchanged. Tune them on the packed shape, not on one
  sample: pass the same `--batch_size` / `--token_batch` to `tune_swap.py`.

## Accounting

`global_step`, `losses`, epochs and `max_steps` count **samples**, not batches,
so an existing run report stays comparable with a `--batch_size 1` run. One
optimizer update covers `grad_accum` forwarded batches. The report adds
`samples_per_s`, `batch_size` and `token_batch`.

## Verification

`tests/test_moss_packed.py` pins the contract:

- packed output equals the per-sample forward for different image sizes and
  caption lengths;
- changing a later sample's latent does not change an earlier sample's output;
- packed loss and `video_embedder` gradients equal the per-sample sum;
- batch planning respects both caps;
- the fused CUDA path equals the reference path.

The 5B throughput number is not claimed here: run `--fit_report` and the
`tune_swap.py` pair on the target card before treating a packed plan as
measured.
