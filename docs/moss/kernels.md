# Fused MOSS kernels

The DiT block ops (`rms_norm_mod`, `qk_norm_rope`, `situ_glu`, `gated_residual`) run as
fused Triton kernels instead of the eager composition in `library/moss/model.py`. The
kernels are the [KohakuMossTrainer](https://github.com/KohakuBlueleaf/KohakuMossTrainer)
op stack, vendored verbatim into `library/moss/vendor/` (provenance and licence in
`library/moss/vendor/__init__.py`); `library/moss/ops.py` is the only place the rest of
the repository touches them.

## Use

Nothing to switch on. `library/moss/ops.py`'s `fuse(model)` runs from `moss-train`,
`moss-generate` and `moss-tune` right before the optional `torch.compile`, and prints
`[kernels] fused block ops on 42 blocks`. It declines (and the model keeps the reference
branch) when any of these holds:

- the host has no CUDA device, or `triton` does not import;
- the DiT's parameters are not fp16 / bf16;
- the model carries the fp32 residual stream (`enable_fp32_residual()`, the other fp16
  guard): the four kernels are 16-bit, so that lane resolves the residual op at fp32
  (`gated_residual` takes the stream dtype apart from the projection's) and the fused
  residual GEMM takes its split path. `enable_stream_scale()` (the default `online`
  lane) keeps the stream 16-bit, so every kernel runs normally;
- one of the four ops has no kernel or `torch.compile` candidate on this arch.

| Variable | Default | Effect |
|---|---|---|
| `MONADFORGE_MOSS_KERNELS` | `1` | `0` forces the reference branch |
| `MONADFORGE_MOSS_TEXT_KERNELS` | `1` | `0` keeps HF's Qwen3.5 text forward instead of the packed varlen ops |
| `KOHAKUMOSS_KERNEL_CACHE` | `~/.cache/kohakumoss_kernels` | where the per-shape tuning decisions are stored |
| `KOHAKUMOSS_LOG_LEVEL` | `INFO` | the vendored stack's own logger |

The branch is all-or-nothing per model, and it is *the same arithmetic*: every kernel is
checked against the eager reference the vendored tree ships, and the block's output and
parameter gradients are checked against the reference block
(`tests/test_moss_kernels.py`). The one deliberate difference is the RoPE: the reference
computes `sin`/`cos` in fp32 and then rounds the rope tensor to bf16 before `rotate`,
while `qk_norm_rope` keeps the phase in fp32 inside the kernel.

The `ref_timestep` modulation (i2i with a clean reference at another timestep) is the one
case the fused branch does not express: it gives the reference tokens a second
`(scale, shift, gate)` set, which is a per-token-segment index rather than a per-sample
one. Those steps fall back to the reference block, as does any fp32 model.

## Measured

RTX 4060 Laptop (sm89), bf16, MOSS DiT shapes (D 3072, FFN 2×8192, 24 heads × 128),
forward + backward, best of 10. `python -m bench.moss_kernels.bench_block_ops --compile`.

Per op, 4096 tokens, device ms:

| op | eager | fused | x |
|---|---|---|---|
| `rms_norm_mod` | 15.44 | 4.15 | 3.7 |
| `situ_glu` | 45.80 | 10.45 | 4.4 |
| `gated_residual` | 8.24 | 3.76 | 2.2 |
| `qk_norm_rope` | 18.97 | 4.31 | 4.4 |

One block, 4288 tokens (a 1 MP image plus its caption):

| | ms |
|---|---|
| reference (eager) | 199.9 |
| `torch.compile` of the reference | 151.4 |
| fused | 149.8 |
| fused + `torch.compile` | 149.6 |

Read that table honestly: the four ops are 2.2–4.4× their eager composition, but a block
is dominated by its six linears, and Inductor already fuses the elementwise chain, so on
this card `torch.compile` recovers almost all of the block-level win by itself (1.32× vs
the fused branch's 1.33×). The fused branch still pays for itself because it is what a
`--no-compile` job runs (no MSVC toolchain, no 30 s compile, a quick smoke) and it adds
1.2 % on top of the compiled block. On sm70 / sm75, where Inductor and SDPA are weaker,
the vendored stack has a TileLang backend instead — `pip install tilelang` makes
`registry.DEFAULT_PRIORITY` bind it, exactly as upstream measured it.

## What is wired

The op surface is one thing; what the trainer and the pipeline reach is another:

| capability | entry point | what it buys |
|---|---|---|
| `attention_varlen` | `library/moss/model.py::forward_packed`, `moss-train --batch_size` | one attention call across a token-major packed batch, with `cu_seqlens` keeping samples apart; `docs/moss/packed_batches.md` |
| `gated_delta_rule_varlen`, `causal_conv_varlen`, `rms_norm_gated`, `attention_varlen_causal`, `rms_norm` | `library/moss/text_kernels.py`, used by `loader.encode_prompt` | the Qwen3.5 text encode runs packed; `MONADFORGE_MOSS_TEXT_KERNELS=0` keeps HF's forward |
| `quant_linear` (int8 / e4m3 weights, per-row scales) | `library/moss/quant.py`, `moss-generate --quant {int8,fp8}` | the frozen 5B DiT base at half the VRAM; a LoRA (not LoKr) still applies on top |
| `AdamW16`, `MuonHybrid` | `library/moss/optim/`, `moss-train --optimizer {adamw16,muon}` | 16-bit optimizer state with fp32 math, or Muon on the LoRA matrices |
| VAE decode in bf16 / fp16, optionally compiled in channels_last | `loader.load_vae(compile=...)`, `moss-generate --vae_dtype`, `--vae_compile` | ~1.5× decode and ~60% of the fp32 VAE's VRAM upstream |
| TileLang backend | vendored source; `uv sync --group kernels-tilelang` | sm70 / sm75, where Inductor and SDPA are weak |

## What is vendored but not wired

`library/moss/vendor/kohakumoss_kernels` carries the whole op surface; the rows below
are not connected, with what stands between each and this model:

| capability | state | what it needs |
|---|---|---|
| `fp8_linear`, `fp8_linear_train`, MXFP8 | available | fp8 compute needs a scaling recipe, fp8 master weights and a scaler; the trainer is bf16 with fp32 masters |
| `quant_linear_rotated` | available, not wired | this repository already ships ConvRot (`library/runtime/convrot`: W8A8 and W8A16, sylvester and regular H, `_int_mm` padding); measured on an RTX 4060 at 4096 tokens, the vendored weight-only kernel was 0.90–1.17× this repository's W8A16 time and plain `quant_linear` was faster still, so a second rotation family buys nothing |
| fused optimizer bundle (`FusedStep`, `HostAccumulator`, streaming AdamW) | available | the trainer's optimizer surface (`library/training/`) owns grad hooks and the offload engines |
| TileLang backend | source only | `tilelang` is not a dependency of this project; sm70 / sm75 would need the extra |

Rotated-quant comparison (RTX 4060 Laptop, bf16, 4096 tokens, group 256, best of 50).
`convrot` is this repository's fused path; `upstream rotated` is the vendored
`quant_linear_rotated` Triton kernel; `upstream plain` is the vendored `quant_linear`
(no rotation) that `library/moss/quant.py` uses.

| shape (N × K) | bf16 | convrot W8A16 | convrot W8A8 | upstream rotated | upstream plain |
|---|---|---|---|---|---|
| q/o 3072 × 3072 | 3.36 | 3.23 | 6.91 | 3.79 | 2.82 |
| ffn_up_gate 16384 × 3072 | 13.04 | 14.92 | 31.87 | 13.37 | 13.19 |
| ffn_down 3072 × 8192 | 6.42 | 7.85 | 15.91 | 7.10 | 6.58 |

ms per forward. Cosines against bf16 are ≥ 0.9999 everywhere. The weight-only paths
save the weight's bytes, not time, on this card; `_int_mm` (W8A8) has no win at these
shapes either, which is why the quantized base uses plain per-row `quant_linear`.

## Layout

`library/moss/ops.py` (the façade) → `library/moss/vendor/kohakumoss_kernels` (ops,
registry, tuning, kernels) → `library/moss/vendor/kohakumoss_common` (logging, io,
device helpers). Only the package root changed in the vendored copy, so an upstream fix
can be diffed in. Nothing outside `library/moss/` imports the vendor tree.

## Reproduce

```bash
python -m pytest tests/test_moss_kernels.py
python -m bench.moss_kernels.bench_block_ops --compile --label <card>
```
