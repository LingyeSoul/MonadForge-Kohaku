# Fused MOSS kernels

The DiT block ops (`rms_norm_mod`, `qk_norm_rope`, `situ_glu`, `gated_residual`) run as
fused Triton kernels instead of the eager composition in `library/moss/model.py`. The
kernels are the [KohakuMossTrainer](https://github.com/KohakuBlueleaf/KohakuMossTrainer)
op stack, vendored verbatim into `library/moss/vendor/` (provenance and licence in
`library/moss/vendor/__init__.py`); `library/moss/ops.py` is the only place the rest of
the repository touches them.

## Use

Nothing to switch on. `library/moss/ops.fuse(model)` runs from `moss-train`,
`moss-generate` and `moss-tune` right before the optional `torch.compile`, and prints
`[kernels] fused block ops on 42 blocks`. It declines (and the model keeps the reference
branch) when any of these holds:

- the host has no CUDA device, or `triton` does not import;
- the DiT's parameters are not fp16 / bf16;
- one of the four ops has no kernel or `torch.compile` candidate on this arch.

| Variable | Default | Effect |
|---|---|---|
| `MONADFORGE_MOSS_KERNELS` | `1` | `0` forces the reference branch |
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

## What is vendored but not wired

`library/moss/vendor/kohakumoss_kernels` carries the whole op surface; only the four
block ops above are connected. The rest, with what stands between it and this model:

| capability | state | what it needs |
|---|---|---|
| `gated_delta_rule_varlen`, `causal_conv_varlen`, `rms_norm_gated`, `attention_varlen_causal` | available through `library.moss.ops`, 2.2× the transformers text encode upstream | transformers' `Qwen3_5GatedDeltaNet.forward` precomputes `g = -exp(A_log)·softplus(a + dt_bias)` and `beta = sigmoid(b)`, while the op takes the raw `a`/`b` and the layer's `a_log`/`dt_bias`. Wiring it means reimplementing that module's forward (projections, conv, norm, `out_proj`) and pinning it to a transformers version |
| `attention_varlen` | available | this model runs one sample per forward with a key mask; the op needs packed `cu_seqlens` batches. It is the right kernel for `forward_batched` only after the joint sequence is packed |
| `fp8_linear`, `fp8_linear_train`, MXFP8 | available | scaling recipe + a custom autograd line placement; not a drop-in for `nn.Linear` in a training run |
| `quant_linear`, `quant_linear_rotated` | available | this repository already ships its own ConvRot (`library/runtime/convrot`, `library/moss/int8.py`); two rotation families on one checkpoint would be a footgun, so the vendored one is left for experiments |
| `adamw_step*`, Muon, the fused optimizer bundle | available | the trainer's optimizer surface (`library/training/`) |
| TileLang backend | source only | `tilelang` is not a dependency of this project; sm70 / sm75 would need the extra |

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
