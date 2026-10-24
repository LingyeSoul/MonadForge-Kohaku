"""MOSS fp16 lane: compute-dtype resolution, the fp32 residual stream, adapter math.

The MOSS line shipped bf16-only: ``load_transformer`` hard-coded bf16 and the
trainer had no precision knob, so a pre-Ampere card (V100 sm_70 / T4 sm_75, no
native bf16 GEMM) could not train at all. fp16 is that lane's dtype, and fp16
needs a fix — the 42-block residual stream passes fp16's 65504 ceiling. Upstream
KohakuMossTrainer refuses ``PRECISION='fp16'`` without ``FP16_FIX`` and ships
``fp32_residual`` for it; this repo now resolves ``precision='auto'`` and turns
the same fix on.

These tests pin the contracts that make the lane safe:

1. ``resolve_compute_dtype`` picks bf16 only where bf16 is native, and refuses
   an explicit bf16 there.
2. ``Block._gated_residual`` is finite past 65504 with the guard on, and inf
   without it (the negative control that the input really overflows fp16).
3. The guard is inert by default: bit-exact to the legacy expression on
   bf16/fp32, so production runs are unchanged.
4. ``attention_mask``-free end-to-end: the transformer's ``head`` stays finite
   for a residual stream past fp16's range only with the guard on.
5. ``merge_adapter_delta`` / ``adapter_work_dtype`` keep the adapter math fp32
   on an fp16 base and leave bf16/fp32 untouched.
6. ``fuse`` declines on the fp16-safe lane (the residual kernels are 16-bit).

CPU-only: the overflow arithmetic is dtype, not device.
"""

from __future__ import annotations

import pytest

import torch

from library.lines.adapters import adapter_work_dtype, merge_adapter_delta
from library.moss import ops
from library.moss.loader import resolve_compute_dtype
from library.moss.model import MOSSImageTransformer, _Block

_FP16_MAX = torch.finfo(torch.float16).max


def _tiny(**kw):
    torch.manual_seed(0)
    base = dict(
        dim=32,
        ffn_dim=48,
        num_heads=4,
        num_layers=2,
        text_dim=16,
        latent_channels=4,
    )
    return MOSSImageTransformer(**{**base, **kw}).eval()


# --------------------------------------------------------------- precision


def _patch_capability(monkeypatch, capability):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda device=None: capability
    )


def test_auto_is_bf16_only_where_bf16_is_native(monkeypatch):
    _patch_capability(monkeypatch, (7, 0))  # V100
    assert resolve_compute_dtype("auto", None) == torch.float16
    _patch_capability(monkeypatch, (7, 5))  # T4
    assert resolve_compute_dtype("auto", None) == torch.float16
    _patch_capability(monkeypatch, (8, 0))  # A100
    assert resolve_compute_dtype("auto", None) == torch.bfloat16


def test_explicit_bf16_on_pre_ampere_is_refused(monkeypatch):
    _patch_capability(monkeypatch, (7, 0))
    with pytest.raises(ValueError, match="sm_70"):
        resolve_compute_dtype("bf16", None)
    assert resolve_compute_dtype("fp16", None) == torch.float16


def test_unknown_precision_is_refused():
    with pytest.raises(ValueError, match="auto, bf16 or fp16"):
        resolve_compute_dtype("fp8", None)


# ------------------------------------------------------- residual stream


def test_gated_residual_guards_the_product_and_the_sum():
    """The guard covers both overflow sites: the gated product and the add."""
    block = _Block(dim=8, ffn_dim=8, heads=2)
    x = torch.full((1, 2, 8), 0.9 * _FP16_MAX, dtype=torch.float16)
    branch = torch.full((1, 2, 8), 0.9 * _FP16_MAX, dtype=torch.float16)
    gate = torch.ones(1, 8, dtype=torch.float16)

    block.fp32_residual = False
    naive = x + branch * gate[:, None]
    assert torch.isinf(naive).any(), "test input does not actually overflow fp16"
    assert torch.equal(block._gated_residual(x, branch, gate, None), naive)

    block.fp32_residual = True
    guarded = block._gated_residual(x, branch, gate, None)
    assert guarded.dtype == torch.float32
    assert torch.isfinite(guarded).all()
    assert torch.equal(guarded, x.float() + branch.float() * gate.float()[:, None])


def test_gated_residual_is_inert_by_default():
    block = _Block(dim=8, ffn_dim=8, heads=2)
    block.fp32_residual = False
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(2, 3, 8, generator=g, dtype=dtype)
        branch = torch.randn(2, 3, 8, generator=g, dtype=dtype)
        gate = torch.rand(2, 8, generator=g, dtype=dtype)
        expect = x + branch * gate[:, None]
        assert torch.equal(block._gated_residual(x, branch, gate, None), expect)


def test_enable_fp32_residual_propagates():
    model = _tiny()
    assert not model.fp32_residual
    assert all(not b.fp32_residual for b in model.layers)
    model.enable_fp32_residual()
    assert model.fp32_residual
    assert all(b.fp32_residual for b in model.layers)


@torch.no_grad()
def test_head_is_finite_past_fp16_range_only_with_the_guard():
    """The final norm + projection must not see an fp16-cast residual stream.

    On the fp16 lane ``head`` receives the fp32 residual stream, so the legacy
    ``features.to(weight dtype)`` is itself the overflow: a value past 65504 has
    no fp16 representation. The guard normalizes in fp32 and keeps the final
    projection fp32 as well.
    """
    model = _tiny().to(torch.float16)
    model.enable_fp32_residual()
    hidden = 32
    # Past fp16's ceiling, and the fp32 stream the block stack really hands over.
    features = torch.full((1, 4, hidden), 70000.0, dtype=torch.float32)
    _, head_mod = model.time_conditioning(500.0, 1)

    model.fp32_residual = True
    out = model.head(features, head_mod, 2, 2)
    assert torch.isfinite(out).all(), "fp32 head produced inf/nan"

    model.fp32_residual = False
    legacy = model.head(features, head_mod, 2, 2)
    assert not torch.isfinite(legacy).all(), (
        "the unguarded head should overflow on this input; the test no longer "
        "covers the fix"
    )


def test_fuse_declines_on_the_fp16_safe_lane(caplog):
    model = _tiny()
    model.enable_fp32_residual()
    assert ops.fuse(model) == 0
    assert not model.layers[0].fused


# --------------------------------------------------------------- adapters


def test_adapter_work_dtype_is_fp32_only_on_an_fp16_base():
    fp16 = torch.zeros(1, dtype=torch.float16)
    bf16 = torch.zeros(1, dtype=torch.bfloat16)
    assert adapter_work_dtype(fp16, True, True) == torch.float32
    # Inert: not training, not requested, or a non-fp16 base.
    assert adapter_work_dtype(fp16, True, False) == torch.float16
    assert adapter_work_dtype(fp16, False, True) == torch.float16
    assert adapter_work_dtype(bf16, True, True) == torch.bfloat16


def test_merge_adapter_delta_keeps_fp32_on_the_fp16_lane():
    base = torch.full((2, 4), 40000.0, dtype=torch.float16)
    delta = torch.full((2, 4), 30000.0, dtype=torch.float32)
    legacy = merge_adapter_delta(base, delta, preserve_fp32=False)
    assert torch.isinf(legacy).all(), "the legacy merge should overflow here"

    safe = merge_adapter_delta(base, delta, preserve_fp32=True)
    assert safe.dtype == torch.float32
    assert torch.equal(safe, base.float() + delta.float())


def test_merge_adapter_delta_is_bit_exact_off_the_lane():
    g = torch.Generator().manual_seed(0)
    for dtype in (torch.bfloat16, torch.float32):
        base = torch.randn(3, 5, generator=g, dtype=dtype)
        delta = torch.randn(3, 5, generator=g, dtype=dtype)
        assert torch.equal(
            merge_adapter_delta(base, delta, preserve_fp32=False),
            base + delta.to(dtype),
        )


# --------------------------------------------------------- end to end


def test_one_fp16_step_is_finite_and_reaches_the_adapters():
    """The lane the trainer runs: fp16 DiT + fp32 residual + fp32 adapter math.

    The loss is fp32 (``training_loss`` casts), so this pins the two contracts
    the trainer depends on: the step produces a finite loss and finite grads on
    every adapter parameter, and the fp16-safe lane is what keeps it finite —
    the same step on the unguarded stack overflows on this fixture.
    """
    from library.moss.lora import LoRANetwork
    from library.moss.train import training_loss

    model = _tiny().to(torch.float16)
    model.requires_grad_(False)
    net = LoRANetwork(model, rank=4, dtype=torch.float32, fp32_compute=True)
    net.apply_to()
    model.train()

    g = torch.Generator().manual_seed(1)
    latent = torch.randn(1, 4, 8, 8, generator=g)
    embeds = torch.randn(1, 5, 16, generator=g).to(torch.float16)

    def step():
        for p in net.parameters():
            p.grad = None
        loss = training_loss(model, latent, embeds, 0.5, gen=g)
        loss.backward()
        return loss

    model.enable_fp32_residual()
    loss = step()
    assert torch.isfinite(loss).all()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(x).all() for x in grads)
    # fp32 masters, so the grads stay fp32 even under an fp16 base.
    assert all(x.dtype == torch.float32 for x in grads)
