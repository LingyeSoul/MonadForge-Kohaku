"""Fused MOSS kernels: parity against the eager reference and the reference branch.

The kernels are CUDA-only, so everything that needs them skips on a host without a
card; the CPU cases pin the guard that keeps the reference branch in charge.
"""

from importlib.util import find_spec

import pytest
import torch

from library.moss import ops
from library.moss.model import MOSSImageTransformer

CUDA = torch.cuda.is_available() and find_spec("triton") is not None
needs_cuda = pytest.mark.skipif(not CUDA, reason="fused MOSS kernels are CUDA-only")


def _tiny(device="cpu", dtype=torch.float32):
    torch.manual_seed(0)
    model = MOSSImageTransformer(
        dim=512, ffn_dim=768, num_heads=4, num_layers=2, text_dim=64, latent_channels=8
    )
    return model.to(device=device, dtype=dtype).eval()


def _inputs(batch=2, s=7):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(batch, 8, 4, 6, generator=g)
    e = torch.randn(batch, s, 64, generator=g)
    mask = torch.ones(batch, s, dtype=torch.bool)
    mask[-1, 3:] = False
    return x, e, mask


def test_cpu_never_fuses():
    model = _tiny()
    assert not ops.available(torch.device("cpu"))
    assert ops.fuse(model) == 0
    assert not any(block.fused for block in model.layers)


def test_env_switch(monkeypatch):
    monkeypatch.setenv(ops.ENV, "0")
    assert not ops.enabled()


@needs_cuda
def test_fuse_is_all_or_nothing():
    model = _tiny("cuda", torch.bfloat16)
    assert ops.fuse(model) == len(model.layers)
    assert all(block.fused for block in model.layers)


@needs_cuda
def test_every_block_op_matches_its_reference():
    device, dtype = torch.device("cuda"), torch.bfloat16
    g = torch.Generator(device=device).manual_seed(3)
    n, dim = 1024, 3072
    x = torch.randn(n, dim, device=device, dtype=dtype, generator=g)
    weight = torch.randn(dim, device=device, generator=g)
    shift = torch.randn(2, dim, device=device, generator=g)
    scale = torch.randn(2, dim, device=device, generator=g)
    seg = ops.segments(2, n // 2, device)

    def ref(op, *args, **static):
        from library.moss.vendor.kohakumoss_kernels import registry

        return registry.resolve(op, device, dtype, backend="eager", **static)(*args)

    def rel(a, b):
        return ((a.float() - b.float()).abs().mean() / b.float().abs().mean()).item()

    assert (
        rel(
            ops.rms_norm_mod(x, weight, shift, scale, seg, 1e-6),
            ref("rms_norm_mod", x, weight, shift, scale, seg, 1e-6),
        )
        < 1e-5
    )
    assert rel(ops.rms_norm(x, weight, 1e-6), ref("rms_norm", x, weight, 1e-6)) < 1e-5

    gate = torch.randn(2, dim, device=device, generator=g)
    o = torch.randn(n, dim, device=device, dtype=dtype, generator=g)
    assert (
        rel(
            ops.gated_residual(x, o, gate, seg, 1.0),
            ref("gated_residual", x, o, gate, seg, 1.0),
        )
        < 1e-5
    )

    up = torch.randn(n, 2 * 16384, device=device, dtype=dtype, generator=g)
    assert rel(ops.situ_glu(up), ref("situ_glu", up)) < 1e-5

    heads, head_dim = 24, 128
    q = torch.randn(n, heads, head_dim, device=device, dtype=dtype, generator=g)
    k = torch.randn(n, heads, head_dim, device=device, dtype=dtype, generator=g)
    qw = torch.randn(head_dim, device=device, generator=g)
    kw = torch.randn(head_dim, device=device, generator=g)
    pos = torch.rand(n, 4, device=device, generator=g) * 64
    bands = torch.rand(head_dim // 8, device=device, generator=g) * 0.5 + 0.5
    got = ops.qk_norm_rope(q, k, qw, kw, pos, bands, 1e-6)
    want = ref("qk_norm_rope", q, k, qw, kw, pos, bands, 1e-6)
    assert rel(got[0], want[0]) < 1e-5 and rel(got[1], want[1]) < 1e-5


@needs_cuda
@torch.no_grad()
def test_fused_forward_matches_reference():
    model = _tiny("cuda", torch.bfloat16)
    x, e, mask = _inputs()
    x, e = x.cuda(), e.cuda()
    want = model(x, 500.0, e, mask).sample
    assert ops.fuse(model) == len(model.layers)
    got = model(x, 500.0, e, mask).sample
    assert torch.allclose(got, want, atol=2e-2, rtol=2e-2)


@needs_cuda
@torch.no_grad()
def test_fused_batched_matches_reference():
    model = _tiny("cuda", torch.bfloat16)
    x, e, mask = _inputs()
    x, e = x.cuda(), e.cuda()
    want = model.forward_batched(x, 500.0, e, mask)
    ops.fuse(model)
    got = model.forward_batched(x, 500.0, e, mask)
    assert torch.allclose(got, want, atol=2e-2, rtol=2e-2)


@needs_cuda
@torch.no_grad()
def test_fused_per_sample_skip_matches_reference():
    model = _tiny("cuda", torch.bfloat16)
    x, e, mask = _inputs()
    x, e = x.cuda(), e.cuda()
    want = model.forward_batched(x, 500.0, e, mask, skip_layers=[(), (1,)])
    ops.fuse(model)
    got = model.forward_batched(x, 500.0, e, mask, skip_layers=[(), (1,)])
    assert torch.allclose(got, want, atol=2e-2, rtol=2e-2)


@needs_cuda
def test_fused_block_backward_matches_reference():
    """The fused ops carry hand-written backward kernels; the block's parameter
    gradients must still land where the reference puts them."""
    model = _tiny("cuda", torch.bfloat16)
    x, e, mask = _inputs()
    x, e = x.cuda(), e.cuda()

    def grads():
        model.zero_grad(set_to_none=True)
        out = model.forward_batched(x, 500.0, e, mask)
        out.square().mean().backward()
        return {
            name: parameter.grad.clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }

    want = grads()
    ops.fuse(model)
    got = grads()
    assert set(got) == set(want)
    for name, gradient in want.items():
        reference_norm = gradient.float().norm().clamp_min(1e-6)
        error = (got[name].float() - gradient.float()).norm() / reference_norm
        assert error < 5e-2, f"{name}: relative gradient error {error:.3f}"


@needs_cuda
@torch.no_grad()
def test_separate_ref_modulation_stays_reference():
    """A ``ref_timestep`` gives the reference tokens their own modulation, which the
    fused branch does not express; the block must fall back and match exactly."""
    model = _tiny("cuda", torch.bfloat16)
    model.ref_timestep = 0.0
    x, e, mask = _inputs()
    x, e = x.cuda(), e.cuda()
    ref = torch.randn(x.shape[0], 8, 2, 3, generator=torch.Generator().manual_seed(4))
    ref = ref.cuda()
    want = model(x, 500.0, e, mask, ref=ref).sample
    ops.fuse(model)
    got = model(x, 500.0, e, mask, ref=ref).sample
    assert torch.allclose(got, want, atol=2e-2, rtol=2e-2)
