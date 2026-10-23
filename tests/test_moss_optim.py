"""The extracted optimisers: AdamW16 and MuonHybrid against torch's AdamW."""

import torch

from library.moss.optim import AdamW16, MuonHybrid, newton_schulz


def _params(shape, dtype):
    torch.manual_seed(0)
    p = torch.randn(shape, dtype=dtype, requires_grad=True)
    return p


def _step_pair(make_opt, steps=5):
    a = _params((16, 8), torch.float32)
    b = a.detach().clone().requires_grad_(True)
    torch.manual_seed(1)
    grads = [torch.randn_like(a) for _ in range(steps)]
    opt_a, opt_b = make_opt([a]), torch.optim.AdamW([b], lr=1e-2, weight_decay=0.0)
    for g in grads:
        for p, opt in ((a, opt_a), (b, opt_b)):
            p.grad = g.clone()
            opt.step()
            opt.zero_grad(set_to_none=True)
    return a, b


def test_adamw16_matches_torch_adamw():
    a, b = _step_pair(lambda params: AdamW16(params, lr=1e-2, betas=(0.9, 0.999)))
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)


def test_muon_adamw_group_matches_torch_adamw():
    a, b = _step_pair(
        lambda params: MuonHybrid([{"params": params, "use_muon": False}], lr=1e-2, betas=(0.9, 0.999))
    )
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)


def test_newton_schulz_is_orthogonal():
    torch.manual_seed(2)
    g = torch.randn(32, 16)
    q = newton_schulz(g)
    gram = q.T @ q
    off = gram - torch.diag(torch.diagonal(gram))
    assert off.abs().max() < 0.3


def test_muon_step_moves_parameters():
    p = _params((16, 8), torch.float32)
    opt = MuonHybrid([{"params": [p], "use_muon": True}], lr=1e-2)
    p.grad = torch.randn_like(p)
    before = p.detach().clone()
    opt.step()
    assert not torch.allclose(p, before)
    assert torch.isfinite(p).all()
