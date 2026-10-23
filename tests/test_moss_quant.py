"""Quantized DiT base: codes round-trip, the linear rejects bad formats, and a
quantized block still computes the same function within the code's precision."""

from importlib.util import find_spec

import pytest
import torch

from library.moss.model import MOSSImageTransformer
from library.moss.quant import QuantLinear, quantize, quantize_transformer

CUDA = torch.cuda.is_available() and find_spec("triton") is not None
needs_cuda = pytest.mark.skipif(not CUDA, reason="quant kernels are CUDA-only")
SM = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)


def _tiny(device="cpu", dtype=torch.float32):
    torch.manual_seed(0)
    model = MOSSImageTransformer(
        dim=512, ffn_dim=768, num_heads=4, num_layers=2, text_dim=64, latent_channels=8
    )
    return model.to(device=device, dtype=dtype).eval()


def test_quantize_round_trip():
    torch.manual_seed(0)
    w = torch.randn(64, 32)
    for fmt, limit, tol in (("int8", 127.0, 0.03), ("fp8", 448.0, 0.08)):
        q, scale = quantize(w, fmt)
        assert q.shape == w.shape and scale.shape == (64,)
        assert q.float().abs().max() <= limit
        dense = (q.float() * scale[:, None]).to(torch.float32)
        assert (dense - w).abs().mean() / w.abs().mean() < tol


def test_unknown_format_rejected():
    with pytest.raises(ValueError, match="not one of"):
        quantize(torch.randn(4, 4), "int4")


@needs_cuda
@pytest.mark.parametrize(
    "fmt,min_cos",
    [
        ("int8", 0.99),
        pytest.param(
            "fp8",
            0.98,
            marks=pytest.mark.skipif(SM < (8, 9), reason="fp8 needs sm89+"),
        ),
    ],
)
@torch.no_grad()
def test_quantized_transformer_matches_bf16(fmt, min_cos):
    model = _tiny("cuda", torch.bfloat16)
    x = torch.randn(2, 8, 4, 6, device="cuda")
    e = torch.randn(2, 7, 64, device="cuda")
    mask = torch.ones(2, 7, dtype=torch.bool, device="cuda")
    want = model(x, 500.0, e, mask).sample
    assert quantize_transformer(model, fmt) == 12
    assert all(isinstance(b.proj["q"], QuantLinear) for b in model.layers)
    got = model(x, 500.0, e, mask).sample
    cos = torch.nn.functional.cosine_similarity(
        got.float().flatten(), want.float().flatten(), dim=0
    )
    assert float(cos) > min_cos
