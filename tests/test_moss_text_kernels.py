"""Kernel text encoder: the packed varlen forward against HF's own Qwen3.5 forward.

The ops are CUDA-only; a host without a card pins the guard that keeps HF's forward in
charge. The fp32 case is the mathematical check (the eager reference is exact there);
the bf16 case is the path a real run takes.
"""

from importlib.util import find_spec

import pytest
import torch

from library.moss import ops, text_kernels

CUDA = torch.cuda.is_available() and find_spec("triton") is not None
needs_cuda = pytest.mark.skipif(not CUDA, reason="kernel text ops are CUDA-only")


class _Wrap:
    def __init__(self, language_model):
        self.language_model = language_model


def _tiny(dtype):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    torch.manual_seed(0)
    config = Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
        max_position_embeddings=128,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [11, 11, 10],
        },
        rms_norm_eps=1e-6,
    )
    return Qwen3_5TextModel(config).to(device="cuda", dtype=dtype).eval()


def test_env_switch(monkeypatch):
    monkeypatch.setenv(text_kernels.ENV, "0")
    assert not text_kernels.enabled()


def test_cpu_never_uses_kernels():
    assert not ops.available(torch.device("cpu"))


@needs_cuda
@torch.no_grad()
def test_fp32_matches_hf_exactly():
    """The eager reference must be the same arithmetic as HF's chunked recurrence."""
    model = _tiny(torch.float32)
    ids = torch.randint(0, 256, (1, 33), device="cuda")
    want = model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state[0]
    got = text_kernels.encode(_Wrap(model), ids)
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5)


@needs_cuda
@torch.no_grad()
def test_bf16_matches_hf():
    model = _tiny(torch.bfloat16)
    ids = torch.randint(0, 256, (1, 33), device="cuda")
    want = model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state[0]
    assert text_kernels.usable(_Wrap(model), torch.device("cuda"))
    got = text_kernels.encode(_Wrap(model), ids)
    cos = torch.nn.functional.cosine_similarity(
        got.float().flatten(), want.float().flatten(), dim=0
    )
    assert float(cos) > 0.9995
    assert torch.allclose(got.float(), want.float(), atol=0.1, rtol=0.1)
