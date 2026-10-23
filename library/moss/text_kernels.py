"""Qwen3.5 text-encoder forward on the vendored varlen kernels.

The HF ``Qwen3_5TextModel.forward`` runs one padding-free prompt at a time and chunks
the Gated DeltaNet recurrence; this module packs the same math into the vendored
``*_varlen`` ops (causal conv, gated delta rule, gated norm, causal GQA attention).
Enablement is ``MONADFORGE_MOSS_TEXT_KERNELS`` (default on). ``usable`` declines when
the model is not a Qwen3.5 text stack, or when an op has no real backend on
(device, dtype); the caller then keeps HF's forward unchanged.

The arithmetic is checked against HF's own forward in ``tests/test_moss_text_kernels``.
Only the text path uses this module; the vision tower's M-RoPE positions stay on HF.
"""

from __future__ import annotations

import logging
import os

import torch
from torch import Tensor

from library.moss import ops

logger = logging.getLogger(__name__)

ENV = "MONADFORGE_MOSS_TEXT_KERNELS"
_OFF = frozenset({"0", "false", "off", "no"})
_BACKENDS = ("tilelang", "triton", "compile")
_READY: dict[tuple[str, int, torch.dtype], bool] = {}


def enabled() -> bool:
    """Whether the kernel text-encoder path is allowed in this process."""
    return os.environ.get(ENV, "1").strip().lower() not in _OFF


def _has_backend(op: str, device, dtype: torch.dtype) -> bool:
    from library.moss.vendor.kohakumoss_kernels import registry

    reasons = registry.available(op, device, dtype)
    return any(reasons.get(name) is None for name in _BACKENDS)


def _ready(device, dtype: torch.dtype) -> bool:
    index = torch.device(device).index
    key = (torch.device(device).type, -1 if index is None else index, dtype)
    cached = _READY.get(key)
    if cached is not None:
        return cached
    ready = all(
        _has_backend(op, device, dtype) for op in ("rms_norm", *ops.TEXT_OPS)
    )
    _READY[key] = ready
    return ready


def usable(te, device) -> bool:
    """Whether ``encode`` can serve this text encoder on ``device``."""
    if not enabled():
        return False
    lm = getattr(te, "language_model", None)
    if lm is None or not hasattr(lm, "layers") or not hasattr(lm, "rotary_emb"):
        return False
    types = getattr(lm.config, "layer_types", None)
    if types is None or len(types) != len(lm.layers):
        return False
    if not ops.available(device):
        return False
    dtype = next(lm.parameters()).dtype
    ready = _ready(device, dtype)
    if not ready:
        logger.info(
            "kernel text encoder off: no non-eager backend for every op on %s %s",
            device,
            dtype,
        )
    return ready


def _norm(x: Tensor, mod) -> Tensor:
    """Qwen3.5 zero-centred RMSNorm: ``x / rms(x) * (1 + w)``."""
    return ops.rms_norm(x, 1.0 + mod.weight.float(), mod.eps)


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Partial rotary: ``cos`` covers the first ``rot`` channels, the rest pass."""
    rot = cos.shape[-1]
    xr, xp = x[..., :rot], x[..., rot:]
    return torch.cat((xr * cos + _rotate_half(xr) * sin, xp), dim=-1)


def _attention_block(
    layer, h: Tensor, cu: Tensor, length: int, rope: tuple[Tensor, Tensor]
) -> Tensor:
    att = layer.self_attn
    hd = att.head_dim
    n = h.shape[0]
    x = _norm(h, layer.input_layernorm)
    qg = att.q_proj(x).view(n, -1, 2 * hd)
    gate = qg[..., hd:].reshape(n, -1)
    q = _norm(qg[..., :hd], att.q_norm)
    k = _norm(att.k_proj(x).view(n, -1, hd), att.k_norm)
    v = att.v_proj(x).view(n, -1, hd)
    cos, sin = rope
    q, k = _rope(q, cos, sin), _rope(k, cos, sin)
    o = ops.attention_causal(q, k, v, cu, length, att.scaling)
    h = h + att.o_proj(o.reshape(n, -1) * torch.sigmoid(gate))
    return h + layer.mlp(_norm(h, layer.post_attention_layernorm))


def _delta_block(layer, h: Tensor, cu: Tensor, length: int) -> Tensor:
    la = layer.linear_attn
    n = h.shape[0]
    x = _norm(h, layer.input_layernorm)
    qkv = ops.causal_conv(la.in_proj_qkv(x), la.conv1d.weight[:, 0], cu)
    q, k, v = qkv.split([la.key_dim, la.key_dim, la.value_dim], dim=-1)
    o = ops.gated_delta_rule(
        q.view(n, -1, la.head_k_dim),
        k.view(n, -1, la.head_k_dim),
        v.view(n, -1, la.head_v_dim),
        la.in_proj_a(x),
        la.in_proj_b(x),
        la.A_log,
        la.dt_bias,
        cu,
        length,
    )
    z = la.in_proj_z(x)
    o = ops.rms_norm_gated(
        o.reshape(-1, la.head_v_dim),
        z.reshape(-1, la.head_v_dim),
        la.norm.weight,
        la.norm.variance_epsilon,
    )
    h = h + la.out_proj(o.reshape(n, -1))
    return h + layer.mlp(_norm(h, layer.post_attention_layernorm))


def encode(te, input_ids: Tensor) -> Tensor:
    """Final hidden states ``(S, D)`` of one prompt, ``input_ids`` ``(1, S)``."""
    lm = te.language_model
    ids = input_ids.reshape(-1)
    length = ids.shape[0]
    device = ids.device
    h = lm.embed_tokens(ids)
    pos = torch.arange(length, device=device)
    cos, sin = lm.rotary_emb(h, pos[None])
    rope = (cos[0][:, None, :], sin[0][:, None, :])
    cu = torch.tensor([0, length], dtype=torch.int32, device=device)
    for layer in lm.layers:
        if layer.block_type == "linear_attention":
            h = _delta_block(layer, h, cu, length)
        else:
            h = _attention_block(layer, h, cu, length, rope)
    return _norm(h, lm.norm)
