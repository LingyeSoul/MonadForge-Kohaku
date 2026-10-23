"""Fused MOSS kernels: the vendored KohakuMoss op stack, bound to this model's layout.

``library.moss.vendor.kohakumoss_kernels`` holds the kernels verbatim (provenance and
licence in ``library.moss.vendor``); this module is the only place the rest of the
repository meets them. Each op has an eager reference in the vendored tree and every
backend is checked against it, so a fused block is the same arithmetic in fewer
launches, not a different model. ``tests/moss/test_moss_kernels.py`` pins that.

Enablement is ``MONADFORGE_MOSS_KERNELS`` (default on). ``fuse`` only flips a model when
the device is CUDA, Triton imports and every op it needs has a real kernel or a
``torch.compile`` candidate for the model's dtype — anything else stays on the
reference branch in ``model.py``. ``MONADFORGE_MOSS_KERNELS=0`` forces the reference.
"""

from __future__ import annotations

import logging
import os

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

ENV = "MONADFORGE_MOSS_KERNELS"
_OFF = frozenset({"0", "false", "off", "no"})
_DTYPES = (torch.float16, torch.bfloat16)
# ops `fuse` needs a non-eager backend for; the text-encoder surface is resolved on use
BLOCK_OPS = ("rms_norm_mod", "qk_norm_rope", "situ_glu", "gated_residual")
TEXT_OPS = (
    "causal_conv_varlen",
    "gated_delta_rule_varlen",
    "rms_norm_gated",
    "attention_varlen_causal",
)
_BOUND: dict[tuple, object] = {}


def enabled() -> bool:
    """Whether fused kernels are allowed in this process."""
    return os.environ.get(ENV, "1").strip().lower() not in _OFF


def available(device) -> bool:
    """Whether this device can run the fused path at all."""
    if not enabled() or torch.device(device).type != "cuda":
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        logger.info("fused MOSS kernels off: triton is not installed")
        return False
    return True


def _registry():
    from library.moss.vendor.kohakumoss_kernels import registry

    return registry


def bind(op: str, device, dtype: torch.dtype, **static) -> object:
    """The resolved implementation of ``op`` for (device, dtype, static options)."""
    index = torch.device(device)
    key = (op, index.type, index.index, dtype, tuple(sorted(static.items())))
    bound = _BOUND.get(key)
    if bound is None:
        bound = _registry().resolve(op, device, dtype, **static)
        _BOUND[key] = bound
    return bound


def describe(device, dtype: torch.dtype) -> dict[str, str]:
    """``{op: the backend that would serve it}`` for the block and text-encoder ops."""
    registry = _registry()
    out: dict[str, str] = {}
    for op in (*BLOCK_OPS, *TEXT_OPS):
        reasons = registry.available(op, device, dtype)
        ready = [
            name
            for name in ("tilelang", "triton", "compile")
            if reasons.get(name) is None
        ]
        out[op] = ready[0] if ready else f"eager only ({reasons.get('triton')})"
    return out


def fuse(model) -> int:
    """Switch ``model.layers`` to the fused branch; returns how many blocks took it."""
    device = next(model.parameters(), torch.empty(0)).device
    dtype = next(model.parameters()).dtype
    if not available(device) or dtype not in _DTYPES:
        return 0
    registry = _registry()
    for op in BLOCK_OPS:
        usable = registry.available(op, device, dtype)
        if not any(
            usable.get(name) is None for name in ("tilelang", "triton", "compile")
        ):
            logger.info("fused MOSS kernels off: %s has no kernel (%s)", op, usable)
            return 0
    for block in model.layers:
        block.fused = True
    return len(model.layers)


def segments(batch: int, tokens: int, device) -> Tensor:
    """``seg`` for ``(batch, tokens)`` packed token-major: ``[0]*tokens, [1]*tokens, ...``."""
    return torch.arange(batch, dtype=torch.int32, device=device).repeat_interleave(
        tokens
    )


# --------------------------------------------------------------------- block ops


def rms_norm(x: Tensor, weight: Tensor | None, eps: float) -> Tensor:
    """``x / rms(x) * weight`` over the last dim, one launch."""
    return bind("rms_norm", x.device, x.dtype)(x, weight, eps)


def rms_norm_mod(
    x: Tensor,
    weight: Tensor | None,
    shift: Tensor | None,
    scale: Tensor | None,
    seg: Tensor,
    eps: float,
) -> Tensor:
    """``rms_norm(x) * (1 + scale[seg]) + shift[seg]``: the norm and the modulation
    ``_affine`` applies, in one pass. ``x (N, D)``; ``shift``/``scale (B, D)``."""
    return bind("rms_norm_mod", x.device, x.dtype)(x, weight, shift, scale, seg, eps)


def qk_norm_rope(
    q: Tensor,
    k: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    pos: Tensor,
    bands: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Per-head RMSNorm of ``q``/``k`` then the multi-axis split-half RoPE, one pass.

    ``q``, ``k``: ``(N, H, Dh)``; ``pos``: ``(N, A)`` (``r, f, h, w`` here); ``bands``:
    ``Dh // (2·A)`` frequencies. Pairs are ``(j, j + Dh/2)``, the layout ``rotate`` uses.
    """
    return bind("qk_norm_rope", q.device, q.dtype)(
        q, k, q_weight, k_weight, pos, bands, eps
    )


def situ_glu(x: Tensor) -> Tensor:
    """SiTU-GLU of the fused up/gate projection: ``x (N, 2F) = [up | gate]`` → ``(N, F)``."""
    return bind("situ_glu", x.device, x.dtype)(x)


def gated_residual(
    x: Tensor, o: Tensor, gate: Tensor | None, seg: Tensor, r: float
) -> Tensor:
    """``x * r + o * gate[seg]`` in one pass (``_gated`` plus the residual add)."""
    return bind("gated_residual", x.device, x.dtype)(x, o, gate, seg, r)


# ------------------------------------------------------- text encoder (forward only)


def causal_conv(x: Tensor, weight: Tensor, cu_seqlens: Tensor) -> Tensor:
    """``silu(causal depthwise conv)`` over packed segments; ``x (N, C)``,
    ``weight (C, W)``, ``cu_seqlens (B + 1,)`` int32. Resets at every segment."""
    return bind("causal_conv_varlen", x.device, x.dtype)(x, weight, cu_seqlens)


def gated_delta_rule(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    b: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,
) -> Tensor:
    """The Qwen3.5 Gated DeltaNet recurrence over packed segments.

    ``q``/``k``: ``(N, Hk, Dk)``; ``v``: ``(N, Hv, Dv)``; ``a``, ``b``: ``(N, Hv)`` raw
    (this op applies ``g = -exp(a_log)·softplus(a + dt_bias)`` and ``β = sigmoid(b)``);
    value head ``h`` reads key head ``h // (Hv // Hk)``. ``q``/``k`` are L2-normalised
    inside. A segment's state resets at its start.
    """
    return bind("gated_delta_rule_varlen", q.device, q.dtype)(
        q, k, v, a, b, a_log, dt_bias, cu_seqlens, max_seqlen
    )


def rms_norm_gated(x: Tensor, z: Tensor, weight: Tensor, eps: float) -> Tensor:
    """``rms_norm(x) * weight * silu(z)`` over the last dim, one pass."""
    return bind("rms_norm_gated", x.device, x.dtype)(x, z, weight, eps)


def attention_causal(
    q: Tensor, k: Tensor, v: Tensor, cu_seqlens: Tensor, max_seqlen: int, scale: float
) -> Tensor:
    """GQA causal attention inside packed segments; ``q (N, H, Dh)``, ``k``/``v``
    ``(N, Hkv, Dh)``. Queries never read another segment or a later token."""
    return bind("attention_varlen_causal", q.device, q.dtype)(
        q, k, v, cu_seqlens, max_seqlen, scale
    )
