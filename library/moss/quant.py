"""Quantized DiT base for inference: per-output-row int8 / fp8 weights.

The 5B transformer is 10.2 GB in bf16; int8 or e4m3 halves that, and the vendored
``quant_linear`` kernel reads the codes directly instead of materializing a dense
weight per call. Only inference uses this: a quantized base is frozen, so the autograd
path (input gradient through the dequantized weight) is not part of a training run.

``quantize_transformer`` swaps every block's six ``nn.Linear`` projections; ``QuantLinear``
stays an ``nn.Linear`` subclass so the LoRA appliers (which patch an instance forward)
still find their targets. Format availability is checked once, and an unsupported
device raises instead of silently keeping bf16 weights.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from library.moss import ops

FP8_MAX = 448.0
INT8_MAX = 127.0
FORMATS = ("int8", "fp8")


def quantize(w: Tensor, fmt: str) -> tuple[Tensor, Tensor]:
    """``(q, scale[out])`` of a 2-D weight; ``scale`` is fp32."""
    wf = w.detach().float()
    amax = wf.abs().amax(dim=1).clamp_min(1e-12)
    if fmt == "int8":
        scale = amax / INT8_MAX
        q = torch.round(wf / scale[:, None]).clamp_(-INT8_MAX, INT8_MAX)
        return q.to(torch.int8), scale
    if fmt == "fp8":
        scale = amax / FP8_MAX
        q = (wf / scale[:, None]).clamp_(-FP8_MAX, FP8_MAX)
        return q.to(torch.float8_e4m3fn), scale
    raise ValueError(f"quant format {fmt!r} not one of {FORMATS}")


def dequantize(q: Tensor, scale: Tensor, dtype: torch.dtype) -> Tensor:
    """The dense weight the codes stand for, rounded once to ``dtype``."""
    return (q.to(torch.float32) * scale.float()[:, None]).to(dtype)


def _check_device(fmt: str, device) -> None:
    if fmt not in FORMATS:
        raise ValueError(f"quant format {fmt!r} not one of {FORMATS}")
    if fmt == "fp8" and torch.device(device).type == "cuda":
        major, minor = torch.cuda.get_device_capability(torch.device(device))
        if major * 10 + minor < 89:
            raise ValueError(
                f"fp8 weights need sm89+, this device is sm{major}{minor}"
            )


class QuantLinear(nn.Linear):
    """A frozen ``nn.Linear`` whose weight is a per-row int8 / fp8 code plus a scale."""

    def __init__(self, linear: nn.Linear, fmt: str):
        super().__init__(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        del self.weight
        q, scale = quantize(linear.weight.detach(), fmt)
        self.fmt = fmt
        self.compute_dtype = linear.weight.dtype
        self.register_buffer("q", q)
        self.register_buffer("scale", scale)
        if linear.bias is not None:
            self.bias.data.copy_(linear.bias.data)

    def forward(self, x: Tensor) -> Tensor:
        return ops.quant_linear(x, self.q, self.scale, self.bias, self.fmt)

    def dense(self, dtype: torch.dtype) -> Tensor:
        """The dequantized weight, for callers that need the dense form."""
        return dequantize(self.q, self.scale, dtype)


def quantize_transformer(transformer, fmt: str) -> int:
    """Swap every ``layers.*.proj.*`` linear for a ``QuantLinear``; returns the count."""
    device = next(transformer.parameters()).device
    _check_device(fmt, device)
    count = 0
    for block in transformer.layers:
        for name, module in list(block.proj.items()):
            if isinstance(module, nn.Linear) and not isinstance(module, QuantLinear):
                block.proj[name] = QuantLinear(module, fmt)
                count += 1
    return count
