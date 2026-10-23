"""The Triton `attention_varlen` registry entry (sm80+ CUDA, XPU): the Triton kernels
chosen per shape against torch's varlen attention (`backends/varlen_choice.py`)."""

from library.moss.vendor.kohakumoss_kernels.backends.triton.attention import varlen
from library.moss.vendor.kohakumoss_kernels.backends.varlen_choice import varlen_choice
from library.moss.vendor.kohakumoss_kernels.registry import register

CHOICE = varlen_choice("triton", varlen.attention_varlen, varlen._key)
register("attention_varlen", "triton", CHOICE, varlen.SUPPORTS)
