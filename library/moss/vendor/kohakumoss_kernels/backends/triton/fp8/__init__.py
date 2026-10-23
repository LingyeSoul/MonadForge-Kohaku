"""Triton fp8 / MXFP8 compute: activation quantizers, fp8 GEMMs and the tuned
`fp8_linear` registration (CUDA sm89+)."""

from library.moss.vendor.kohakumoss_kernels.backends.triton.fp8 import linear  # noqa: F401  (registers)
