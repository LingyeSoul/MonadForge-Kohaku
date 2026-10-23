"""Kernel library for KohakuMoss: ops, backends, constraints and tuning. Importing it
has no side effect: the backends, the host toolchain and the compile-cache key tag are
set up on the first kernel build (`registry.load_backends`)."""

from library.moss.vendor.kohakumoss_kernels import cache_tag, registry
from library.moss.vendor.kohakumoss_kernels.hardware.arch import Arch, arch_of
from library.moss.vendor.kohakumoss_kernels.registry import (
    available,
    load_backends,
    register,
    resolve,
    set_priority,
)

__all__ = [
    "Arch",
    "arch_of",
    "available",
    "cache_tag",
    "load_backends",
    "register",
    "registry",
    "resolve",
    "set_priority",
]
