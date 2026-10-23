"""Vendored MOSS kernel stack, extracted from KohakuMossTrainer.

Source: https://github.com/KohakuBlueleaf/KohakuMossTrainer (``src/kohakumoss_kernels``
and ``src/kohakumoss_common``), Copyright 2026 Shih-Ying Yeh (KohakuBlueLeaf) and
contributors, under the KohakuMoss License 1.0 — Apache-2.0 plus a naming requirement,
see ``LICENSE-KohakuMoss`` in this directory. Only the package root changed
(``kohakumoss_*`` -> ``library.moss.vendor.kohakumoss_*``); the code is otherwise
verbatim, so any fix upstream applies can be diffed in.

What is here: the op registry with per-arch backend priority and fallback, the tuning
table that binds the faster of kernel / ``torch.compile`` per shape, the Triton and
TileLang kernels (varlen attention, the MOSS block ops, the Qwen3.5 gated-delta-rule
text-encoder kernels, ConvRot-quantised linears, fp8), and the eager reference every
kernel is checked against.

``library.moss.ops`` is the façade the rest of this repository imports; nothing outside
``library/moss/`` should reach into this tree.
"""
