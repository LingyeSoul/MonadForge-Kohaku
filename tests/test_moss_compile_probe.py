"""Compiling a fused MOSS block is guarded by a one-off Triton trace probe.

Dynamo builds a Triton kernel's mutated-arg list by generating TTIR from example
arguments, where a runtime scalar (a norm ``eps``) arrives as a Python float. The
vendored kernels call ``eps.to(tl.float32)``, so an incompatible torch/triton pair
throws inside that pass; torch then assumes every input is mutated and the graph
reinterprets a buffer at the wrong dtype. The probe turns that crash into a skip.
"""

from __future__ import annotations

import torch

from library.moss import loader


class _Block:
    """Just enough of a MOSS block for ``compile_blocks``."""

    def __init__(self, fused: bool):
        self.fused = fused
        self.forward = lambda x: x


def test_fused_blocks_are_not_compiled_when_the_probe_fails(monkeypatch):
    monkeypatch.setattr(loader, "_PROBE", False)
    blocks = [_Block(fused=True) for _ in range(3)]
    originals = [block.forward for block in blocks]

    def unexpected(*_args, **_kwargs):
        raise AssertionError("torch.compile must not run when the probe declined")

    monkeypatch.setattr(torch, "compile", unexpected)
    loader.compile_blocks(blocks)
    assert [block.forward for block in blocks] == originals


def test_fused_blocks_are_compiled_when_the_probe_passes(monkeypatch):
    monkeypatch.setattr(loader, "_PROBE", True)
    compiled: list[object] = []

    def fake_compile(fn, **_kwargs):
        compiled.append(fn)
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    loader.compile_blocks([_Block(fused=True) for _ in range(2)])
    assert len(compiled) == 2


def test_the_reference_branch_compiles_without_probing(monkeypatch):
    """Nothing to trace means nothing to probe: an unfused block always compiles."""
    monkeypatch.setattr(loader, "_PROBE", None)

    def probe_must_not_run():
        raise AssertionError("the reference branch has no Triton kernel to trace")

    monkeypatch.setattr(loader, "triton_compile_ok", probe_must_not_run)
    compiled: list[object] = []

    def fake_compile(fn, **_kwargs):
        compiled.append(fn)
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    loader.compile_blocks([_Block(fused=False)])
    assert len(compiled) == 1


def test_the_probe_rejects_a_right_answer_at_the_wrong_dtype():
    """A fp32 result with the right numbers is still the broken graph."""
    reference = torch.ones(4, 4, dtype=torch.float16)
    assert loader._probe_matches(reference, torch.ones(4, 4, dtype=torch.float16))
    assert not loader._probe_matches(reference, torch.ones(4, 4, dtype=torch.float32))
    assert not loader._probe_matches(reference, torch.zeros(4, 4, dtype=torch.float16))
    assert not loader._probe_matches(reference, torch.ones(4, 3, dtype=torch.float16))


class _ParamModel:
    """Stand-in for the DiT: ``fuse`` reads the device and dtype of its parameters."""

    def __init__(self, dtype):
        self.layers = [_Block(fused=False)]
        self._parameter = torch.empty(0, dtype=dtype)

    def parameters(self):
        return iter([self._parameter])


def test_fusion_declines_on_the_fp32_adapter_fp16_lane(monkeypatch, caplog):
    """fp32 adapter merges return fp32; the 16-bit block ops cannot consume that."""
    from library.moss import ops

    monkeypatch.setattr(ops, "available", lambda _device: True)
    with caplog.at_level("INFO", logger="library.moss.ops"):
        fused = ops.fuse(_ParamModel(torch.float16), fp32_adapters=True)
    assert fused == 0
    assert "fp32" in caplog.text
    # bf16 has no fp32 merges, and an fp16 run whose adapters stay 16-bit still fuses.
    assert ops.fuse(_ParamModel(torch.bfloat16), fp32_adapters=True) == 1
    assert ops.fuse(_ParamModel(torch.float16), fp32_adapters=False) == 1


def test_the_probe_is_cached_and_honours_the_opt_out(monkeypatch):
    # A box without a card short-circuits to "nothing to fuse"; the probe only
    # runs on a CUDA host (this suite runs on CPU too).
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(loader, "_PROBE", None)
    monkeypatch.setattr(loader, "_probe_triton_compile", lambda: False)
    assert loader.triton_compile_ok() is False

    def probed_twice():
        raise AssertionError("probed twice")

    monkeypatch.setattr(loader, "_probe_triton_compile", probed_twice)
    assert loader.triton_compile_ok() is False

    monkeypatch.setattr(loader, "_PROBE", None)
    monkeypatch.setenv(loader.PROBE_ENV, "0")
    assert loader.triton_compile_ok() is True
