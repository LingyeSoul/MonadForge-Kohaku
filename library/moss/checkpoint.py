"""ComfyUI single-file DiT checkpoints for MOSS-Image.

A ComfyUI file is one `.safetensors` holding **only the DiT** — the text encoder
and VAE stay in `--model_dir`. Two things differ from the published diffusers
`transformer/` folder:

- the joint blocks store q / k / v **fused** as `layers.N.proj.qkv.weight`
  (ComfyUI does `self.proj["qkv"](y).chunk(3, dim=-1)`, so the rows are q ; k ; v);
- an fp16-fix file carries the residual-stream fold **in its weights**, plus a
  `kohakumoss_fp16_fix` metadata record.

`load_transformer(dit=...)` therefore reads the header first (`inspect`), infers
the architecture from tensor shapes (`arch_config`), and materialises one tensor
at a time (`read_state`) — a 10 GB file never holds a second copy.

The fold is **undone on load** (`unfold_multiplier`): a LoRA delta trained on a
folded base would otherwise be in the folded parameterisation, and merging it
onto the official weights would be off by the fold's row scales. Unfolding is
exact for the shipped `c = 1/16` (a power of two), and afterwards the file is
interchangeable with the diffusers folder — `--precision fp16` re-applies the
fold at run time exactly as it does there.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from pathlib import Path

import torch
from safetensors import safe_open
from torch import Tensor, nn


class CheckpointFormatError(ValueError):
    """A single-file DiT does not satisfy the MOSS-Image transformer contract."""


# State-dict prefixes ComfyUI exporters put in front of every key ("" if none).
PREFIXES = ("model.diffusion_model.", "diffusion_model.", "transformer.")
# Keys the model derives instead of reading (the RoPE frequencies are rebuilt
# from `rope_base`, which this module infers from them).
DERIVED = ("rope.bands",)
# The upstream fp16-fix record: `{"c", "via", "s_attn", "s_ffn", "eps"}` (or the
# newer `{"scales", "via", "pregate", "eps"}`); see `fold_record`.
FOLD_KEY = "kohakumoss_fp16_fix"

FUSED_QKV = re.compile(r"^(layers\.\d+\.proj)\.qkv\.weight$")
_RUNTIME_HINT = re.compile(r"^layers\.\d+\.comfy_attention\.")
_TEXT_WRITER = re.compile(
    r"^text_embedder\.blocks\.\d+\.(?:proj|ffn_down)\.(?:weight|bias)$"
)
_JOINT_O = re.compile(r"^layers\.\d+\.proj\.o\.weight$")
_JOINT_FFN_DOWN = re.compile(r"^layers\.\d+\.proj\.ffn_down\.weight$")
# Folded by 1/c on load (upstream `writer_scales`: the embedder and the text
# refiner's own stream writers).
_FOLDED_WRITERS = frozenset(
    {
        "video_embedder.weight",
        "video_embedder.bias",
        "text_embedder.proj.weight",
        "text_embedder.proj.bias",
    }
)
_QUANT_SUFFIXES = (".comfy_quant", ".weight_scale")
_SUPPORTED_DTYPES = ("F16", "BF16", "F32")
_KNOWN_ROPE_BASE = 10000.0
_TEXT_BLOCKS = 2


def _open(path: Path):
    return safe_open(str(path), framework="pt", device="cpu")


def metadata(path: str | Path) -> dict:
    """The safetensors metadata (an empty dict when the file carries none)."""
    with _open(Path(path)) as handle:
        return handle.metadata() or {}


def header(path: str | Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    """`key -> (dtype, shape)` from the header; no tensor is materialised."""
    with _open(Path(path)) as handle:
        keys = handle.keys()
        return {
            key: (
                handle.get_slice(key).get_dtype(),
                tuple(handle.get_slice(key).get_shape()),
            )
            for key in keys
        }


def prefix_of(keys: list[str]) -> str:
    """The format prefix every key carries ("" if none)."""
    for prefix in PREFIXES:
        if keys and all(key.startswith(prefix) for key in keys):
            return prefix
    return ""


def _layout(
    head: Mapping[str, tuple[str, tuple[int, ...]]],
) -> tuple[str, dict[str, tuple[str, int | None, str, tuple[int, ...]]], int]:
    """`(prefix, name -> (source key, qkv chunk, dtype, shape), hint count)`.

    Fused joint q / k / v are split into three names (the source tensor is read
    once per layer by :func:`read_state`); derived keys and ComfyUI's runtime
    hint tensors are left out.
    """
    raw = list(head)
    prefix = prefix_of(raw)
    mapping: dict[str, tuple[str, int | None, str, tuple[int, ...]]] = {}
    hints = 0
    for key, (dtype, shape) in head.items():
        name = key[len(prefix) :]
        if name in DERIVED:
            continue
        if _RUNTIME_HINT.match(name):
            hints += 1
            continue
        m = FUSED_QKV.match(name)
        if m is None:
            mapping[name] = (key, None, dtype, shape)
            continue
        if shape[0] % 3:
            raise CheckpointFormatError(
                f"fused qkv weight {key} has {shape[0]} output rows, "
                "not a multiple of 3"
            )
        part = (shape[0] // 3, *shape[1:])
        for index, layer in enumerate("qkv"):
            mapping[f"{m.group(1)}.{layer}.weight"] = (key, index, dtype, part)
    return prefix, mapping, hints


def fold_record(path: str | Path) -> dict | None:
    """The weight fold a file carries (None for plain 16-bit weights).

    Only a uniform gate fold (upstream's fp16-fix recipe) can be undone here: a
    per-block (via="proj") fold needs a per-block residual rescale the model
    does not implement, so it is refused rather than loaded wrong.

    The record's ``eps`` is informational: unfolding restores the official
    weights, and the norms keep the stock eps those weights already assume —
    the exact-eps mode only changes runtime norms, never the tensors.
    """
    raw = metadata(path).get(FOLD_KEY)
    if not raw:
        return None
    try:
        record = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CheckpointFormatError(
            f"{Path(path).name}: {FOLD_KEY} is not JSON ({exc})"
        ) from exc
    scales = record.get("scales", record.get("c"))
    if scales is None:
        raise CheckpointFormatError(f"{Path(path).name}: {FOLD_KEY} has no c / scales")
    if isinstance(scales, (int, float)):
        c = float(scales)
    else:
        values = [float(value) for value in scales]
        if len(set(values)) != 1:
            raise CheckpointFormatError(
                f"{Path(path).name}: per-block fold scales {values} are unsupported; "
                "load a uniform-gate fp16-fix file (or the un-folded weights)"
            )
        c = values[0]
    if not c > 0:
        raise CheckpointFormatError(f"{Path(path).name}: fold c={c!r} is not positive")
    via = str(record.get("via", "gate"))
    if via != "gate":
        raise CheckpointFormatError(
            f"{Path(path).name}: fold via={via!r} is unsupported (only the uniform "
            "'gate' fold the fp16-fix files carry can be undone)"
        )
    pregate = record.get("pregate") or (1.0, 1.0)
    return {
        "c": c,
        "via": via,
        "s_attn": float(record.get("s_attn", pregate[0])),
        "s_ffn": float(record.get("s_ffn", pregate[1])),
        "eps": str(record.get("eps", "stock")),
    }


def assert_compute_weights(path: str | Path) -> None:
    """Refuse a quantized single file: this loader reads 16-bit weights only.

    `library/moss/quant.py` quantizes a *loaded* base for inference; a file that
    is already int8 / fp8 (ComfyUI's `comfy_quant` + `weight_scale` records) has
    no 16-bit weights to train through, and no dequantization happens here.
    """
    head = header(path)
    quant = sorted(key for key in head if key.endswith(_QUANT_SUFFIXES))
    if quant:
        raise CheckpointFormatError(
            f"quantized MOSS-Image checkpoint is unsupported: {Path(path).name}; "
            f"found {quant[0]}. Select a 16-bit (BF16 / FP16 / FP32) single file — "
            "automatic dequantization is disabled"
        )
    for key, (dtype, _shape) in head.items():
        if dtype not in _SUPPORTED_DTYPES:
            raise CheckpointFormatError(
                f"unsupported weight dtype in {Path(path).name}: {key} = {dtype}; "
                "select a 16-bit (BF16 / FP16 / FP32) single file"
            )


def _rope_base(path: Path, head: Mapping[str, tuple[str, tuple[int, ...]]]) -> float:
    """`rope_base` from the file's own `rope.bands` (`bands[i] = base**(-i/n)`)."""
    if "rope.bands" not in head:
        return _KNOWN_ROPE_BASE
    with _open(path) as handle:
        bands = handle.get_tensor("rope.bands").float()
    n = int(bands.numel())
    if abs(float(bands[0]) - 1.0) > 1e-3:
        raise CheckpointFormatError(
            f"{path.name}: rope.bands is not the expected base**(-i/n) table"
        )
    if n < 2:
        # One band is base-independent (only the exponent-0 entry), so a tiny
        # model's table cannot tell a custom base from the default.
        return _KNOWN_ROPE_BASE
    base = float((1.0 / float(bands[1])) ** n)
    rounded = round(base)
    if rounded > 0 and abs(base / rounded - 1.0) < 1e-6:
        # The table is fp32, so the reconstruction carries a few ulps of error;
        # every shipped base is a whole number (10000 for MOSS-Image).
        return float(rounded)
    return base


def _sidecars(path: Path) -> list[Path]:
    """`<file>.json` config sidecars beside a checkpoint.

    Both readings are accepted — `foo.json` (`with_suffix`) and the literal
    `foo.safetensors.json` — because the docs say "<file>.json" and users write
    both. Two at once is ambiguous, so `arch_config` refuses it instead of
    silently picking one.
    """
    return [
        candidate
        for candidate in (path.with_suffix(".json"), Path(f"{path}.json"))
        if candidate.is_file()
    ]


def arch_config(path: str | Path, head=None) -> dict:
    """`MOSSImageTransformer` kwargs, inferred from the file's tensor shapes.

    A `<file>.json` beside the checkpoint — `foo.json` or the literal
    `foo.safetensors.json`, a diffusers `config.json` the user copied over —
    overrides the inference and must agree with it. There is one MOSS-Image
    architecture (42 joint blocks, 2 text-refiner blocks), so an unknown shape
    is an error rather than a guess.
    """
    path = Path(path)
    head = head if head is not None else header(path)
    _prefix, mapping, _hints = _layout(head)

    def shape(name: str) -> tuple[int, ...]:
        if name not in mapping:
            raise CheckpointFormatError(
                f"{path.name} is not a MOSS-Image DiT: missing {name}"
            )
        return mapping[name][3]

    modulation_rows = shape("shared_block_modulation.weight")[0]
    if modulation_rows % 6:
        raise CheckpointFormatError(
            f"{path.name}: shared_block_modulation has {modulation_rows} output "
            "rows, not 6 × dim"
        )
    dim = modulation_rows // 6
    head_dim = shape("layers.0.q_norm.weight")[0]
    depth = 1 + max(
        int(name.split(".")[1]) for name in mapping if name.startswith("layers.")
    )
    text_blocks = {
        int(name.split(".")[2])
        for name in mapping
        if name.startswith("text_embedder.blocks.")
    }
    if text_blocks != set(range(_TEXT_BLOCKS)):
        raise CheckpointFormatError(
            f"{path.name}: {len(text_blocks)} text-refiner blocks, expected "
            f"{_TEXT_BLOCKS}"
        )
    if dim % head_dim:
        raise CheckpointFormatError(
            f"{path.name}: dim={dim} is not a multiple of head_dim={head_dim}"
        )
    cfg = {
        "dim": dim,
        "num_heads": dim // head_dim,
        "num_layers": depth,
        "ffn_dim": shape("layers.0.proj.ffn_up_gate.weight")[0] // 2,
        "text_dim": shape("text_embedder.proj.weight")[1],
        "freq_dim": shape("time_embedding.0.weight")[1],
        "latent_channels": shape("video_embedder.weight")[1],
        "rope_base": _rope_base(path, head),
    }
    sidecars = _sidecars(path)
    if len(sidecars) > 1:
        raise CheckpointFormatError(
            f"{path.name}: two config sidecars ({sidecars[0].name}, "
            f"{sidecars[1].name}) — keep one"
        )
    if sidecars:
        sibling = sidecars[0]
        try:
            declared = json.loads(sibling.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CheckpointFormatError(f"{sibling.name} is not JSON ({exc})") from exc
        updates = {}
        for key, value in cfg.items():
            if key not in declared:
                continue
            if declared[key] != value:
                raise CheckpointFormatError(
                    f"{sibling.name} disagrees with {path.name}: {key}="
                    f"{declared[key]!r}, the file says {value!r}"
                )
            updates[key] = declared[key]
        cfg.update(updates)
    return cfg


def inspect(
    path: str | Path, expected: Mapping[str, tuple[int, ...]]
) -> dict[str, tuple[str, int | None]]:
    """Validate the whole checkpoint against `expected` without loading it.

    Returns `name -> (source key, qkv chunk)` for the tensors to assign. Every
    key, shape and dtype is checked from the header first, so a wrong file fails
    before a single weight is read.
    """
    head = header(path)
    _prefix, mapping, _hints = _layout(head)
    missing = [name for name in expected if name not in mapping and name not in DERIVED]
    unexpected = [name for name in mapping if name not in expected]
    if missing or unexpected:
        raise CheckpointFormatError(
            f"incomplete MOSS-Image checkpoint {Path(path).name}: missing "
            f"{missing[:8]} ({len(missing)}), unexpected {unexpected[:8]} "
            f"({len(unexpected)})"
        )
    for name, (source, _chunk, dtype, shape) in mapping.items():
        if name not in expected:
            continue
        if dtype not in _SUPPORTED_DTYPES:
            raise CheckpointFormatError(
                f"unsupported weight dtype in {Path(path).name}: {source} = {dtype}"
            )
        if shape != tuple(expected[name]):
            raise CheckpointFormatError(
                f"wrong tensor shape in {Path(path).name}: {source} has "
                f"{list(shape)}, expected {list(expected[name])}"
            )
    return {name: (source, chunk) for name, (source, chunk, _, _) in mapping.items()}


def unfold_multiplier(
    fold: Mapping | None, name: str, shape: tuple[int, ...]
) -> float | Tensor | None:
    """The factor that undoes `fold` on parameter `name` (per-row for the gates).

    Mirrors upstream's `writer_scales` / `_row_scales` (via gate): the embedder
    and the text refiner's writers carry `c`, the shared modulation's two gate
    row bands carry `c / s`, and the pre-gate scales ride `proj.o` /
    `proj.ffn_down`.
    """
    if fold is None:
        return None
    c = float(fold["c"])
    if name in _FOLDED_WRITERS or _TEXT_WRITER.match(name):
        return 1.0 / c
    if name.startswith("shared_block_modulation."):
        dim = shape[0] // 6
        if dim * 6 != shape[0]:
            raise CheckpointFormatError(f"{name} has {shape[0]} rows, not 6 * dim")
        rows = torch.ones(shape[0], dtype=torch.float32)
        rows[2 * dim : 3 * dim] = float(fold["s_attn"]) / c
        rows[5 * dim : 6 * dim] = float(fold["s_ffn"]) / c
        return rows
    if _JOINT_O.match(name) and float(fold["s_attn"]) != 1.0:
        return 1.0 / float(fold["s_attn"])
    if _JOINT_FFN_DOWN.match(name) and float(fold["s_ffn"]) != 1.0:
        return 1.0 / float(fold["s_ffn"])
    return None


def _unfold(t: Tensor, multiplier: float | Tensor | None) -> Tensor:
    """`t` with the fold divided out (fp32 math, one rounding back to `t.dtype`)."""
    if multiplier is None:
        return t
    wide = torch.promote_types(t.dtype, torch.float32)
    if isinstance(multiplier, Tensor):
        multiplier = multiplier.reshape(-1, *([1] * (t.dim() - 1))).to(wide)
    return (t.to(wide) * multiplier).to(t.dtype)


def read_state(
    path: str | Path,
    plan: Mapping[str, tuple[str, int | None]],
    dtype: torch.dtype,
    fold: Mapping | None = None,
) -> Iterator[tuple[str, Tensor]]:
    """`(name, tensor)` for every planned key: read, split, un-fold, cast.

    One tensor lives at a time (the fused qkv source is cached per layer because
    three names share it), so the host peak is the model plus one weight.
    """
    with _open(Path(path)) as handle:
        fused: tuple[str, Tensor] | None = None
        for name, (source, chunk) in plan.items():
            if chunk is None:
                tensor = handle.get_tensor(source)
            else:
                if fused is None or fused[0] != source:
                    fused = (source, handle.get_tensor(source))
                tensor = fused[1].chunk(3, 0)[chunk]
            tensor = _unfold(tensor, unfold_multiplier(fold, name, tuple(tensor.shape)))
            yield name, tensor.to(dtype).contiguous()


def assign(model: nn.Module, name: str, value: Tensor) -> None:
    """Put `value` into `model`'s parameter (or buffer) called `name`."""
    path, _, attribute = name.rpartition(".")
    module = model.get_submodule(path)
    if attribute in module._parameters:
        module._parameters[attribute] = nn.Parameter(value, requires_grad=False)
    else:
        module._buffers[attribute] = value
