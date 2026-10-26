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
) -> tuple[str, dict[str, tuple[str, int | None, str, tuple[int, ...]]]]:
    """`(prefix, name -> (source key, qkv chunk, dtype, shape))`.

    Fused joint q / k / v are split into three names (the source tensor is read
    once per layer by :func:`read_state`); derived keys and ComfyUI's runtime
    hint tensors are left out.
    """
    raw = list(head)
    prefix = prefix_of(raw)
    mapping: dict[str, tuple[str, int | None, str, tuple[int, ...]]] = {}
    for key, (dtype, shape) in head.items():
        name = key[len(prefix) :]
        if name in DERIVED:
            continue
        if _RUNTIME_HINT.match(name):
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
    return prefix, mapping


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
    name = Path(path).name
    scales = record.get("scales", record.get("c"))
    if scales is None:
        raise CheckpointFormatError(f"{name}: {FOLD_KEY} has no c / scales")
    if isinstance(scales, bool) or not isinstance(scales, (int, float, list, tuple)):
        raise CheckpointFormatError(
            f"{name}: {FOLD_KEY} scales must be a number or a list of numbers, "
            f"got {type(scales).__name__}"
        )
    if isinstance(scales, (int, float)):
        c = float(scales)
    else:
        try:
            values = [float(value) for value in scales]
        except (TypeError, ValueError) as exc:
            raise CheckpointFormatError(
                f"{name}: {FOLD_KEY} scales are not numbers ({exc})"
            ) from exc
        if not values:
            raise CheckpointFormatError(f"{name}: {FOLD_KEY} scales are empty")
        if len(set(values)) != 1:
            raise CheckpointFormatError(
                f"{name}: per-block fold scales {values} are unsupported; "
                "load a uniform-gate fp16-fix file (or the un-folded weights)"
            )
        c = values[0]
    if not c > 0:
        raise CheckpointFormatError(f"{name}: fold c={c!r} is not positive")
    via = str(record.get("via", "gate"))
    if via != "gate":
        raise CheckpointFormatError(
            f"{name}: fold via={via!r} is unsupported (only the uniform "
            "'gate' fold the fp16-fix files carry can be undone)"
        )
    pregate = record.get("pregate")
    if pregate is None:
        pregate = (1.0, 1.0)
    if not isinstance(pregate, (list, tuple)) or len(pregate) != 2:
        raise CheckpointFormatError(
            f"{name}: {FOLD_KEY} pregate must be [s_attn, s_ffn], got {pregate!r}"
        )
    try:
        s_attn, s_ffn = (float(value) for value in pregate)
    except (TypeError, ValueError) as exc:
        raise CheckpointFormatError(
            f"{name}: {FOLD_KEY} pregate is not numeric ({exc})"
        ) from exc
    return {
        "c": c,
        "via": via,
        "s_attn": float(record.get("s_attn", s_attn)),
        "s_ffn": float(record.get("s_ffn", s_ffn)),
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


def _rope_base(
    path: Path,
    head: Mapping[str, tuple[str, tuple[int, ...]]],
    prefix: str,
    declared: float | None,
) -> float:
    """`rope_base` from the file's own `rope.bands` (`bands[i] = base**(-i/n)`).

    Only an fp32 table reconstructs the base exactly. A 16-bit table loses
    enough mantissa that the round trip lands 0.1 %–20 % off (the exponent
    amplifies the band's own rounding), and a one-entry table says nothing about
    the base at all; those files must name the base in their `<file>.json`
    sidecar. Guessing would shift every RoPE frequency of the run without a
    single warning, so it is refused instead.
    """
    key = f"{prefix}rope.bands"
    source = head.get(key)
    if source is None:
        return _KNOWN_ROPE_BASE if declared is None else float(declared)
    dtype = source[0]
    with _open(path) as handle:
        bands = handle.get_tensor(key).float()
    n = int(bands.numel())
    if abs(float(bands[0]) - 1.0) > 1e-3:
        raise CheckpointFormatError(
            f"{path.name}: {key} is not the expected base**(-i/n) table"
        )
    if n < 2:
        # A one-entry table holds only the exponent-0 band (1.0), which is the
        # same for every base: the default is the only thing it can mean.
        return _KNOWN_ROPE_BASE if declared is None else float(declared)
    if dtype != "F32":
        if declared is not None:
            return float(declared)
        raise CheckpointFormatError(
            f"{path.name}: stores {key} as {dtype}, so rope_base cannot be "
            f'recovered exactly; put {{"rope_base": <value>}} in '
            f"{path.name}.json (or load a file that keeps rope.bands in fp32)"
        )
    base = float((1.0 / float(bands[1])) ** n)
    rounded = round(base)
    if not (rounded > 0 and abs(base / rounded - 1.0) < 1e-6):
        # The table is fp32, so the reconstruction carries a few ulps of error;
        # every shipped base is a whole number (10000 for MOSS-Image).
        if declared is None:
            raise CheckpointFormatError(
                f"{path.name}: rope.bands reconstructs rope_base={base!r}, not a "
                f'whole number; put {{"rope_base": <value>}} in {path.name}.json'
            )
        return float(declared)
    if declared is not None and abs(float(declared) - rounded) > 1e-6 * rounded:
        raise CheckpointFormatError(
            f"{path.name}.json disagrees with {path.name}: rope_base="
            f"{declared!r}, the fp32 bands say {float(rounded)!r}"
        )
    return float(rounded)


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


def _declared(path: Path) -> dict:
    """The one config sidecar beside `path` (empty when there is none)."""
    sidecars = _sidecars(path)
    if len(sidecars) > 1:
        raise CheckpointFormatError(
            f"{path.name}: two config sidecars ({sidecars[0].name}, "
            f"{sidecars[1].name}) — keep one"
        )
    if not sidecars:
        return {}
    sibling = sidecars[0]
    try:
        declared = json.loads(sibling.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CheckpointFormatError(f"{sibling.name} is not JSON ({exc})") from exc
    if not isinstance(declared, dict):
        raise CheckpointFormatError(f"{sibling.name} is not a JSON object")
    return declared


def arch_config(path: str | Path, head=None) -> dict:
    """`MOSSImageTransformer` kwargs, inferred from the file's tensor shapes.

    A `<file>.json` beside the checkpoint — `foo.json` or the literal
    `foo.safetensors.json`, a diffusers `config.json` the user copied over —
    may **override** `rope_base`, the one value a 16-bit file cannot state
    exactly. Every other key is read from the shapes, so a sidecar that
    disagrees with them is an error (a wrong architecture must not load).
    There is one MOSS-Image architecture (42 joint blocks, 2 text-refiner
    blocks), so an unknown shape is an error rather than a guess.
    """
    path = Path(path)
    head = head if head is not None else header(path)
    declared = _declared(path)
    prefix, mapping = _layout(head)

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
    }
    declared_base = declared.pop("rope_base", None)
    if declared_base is not None and not isinstance(declared_base, (int, float)):
        raise CheckpointFormatError(
            f"{path.name}.json: rope_base must be a number, got {declared_base!r}"
        )
    cfg["rope_base"] = _rope_base(path, head, prefix, declared_base)
    for key, value in declared.items():
        if key not in cfg:
            continue
        if cfg[key] != value:
            raise CheckpointFormatError(
                f"{path.name}.json disagrees with {path.name}: {key}="
                f"{value!r}, the file says {cfg[key]!r}"
            )
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
    _prefix, mapping = _layout(head)
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


# --------------------------------------------------------------------- VAE

# ComfyUI / BFL single-file VAEs use the ldm key layout (`flux2-vae`); a
# diffusers export keeps `down_blocks` / `up_blocks` names. Both are accepted:
# the layout is detected from the key set, the BFL keys map onto
# `AutoencoderKLFlux2`'s names, and the config is inferred from the shapes (the
# architecture is fixed apart from widths, layers and attention).
VAE_BFL_MARK = "encoder.down.0.block.0.conv1.weight"
_VAE_RESNET_PARTS = {"nin_shortcut": "conv_shortcut"}
_VAE_ATTN_PARTS = {
    "norm": "group_norm",
    "q": "to_q",
    "k": "to_k",
    "v": "to_v",
    "proj_out": "to_out.0",
}
_VAE_SIDE_PARTS = {
    "norm_out": "conv_norm_out",
    "conv_in": "conv_in",
    "conv_out": "conv_out",
}
_VAE_TOP_LEVEL = {
    "encoder.quant_conv": "quant_conv",
    "decoder.post_quant_conv": "post_quant_conv",
}
_VAE_STOCK = {
    "act_fn": "silu",
    "norm_num_groups": 32,
    "sample_size": 1024,
    "batch_norm_eps": 1e-4,
    "batch_norm_momentum": 0.1,
}
_VAE_DOWN = re.compile(r"^encoder\.down\.(\d+)\.(block\.(\d+)|downsample)\.(.+)$")
_VAE_UP = re.compile(r"^decoder\.up\.(\d+)\.(block\.(\d+)|upsample)\.(.+)$")
_VAE_MID_BLOCK = re.compile(r"^(encoder|decoder)\.mid\.block_(\d+)\.(.+)$")
_VAE_MID_ATTN = re.compile(
    r"^(encoder|decoder)\.mid\.attn_1\.([a-z_]+)\.(weight|bias)$"
)
_VAE_SIDE = re.compile(r"^(encoder|decoder)\.([a-z_]+)\.(weight|bias)$")
_VAE_TOP = re.compile(
    r"^(encoder\.quant_conv|decoder\.post_quant_conv)\.(weight|bias)$"
)


def vae_is_bfl(keys) -> bool:
    """Whether a key set is BFL / ldm layout (ComfyUI's `flux2-vae`)."""
    return VAE_BFL_MARK in keys


def _vae_up_blocks(keys) -> int:
    """The decoder up-block count of a BFL key set."""
    found = {int(m.group(1)) for key in keys if (m := _VAE_UP.match(key))}
    return 1 + max(found) if found else 0


def _vae_resnet(rest: str) -> str:
    part, _, tail = rest.partition(".")
    return f"{_VAE_RESNET_PARTS.get(part, part)}.{tail}"


def vae_key(key: str, n_up: int) -> str:
    """The diffusers key of BFL key `key` (`n_up` decoder up blocks)."""
    if key.startswith("bn."):
        return key
    if m := _VAE_TOP.match(key):
        return f"{_VAE_TOP_LEVEL[m.group(1)]}.{m.group(2)}"
    if m := _VAE_DOWN.match(key):
        index, rest = m.group(1), m.group(4)
        if m.group(3) is None:
            return f"encoder.down_blocks.{index}.downsamplers.0.{rest}"
        return f"encoder.down_blocks.{index}.resnets.{m.group(3)}.{_vae_resnet(rest)}"
    if m := _VAE_UP.match(key):
        index, rest = n_up - 1 - int(m.group(1)), m.group(4)
        if m.group(3) is None:
            return f"decoder.up_blocks.{index}.upsamplers.0.{rest}"
        return f"decoder.up_blocks.{index}.resnets.{m.group(3)}.{_vae_resnet(rest)}"
    if m := _VAE_MID_BLOCK.match(key):
        index = int(m.group(2)) - 1
        return f"{m.group(1)}.mid_block.resnets.{index}.{_vae_resnet(m.group(3))}"
    if (m := _VAE_MID_ATTN.match(key)) and m.group(2) in _VAE_ATTN_PARTS:
        part = _VAE_ATTN_PARTS[m.group(2)]
        return f"{m.group(1)}.mid_block.attentions.0.{part}.{m.group(3)}"
    if (m := _VAE_SIDE.match(key)) and m.group(2) in _VAE_SIDE_PARTS:
        return f"{m.group(1)}.{_VAE_SIDE_PARTS[m.group(2)]}.{m.group(3)}"
    raise CheckpointFormatError(f"{key!r} is not a FLUX.2 VAE key of the BFL layout")


def _vae_attn_weight(key: str) -> bool:
    m = _VAE_MID_ATTN.match(key)
    return m is not None and m.group(2) != "norm" and m.group(3) == "weight"


def vae_to_diffusers(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """A BFL / ldm VAE state on `AutoencoderKLFlux2` keys (mid attention's 1x1 conv
    weights viewed as `[C, C]`)."""
    n_up = _vae_up_blocks(state)
    out = {}
    for key, tensor in state.items():
        name = vae_key(key, n_up)
        out[name] = (
            tensor.reshape(tensor.shape[:2]) if _vae_attn_weight(key) else tensor
        )
    return out


def _vae_diffusers_shapes(shapes: Mapping[str, tuple]) -> dict[str, tuple]:
    """Key -> shape on `AutoencoderKLFlux2`'s keys, from either key style."""
    if not vae_is_bfl(shapes):
        return dict(shapes)
    n_up = _vae_up_blocks(shapes)
    return {
        vae_key(key, n_up): (
            tuple(shape[:2]) if _vae_attn_weight(key) else tuple(shape)
        )
        for key, shape in shapes.items()
    }


def _count_keys(shapes, pattern: str) -> int:
    count = 0
    while pattern.format(count) in shapes:
        count += 1
    return count


def infer_vae_config(shapes: Mapping[str, tuple]) -> dict:
    """`AutoencoderKLFlux2` kwargs of a state given as key -> shape (either layout).

    Widths, layers per block, channels and conv / attention presence come from the
    shapes; the rest is the stock config (silu, 32 groups, patch 2x2 when `bn` says
    so). There is one FLUX.2 VAE family, so a shape the inference cannot explain is
    an error, not a guess.
    """
    shapes = _vae_diffusers_shapes(shapes)
    anchors = (
        "encoder.conv_in.weight",
        "decoder.conv_in.weight",
        "decoder.conv_out.weight",
        "bn.running_mean",
    )
    absent = [name for name in anchors if name not in shapes]
    if absent:
        raise CheckpointFormatError(
            f"not a FLUX.2 VAE: missing {absent} (of {len(shapes)} keys)"
        )
    down = _count_keys(shapes, "encoder.down_blocks.{}.resnets.0.conv1.weight")
    up = _count_keys(shapes, "decoder.up_blocks.{}.resnets.0.conv2.weight")
    if down == 0 or up != down:
        raise CheckpointFormatError(
            f"a FLUX.2 VAE has as many up as down blocks, got {down} / {up}"
        )
    layers = _count_keys(shapes, "encoder.down_blocks.0.resnets.{}.conv1.weight")
    decoded = _count_keys(shapes, "decoder.up_blocks.0.resnets.{}.conv1.weight")
    if decoded != layers + 1:
        raise CheckpointFormatError(
            f"decoder blocks hold {decoded} resnets, expected {layers + 1}"
        )
    latent = shapes["decoder.conv_in.weight"][1]
    patches = shapes["bn.running_mean"][0] // latent
    patch = round(patches**0.5)
    widths = [
        shapes[f"encoder.down_blocks.{i}.resnets.0.conv1.weight"][0]
        for i in range(down)
    ]
    decoded_widths = [
        shapes[f"decoder.up_blocks.{up - 1 - i}.resnets.0.conv2.weight"][0]
        for i in range(up)
    ]
    return {
        **_VAE_STOCK,
        "in_channels": shapes["encoder.conv_in.weight"][1],
        "out_channels": shapes["decoder.conv_out.weight"][0],
        "down_block_types": ("DownEncoderBlock2D",) * down,
        "up_block_types": ("UpDecoderBlock2D",) * up,
        "block_out_channels": tuple(widths),
        "decoder_block_out_channels": tuple(decoded_widths),
        "layers_per_block": layers,
        "latent_channels": latent,
        "use_quant_conv": "quant_conv.weight" in shapes,
        "use_post_quant_conv": "post_quant_conv.weight" in shapes,
        "mid_block_add_attention": "encoder.mid_block.attentions.0.to_q.weight"
        in shapes,
        "patch_size": (patch, patch),
    }


def read_state_all(path: str | Path) -> dict[str, Tensor]:
    """Every tensor of one safetensors file on the host (a VAE is a few hundred MB)."""
    with _open(Path(path)) as handle:
        keys = handle.keys()
        return {key: handle.get_tensor(key) for key in keys}


def load_vae_file(path: str | Path, dtype=torch.float32) -> nn.Module:
    """`AutoencoderKLFlux2` from one safetensors file of either layout, in `dtype`.

    The config is inferred from the shapes; the state is then loaded strictly, so a
    wrong file fails with its missing / unexpected keys instead of silently
    half-loading. Built on the meta device, then one fp32 load of the state.
    """
    from diffusers import AutoencoderKLFlux2

    path = Path(path)
    state = read_state_all(path)
    if vae_is_bfl(state):
        state = vae_to_diffusers(state)
    config = infer_vae_config({key: tuple(t.shape) for key, t in state.items()})
    with torch.device("meta"):
        vae = AutoencoderKLFlux2(**config)
    want = set(vae.state_dict())
    missing = sorted(want - set(state))
    unexpected = sorted(set(state) - want)
    if missing or unexpected:
        raise CheckpointFormatError(
            f"{path.name}: missing {missing[:6]} ({len(missing)}), unexpected "
            f"{unexpected[:6]} ({len(unexpected)})"
        )
    # Load in the target dtype directly: `to(dtype)` on a diffusers model warns
    # about its float32-only modules (none here) and would cast twice.
    keep_fp32 = tuple(getattr(vae, "_keep_in_fp32_modules", None) or ())

    def placed(key: str, value: Tensor) -> Tensor:
        if not value.is_floating_point():
            return value
        if keep_fp32 and any(marker in key for marker in keep_fp32):
            return value.float()
        return value.to(dtype)

    vae.load_state_dict(
        {key: placed(key, value) for key, value in state.items()},
        strict=True,
        assign=True,
    )
    left = [
        name
        for name, tensor in (*vae.named_parameters(), *vae.named_buffers())
        if tensor.is_meta
    ]
    if left:
        raise CheckpointFormatError(f"{path.name}: tensors left unloaded {left[:6]}")
    return vae.eval().requires_grad_(False)
