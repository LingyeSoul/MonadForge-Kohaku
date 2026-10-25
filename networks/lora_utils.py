import os
import re
from typing import Dict, Iterable, List, Optional, Union
import torch
from tqdm import tqdm
from library.runtime.device import synchronize_device
from library.io.safetensors import (
    MemoryEfficientSafeOpen,
    TensorWeightAdapter,
    WeightTransformHooks,
    get_split_weight_filenames,
)
from library.log import setup_logging
from networks.attn_fuse import ATTN_FUSE_SPECS

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)

# AdaLN LoRA key layouts. The same modulation up-projection has two names: the
# in-repo runtime module name (``adaln_up_{branch}``, post ``_dit_rename_hook``)
# and the ComfyUI state-dict path (``adaln_modulation_{branch}.2`` → LoRA key
# ``adaln_modulation_{branch}_2``). ComfyUI's generic key map only knows the
# latter; the in-repo loader only knows the former. These two helpers convert a
# flat LoRA state dict between them so one trained checkpoint round-trips both
# ecosystems (see adaln.md §Key-naming contract). Non-adaln keys pass through.
_ADALN_BRANCHES = "self_attn|cross_attn|mlp"
_ADALN_RUNTIME_KEY_RE = re.compile(
    rf"^(lora_unet_blocks_\d+_)adaln_up_({_ADALN_BRANCHES})(\..+)$"
)
_ADALN_COMFY_KEY_RE = re.compile(
    rf"^(lora_unet_blocks_\d+_)adaln_modulation_({_ADALN_BRANCHES})_2(\..+)$"
)


def relayout_adaln_runtime_to_comfy(
    weights_sd: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Rename runtime adaln keys (``adaln_up_{br}``) to the ComfyUI layout
    (``adaln_modulation_{br}_2``) so the file loads natively in ComfyUI."""
    return {
        (
            f"{m.group(1)}adaln_modulation_{m.group(2)}_2{m.group(3)}"
            if (m := _ADALN_RUNTIME_KEY_RE.match(k))
            else k
        ): v
        for k, v in weights_sd.items()
    }


def relayout_adaln_comfy_to_runtime(
    weights_sd: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Inverse of :func:`relayout_adaln_runtime_to_comfy` — ComfyUI adaln keys
    back to in-repo runtime module names so a comfy-native adaln LoRA also loads
    through ``create_network_from_weights`` (in-repo inference / merge)."""
    return {
        (
            f"{m.group(1)}adaln_up_{m.group(2)}{m.group(3)}"
            if (m := _ADALN_COMFY_KEY_RE.match(k))
            else k
        ): v
        for k, v in weights_sd.items()
    }


def has_comfy_adaln_keys(weights_sd: Dict[str, torch.Tensor]) -> bool:
    """True if any key is in the ComfyUI adaln layout (needs comfy→runtime rename)."""
    return any(_ADALN_COMFY_KEY_RE.match(k) for k in weights_sd)


# PEFT / diffusion-pipe LoRA key layout → kohya (ComfyUI) key layout.
#
# diffusion-pipe and plain PEFT exports name the factors after the *checkpoint*
# module path with PEFT's spelling, e.g.
# ``diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight``. Everything in
# this repo (trainer, merge, inference) speaks the kohya layout
# ``lora_unet_<path-with-underscores>.lora_down.weight``. ComfyUI bridges the
# two with ``comfy.lora_convert.convert_lora``; this is the in-repo equivalent,
# so a PEFT file loads with no external conversion step.
#
# Longest prefixes first: ``model.diffusion_model.`` must win over ``model.``.
_PEFT_WRAPPER_PREFIXES = (
    "base_model.model.",
    "base_model.",
    "model.diffusion_model.",
    "diffusion_model.",
    "model.",
    "net.",
)
# PEFT factor tensors: ``<module>.lora_A.weight`` (+ optional adapter name,
# e.g. ``.lora_A.default.weight``).
_PEFT_FACTOR_RE = re.compile(
    r"^(?P<module>.+?)\.(?P<factor>lora_A|lora_B)"
    r"(?:\.(?P<adapter>[^.]+))?\.weight$"
)
_PEFT_ALPHA_RE = re.compile(r"^(?P<module>.+)\.lora_alpha$")
_PEFT_DORA_RE = re.compile(
    r"^(?P<module>.+)\.lora_magnitude_vector(?:\.(?P<adapter>[^.]+))?$"
)


def _strip_peft_wrapper(module: str) -> str:
    """Drop stacked trainer wrappers (``base_model.model.diffusion_model.`` …)."""
    stripped = module
    while True:
        for prefix in _PEFT_WRAPPER_PREFIXES:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :]
                break
        else:
            return stripped


def has_peft_lora_keys(weights_sd: Iterable[str]) -> bool:
    """True if the keys carry PEFT ``lora_A`` / ``lora_B`` factors."""
    return any(_PEFT_FACTOR_RE.match(k) for k in weights_sd)


def convert_peft_lora_state_dict(
    weights_sd: Dict[str, torch.Tensor],
    alpha_override: Optional[float],
) -> Dict[str, torch.Tensor]:
    """Rewrite PEFT / diffusion-pipe keys to the kohya ``lora_unet_*`` layout.

    Pure: a converted file comes back as a new dict, the input untouched; a file
    already in the kohya layout (or native LoKr/LoHa/GLoKr) is returned as-is.

    ``alpha_override`` wins over a per-module ``lora_alpha`` tensor. When
    neither exists the rank is used (scale = 1.0, the LyCORIS ``alpha ==
    lora_dim`` convention) and a warning names the affected modules — a file
    trained with ``alpha != rank`` needs compensation (``--lora_multiplier``
    for a uniform alpha, ``network_args peft_alpha=<a>`` otherwise).

    Raises rather than guessing on DoRA magnitude vectors and on factor pairs
    that cannot be reconstructed (an ``lora_B`` without its ``lora_A``).
    """
    if not has_peft_lora_keys(weights_sd):
        return weights_sd

    dora_keys = [k for k in weights_sd if _PEFT_DORA_RE.match(k)]
    if dora_keys:
        raise RuntimeError(
            f"PEFT checkpoint carries {len(dora_keys)} DoRA magnitude vector(s) "
            f"(first: {dora_keys[0]!r}). The decompose renormalization cannot be "
            "reproduced by this repo's LoRA modules — load it in ComfyUI or "
            "merge the file with official PEFT/LyCORIS tooling first."
        )

    alpha_by_module: Dict[str, float] = {}
    alpha_source_keys: set[str] = set()
    for key, value in weights_sd.items():
        match = _PEFT_ALPHA_RE.match(key)
        if match is None:
            continue
        if value.numel() != 1:
            raise ValueError(
                f"PEFT alpha at {key!r} is not a scalar (numel={value.numel()}); "
                "cannot derive the per-module scale."
            )
        alpha_by_module[_strip_peft_wrapper(match.group("module"))] = float(
            value.item()
        )
        alpha_source_keys.add(key)

    converted: Dict[str, torch.Tensor] = {}
    module_by_stem: Dict[str, str] = {}
    for key, value in weights_sd.items():
        match = _PEFT_FACTOR_RE.match(key)
        if match is None:
            continue
        module = _strip_peft_wrapper(match.group("module"))
        if not module:
            raise ValueError(f"PEFT key {key!r} has no module path after prefix strip.")
        stem = "lora_unet_" + module.replace(".", "_")
        factor = "lora_down" if match.group("factor") == "lora_A" else "lora_up"
        new_key = f"{stem}.{factor}.weight"
        if new_key in converted or new_key in weights_sd:
            raise RuntimeError(
                f"PEFT conversion collision: {key!r} maps to {new_key!r}, which "
                "already exists in the state dict — the file mixes both layouts "
                "for the same module."
            )
        converted[new_key] = value
        module_by_stem[stem] = module

    defaulted_alphas: List[str] = []
    for stem, module in module_by_stem.items():
        alpha = (
            alpha_override
            if alpha_override is not None
            else alpha_by_module.get(module)
        )
        if alpha is None:
            down = converted.get(f"{stem}.lora_down.weight")
            if down is None:
                raise RuntimeError(
                    f"PEFT module {module!r} has an lora_B factor but no lora_A — "
                    "the rank and alpha are unrecoverable; the file is malformed."
                )
            alpha = float(down.shape[0])
            defaulted_alphas.append(stem)
        converted[f"{stem}.alpha"] = torch.tensor(float(alpha), dtype=torch.float32)

    merged = {
        key: value
        for key, value in weights_sd.items()
        if key not in alpha_source_keys and _PEFT_FACTOR_RE.match(key) is None
    }
    merged.update(converted)
    logger.info(
        f"PEFT LoRA: converted {len(module_by_stem)} module(s) to the "
        "kohya/ComfyUI key layout (lora_A/lora_B → lora_down/lora_up)"
    )
    if defaulted_alphas:
        logger.warning(
            f"PEFT LoRA has no lora_alpha for {len(defaulted_alphas)} module(s) "
            f"(first: {defaulted_alphas[0]!r}) — assuming alpha = rank "
            "(scale 1.0). If the adapter was trained with alpha != rank, pass "
            "network_args peft_alpha=<alpha>, or compensate a uniform mismatch "
            "exactly with --lora_multiplier <alpha/rank>."
        )
    return merged


def filter_lora_state_dict(
    weights_sd: Dict[str, torch.Tensor],
    include_pattern: Optional[str] = None,
    exclude_pattern: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    original_key_count = len(weights_sd.keys())
    if include_pattern is not None:
        regex_include = re.compile(include_pattern)
        weights_sd = {k: v for k, v in weights_sd.items() if regex_include.search(k)}
        logger.info(
            f"Filtered keys with include pattern {include_pattern}: {original_key_count} -> {len(weights_sd.keys())}"
        )

    if exclude_pattern is not None:
        original_key_count_ex = len(weights_sd.keys())
        regex_exclude = re.compile(exclude_pattern)
        weights_sd = {
            k: v for k, v in weights_sd.items() if not regex_exclude.search(k)
        }
        logger.info(
            f"Filtered keys with exclude pattern {exclude_pattern}: {original_key_count_ex} -> {len(weights_sd.keys())}"
        )

    if len(weights_sd) != original_key_count:
        remaining_keys = list(set([k.split(".", 1)[0] for k in weights_sd.keys()]))
        remaining_keys.sort()
        logger.info(f"Remaining LoRA modules after filtering: {remaining_keys}")
        if len(weights_sd) == 0:
            logger.warning("No keys left after filtering.")

    return weights_sd


# Derived from `ATTN_FUSE_SPECS`. Save / adapter-load walk the underscore-fragment form via
# `iter_split_groups`; this base-model merge path needs the dotted module-path
# form (e.g. `self_attn.qkv_proj` matches against `blocks.0.self_attn.qkv_proj`).
_FUSED_PROJ_FALLBACK = {
    f"{spec.attn_type}.{spec.fused_letters}_proj": (
        f"{spec.attn_type}.{{}}_proj",
        spec.component_letters,
    )
    for spec in ATTN_FUSE_SPECS
}


def _try_merge_unfused_lora(
    model_key_without_weight: str,
    lora_weight_keys: set,
    lora_sd: dict,
    multiplier: float,
    calc_device: torch.device,
    model_weight: torch.Tensor,
) -> "torch.Tensor | None":
    """Try to merge old unfused LoRA keys (q_proj/k_proj/v_proj) into a fused model weight.

    Returns the concatenated delta tensor, or None if no old keys found.
    """
    for fused_fragment, (template, suffixes) in _FUSED_PROJ_FALLBACK.items():
        if fused_fragment not in model_key_without_weight:
            continue

        deltas = []
        all_found = True
        keys_to_remove = []
        for suffix in suffixes:
            old_module = model_key_without_weight.replace(
                fused_fragment, template.format(suffix)
            )
            found_prefix = None
            for prefix in ["lora_unet_", ""]:
                lora_name = prefix + old_module.replace(".", "_")
                down_key = lora_name + ".lora_down.weight"
                up_key = lora_name + ".lora_up.weight"
                alpha_key = lora_name + ".alpha"
                if down_key in lora_weight_keys and up_key in lora_weight_keys:
                    found_prefix = prefix
                    break
            if found_prefix is None:
                all_found = False
                break

            down = lora_sd[down_key].to(calc_device)
            up = lora_sd[up_key].to(calc_device)
            dim = down.size(0)
            alpha = lora_sd.get(alpha_key, dim)
            scale = alpha / dim

            delta = multiplier * (up @ down) * scale
            deltas.append(delta)
            keys_to_remove.extend(
                [k for k in (down_key, up_key, alpha_key) if k in lora_weight_keys]
            )

        if not all_found or not deltas:
            continue

        for k in keys_to_remove:
            lora_weight_keys.discard(k)

        fused_delta = torch.cat(deltas, dim=0)
        return fused_delta

    return None


def load_safetensors_with_lora(
    model_files: Union[str, List[str]],
    lora_weights_list: Optional[List[Dict[str, torch.Tensor]]],
    lora_multipliers: Optional[List[float]],
    calc_device: torch.device,
    move_to_device: bool = False,
    dit_weight_dtype: Optional[torch.dtype] = None,
    disable_numpy_memmap: bool = False,
    weight_transform_hooks: Optional[WeightTransformHooks] = None,
) -> dict[str, torch.Tensor]:
    """
    Merge LoRA weights into the state dict of a model.

    Args:
        model_files (Union[str, List[str]]): Path to the model file or list of paths. If the path matches a pattern like `00001-of-00004`, it will load all files with the same prefix.
        lora_weights_list (Optional[List[Dict[str, torch.Tensor]]]): List of dictionaries of LoRA weight tensors to load.
        lora_multipliers (Optional[List[float]]): List of multipliers for LoRA weights.
        calc_device (torch.device): Device to calculate on.
        move_to_device (bool): Whether to move tensors to the calculation device after loading.
        disable_numpy_memmap (bool): Whether to disable numpy memmap when loading safetensors.
        weight_transform_hooks (Optional[WeightTransformHooks]): Hooks for transforming weights during loading.
    """

    if isinstance(model_files, str):
        model_files = [model_files]

    extended_model_files = []
    for model_file in model_files:
        split_filenames = get_split_weight_filenames(model_file)
        if split_filenames is not None:
            extended_model_files.extend(split_filenames)
        else:
            extended_model_files.append(model_file)
    model_files = extended_model_files
    logger.info(f"Loading model files: {model_files}")

    weight_hook = None
    if lora_weights_list is None or len(lora_weights_list) == 0:
        lora_weights_list = []
        lora_multipliers = []
        list_of_lora_weight_keys = []
    else:
        list_of_lora_weight_keys = []
        for i, lora_sd in enumerate(lora_weights_list):
            # Strip _orig_mod_ from LoRA keys (inserted by torch.compile during training)
            normalized = {}
            for k, v in lora_sd.items():
                normalized[k.replace("__orig_mod_", "_")] = v
            # Shipped checkpoints carry adaln LoRA keys in the ComfyUI layout
            # (``adaln_modulation_{br}_2`` — lora_save relays them at write
            # time), but this hook sees the DiT's *runtime* key names
            # (``adaln_up_{br}``, post ``_dit_rename_hook``). Without the
            # rename the adaln rows of every ``train_adaln`` LoRA silently
            # fell into the "not all LoRA keys are used" warning on the
            # static-merge path (create_network_from_weights already did
            # this for the live-hook path). Presence-gated: no-op otherwise.
            if has_comfy_adaln_keys(normalized):
                normalized = relayout_adaln_comfy_to_runtime(normalized)
            lora_weights_list[i] = normalized
            lora_weight_keys = set(normalized.keys())
            list_of_lora_weight_keys.append(lora_weight_keys)

        if lora_multipliers is None:
            lora_multipliers = [1.0] * len(lora_weights_list)
        # A scalar multiplier (e.g. the `--lora_multiplier` argparse default of
        # 1.0, which is a bare float under nargs="*") broadcasts to all adapters.
        # Coerce to a list so the length checks below don't len() a float.
        if isinstance(lora_multipliers, (int, float)):
            lora_multipliers = [float(lora_multipliers)] * len(lora_weights_list)
        while len(lora_multipliers) < len(lora_weights_list):
            lora_multipliers.append(1.0)
        if len(lora_multipliers) > len(lora_weights_list):
            lora_multipliers = lora_multipliers[: len(lora_weights_list)]

        logger.info(
            f"Merging LoRA weights into state dict. multipliers: {lora_multipliers}"
        )

        def weight_hook_func(
            model_weight_key, model_weight: torch.Tensor, keep_on_calc_device=False
        ):
            nonlocal \
                list_of_lora_weight_keys, \
                lora_weights_list, \
                lora_multipliers, \
                calc_device

            if not model_weight_key.endswith(".weight"):
                return model_weight

            original_device = model_weight.device
            if original_device != calc_device:
                model_weight = model_weight.to(calc_device)

            for lora_weight_keys, lora_sd, multiplier in zip(
                list_of_lora_weight_keys, lora_weights_list, lora_multipliers
            ):
                lora_name_without_prefix = model_weight_key.rsplit(".", 1)[0]
                found = False
                for prefix in ["lora_unet_", ""]:
                    lora_name = prefix + lora_name_without_prefix.replace(".", "_")
                    down_key = lora_name + ".lora_down.weight"
                    up_key = lora_name + ".lora_up.weight"
                    alpha_key = lora_name + ".alpha"
                    if down_key in lora_weight_keys and up_key in lora_weight_keys:
                        found = True
                        break
                if not found:
                    # Fallback: check for old unfused LoRA keys on fused projections
                    fused_delta = _try_merge_unfused_lora(
                        lora_name_without_prefix,
                        lora_weight_keys,
                        lora_sd,
                        multiplier,
                        calc_device,
                        model_weight,
                    )
                    if fused_delta is not None:
                        model_weight = model_weight + fused_delta
                    continue

                down_weight = lora_sd[down_key]
                up_weight = lora_sd[up_key]

                dim = down_weight.size()[0]
                alpha = lora_sd.get(alpha_key, dim)
                scale = alpha / dim

                down_weight = down_weight.to(calc_device)
                up_weight = up_weight.to(calc_device)

                if len(model_weight.size()) == 2:
                    # linear
                    if len(up_weight.size()) == 4:  # use linear projection mismatch
                        up_weight = up_weight.squeeze(3).squeeze(2)
                        down_weight = down_weight.squeeze(3).squeeze(2)
                    delta = multiplier * (up_weight @ down_weight) * scale
                elif down_weight.size()[2:4] == (1, 1):
                    # conv2d 1x1
                    delta = (
                        multiplier
                        * (
                            up_weight.squeeze(3).squeeze(2)
                            @ down_weight.squeeze(3).squeeze(2)
                        )
                        .unsqueeze(2)
                        .unsqueeze(3)
                        * scale
                    )
                else:
                    # conv2d 3x3
                    conved = torch.nn.functional.conv2d(
                        down_weight.permute(1, 0, 2, 3), up_weight
                    ).permute(1, 0, 2, 3)
                    delta = multiplier * conved * scale

                model_weight = model_weight + delta

                lora_weight_keys.remove(down_key)
                lora_weight_keys.remove(up_key)
                if alpha_key in lora_weight_keys:
                    lora_weight_keys.remove(alpha_key)

            if not keep_on_calc_device and original_device != calc_device:
                model_weight = model_weight.to(original_device)
            return model_weight

        weight_hook = weight_hook_func

    state_dict = _load_safetensors_with_hook(
        model_files,
        calc_device,
        move_to_device,
        dit_weight_dtype,
        weight_hook=weight_hook,
        disable_numpy_memmap=disable_numpy_memmap,
        weight_transform_hooks=weight_transform_hooks,
    )

    for lora_weight_keys in list_of_lora_weight_keys:
        if len(lora_weight_keys) > 0:
            logger.warning(
                f"Warning: not all LoRA keys are used: {', '.join(lora_weight_keys)}"
            )

    return state_dict


def _load_safetensors_with_hook(
    model_files: list[str],
    calc_device: torch.device,
    move_to_device: bool = False,
    dit_weight_dtype: Optional[torch.dtype] = None,
    weight_hook: callable = None,
    disable_numpy_memmap: bool = False,
    weight_transform_hooks: Optional[WeightTransformHooks] = None,
) -> dict[str, torch.Tensor]:
    """Load state dict from safetensors files and run an optional per-tensor hook."""
    logger.info(
        f"Loading state dict. Dtype of weight: {dit_weight_dtype}, hook enabled: {weight_hook is not None}"
    )
    state_dict = {}
    for model_file in model_files:
        with MemoryEfficientSafeOpen(
            model_file, disable_numpy_memmap=disable_numpy_memmap
        ) as original_f:
            f = (
                TensorWeightAdapter(weight_transform_hooks, original_f)
                if weight_transform_hooks is not None
                else original_f
            )
            for key in tqdm(
                f.keys(),
                desc=f"Loading {os.path.basename(model_file)}",
                leave=False,
            ):
                if weight_hook is None and move_to_device:
                    value = f.get_tensor(
                        key, device=calc_device, dtype=dit_weight_dtype
                    )
                else:
                    value = f.get_tensor(
                        key
                    )  # we cannot directly load to device because get_tensor does non-blocking transfer
                    if weight_hook is not None:
                        value = weight_hook(
                            key, value, keep_on_calc_device=move_to_device
                        )
                    if move_to_device:
                        value = value.to(
                            calc_device, dtype=dit_weight_dtype, non_blocking=True
                        )
                    elif dit_weight_dtype is not None:
                        value = value.to(dit_weight_dtype)

                state_dict[key] = value
    if move_to_device:
        synchronize_device(calc_device)

    return state_dict
