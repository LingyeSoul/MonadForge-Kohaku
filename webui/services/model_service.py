"""Model service — listing and management of trained adapter weights.

Aggregates every final weight below ``output/ckpt`` (canonical and legacy
layouts via :func:`discover_weights`) and enriches each entry with the
adapter type read from the safetensors header metadata. Metadata reads are
header-only, so listing never loads tensors.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from webui.services.config_service import ROOT
from webui.services.paths import resolve_path

# NETWORK_REGISTRY key (stamped as ``ss_network_spec``) → display label.
_SPEC_LABELS = {
    "lora": "LoRA",
    "lokr": "LoKr",
    "glokr": "GLoKr",
    "loha": "LoHa",
    "ortho": "OrthoLoRA",
    "ortho_init": "OrthoInit",
    "hydra": "Hydra",
    "ortho_hydra": "OrthoHydra",
    "chimera_hydra": "ChimeraHydra",
    "step_expert": "StepExpert",
    "stacked_experts_global_fei": "StackedExperts",
    "vera": "VeRA",
    "dylora": "DyLoRA",
}

# Bulky aggregate JSON blobs that would bloat the metadata dialog payload.
_SKIP_METADATA_KEYS = frozenset(
    {
        "ss_tag_frequency",
        "ss_datasets",
        "ss_bucket_info",
        "ss_dataset_dirs",
        "ss_reg_dataset_dirs",
    }
)


# Every model line writes its final adapter somewhere else. Keep this list in
# sync with ``library/<line>/requests.py``'s output defaults; the Models page is
# the shared weight manager, not an Anima-only view.
LINE_OUTPUT_DIRS: tuple[tuple[str, str], ...] = (
    ("anima", "output/ckpt"),
    ("qwen21", "output/qwen21"),
    ("moss", "output/ckpt_moss"),
)

# Cache trees live inside a line's output root (Qwen's ``output/qwen21/cache``
# is the default) and hold latents/TE embeddings, not adapters.
_NON_WEIGHT_DIRS = frozenset({"cache", "crops", "test"})
_NON_WEIGHT_SUFFIXES = (".te.safetensors", ".latent.safetensors")


def default_output_dir() -> Path:
    return ROOT / "output" / "ckpt"


def model_roots() -> tuple[tuple[str, Path], ...]:
    """``(family, directory)`` for every model line's final-weight root."""
    return tuple((family, ROOT / rel) for family, rel in LINE_OUTPUT_DIRS)


def _is_cached_pair(path: Path) -> bool:
    """Whether *path* is a Qwen cache file rather than a trained adapter."""
    if path.name.endswith(_NON_WEIGHT_SUFFIXES):
        return True
    return any(parent.name in _NON_WEIGHT_DIRS for parent in path.parents)


def read_weight_header(path: Path) -> dict:
    """Return safetensors header metadata, or {} when absent/unreadable."""
    if path.suffix.lower() != ".safetensors":
        return {}
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="numpy") as f:
            return dict(f.metadata() or {})
    except Exception:
        return {}


def classify_spec(metadata: dict) -> str:
    """Map header metadata to a NETWORK_REGISTRY-style spec key."""
    spec = str(metadata.get("ss_network_spec", "")).strip().lower()
    if spec:
        return spec
    # Older saves carry only the network_args flags (or lycoris-style algo).
    args_raw = metadata.get("ss_network_args")
    if args_raw:
        try:
            args = json.loads(args_raw)
        except ValueError:
            args = None
        if isinstance(args, dict):
            if args.get("use_glokr"):
                return "glokr"
            if args.get("use_lokr"):
                return "lokr"
            if args.get("use_loha"):
                return "loha"
            if args.get("use_dylora"):
                return "dylora"
            algo = str(args.get("algo", "")).strip().lower()
            if algo:
                return algo
    if str(metadata.get("ss_network_module", "")).strip():
        return "lora"
    return "unknown"


def type_label(spec: str) -> str:
    if spec == "unknown":
        return "Unknown"
    return _SPEC_LABELS.get(spec, spec.upper())


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def list_models() -> list[dict]:
    """List every final adapter weight across all model lines.

    Anima, Qwen-Image 2.1 and MOSS write under different roots; each entry
    carries its ``family`` so the page can label and filter them.
    """
    from library.io.output_layout import discover_weights

    models: list[dict] = []
    seen: set[str] = set()
    for family, base in model_roots():
        if not base.exists():
            continue
        for p in discover_weights(base):
            key = str(p.resolve())
            if key in seen or _is_cached_pair(p):
                continue
            seen.add(key)
            stat = p.stat()
            metadata = read_weight_header(p)
            spec = classify_spec(metadata)
            try:
                directory = str(p.parent.relative_to(ROOT))
            except ValueError:
                directory = str(p.parent)
            models.append(
                {
                    "family": family,
                    "name": p.name,
                    "path": str(p),
                    "directory": directory,
                    "size": stat.st_size,
                    "size_human": _human_size(stat.st_size),
                    "mtime": _iso(stat.st_mtime),
                    "created": _iso(stat.st_ctime),
                    "spec": spec,
                    "type_label": type_label(spec),
                    "dim": metadata.get("ss_network_dim", "")
                    or metadata.get("rank", ""),
                    "alpha": metadata.get("ss_network_alpha", "")
                    or metadata.get("alpha", ""),
                    "base_model": metadata.get("ss_base_model_version")
                    or metadata.get("ss_sd_model_name", "")
                    or metadata.get("base_model", ""),
                    "output_name": metadata.get("ss_output_name", ""),
                }
            )
    models.sort(key=lambda m: (m["mtime"], m["path"]), reverse=True)
    return models


def read_model_metadata(path: str) -> dict | None:
    """Return the ss_* header metadata of one weight for the details dialog."""
    p = resolve_path(path, expect_file=True)
    if p is None:
        return None
    metadata = read_weight_header(p)
    network_args: dict | None = None
    args_raw = metadata.get("ss_network_args")
    if args_raw:
        try:
            parsed = json.loads(args_raw)
            network_args = parsed if isinstance(parsed, dict) else None
        except ValueError:
            network_args = None
    spec = classify_spec(metadata)
    return {
        "path": str(p),
        "name": p.name,
        "size": p.stat().st_size,
        "size_human": _human_size(p.stat().st_size),
        "spec": spec,
        "type_label": type_label(spec),
        "network_args": network_args,
        "metadata": {k: v for k, v in metadata.items() if k not in _SKIP_METADATA_KEYS},
    }


def delete_model(path: str) -> tuple[bool, str]:
    """Delete one final adapter weight.

    Refuses anything that is not a final weight — trajectory checkpoints and
    ``-state`` resume directories belong to the trainer, not this page.
    """
    from library.io.output_layout import (
        OUTPUT_WEIGHT_EXTENSIONS,
        _is_managed_nonfinal_path,
        is_checkpoint_weight,
        remove_path_with_retry,
    )

    p = resolve_path(path, expect_file=True)
    if p is None:
        return False, f"File not found: {path}"
    if (
        p.suffix.lower() not in OUTPUT_WEIGHT_EXTENSIONS
        or not is_checkpoint_weight(p)
        or _is_managed_nonfinal_path(p.resolve(), Path(p.resolve().anchor))
    ):
        return False, (
            "Refusing to delete: not a final adapter weight. "
            "Checkpoint trajectories and resume states are managed by the trainer."
        )
    try:
        remove_path_with_retry(p)
    except OSError as e:
        return False, f"Delete failed: {e}"
    return True, ""


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} TB"
