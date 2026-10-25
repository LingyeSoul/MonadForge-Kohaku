"""System API — model group status, model paths, environment info, and hw stats."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from library.config.model_paths import (
    MODEL_CONFIG_DEFAULTS,
    MODEL_CONFIG_KEYS,
    load_model_config,
)
from webui.services.config_service import ROOT
from webui.services.config_storage import CONFIG_WRITE_LOCK, atomic_write_text

router = APIRouter()


@router.get("/runtimes")
def get_runtimes():
    """Registered execution profiles; metadata only, never a torch import."""
    from library.runtime.profiles import list_runtime_profiles

    return {"runtimes": list_runtime_profiles(ROOT / "configs")}


# ── Update status ──────────────────────────────────────────────


@router.get("/update-status")
def get_update_status(remote: bool = False):
    """Installed build plus, with ``remote=true``, the newest GitHub target.

    ``remote`` stays opt-in: the System page renders the installed label on
    mount, and only the check button spends a GitHub API request.
    """
    from scripts.update import (
        REPO,
        UpdateSourceError,
        describe_available,
        describe_installed,
    )

    installed = describe_installed()
    payload = {
        "repo": REPO,
        "installed": installed,
        "available": None,
        "up_to_date": None,
    }
    if not remote:
        return payload
    try:
        available = describe_available()
    except UpdateSourceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    payload["available"] = available
    payload["up_to_date"] = (
        installed["sha"] is not None and installed["sha"] == available["sha"]
    )
    return payload


# Model groups — the anima group reads the effective model config; every other
# group is derived from ``library.downloads`` so the status list can never fall
# behind the catalog again.
_ANIMA_PATH_KEYS = [(key, default) for key, default in MODEL_CONFIG_DEFAULTS.items()]


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def _check_anima_group() -> dict:
    model = load_model_config(ROOT / "configs")
    files_info = []
    for toml_key, default_ in _ANIMA_PATH_KEYS:
        raw = model.get(toml_key, default_)
        resolved = _resolve_path(raw)
        files_info.append({"path": raw, "exists": resolved.is_file()})
    installed = all(f["exists"] for f in files_info)
    return {
        "id": "anima",
        "title": "Anima base",
        "installed": installed,
        "files": files_info,
    }


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _asset_files(asset) -> list[dict]:
    """One ``{path, exists}`` row per required/optional file of a catalog asset."""
    missing = set(asset.missing())
    rows: list[dict] = []
    for name in asset.files:
        if asset.dest is None:
            rows.append({"path": f"{asset.repo}/{name}", "exists": not missing})
            continue
        path = asset.dest / Path(name).name
        rows.append({"path": _display_path(path), "exists": path.exists()})
    for name in asset.optional:
        if asset.dest is None:
            continue
        path = asset.dest / Path(name).name
        if path.exists():
            rows.append({"path": _display_path(path), "exists": True})
    return rows


def _catalog_groups() -> list[dict]:
    """Download packs from the canonical catalog, minus the dynamic anima group."""
    from library import downloads

    groups: list[dict] = []
    for pack_id, assets in downloads.by_pack().items():
        if pack_id == "anima":
            continue
        pack = downloads.PACK_BY_ID[pack_id]
        groups.append(
            {
                "id": pack_id,
                "title": pack.title,
                "installed": all(not asset.missing() for asset in assets),
                "files": [row for asset in assets for row in _asset_files(asset)],
            }
        )
    return groups


@router.get("/models")
def get_model_groups():
    """Check installation status of every downloadable pack."""
    return {"groups": [_check_anima_group(), *_catalog_groups()]}


# ── Model paths (configurable from System page) ────────────────

# Model path keys exposed by the System page.
_MODEL_PATH_KEYS: list[tuple[str, str, str]] = [
    (
        "anima_dit",
        "pretrained_model_name_or_path",
        MODEL_CONFIG_DEFAULTS["pretrained_model_name_or_path"],
    ),
    ("anima_te", "qwen3", MODEL_CONFIG_DEFAULTS["qwen3"]),
    ("anima_vae", "vae", MODEL_CONFIG_DEFAULTS["vae"]),
]


class ModelPathUpdate(BaseModel):
    key: str
    value: str


def _resolve_and_check(path_str: str) -> tuple[str, bool]:
    """Resolve a path (relative to ROOT or absolute) and check existence."""
    p = Path(path_str)
    if not p.is_absolute():
        p = ROOT / p
    return str(p), p.is_file()


@router.get("/model-paths")
def get_model_paths():
    """Return the configured model paths with existence status."""
    model = load_model_config(ROOT / "configs")
    paths = []
    for id_, toml_key, default_ in _MODEL_PATH_KEYS:
        raw = model.get(toml_key, default_)
        resolved, exists = _resolve_and_check(raw)
        paths.append(
            {
                "id": id_,
                "toml_key": toml_key,
                "path": raw,
                "resolved": resolved,
                "exists": exists,
            }
        )
    return {"paths": paths}


@router.put("/model-paths")
def update_model_paths(body: list[ModelPathUpdate]):
    """Update machine-local paths in gitignored custom/model.toml."""
    import toml as _toml

    custom_path = ROOT / "configs" / "custom" / "model.toml"
    with CONFIG_WRITE_LOCK:
        custom = (
            _toml.loads(custom_path.read_text(encoding="utf-8"))
            if custom_path.exists()
            else {}
        )
        for item in body:
            if item.key in MODEL_CONFIG_KEYS:
                custom[item.key] = item.value
        atomic_write_text(custom_path, _toml.dumps(custom))
    return {"ok": True}


# ── Hardware stats ─────────────────────────────────────────────


@router.get("/hw-stats")
def get_hw_stats():
    """Return live GPU / CPU / memory statistics."""
    import psutil

    # CPU & system memory
    cpu_percent = psutil.cpu_percent(interval=0)
    vm = psutil.virtual_memory()

    stats: dict = {
        "cpu_percent": cpu_percent,
        "mem_used_gb": round(vm.used / (1024**3), 1),
        "mem_total_gb": round(vm.total / (1024**3), 1),
        "mem_percent": vm.percent,
    }

    # GPU detection: nvidia-smi primary (system-wide GPU memory/util/temp),
    # torch.cuda supplement (device name, per-process reserved memory).
    #
    # nvidia-smi reports the ACTUAL system-wide memory usage — torch.cuda only
    # sees the current process's allocation (0 for the WebUI server process
    # since training runs in a separate subprocess).  So nvidia-smi values
    # always win for memory / util / temp.
    try:
        import subprocess

        from library.runtime.proc import no_window_kwargs

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            **no_window_kwargs(),
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 5:
                stats["gpu_name"] = parts[0]
                stats["gpu_mem_used_gb"] = round(int(parts[1]) / 1024, 1)
                stats["gpu_mem_total_gb"] = round(int(parts[2]) / 1024, 1)
                stats["gpu_util_percent"] = int(parts[3])
                stats["gpu_temp_c"] = int(parts[4])
    except Exception:
        pass

    # torch.cuda: supplement with device name + reserved memory when available
    try:
        import torch

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            stats.setdefault("gpu_name", torch.cuda.get_device_name(idx))
            stats.setdefault(
                "gpu_mem_total_gb",
                round(
                    torch.cuda.get_device_properties(idx).total_memory / (1024**3), 1
                ),
            )
            # Per-process allocated + reserved (useful for debugging OOM)
            stats["gpu_mem_reserved_gb"] = round(
                torch.cuda.memory_reserved(idx) / (1024**3), 1
            )
    except Exception:
        pass

    return stats
