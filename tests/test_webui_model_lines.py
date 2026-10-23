"""WebUI model manager + System page against every model line.

The Models tab and the System download list used to know only Anima. These
tests pin the three-root listing (Anima / Qwen-Image 2.1 / MOSS), the cache-file
filter, and the catalog-driven download groups.
"""

from __future__ import annotations

from pathlib import Path

from webui.api import system
from webui.services import model_service


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a real safetensors header")
    return path


def test_list_models_covers_every_line_and_skips_cached_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr(model_service, "ROOT", tmp_path)
    anima = _touch(tmp_path / "output" / "ckpt" / "anima_lora.safetensors")
    qwen = _touch(tmp_path / "output" / "qwen21" / "qwen21_lora.safetensors")
    moss = _touch(tmp_path / "output" / "ckpt_moss" / "moss_lora.safetensors")
    cached = _touch(
        tmp_path / "output" / "qwen21" / "cache" / "img_1024.latent.safetensors"
    )

    models = model_service.list_models()
    by_path = {m["path"]: m for m in models}

    assert str(anima) in by_path
    assert str(qwen) in by_path
    assert str(moss) in by_path
    assert str(cached) not in by_path
    assert by_path[str(qwen)]["family"] == "qwen21"
    assert by_path[str(moss)]["family"] == "moss"
    assert by_path[str(anima)]["family"] == "anima"
    assert [m["family"] for m in models].count("anima") == 1


def test_system_model_groups_follow_the_download_catalog():
    groups = system.get_model_groups()["groups"]
    ids = [group["id"] for group in groups]

    assert len(ids) == len(set(ids))
    assert ids[0] == "anima"
    # Packs the old hardcoded list could never report.
    assert "grouping" in ids  # PE-Spatial
    assert "moss-base" in ids
    assert "cjk" in ids
    for group in groups:
        assert isinstance(group["installed"], bool)
        assert isinstance(group["files"], list)
