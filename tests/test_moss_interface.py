"""MOSS control-plane validation; no tensors, models or worker processes."""

from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from scripts.moss.workflow import parse_workflow
from webui.api.moss import router, task_service
from webui.services import moss_service


def test_moss_http_schema_strict_inputs_and_runtime(monkeypatch):
    calls = []

    async def start(command, args, **kwargs):
        calls.append((command, args, kwargs))
        return SimpleNamespace(id="moss-fixture", command=command)

    monkeypatch.setattr(task_service, "start_task", start)
    app = FastAPI()
    app.include_router(router, prefix="/moss")
    with TestClient(app) as client:
        response = client.get("/moss/schema")
        assert response.status_code == 200
        assert set(response.json()) == {
            "model_field_names",
            "cache",
            "train",
            "generate",
            "tune",
        }
        assert response.json()["model_field_names"] == ["model_dir"]
        invalid = client.post("/moss/jobs/train", json={"values": {"rank": "16"}})
        assert invalid.status_code == 400
        assert not calls
        submitted = client.post(
            "/moss/jobs/train", json={"values": {"rank": 32}, "runtime_profile": "v100"}
        )
        assert submitted.status_code == 200
    assert calls[0][0] == "moss-train"
    assert calls[0][2] == {
        "context": {"model_type": "moss", "phase": "train"},
        "runtime_profile": "v100",
    }


def test_moss_preflight_accepts_a_single_file_dit(tmp_path):
    """A ComfyUI single file replaces the transformer folder, not the other paths."""
    import numpy as np
    from safetensors.numpy import save_file

    path = tmp_path / "dit.safetensors"
    save_file(
        {
            "shared_block_modulation.weight": np.zeros((6, 2), np.float32),
            "layers.0.proj.q.weight": np.zeros((2, 2), np.float32),
        },
        str(path),
    )
    app = FastAPI()
    app.include_router(router, prefix="/moss")
    with TestClient(app) as client:
        preflight = client.post(
            "/moss/preflight",
            json={"values": {"dit": str(path), "cache": str(tmp_path / "cache")}},
        )
        assert preflight.status_code == 200, preflight.text
        resolved = client.post("/moss/resolve", json={"values": {"dit": str(path)}})
        assert resolved.status_code == 200
        assert resolved.json()["dit"] == str(path)


def test_moss_preflight_rejects_a_quantized_single_file(tmp_path):
    import numpy as np
    from safetensors.numpy import save_file

    path = tmp_path / "int8.safetensors"
    save_file(
        {
            "shared_block_modulation.weight": np.zeros((6, 2), np.float32),
            "layers.0.proj.q.weight": np.zeros((2, 2), np.float32),
            "layers.0.proj.o.weight_scale": np.zeros(1, np.float32),
        },
        str(path),
    )
    app = FastAPI()
    app.include_router(router, prefix="/moss")
    with TestClient(app) as client:
        response = client.post(
            "/moss/preflight",
            json={"values": {"dit": str(path), "cache": str(tmp_path / "cache")}},
        )
        assert response.status_code == 400
        assert "Quantized" in response.json()["detail"]


def test_moss_chain_pins_train_to_cache_output():
    cache, train = parse_workflow(
        [
            "--cache-args",
            json.dumps(["--out", "output/moss/new-cache"]),
            "--train-args",
            json.dumps(["--cache", "output/moss/old-cache"]),
        ]
    )
    assert train.cache == cache.out == "output/moss/new-cache"


def test_moss_manifest_rejects_non_object_before_request_parsing(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setattr(moss_service, "output_path", lambda value: tmp_path)
    (tmp_path / "manifest.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        moss_service.read_results(str(tmp_path))
