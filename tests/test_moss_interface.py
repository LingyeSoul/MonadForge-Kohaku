"""MOSS control-plane validation; no tensors, models or worker processes."""

from __future__ import annotations

import json
from pathlib import Path
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


def test_moss_resolve_shows_the_single_file_vae(tmp_path):
    """The cache/generate `vae` override is not a TrainRequest field, but the
    resolved-paths panel must still show the file the jobs will load."""
    dit = tmp_path / "dit.safetensors"
    vae = tmp_path / "flux2-vae.safetensors"
    dit.write_bytes(b"")
    vae.write_bytes(b"")
    resolved = moss_service.resolve_values({"vae": str(vae), "dit": str(dit)})
    assert resolved.vae == str(vae)
    assert resolved.dit == str(dit)
    default = moss_service.resolve_values({})
    assert Path(default.vae).parts[-2:] == ("MOSS-Image-Base", "vae")
    # A quantized TE is one file, not the catalog's folder row.
    from library import downloads

    assert default.text_encoder == str(
        downloads.by_id()["moss_te_int8"].dest / downloads.MOSS_TE_INT8_FILE
    )
    assert moss_service.resolve_values({"text_encoder": "bf16"}).text_encoder == str(
        Path(default.model_dir) / "MOSS-Image-Base" / "text_encoder"
    )


def test_moss_tokenizer_requires_the_chat_template(tmp_path):
    """ComfyUI's tokenizer folder lacks chat_template.jinja, which prompt_ids needs."""
    import pytest

    folder = tmp_path / "tokenizer"
    folder.mkdir()
    config = folder / "tokenizer_config.json"
    config.write_text('{"eos_token": "<|im_end|>"}', encoding="utf-8")
    with pytest.raises(ValueError, match="chat template"):
        moss_service.require_tokenizer(folder)

    (folder / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    moss_service.require_tokenizer(folder)

    (folder / "chat_template.jinja").unlink()
    config.write_text('{"chat_template": "{{ messages }}"}', encoding="utf-8")
    moss_service.require_tokenizer(folder)

    # The key existing is not the template: `null` fails in transformers.
    config.write_text('{"chat_template": null}', encoding="utf-8")
    with pytest.raises(ValueError, match="chat template"):
        moss_service.require_tokenizer(folder)


def _quantized_te(tmp_path, prefix="language_model."):
    import numpy as np
    from safetensors.numpy import save_file

    folder = tmp_path / "int8"
    folder.mkdir()
    (folder / "config.json").write_text("{}", encoding="utf-8")
    path = folder / "qwen35_9b_int8_convrot.safetensors"
    save_file(
        {
            f"{prefix}embed_tokens.weight": np.zeros((4, 4), np.int8),
            f"{prefix}layers.0.mlp.up_proj.weight": np.zeros((4, 4), np.int8),
            f"{prefix}layers.0.mlp.up_proj.weight_scale": np.zeros(1, np.float32),
            f"{prefix}layers.0.mlp.up_proj.comfy_quant": np.frombuffer(
                b'{"format": "int8_tensorwise", "convrot": true}', dtype=np.uint8
            ),
        },
        str(path),
    )
    return path


def test_moss_cache_accepts_the_quantized_default_text_encoder(tmp_path, monkeypatch):
    """int8/fp8 is the shipped TE; the bf16-only component check must not refuse it."""
    te = _quantized_te(tmp_path)
    monkeypatch.setattr(moss_service, "quantized_te_file", lambda precision: te)
    base = tmp_path / "MOSS-Image-Base"
    (base / "tokenizer").mkdir(parents=True)
    (base / "tokenizer" / "tokenizer_config.json").write_text(
        '{"chat_template": "{{ messages }}"}', encoding="utf-8"
    )
    source = tmp_path / "images"
    source.mkdir()

    argv = moss_service.prepare_job_args(
        "moss-cache",
        ["--src", str(source), "--model_dir", str(tmp_path), "--skip_latents"],
    )
    assert "--src" in argv


def test_moss_cache_rejects_a_text_encoder_without_the_moss_namespace(
    tmp_path, monkeypatch
):
    """ComfyUI repacks use `model.language_model.*`; the worker's loader would not map them."""
    import pytest

    te = _quantized_te(tmp_path, prefix="model.language_model.")
    monkeypatch.setattr(moss_service, "quantized_te_file", lambda precision: te)
    (tmp_path / "MOSS-Image-Base" / "tokenizer").mkdir(parents=True)
    source = tmp_path / "images"
    source.mkdir()

    with pytest.raises(ValueError, match="language_model"):
        moss_service.prepare_job_args(
            "moss-cache",
            ["--src", str(source), "--model_dir", str(tmp_path), "--skip_latents"],
        )


def test_moss_cache_points_at_the_download_when_the_text_encoder_is_absent(
    tmp_path, monkeypatch
):
    import pytest

    missing = tmp_path / "int8" / "qwen35_9b_int8_convrot.safetensors"
    monkeypatch.setattr(moss_service, "quantized_te_file", lambda precision: missing)
    (tmp_path / "MOSS-Image-Base" / "tokenizer").mkdir(parents=True)
    source = tmp_path / "images"
    source.mkdir()

    with pytest.raises(FileNotFoundError, match="moss-te-int8"):
        moss_service.prepare_job_args(
            "moss-cache",
            ["--src", str(source), "--model_dir", str(tmp_path), "--skip_latents"],
        )


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
