"""MOSS i2i: the task recipes load against their requests, ``generate`` expands a
reference into sized jobs, and the GUI's i2i tab chains its jobs through the panel."""

from __future__ import annotations


import pytest
from PIL import Image

from library.moss.requests import (
    I2I_CACHE,
    GenerateRequest,
    TrainRequest,
    i2i_config,
)


@pytest.mark.parametrize("task", sorted(I2I_CACHE))
def test_task_recipes_load(task):
    cache = I2I_CACHE[task].load_config(i2i_config(task, "cache"))
    train = TrainRequest.load_config(i2i_config(task, "train"))
    gen = GenerateRequest.load_config(i2i_config(task, "generate"))
    out = cache.get("out", I2I_CACHE[task]().out)
    assert train["cache"] == out
    # The test reference matches what the cache encoded.
    ref_scale = cache.get("ref_scale", I2I_CACHE[task]().ref_scale)
    assert gen.get("ref_scale", GenerateRequest().ref_scale) == ref_scale


def test_ref_jobs_follow_each_reference(tmp_path):
    from library.moss.generate import _fingerprint, _jobs

    Image.new("RGB", (1200, 1700)).save(tmp_path / "page one.png")
    Image.new("RGB", (800, 800)).save(tmp_path / "b.webp")
    (tmp_path / "notes.txt").write_text("x")
    req = GenerateRequest(prompt="", ref_image=str(tmp_path), ref_vision=True)
    jobs = _jobs(req)
    assert [(j["key"], j["width"], j["height"]) for j in jobs] == [
        ("000_b", 800, 800),  # never upscaled
        ("000_page_one", 864, 1216),
    ]
    assert all(j["prompt"] == "" and j["ref_vision"] for j in jobs)
    # A reference edit invalidates the saved latent.
    assert _fingerprint(req, jobs[0])["ref_mtime"] == jobs[0]["ref_mtime"]

    with pytest.raises(SystemExit, match="no prompts"):
        _jobs(GenerateRequest(prompt="", ref_image=str(tmp_path)))


def test_plain_jobs_unchanged():
    """No reference: the job dicts (and so saved-latent fingerprints) carry no
    ref keys."""
    from library.moss.generate import _jobs

    (job,) = _jobs(GenerateRequest(prompt="a", prompt_format="text"))
    assert not {k for k in job if k.startswith("ref")}


def test_a_different_dit_or_quant_invalidates_a_saved_latent():
    """`--dit` / `--quant` change the denoising base, so a resume must not reuse."""
    from dataclasses import replace
    from pathlib import Path

    from library.moss.generate import _fingerprint, _jobs, _latent_path

    plain = GenerateRequest(prompt="a", prompt_format="text")
    (job,) = _jobs(plain)
    folder = _fingerprint(plain, job)
    dit_a = _fingerprint(replace(plain, dit="a.safetensors"), job)
    dit_b = _fingerprint(replace(plain, dit="b.safetensors"), job)
    int8 = _fingerprint(replace(plain, quant="int8"), job)
    assert folder != dit_a and dit_a != dit_b and folder != int8
    out = Path("out")
    assert _latent_path(out, dit_a).name != _latent_path(out, dit_b).name
    # A folder run keeps the names its earlier latents already have.
    assert _latent_path(out, folder).name.endswith(
        f"_{folder['model']}_s{folder['seed']}.safetensors"
    )
