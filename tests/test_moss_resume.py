"""Resume must replay the uninterrupted packed run, not just warm-start it.

One epoch is checkpointed, then resumed to two epochs; the adapter must equal a
single two-epoch run tensor-for-tensor (optimizer, LR schedule and the three
caption RNGs included).
"""

import pytest
import torch
from safetensors.torch import load_file

import library.moss.train as train_module
from library.moss.model import MOSSImageTransformer
from library.moss.requests import TrainRequest
from library.moss.train import Sample, train


def _samples():
    generator = torch.Generator().manual_seed(11)
    samples = []
    for index in range(4):
        height, width = 4 + index % 2, 6 - index % 2
        n_text = 5 + index
        latent = torch.randn(8, height, width, generator=generator)
        embeds = torch.randn(n_text, 32, generator=generator).to(torch.bfloat16)
        samples.append(
            Sample(
                f"s{index}",
                height * width,
                n_text,
                lambda latent=latent, embeds=embeds: (latent, embeds),
                latent_hw=(height, width),
            )
        )
    return samples


def _request(path, epochs, resume=""):
    return TrainRequest(
        cache="unused",
        output=str(path),
        epochs=epochs,
        batch_size=2,
        rank=2,
        compile=False,
        grad_checkpointing=False,
        blocks_to_swap=0,
        plain_caption_rate=0.0,
        lr=1e-3,
        channel_scaling_alpha=0.0,
        dc_isolate=False,
        down_init="kaiming",
        resume=resume,
        seed=3,
    )


def test_the_resume_signature_covers_the_dit_file():
    """A state trained on one DiT must not resume onto another."""
    from library.moss.train import _RESUME_KEYS, resume_mismatch

    assert "dit" in _RESUME_KEYS
    current = {"model": "base", "dit": ""}
    # A state written before `dit` existed is exactly an empty-dit run.
    assert resume_mismatch({"model": "base"}, current) == {}
    mismatch = resume_mismatch({"model": "base"}, {**current, "dit": "a.safetensors"})
    assert set(mismatch) == {"dit"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the trainer targets CUDA")
def test_resume_replays_the_packed_run(monkeypatch, tmp_path):
    def tiny(*args, **kwargs):
        return MOSSImageTransformer(
            dim=128,
            ffn_dim=128,
            num_heads=4,
            num_layers=2,
            text_dim=32,
            latent_channels=8,
        ).to("cuda", torch.bfloat16)

    monkeypatch.setattr(train_module, "load_transformer", tiny)
    monkeypatch.setattr(train_module, "place", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_module, "fuse", lambda *args, **kwargs: 0)
    monkeypatch.setattr(train_module, "compile_blocks", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_module, "checkpoint_blocks", lambda *args, **kwargs: None)

    full = tmp_path / "full.safetensors"
    first = tmp_path / "first.safetensors"
    resumed = tmp_path / "resumed.safetensors"

    train(_request(full, 2), _samples())
    train(_request(first, 1), _samples())
    state = first.with_name(first.name + ".state.pt")
    assert state.is_file()
    train(_request(resumed, 2, resume=str(state)), _samples())

    expected = load_file(str(full))
    actual = load_file(str(resumed))
    assert set(actual) == set(expected)
    for key in expected:
        assert torch.equal(actual[key], expected[key]), key
