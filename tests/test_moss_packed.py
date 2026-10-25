"""Packed MOSS training: varlen attention must be the per-sample forward.

The packed path is the throughput lever (one block pass for several samples),
so its contract is exact equivalence on the text-to-image path plus the batch
planner's caps. The fused-kernel variant is checked when a CUDA device can bind
the vendored stack.
"""

from types import SimpleNamespace

import pytest
import torch

from library.moss.model import MOSSImageTransformer
from library.moss.ops import fuse
from library.moss.train import (
    make_batches,
    max_packed_joint,
    training_loss,
    training_loss_packed,
)


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    return MOSSImageTransformer(
        dim=96, ffn_dim=128, num_heads=4, num_layers=4, text_dim=32, latent_channels=8
    ).eval()


def _samples(seed=1):
    g = torch.Generator().manual_seed(seed)
    return [
        (torch.randn(1, 8, 4, 6, generator=g), torch.randn(1, 7, 32, generator=g)),
        (torch.randn(1, 8, 5, 3, generator=g), torch.randn(1, 5, 32, generator=g)),
        (torch.randn(1, 8, 3, 7, generator=g), torch.randn(1, 9, 32, generator=g)),
    ]


@torch.no_grad()
def test_packed_matches_per_sample(tiny):
    samples = _samples()
    latents = [latent for latent, _ in samples]
    embeds = [embed for _, embed in samples]
    timesteps = torch.tensor([500.0, 250.0, 750.0])
    expected = [
        tiny(latent, timestep, embed).sample
        for latent, embed, timestep in zip(latents, embeds, timesteps)
    ]
    packed = tiny.forward_packed(latents, timesteps, embeds)
    assert [tuple(out.shape) for out in packed] == [
        tuple(out.shape) for out in expected
    ]
    for actual, wanted in zip(packed, expected):
        assert torch.allclose(actual, wanted, atol=1e-6)


@torch.no_grad()
def test_packed_does_not_cross_samples(tiny):
    samples = _samples()
    first = tiny.forward_packed(
        [samples[0][0], samples[1][0]],
        torch.tensor([500.0, 250.0]),
        [samples[0][1], samples[1][1]],
    )[0]
    changed = samples[2][0]
    second = tiny.forward_packed(
        [samples[0][0], changed],
        torch.tensor([500.0, 250.0]),
        [samples[0][1], samples[2][1]],
    )[0]
    assert torch.equal(first, second)


def test_packed_loss_matches_per_sample_and_gradient(tiny):
    samples = _samples()
    latents = [latent for latent, _ in samples]
    embeds = [embed for _, embed in samples]
    sigmas = [0.2, 0.55, 0.8]

    per_model = MOSSImageTransformer(
        dim=96, ffn_dim=128, num_heads=4, num_layers=4, text_dim=32, latent_channels=8
    ).train()
    packed_model = MOSSImageTransformer(
        dim=96, ffn_dim=128, num_heads=4, num_layers=4, text_dim=32, latent_channels=8
    ).train()
    packed_model.load_state_dict(per_model.state_dict())

    per_gen = torch.Generator().manual_seed(7)
    per_losses = [
        training_loss(per_model, latent, embed, sigma, gen=per_gen)
        for latent, embed, sigma in zip(latents, embeds, sigmas)
    ]
    sum(per_losses).backward()

    packed_gen = torch.Generator().manual_seed(7)
    packed_loss, packed_per = training_loss_packed(
        packed_model, latents, embeds, sigmas, gen=packed_gen
    )
    (packed_loss * len(latents)).backward()

    assert torch.allclose(
        packed_loss * len(latents), torch.stack(per_losses).sum(), atol=1e-6
    )
    for actual, wanted in zip(packed_per, per_losses):
        assert torch.allclose(actual, wanted, atol=1e-6)
    assert torch.allclose(
        packed_model.video_embedder.weight.grad,
        per_model.video_embedder.weight.grad,
        atol=1e-6,
    )


def _fake(n_img, n_text, n_ref=0):
    return SimpleNamespace(n_img=n_img, n_text=n_text, n_ref=n_ref)


def test_make_batches_caps_samples_and_tokens():
    samples = [_fake(100, 10), _fake(200, 20), _fake(50, 5), _fake(300, 30)]
    assert [len(batch) for batch in make_batches(samples, 2, 0)] == [2, 2]
    batches = make_batches(samples, 4, 300)
    assert [len(batch) for batch in batches] == [1, 2, 1]
    # A single sample larger than the cap still gets its own batch.
    assert [len(batch) for batch in make_batches([_fake(1000, 10)], 4, 100)] == [1]

    assert max_packed_joint(samples, 2, 0) == 550
    assert max_packed_joint(samples, 4, 300) == 330


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA for the fused stack"
)
@torch.no_grad()
def test_fused_packed_matches_reference():
    pytest.importorskip("triton")
    torch.manual_seed(0)
    model = (
        MOSSImageTransformer(
            dim=96,
            ffn_dim=128,
            num_heads=4,
            num_layers=2,
            text_dim=32,
            latent_channels=8,
        )
        .eval()
        .to("cuda", torch.bfloat16)
    )
    if not fuse(model):
        pytest.skip("no fused backend for this device/dtype")
    samples = [
        (
            torch.randn(1, 8, 4, 6, generator=torch.Generator().manual_seed(1))
            .cuda()
            .bfloat16(),
            torch.randn(1, 7, 32, generator=torch.Generator().manual_seed(2))
            .cuda()
            .bfloat16(),
        ),
        (
            torch.randn(1, 8, 5, 3, generator=torch.Generator().manual_seed(3))
            .cuda()
            .bfloat16(),
            torch.randn(1, 5, 32, generator=torch.Generator().manual_seed(4))
            .cuda()
            .bfloat16(),
        ),
    ]
    timesteps = torch.tensor([500.0, 250.0], device="cuda")
    expected = [
        model(latent, timestep, embed).sample
        for (latent, embed), timestep in zip(samples, timesteps)
    ]
    packed = model.forward_packed(
        [latent for latent, _ in samples],
        timesteps,
        [embed for _, embed in samples],
    )
    for actual, wanted in zip(packed, expected):
        assert torch.allclose(actual, wanted, atol=2e-3)
