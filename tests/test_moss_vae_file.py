"""MOSS ComfyUI single-file VAE: config inference, key layouts, the loader.

A ComfyUI VAE is one .safetensors holding the FLUX.2 VAE, in either the BFL /
ldm layout (`flux2-vae`) or diffusers names. These tests pin the contracts:

1. Config inference: widths, layers per block, latent channels, patch size and
   attention presence come from the tensor shapes.
2. Both key layouts load: BFL keys map onto diffusers names (up blocks reversed,
   mid attention's 1x1 convs viewed as linears) and a diffusers-key file loads as
   it is.
3. The state loads strictly: a wrong file fails with its missing / unexpected
   keys, and the result is the requested dtype with a working decode.
4. `--vae` reaches the loader from every entry point and the browser preflight
   accepts the file instead of the `vae/` folder.

CPU-only and tiny: two down / up blocks, 32 / 64 channels, 4 latent channels.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
from diffusers import AutoencoderKLFlux2
from safetensors.torch import save_file

from library.moss import checkpoint
from library.moss.checkpoint import CheckpointFormatError
from library.moss.loader import load_vae
from library.moss.requests import CacheRequest, ColorizeCacheRequest, GenerateRequest
from library.moss.validation import validate_request
from webui.services import moss_service

CONFIG = {
    "in_channels": 3,
    "out_channels": 3,
    "down_block_types": ("DownEncoderBlock2D",) * 2,
    "up_block_types": ("UpDecoderBlock2D",) * 2,
    "block_out_channels": (32, 64),
    "layers_per_block": 1,
    "latent_channels": 4,
    "sample_size": 32,
    "mid_block_add_attention": False,
    "patch_size": (2, 2),
}


def tiny_vae() -> AutoencoderKLFlux2:
    torch.manual_seed(0)
    return AutoencoderKLFlux2(**CONFIG).eval()


def write_vae(path: Path, model: AutoencoderKLFlux2) -> Path:
    save_file(
        {key: value.contiguous() for key, value in model.state_dict().items()},
        str(path),
    )
    return path


# ------------------------------------------------------------- config inference


def test_config_is_inferred_from_the_shapes(tmp_path):
    path = write_vae(tmp_path / "vae.safetensors", tiny_vae())
    shapes = checkpoint.header(path)
    config = checkpoint.infer_vae_config(
        {key: value[1] for key, value in shapes.items()}
    )
    assert config == {
        "act_fn": "silu",
        "norm_num_groups": 32,
        "sample_size": 1024,
        "batch_norm_eps": 1e-4,
        "batch_norm_momentum": 0.1,
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": ("DownEncoderBlock2D",) * 2,
        "up_block_types": ("UpDecoderBlock2D",) * 2,
        "block_out_channels": (32, 64),
        "decoder_block_out_channels": (32, 64),
        "layers_per_block": 1,
        "latent_channels": 4,
        "use_quant_conv": True,
        "use_post_quant_conv": True,
        "mid_block_add_attention": False,
        "patch_size": (2, 2),
    }


def test_asymmetric_blocks_are_refused():
    shapes = {
        "encoder.conv_in.weight": (32, 3, 3, 3),
        "encoder.down_blocks.0.resnets.0.conv1.weight": (32, 32, 3, 3),
        "decoder.conv_in.weight": (512, 4, 3, 3),
        "decoder.conv_out.weight": (3, 32, 3, 3),
        "decoder.up_blocks.0.resnets.0.conv2.weight": (32, 32, 3, 3),
        "decoder.up_blocks.1.resnets.0.conv2.weight": (32, 32, 3, 3),
        "bn.running_mean": (16,),
    }
    with pytest.raises(CheckpointFormatError, match="as many up as down"):
        checkpoint.infer_vae_config(shapes)


# --------------------------------------------------------------- key layouts


@pytest.mark.parametrize(
    ("bfl", "diffusers"),
    [
        (
            "encoder.down.0.block.0.conv1.weight",
            "encoder.down_blocks.0.resnets.0.conv1.weight",
        ),
        (
            "encoder.down.0.block.0.nin_shortcut.weight",
            "encoder.down_blocks.0.resnets.0.conv_shortcut.weight",
        ),
        (
            "encoder.down.1.downsample.conv.weight",
            "encoder.down_blocks.1.downsamplers.0.conv.weight",
        ),
        (
            "encoder.mid.block_1.conv1.weight",
            "encoder.mid_block.resnets.0.conv1.weight",
        ),
        ("decoder.mid.attn_1.q.weight", "decoder.mid_block.attentions.0.to_q.weight"),
        (
            "decoder.mid.attn_1.norm.weight",
            "decoder.mid_block.attentions.0.group_norm.weight",
        ),
        ("decoder.norm_out.weight", "decoder.conv_norm_out.weight"),
        ("encoder.quant_conv.weight", "quant_conv.weight"),
        ("decoder.post_quant_conv.weight", "post_quant_conv.weight"),
        ("bn.running_mean", "bn.running_mean"),
    ],
)
def test_bfl_keys_map_to_diffusers(bfl, diffusers):
    assert checkpoint.vae_key(bfl, n_up=2) == diffusers


def test_bfl_up_blocks_are_reversed():
    assert checkpoint.vae_key("decoder.up.1.block.0.conv1.weight", n_up=2) == (
        "decoder.up_blocks.0.resnets.0.conv1.weight"
    )
    assert checkpoint.vae_key("decoder.up.0.block.0.conv1.weight", n_up=2) == (
        "decoder.up_blocks.1.resnets.0.conv1.weight"
    )


def test_bfl_mid_attention_convs_become_linears():
    weight = torch.arange(12, dtype=torch.float32).reshape(4, 3, 1, 1)
    state = checkpoint.vae_to_diffusers({"decoder.mid.attn_1.q.weight": weight})
    assert state["decoder.mid_block.attentions.0.to_q.weight"].shape == (4, 3)


def test_unknown_bfl_key_is_named():
    with pytest.raises(CheckpointFormatError, match="not a FLUX.2 VAE key"):
        checkpoint.vae_key("decoder.mystery.weight", n_up=2)


# ---------------------------------------------------------------- loading


def test_single_file_vae_round_trips(tmp_path):
    model = tiny_vae()
    state = model.state_dict()
    path = write_vae(tmp_path / "vae.safetensors", model)
    loaded = checkpoint.load_vae_file(path, torch.float32)
    for key, value in state.items():
        assert torch.equal(loaded.state_dict()[key], value), key
    latent = torch.randn(1, 4, 4, 4, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        image = loaded.decode(latent).sample
    assert image.shape == (1, 3, 8, 8)
    assert torch.isfinite(image).all()


def test_single_file_vae_takes_the_requested_dtype(tmp_path):
    path = write_vae(tmp_path / "vae.safetensors", tiny_vae())
    loaded = checkpoint.load_vae_file(path, torch.float16)
    assert next(loaded.parameters()).dtype == torch.float16
    assert loaded.bn.running_mean.dtype == torch.float16


def test_loader_entry_point_accepts_a_vae_file(tmp_path):
    model = tiny_vae()
    path = write_vae(tmp_path / "vae.safetensors", model)
    loaded = load_vae("", dtype=torch.float32, vae=str(path))
    assert torch.equal(loaded.decoder.conv_in.weight, model.decoder.conv_in.weight)


def test_wrong_file_reports_missing_and_unexpected_keys(tmp_path):
    model = tiny_vae()
    state = {key: value.contiguous() for key, value in model.state_dict().items()}
    state.pop("encoder.conv_in.weight")
    state["decoder.bogus.weight"] = torch.zeros(2, 2)
    path = tmp_path / "bad.safetensors"
    save_file(state, str(path))
    with pytest.raises(CheckpointFormatError, match="missing"):
        checkpoint.load_vae_file(path, torch.float32)


# ------------------------------------------------------------ request wiring


def test_requests_round_trip_the_vae_flag(tmp_path):
    path = tmp_path / "vae.safetensors"
    for request in (
        CacheRequest(vae=str(path)),
        ColorizeCacheRequest(vae=str(path)),
        GenerateRequest(vae=str(path), prompt="x"),
    ):
        argv = request.to_argv()
        assert "--vae" in argv
        assert type(request).from_argv(argv).vae == str(path)


def test_validation_rejects_a_missing_vae(tmp_path):
    missing = str(tmp_path / "nope.safetensors")
    with pytest.raises(ValueError, match="vae does not exist"):
        validate_request(CacheRequest(vae=missing))
    with pytest.raises(ValueError, match="vae does not exist"):
        validate_request(GenerateRequest(vae=missing, prompt="x"))


@pytest.mark.parametrize("name", ("cache", "generate", "colorize"))
def test_entry_points_forward_the_vae_to_the_loader(name):
    """Every `load_vae` call must carry `vae=` (or the file would be ignored)."""
    source = (
        Path(__file__).resolve().parents[1] / "library/moss" / f"{name}.py"
    ).read_text(encoding="utf-8")
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_vae"
    ]
    assert calls, f"{name}.py no longer loads a VAE"
    for call in calls:
        if "vae" not in {keyword.arg for keyword in call.keywords}:
            raise AssertionError(
                f"{name}.py:{call.lineno}: load_vae call misses vae=..."
            )


# ------------------------------------------------------- browser preflight


def test_preflight_accepts_a_vae_file(tmp_path):
    path = write_vae(tmp_path / "vae.safetensors", tiny_vae())
    moss_service.validate_paths(
        CacheRequest(
            src=str(tmp_path),
            vae=str(path),
            skip_text=True,
            out="post_image_dataset/moss/vae-preflight",
        )
    )


def test_preflight_accepts_a_vae_file_for_generate(tmp_path):
    path = write_vae(tmp_path / "vae.safetensors", tiny_vae())
    moss_service.validate_paths(
        GenerateRequest(
            vae=str(path),
            decode_only=True,
            out_dir="output/moss/vae-preflight",
        )
    )


def test_preflight_rejects_a_non_vae_file(tmp_path):
    path = tmp_path / "other.safetensors"
    save_file({"bogus.weight": torch.zeros(2, 2)}, str(path))
    with pytest.raises(ValueError, match="not a FLUX.2 VAE"):
        moss_service.require_vae_file(path, "MOSS VAE")


def test_preflight_still_requires_the_folder_without_vae(tmp_path):
    with pytest.raises(FileNotFoundError, match="MOSS VAE"):
        moss_service.validate_paths(
            CacheRequest(
                src=str(tmp_path),
                skip_text=True,
                out="post_image_dataset/moss/vae-preflight",
            )
        )
