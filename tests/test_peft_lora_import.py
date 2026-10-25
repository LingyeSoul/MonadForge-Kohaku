"""PEFT / diffusion-pipe LoRA imports: key conversion + end-to-end load."""

import logging

import pytest
import torch
import torch.nn as nn

from networks.lora_utils import (
    convert_peft_lora_state_dict,
    has_peft_lora_keys,
)


def _peft_sd(prefix: str, module: str, rank: int, in_dim: int, out_dim: int) -> dict:
    """One PEFT module: ``lora_A`` (r, in) + ``lora_B`` (out, r)."""
    return {
        f"{prefix}{module}.lora_A.weight": torch.randn(rank, in_dim),
        f"{prefix}{module}.lora_B.weight": torch.randn(out_dim, rank),
    }


def test_convert_strips_trainer_wrappers_and_renames_factors():
    sd = {}
    for letter in ("q", "k", "v"):
        sd |= _peft_sd(
            "base_model.model.diffusion_model.",
            f"blocks.0.self_attn.{letter}_proj",
            2,
            8,
            8,
        )
    sd |= _peft_sd("diffusion_model.", "x_embedder.proj.1", 2, 8, 16)
    # A PEFT adapter name must not leak into the module path.
    sd |= _peft_sd("diffusion_model.", "blocks.0.mlp.layer1", 4, 8, 32)
    sd = {
        k.replace("layer1.lora_A.weight", "layer1.lora_A.default.weight"): v
        for k, v in sd.items()
    }

    input_keys = set(sd)
    converted = convert_peft_lora_state_dict(sd, None)

    assert set(sd) == input_keys  # pure: the caller's dict is untouched
    assert not any(k.startswith(("diffusion_model.", "base_model.")) for k in converted)
    assert not any("lora_A" in k or "lora_B" in k for k in converted)
    for letter in ("q", "k", "v"):
        stem = f"lora_unet_blocks_0_self_attn_{letter}_proj"
        assert torch.equal(
            converted[f"{stem}.lora_down.weight"],
            sd[
                f"base_model.model.diffusion_model.blocks.0.self_attn.{letter}_proj.lora_A.weight"
            ],
        )
        assert torch.equal(
            converted[f"{stem}.lora_up.weight"],
            sd[
                f"base_model.model.diffusion_model.blocks.0.self_attn.{letter}_proj.lora_B.weight"
            ],
        )
        assert converted[f"{stem}.alpha"].item() == 2.0  # rank, scale 1.0
    assert "lora_unet_x_embedder_proj_1.lora_down.weight" in converted
    assert "lora_unet_blocks_0_mlp_layer1.lora_up.weight" in converted
    assert converted["lora_unet_blocks_0_mlp_layer1.alpha"].item() == 4.0


def test_convert_honors_lora_alpha_and_drops_its_source_key():
    sd = _peft_sd("diffusion_model.", "blocks.1.mlp.layer2", 4, 8, 16)
    sd["diffusion_model.blocks.1.mlp.layer2.lora_alpha"] = torch.tensor(8.0)

    converted = convert_peft_lora_state_dict(sd, None)

    assert converted["lora_unet_blocks_1_mlp_layer2.alpha"].item() == 8.0
    assert not any("lora_alpha" in k for k in converted)


def test_convert_alpha_override_beats_file_alpha():
    sd = _peft_sd("diffusion_model.", "blocks.1.mlp.layer2", 4, 8, 16)
    sd["diffusion_model.blocks.1.mlp.layer2.lora_alpha"] = torch.tensor(8.0)

    converted = convert_peft_lora_state_dict(sd, 2.0)

    assert converted["lora_unet_blocks_1_mlp_layer2.alpha"].item() == 2.0


def test_convert_warns_when_alpha_is_absent(caplog):
    sd = _peft_sd("diffusion_model.", "blocks.2.self_attn.q_proj", 4, 8, 8)

    with caplog.at_level(logging.WARNING):
        converted = convert_peft_lora_state_dict(sd, None)

    assert converted["lora_unet_blocks_2_self_attn_q_proj.alpha"].item() == 4.0
    assert "no lora_alpha" in caplog.text
    assert "peft_alpha" in caplog.text


def test_convert_is_identity_for_kohya_and_native_adapters():
    kohya = {
        "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight": torch.zeros(2, 8),
        "lora_unet_blocks_0_self_attn_q_proj.lora_up.weight": torch.zeros(8, 2),
        "lora_unet_blocks_0_self_attn_q_proj.alpha": torch.tensor(2.0),
    }
    native = {
        "lora_unet_blocks_0_self_attn_q_proj.lokr_w1": torch.zeros(2, 2),
        "lora_unet_blocks_0_self_attn_q_proj.hada_w1_a": torch.zeros(8, 2),
    }

    assert not has_peft_lora_keys(kohya)
    assert not has_peft_lora_keys(native)
    assert convert_peft_lora_state_dict(kohya, None) is kohya
    assert convert_peft_lora_state_dict(native, None) is native


def test_convert_refuses_dora_magnitude_vectors():
    sd = _peft_sd("diffusion_model.", "blocks.0.self_attn.q_proj", 2, 8, 8)
    sd["diffusion_model.blocks.0.self_attn.q_proj.lora_magnitude_vector"] = torch.ones(
        8
    )

    with pytest.raises(RuntimeError, match="DoRA"):
        convert_peft_lora_state_dict(sd, None)


def test_convert_refuses_orphan_up_factor():
    sd = {
        "diffusion_model.blocks.0.self_attn.q_proj.lora_B.weight": torch.zeros(8, 2),
    }

    with pytest.raises(RuntimeError, match="no lora_A"):
        convert_peft_lora_state_dict(sd, None)


# --- end-to-end: the factory accepts a PEFT export as a normal adapter -------


class Block(nn.Module):
    """Named ``Block`` so ``ANIMA_TARGET_REPLACE_MODULE`` matches it."""

    def __init__(self, in_dim: int, out_per_component: int) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.qkv_proj = nn.Linear(in_dim, 3 * out_per_component, bias=False)


class _FakeDiT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block(8, 8)])


def _peft_qkv_sd(rank: int) -> dict:
    sd = {}
    for letter in ("q", "k", "v"):
        sd |= _peft_sd(
            "diffusion_model.", f"blocks.0.self_attn.{letter}_proj", rank, 8, 8
        )
    return sd


def _load(sd, peft_alpha):
    from networks import lora_anima

    model = _FakeDiT()
    return lora_anima.create_network_from_weights(
        multiplier=1.0,
        file=None,
        ae=None,
        text_encoders=[],
        unet=model,
        weights_sd=sd,
        for_inference=True,
        peft_alpha=peft_alpha,
    )


def test_factory_loads_peft_export_into_fused_runtime_module():
    sd = _peft_qkv_sd(2)

    network, weights_sd = _load(sd, None)
    network.apply_to([], _FakeDiT(), apply_text_encoder=False, apply_unet=True)
    info = network.load_state_dict(weights_sd, strict=False)

    assert info.missing_keys == []
    assert info.unexpected_keys == []
    assert len(network.unet_loras) == 1

    stem = "lora_unet_blocks_0_self_attn_qkv_proj"
    expected_down = torch.cat(
        [
            sd[f"diffusion_model.blocks.0.self_attn.{x}_proj.lora_A.weight"]
            for x in "qkv"
        ],
        dim=0,
    )
    expected_up = torch.zeros(24, 6)
    for i, x in enumerate("qkv"):
        expected_up[i * 8 : (i + 1) * 8, i * 2 : (i + 1) * 2] = sd[
            f"diffusion_model.blocks.0.self_attn.{x}_proj.lora_B.weight"
        ]
    assert torch.equal(weights_sd[f"{stem}.lora_down.weight"], expected_down)
    assert torch.equal(weights_sd[f"{stem}.lora_up.weight"], expected_up)
    # alpha == fused rank, so the runtime scale stays 1.0
    assert weights_sd[f"{stem}.alpha"].item() == 6.0


def test_factory_peft_alpha_override_rescales_the_fused_delta():
    sd = _peft_qkv_sd(2)

    _, weights_sd = _load(sd, "1.0")

    stem = "lora_unet_blocks_0_self_attn_qkv_proj"
    expected_up = torch.zeros(24, 6)
    for i, x in enumerate("qkv"):
        # scale = alpha / rank = 1.0 / 2 = 0.5 folded into the up factors
        expected_up[i * 8 : (i + 1) * 8, i * 2 : (i + 1) * 2] = (
            sd[f"diffusion_model.blocks.0.self_attn.{x}_proj.lora_B.weight"] * 0.5
        )
    assert torch.equal(weights_sd[f"{stem}.lora_up.weight"], expected_up)
    assert weights_sd[f"{stem}.alpha"].item() == 6.0


def test_factory_loads_peft_file_from_path(tmp_path):
    from safetensors.torch import save_file

    from networks import lora_anima

    sd = _peft_qkv_sd(2)
    path = tmp_path / "peft.safetensors"
    save_file(sd, str(path))

    network, weights_sd = lora_anima.create_network_from_weights(
        multiplier=1.0,
        file=str(path),
        ae=None,
        text_encoders=[],
        unet=_FakeDiT(),
        for_inference=True,
    )

    stem = "lora_unet_blocks_0_self_attn_qkv_proj"
    assert f"{stem}.lora_down.weight" in weights_sd
    assert len(network.unet_loras) == 1


def test_merge_toolkit_reads_peft_factors(tmp_path):
    from safetensors.torch import save_file

    from scripts.toolkits.merge_loras import load_lora

    sd = _peft_sd("diffusion_model.", "blocks.0.self_attn.q_proj", 2, 8, 4)
    path = tmp_path / "peft.safetensors"
    save_file(sd, str(path))

    downs, ups, alphas = load_lora(str(path))

    stem = "lora_unet_blocks_0_self_attn_q_proj"
    assert set(downs) == {stem}
    assert set(ups) == {stem}
    assert torch.equal(
        downs[stem],
        sd["diffusion_model.blocks.0.self_attn.q_proj.lora_A.weight"].float(),
    )
    assert alphas[stem] == 2.0


def test_inference_loader_converts_peft_and_keeps_unet_keys(tmp_path):
    from safetensors.torch import save_file

    from library.inference.models import _load_unet_lora_state

    sd = _peft_sd("diffusion_model.", "blocks.0.self_attn.q_proj", 2, 8, 4)
    # TE rows already in the kohya layout are dropped by the unet filter.
    sd["lora_te_model.layers.0.self_attn.q_proj.lora_down.weight"] = torch.zeros(2, 8)
    path = tmp_path / "peft.safetensors"
    save_file(sd, str(path))

    loaded = _load_unet_lora_state(str(path))

    assert set(loaded) == {
        "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight",
        "lora_unet_blocks_0_self_attn_q_proj.lora_up.weight",
        "lora_unet_blocks_0_self_attn_q_proj.alpha",
    }
