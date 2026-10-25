"""`bench/repa/compare_repa_ckpts.delta_w_by_block`: never an empty table.

The reader is Anima-only (blocks are `blocks_N_` in the file stem). A file from
another line must be refused by name — an empty dict renders as a per-block
table with no rows, which reads as "this adapter changed nothing".
"""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from bench.repa.compare_repa_ckpts import delta_w_by_block


def _lora_sd(prefix: str, modules: list[str]) -> dict[str, torch.Tensor]:
    """A plain-LoRA state dict: rank-2 down/up plus an alpha per module."""
    sd: dict[str, torch.Tensor] = {}
    for i, module in enumerate(modules):
        stem = prefix + module
        sd[f"{stem}.lora_down.weight"] = torch.eye(4) * (0.5 + 0.1 * i)
        sd[f"{stem}.lora_up.weight"] = torch.eye(4) * (0.25 + 0.05 * i)
        sd[f"{stem}.alpha"] = torch.tensor(2.0)
    return sd


def test_anima_blocks_are_grouped_per_block(tmp_path):
    sd = _lora_sd(
        "lora_unet_",
        [
            "blocks_0_self_attn_q_proj",
            "blocks_0_mlp_layer1",
            "blocks_3_self_attn_q_proj",
        ],
    )
    path = tmp_path / "anima.safetensors"
    save_file(sd, str(path))

    out = delta_w_by_block(str(path))
    assert sorted(out) == [0, 3]
    assert out[0]["n"] == 2 and out[3]["n"] == 1
    assert out[0]["fro"] > 0 and out[0]["srank"] > 0


def test_moss_layers_naming_is_refused_not_scored_zero(tmp_path):
    """MOSS-Image names blocks `layers.N` -> stems `lora_unet_layers_N_...`."""
    sd = _lora_sd(
        "lora_unet_",
        [
            "layers_0_proj_q",
            "layers_0_proj_ffn_up_gate",
            "layers_1_proj_v",
        ],
    )
    path = tmp_path / "moss.safetensors"
    save_file(sd, str(path))

    with pytest.raises(ValueError) as exc:
        delta_w_by_block(str(path))
    message = str(exc.value)
    assert "blocks_<n>_" in message
    assert "layers.N" in message  # names the other line, so the fix is obvious


def test_non_block_targets_alone_are_refused(tmp_path):
    """Only x_embedder / final_layer: no block at all, still not an empty table."""
    sd = _lora_sd("lora_unet_", ["x_embedder_proj_1", "final_layer_linear"])
    path = tmp_path / "head_only.safetensors"
    save_file(sd, str(path))

    with pytest.raises(ValueError, match="empty ΔW table"):
        delta_w_by_block(str(path))


def test_file_without_plain_lora_factors_is_refused(tmp_path):
    path = tmp_path / "lokr.safetensors"
    save_file({"lora_unet_blocks_0_self_attn_q_proj.lokr_w1": torch.eye(4)}, str(path))

    with pytest.raises(ValueError, match="lora_down"):
        delta_w_by_block(str(path))


def test_unpaired_down_is_refused(tmp_path):
    path = tmp_path / "half.safetensors"
    save_file(
        {"lora_unet_blocks_0_self_attn_q_proj.lora_down.weight": torch.eye(4)},
        str(path),
    )

    with pytest.raises(ValueError, match="without `.lora_up.weight`"):
        delta_w_by_block(str(path))
