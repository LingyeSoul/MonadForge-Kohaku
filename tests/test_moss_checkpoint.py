"""MOSS ComfyUI single-file DiT: layout mapping, arch inference, fp16-fix unfold.

A ComfyUI checkpoint is one .safetensors with the joint q / k / v fused and its
keys optionally prefixed; an fp16-fix file additionally carries the fold in its
weights. These tests pin the three contracts that make it interchangeable with
the published diffusers folder:

1. Layout: fused qkv splits into q / k / v in that order, prefixes strip, and the
   derived rope.bands is rebuilt from an inferred rope_base instead of loaded.
2. Fold: a folded file loads back to the **unfolded** weights bit-exactly (so a
   LoRA trains in the official parameterisation), and the fp16 lane then applies
   its runtime fold exactly once.
3. Refusals: quantized files, per-block folds, missing / unexpected keys, wrong
   shapes and a directory passed as --dit fail with a clear error.

CPU-only and tiny: key mapping, shapes and the fold arithmetic, no kernels.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from library.moss import checkpoint
from library.moss.checkpoint import CheckpointFormatError
from library.moss.loader import load_transformer
from library.moss.model import _EPS, MOSSImageTransformer
from library.moss.requests import GenerateRequest, TrainRequest
from library.moss.validation import validate_request
from webui.services import moss_service

FOLD = {
    "c": 0.0625,
    "via": "gate",
    "s_attn": 1.0,
    "s_ffn": 1.0,
    "eps": "stock",
}


def tiny(**kw) -> MOSSImageTransformer:
    torch.manual_seed(0)
    base = {
        "dim": 32,
        "ffn_dim": 48,
        "num_heads": 4,
        "num_layers": 2,
        "text_dim": 16,
        "latent_channels": 4,
        "freq_dim": 8,
    }
    return MOSSImageTransformer(**{**base, **kw}).eval()


def comfy_state(
    model: MOSSImageTransformer, prefix: str = "", fold_c: float | None = None
) -> dict[str, torch.Tensor]:
    """`model`'s state dict in ComfyUI layout, optionally folded by `fold_c`."""
    state = {key: value.clone() for key, value in model.state_dict().items()}
    for index in range(model.config.num_layers):
        base = f"layers.{index}.proj"
        q, k, v = (state.pop(f"{base}.{part}.weight") for part in "qkv")
        state[f"{base}.qkv.weight"] = torch.cat((q, k, v), 0)
    if fold_c is not None:
        _fold_in_place(state, model.config.dim, fold_c)
    return {prefix + key: value for key, value in state.items()}


def _scale(state, name, factor) -> None:
    state[name] = (state[name].float() * factor).to(state[name].dtype)


def _fold_in_place(state, dim: int, c: float) -> None:
    """Upstream's uniform gate fold: writers and the two gate row bands carry c."""
    for name in (
        "video_embedder.weight",
        "video_embedder.bias",
        "text_embedder.proj.weight",
        "text_embedder.proj.bias",
    ):
        _scale(state, name, c)
    for index in range(2):
        for linear in ("proj", "ffn_down"):
            _scale(state, f"text_embedder.blocks.{index}.{linear}.weight", c)
            _scale(state, f"text_embedder.blocks.{index}.{linear}.bias", c)
    rows = torch.ones(6 * dim)
    rows[2 * dim : 3 * dim] = c
    rows[5 * dim : 6 * dim] = c
    _scale(state, "shared_block_modulation.weight", rows[:, None])
    _scale(state, "shared_block_modulation.bias", rows)


def write(path: Path, state: dict[str, torch.Tensor], fold: dict | None = None) -> Path:
    metadata = {"kohakumoss_fp16_fix": json.dumps(fold)} if fold else None
    save_file(
        {key: value.contiguous() for key, value in state.items()},
        str(path),
        metadata=metadata,
    )
    return path


def load(path: Path, dtype=torch.float32, **kw) -> MOSSImageTransformer:
    return load_transformer("", dit=str(path), dtype=dtype, **kw)


# --------------------------------------------------------------- key layout


def test_single_file_round_trip_matches_the_folder_layout(tmp_path):
    model = tiny()
    path = write(tmp_path / "plain.safetensors", comfy_state(model))
    loaded = load(path)
    for key, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value), key
    assert loaded.base_file == str(path)
    assert loaded.base_fold is None


def test_prefixed_keys_are_stripped(tmp_path):
    model = tiny()
    state = comfy_state(model, prefix="model.diffusion_model.")
    loaded = load(write(tmp_path / "prefixed.safetensors", state))
    assert torch.equal(
        loaded.state_dict()["layers.0.proj.q.weight"],
        model.state_dict()["layers.0.proj.q.weight"],
    )


def test_fused_qkv_splits_in_q_k_v_row_order(tmp_path):
    model = tiny()
    # Each projection is a distinct constant, so a wrong split order shows up.
    with torch.no_grad():
        for name, value in (("q", 1.0), ("k", 2.0), ("v", 3.0)):
            model.layers[0].proj[name].weight.fill_(value)
    loaded = load(write(tmp_path / "qkv.safetensors", comfy_state(model)))
    for name, value in (("q", 1.0), ("k", 2.0), ("v", 3.0)):
        weight = loaded.state_dict()[f"layers.0.proj.{name}.weight"]
        assert torch.equal(weight, torch.full_like(weight, value)), name


def test_comfy_attention_hints_and_rope_bands_are_not_loaded(tmp_path):
    model = tiny()
    state = comfy_state(model)
    state["layers.0.comfy_attention.scale"] = torch.ones(1)
    loaded = load(write(tmp_path / "hints.safetensors", state))
    assert torch.equal(loaded.rope.bands, model.rope.bands)
    assert not hasattr(loaded.layers[0], "comfy_attention")


# ------------------------------------------------------------ fp16-fix fold


def test_folded_file_is_undone_to_the_official_weights(tmp_path):
    model = tiny()
    path = write(
        tmp_path / "fixed.safetensors", comfy_state(model, fold_c=FOLD["c"]), FOLD
    )
    loaded = load(path)
    for key, value in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value), key
    assert loaded.base_fold == {
        "c": FOLD["c"],
        "via": "gate",
        "s_attn": 1.0,
        "s_ffn": 1.0,
        "eps": "stock",
    }


def test_fp16_lane_folds_exactly_once_on_an_unfolded_base(tmp_path):
    model = tiny()
    path = write(
        tmp_path / "fixed.safetensors", comfy_state(model, fold_c=FOLD["c"]), FOLD
    )
    loaded = load(path, dtype=torch.float16, fp16_fix="online", fp16_fix_c=FOLD["c"])
    # The file's fold was undone on load, so the guard's c is applied once, by
    # the runtime lane: eps = base * c^2 and the gate rows are the plain weights.
    assert loaded.layers[0].norm1.eps == pytest.approx(_EPS * FOLD["c"] ** 2)
    assert torch.equal(
        loaded.state_dict()["shared_block_modulation.weight"],
        model.state_dict()["shared_block_modulation.weight"].half(),
    )


def test_plain_fp16_file_still_gets_the_runtime_fold(tmp_path):
    model = tiny()
    path = write(tmp_path / "plain.safetensors", comfy_state(model))
    loaded = load(path, dtype=torch.float16, fp16_fix="online", fp16_fix_c=0.0625)
    assert loaded.base_fold is None
    assert loaded.layers[0].norm1.eps == pytest.approx(_EPS * 0.0625**2)


# --------------------------------------------------------------- refusals


def test_quantized_single_file_is_refused(tmp_path):
    model = tiny()
    state = comfy_state(model)
    state["layers.0.proj.o.weight_scale"] = torch.ones(1)
    path = write(tmp_path / "int8.safetensors", state)
    with pytest.raises(CheckpointFormatError, match="weight_scale"):
        load(path)


def test_per_block_fold_is_refused(tmp_path):
    model = tiny()
    state = comfy_state(model)
    path = write(
        tmp_path / "proj-fold.safetensors",
        state,
        {"scales": [0.5, 0.25], "via": "gate", "pregate": [1.0, 1.0], "eps": "stock"},
    )
    with pytest.raises(CheckpointFormatError, match="per-block fold scales"):
        load(path)


def test_projection_fold_is_refused(tmp_path):
    model = tiny()
    state = comfy_state(model)
    path = write(
        tmp_path / "via-proj.safetensors",
        state,
        {"c": 0.5, "via": "proj", "pregate": [1.0, 1.0], "eps": "stock"},
    )
    with pytest.raises(CheckpointFormatError, match="via='proj'"):
        load(path)


def test_missing_and_unexpected_keys_are_reported(tmp_path):
    model = tiny()
    state = comfy_state(model)
    state.pop("final_norm_video.weight")
    path = write(tmp_path / "missing.safetensors", state)
    with pytest.raises(CheckpointFormatError, match="missing"):
        load(path)

    state = comfy_state(model)
    state["bogus.weight"] = torch.ones(2, 2)
    path = write(tmp_path / "unexpected.safetensors", state)
    with pytest.raises(CheckpointFormatError, match="unexpected"):
        load(path)


def test_wrong_shape_is_rejected_before_any_weight_is_read(tmp_path):
    model = tiny()
    state = comfy_state(model)
    state["video_embedder.weight"] = torch.ones(3, 5)
    path = write(tmp_path / "shape.safetensors", state)
    with pytest.raises(CheckpointFormatError, match="wrong tensor shape"):
        load(path)


# ------------------------------------------------------- architecture + input


def test_arch_config_is_inferred_from_the_header(tmp_path):
    model = tiny()
    path = write(tmp_path / "arch.safetensors", comfy_state(model))
    assert checkpoint.arch_config(path) == {
        "dim": 32,
        "num_heads": 4,
        "num_layers": 2,
        "ffn_dim": 48,
        "text_dim": 16,
        "freq_dim": 8,
        "latent_channels": 4,
        "rope_base": 10000.0,
    }


def test_custom_rope_base_is_recovered_from_rope_bands(tmp_path):
    model = tiny(num_heads=2, rope_base=5000.0)
    path = write(tmp_path / "rope.safetensors", comfy_state(model))
    assert checkpoint.arch_config(path)["rope_base"] == 5000.0
    loaded = load(path)
    assert torch.equal(loaded.rope.bands, model.rope.bands)


def test_dit_must_be_a_safetensors_file_not_a_folder(tmp_path):
    with pytest.raises(ValueError, match="folder"):
        load_transformer("", dit=str(tmp_path), dtype=torch.float32)
    with pytest.raises(ValueError, match="not a file"):
        load_transformer(
            "", dit=str(tmp_path / "nope.safetensors"), dtype=torch.float32
        )


def test_validation_rejects_a_missing_or_foldered_dit(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        validate_request(TrainRequest(dit=str(tmp_path / "nope.safetensors")))
    folder = tmp_path / "weights"
    folder.mkdir()
    with pytest.raises(ValueError, match="not a folder"):
        validate_request(TrainRequest(dit=str(folder)))


def test_request_round_trips_the_dit_flag(tmp_path):
    path = tmp_path / "weights.safetensors"
    req = TrainRequest(dit=str(path))
    assert "--dit" in req.to_argv()
    assert TrainRequest.from_argv(req.to_argv()).dit == str(path)


# --------------------------------------------- browser/daemon preflight (torch-free)


def test_service_preflight_accepts_a_single_file_dit(tmp_path):
    """The WebUI/daemon path must not require the diffusers transformer folder."""
    path = write(tmp_path / "dit.safetensors", comfy_state(tiny()))
    moss_service.validate_paths(TrainRequest(dit=str(path)))


def test_service_preflight_still_requires_the_folder_without_dit(tmp_path):
    with pytest.raises(FileNotFoundError, match="MOSS transformer"):
        moss_service.validate_paths(TrainRequest(model_dir=str(tmp_path)))


def test_service_preflight_refuses_a_quantized_single_file(tmp_path):
    state = comfy_state(tiny())
    state["layers.0.proj.o.weight_scale"] = torch.ones(1)
    path = write(tmp_path / "int8.safetensors", state)
    with pytest.raises(ValueError, match="Quantized"):
        moss_service.validate_paths(TrainRequest(dit=str(path)))


def test_service_preflight_refuses_a_non_moss_safetensors(tmp_path):
    path = tmp_path / "other.safetensors"
    save_file({"bogus.weight": torch.ones(2, 2)}, str(path))
    with pytest.raises(ValueError, match="not a MOSS-Image DiT"):
        moss_service.validate_paths(TrainRequest(dit=str(path)))


def test_service_resolve_reports_the_dit_file(tmp_path):
    path = write(tmp_path / "dit.safetensors", comfy_state(tiny()))
    assert moss_service.resolve_values({"dit": str(path)}).dit == str(path)
    assert Path(moss_service.resolve_values({}).dit).name == "transformer"


def test_service_pins_the_dit_for_the_daemon(tmp_path, monkeypatch):
    from webui.services import moss_workspace
    from webui.services.qwen21_workspace import CacheStatus

    monkeypatch.setattr(
        moss_workspace,
        "inspect_cache",
        lambda directory: CacheStatus(
            directory=str(directory),
            pairs=1,
            missing_text=0,
            missing_latents=0,
            errors=[],
            ready=True,
        ),
    )
    path = write(tmp_path / "dit.safetensors", comfy_state(tiny()))
    args = moss_service.prepare_job_args(
        "moss-train", [*TrainRequest(dit=str(path)).to_argv(), "--rank", "4"]
    )
    assert args[args.index("--dit") + 1] == str(path)


def test_service_job_context_records_the_dit(tmp_path):
    path = write(tmp_path / "dit.safetensors", comfy_state(tiny()))
    args = TrainRequest(dit=str(path)).to_argv()
    assert moss_service.task_context("moss-train", args)["dit"] == path.name


def test_service_generate_still_requires_the_vae(tmp_path):
    """A single-file DiT replaces the DiT only: VAE / TE / tokenizer still load."""
    path = write(tmp_path / "dit.safetensors", comfy_state(tiny()))
    req = GenerateRequest(dit=str(path), prompt="x", out_dir="output/moss/test")
    with pytest.raises(FileNotFoundError, match="MOSS VAE"):
        moss_service.validate_paths(req)


# ------------------------------------------------ a folded base trains one step


def test_folded_single_file_base_trains_one_step(tmp_path):
    """The fp16-fix base is a normal training run: finite loss, adapter grads.

    Loaded through the same path the trainer uses (un-folded on load, fp16 lane
    re-folds once), then one loss / backward exactly as `train.train` does.
    """
    from library.moss.lora import LoRANetwork
    from library.moss.train import training_loss

    path = write(
        tmp_path / "fixed.safetensors",
        comfy_state(tiny(), fold_c=FOLD["c"]),
        FOLD,
    )
    model = load(path, dtype=torch.float16, fp16_fix="online", fp16_fix_c=FOLD["c"])
    model.requires_grad_(False)
    net = LoRANetwork(model, rank=4, dtype=torch.float32, fp32_compute=True)
    net.apply_to()
    model.train()

    g = torch.Generator().manual_seed(1)
    latent = torch.randn(1, 4, 8, 8, generator=g)
    embeds = torch.randn(1, 5, 16, generator=g).to(torch.float16)
    loss = training_loss(model, latent, embeds, 0.5, gen=g)
    loss.backward()

    assert torch.isfinite(loss).all()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(x).all() for x in grads)
    assert all(x.dtype == torch.float32 for x in grads)


# ------------------------------------------------- entry-point wiring (static)

_ENTRY_POINTS = ("train", "generate", "tune")


@pytest.mark.parametrize("name", _ENTRY_POINTS)
def test_entry_points_forward_the_dit_to_the_loader(name):
    """Every ``load_transformer`` call in an entry point must carry ``dit=``."""
    import ast

    source = (
        Path(__file__).resolve().parents[1] / "library/moss" / f"{name}.py"
    ).read_text(encoding="utf-8")
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_transformer"
    ]
    assert calls, f"{name}.py no longer loads a transformer"
    for call in calls:
        if "dit" not in {keyword.arg for keyword in call.keywords}:
            raise AssertionError(
                f"{name}.py:{call.lineno}: load_transformer call misses dit=req.dit"
            )


# ------------------------------------------------ metadata / file refusals


def test_corrupt_fold_metadata_is_reported(tmp_path):
    model = tiny()
    path = tmp_path / "bad-metadata.safetensors"
    save_file(
        {key: value.contiguous() for key, value in comfy_state(model).items()},
        str(path),
        metadata={"kohakumoss_fp16_fix": "{not json"},
    )
    with pytest.raises(CheckpointFormatError, match="not JSON"):
        load(path)


def test_non_moss_file_is_rejected_by_the_loader(tmp_path):
    path = tmp_path / "other.safetensors"
    save_file({"bogus.weight": torch.ones(4, 4)}, str(path))
    with pytest.raises(CheckpointFormatError, match="not a MOSS-Image DiT"):
        load(path)


def test_malformed_modulation_rows_are_reported(tmp_path):
    model = tiny()
    state = comfy_state(model)
    state["shared_block_modulation.weight"] = torch.zeros(6 * 32 + 1, 32)
    state["shared_block_modulation.bias"] = torch.zeros(6 * 32 + 1)
    path = write(tmp_path / "odd-rows.safetensors", state)
    with pytest.raises(CheckpointFormatError, match="6 × dim"):
        load(path)


@pytest.mark.parametrize("sidecar", ("arch.json", "arch.safetensors.json"))
def test_sidecar_config_is_read_under_both_names(tmp_path, sidecar):
    model = tiny()
    path = write(tmp_path / "arch.safetensors", comfy_state(model))
    (tmp_path / sidecar).write_text(json.dumps({"num_layers": 3}), encoding="utf-8")
    with pytest.raises(CheckpointFormatError, match="disagrees"):
        load(path)


def test_two_sidecar_configs_are_refused(tmp_path):
    model = tiny()
    path = write(tmp_path / "two.safetensors", comfy_state(model))
    for sidecar in ("two.json", "two.safetensors.json"):
        (tmp_path / sidecar).write_text("{}", encoding="utf-8")
    with pytest.raises(CheckpointFormatError, match="two config sidecars"):
        load(path)
