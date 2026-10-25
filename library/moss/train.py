"""Train a LoRA on a precached folder: flow matching, batch 1.

Reads only what ``cache.py`` wrote (latents + text embeddings), so neither the
Qwen3.5-9B TE nor the VAE is loaded. ``--caption_dropout`` reads the empty prompt
from the shipped ``library/moss/asset/empty_prompt.safetensors`` (key ``embeds``;
one file for every cache, re-encoded only if it is missing).

A layout-JSON cache also holds each image's plain Anima caption
(``embeds_plain``); a step trains on it with probability ``plain_caption_rate``,
else on one of the TE file's caption variants (``embeds_v{k}``) with probability
``caption_variant_rate``, then on the empty prompt with ``caption_dropout``
(``choose_embeds``). Each of the three draws has its own RNG, so the shuffle and
σ sequence match a run with none of them. Samples stay in RAM up to
``PRELOAD_LIMIT_GB``; a larger set is read from disk each step (variants always are).
With ``--fit_report``, step 1 is the largest sample with its longest caption (full,
plain or a variant) and no dropout, so the report bounds the worst case.

Objective (report App. A.2): ``z_σ = (1−σ)·z0 + σ·ε``, the DiT predicts
``ε − z0`` at timestep ``1000·σ``, MSE averaged over image tokens and channels
(text tokens never reach the head), with no σ/SNR weighting.

σ follows the final Stage-3 interval exactly: ``Z ~ N(μ, s²)``, ``S = 1 − sigmoid(Z)``,
truncated to ``S ≤ cutoff`` and rescaled onto the grid ``{1/1000 … 1}``
(``σ = ⌈1000·S/cutoff⌉ / 1000``), mixed with 1 % discrete uniform. Shift 1, no
resolution shift: that interval trained at area ≤ 1024², the size the cache
targets.

Default memory plan: no checkpointing, 12 blocks swapped, and the compiled
blocks' saved set capped by ``activation_memory_budget`` (0.1), which is faster
than checkpointing on 16 GB.
``--grad_checkpointing`` wraps each block's ``forward`` (``checkpoint_blocks``;
not ``__call__``), so the block swapper's forward hooks fire once per step and
never again during the recompute.

    make daemon-run ARGS="scripts/moss/train.py --cache post_image_dataset/moss/<set> --epochs 8"
"""

from __future__ import annotations

import functools
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from library.env import resolve_under_home
from library.moss.loader import (
    compile_blocks,
    empty_cache,
    free_vram_gb,
    load_transformer,
    resolve_compute_dtype,
)
from library.moss.lora import LoRANetwork, load_channel_stats
from library.moss.ops import fuse
from library.moss.requests import LORA_ALPHA, TrainRequest
from library.moss.validation import validate_request
from library.training.progress import ProgressSink, command_job_path, run_scope

EMPTY_PROMPT_PATH = "library/moss/asset/empty_prompt.safetensors"
NUM_TRAIN_TIMESTEPS = 1000
PRELOAD_LIMIT_GB = 16  # above this the samples are read from disk each step
PROGRESS_EVERY = 2  # steps per "  progress i/n" line (the GUI bar reads them)


@dataclass
class Sample:
    """One training pair, lazily loadable. ``n_img`` / ``n_text`` order the run."""

    stem: str
    n_img: int
    n_text: int
    load: Callable[
        [], tuple[torch.Tensor, torch.Tensor]
    ]  # latent (128,h,w), embeds (S,4096)
    load_plain: Callable[[], torch.Tensor] | None = None  # embeds_plain (S',4096)
    n_ref: int = 0
    load_ref: Callable[[], torch.Tensor] | None = None  # clean ref latent (128,h,w)
    latent_path: Path | None = None  # finds the REPA target (latent metadata)
    n_variants: int = 0
    load_variant: Callable[[int], torch.Tensor] | None = None  # embeds_v{k}, from disk
    latent_hw: tuple[int, int] | None = None  # (h, w) latent tokens, from the file name


def samples_from_cache(cache_dir: Path) -> list[Sample]:
    from library.moss import cache as moss_cache

    out = []
    for item in moss_cache.list_items(cache_dir):
        load_plain = (
            functools.partial(moss_cache.load_plain, item)
            if item.n_text_plain
            else None
        )
        out.append(
            Sample(
                item.stem,
                (item.width // 16) * (item.height // 16),
                max(item.n_text, item.n_text_plain, item.n_text_variants),
                functools.partial(moss_cache.load_item, item),
                load_plain,
                item.n_ref,
                functools.partial(moss_cache.load_ref, item) if item.n_ref else None,
                item.latent_path,
                item.n_variants,
                functools.partial(moss_cache.load_variant, item)
                if item.n_variants
                else None,
                (item.height // 16, item.width // 16),
            )
        )
    return out


def choose_embeds(
    embeds: torch.Tensor,
    plain: torch.Tensor | None = None,
    n_variants: int = 0,
    load_variant: Callable[[int], torch.Tensor] | None = None,
    empty: torch.Tensor | None = None,
    *,
    plain_rate: float = 0.0,
    variant_rate: float = 0.0,
    dropout: float = 0.0,
    plain_rng: random.Random | None = None,
    variant_rng: random.Random | None = None,
    drop_rng: random.Random | None = None,
    worst_case: bool = False,
) -> tuple[torch.Tensor, str]:
    """The text one step trains on and its kind (``full`` / ``plain`` / ``v{k}`` /
    ``empty``): the plain caption at ``plain_rate``, else a variant (loaded on
    demand) at ``variant_rate``, then the empty prompt at ``dropout``. A candidate
    that is None / 0 draws nothing from its RNG. ``worst_case`` (the fit-report
    step) draws nothing and returns the longest of full, plain and every variant."""
    if worst_case:
        cands = [("full", embeds)]
        if plain is not None:
            cands.append(("plain", plain))
        cands += [(f"v{k}", load_variant(k)) for k in range(1, n_variants + 1)]
        kind, x = max(cands, key=lambda c: c[1].shape[0])
        return x, kind
    kind, x = "full", embeds
    if plain is not None and plain_rng.random() < plain_rate:
        kind, x = "plain", plain
    elif n_variants and variant_rng.random() < variant_rate:
        k = variant_rng.randint(1, n_variants)
        kind, x = f"v{k}", load_variant(k)
    if empty is not None and drop_rng.random() < dropout:
        kind, x = "empty", empty
    return x, kind


def sample_sigma(
    gen: torch.Generator,
    mu: float,
    s: float,
    cutoff: float = 0.95,
    uniform: float = 0.01,
    total: int = NUM_TRAIN_TIMESTEPS,
) -> float:
    """One σ on the ``{1/T … 1}`` grid, from report eq. (4)."""
    if torch.rand((), generator=gen).item() < uniform:
        j = int(torch.randint(1, total + 1, (), generator=gen))
        return j / total
    while True:
        z = torch.randn((), generator=gen).item() * s + mu
        noise = 1.0 - 1.0 / (1.0 + math.exp(-z))
        if noise <= cutoff:
            break
    j = min(total, max(1, math.ceil(noise * total / cutoff)))
    return j / total


def training_loss(transformer, latent, embeds, sigma: float, gen=None, ref=None):
    """``latent (1,128,h,w)`` fp32, ``embeds (1,S,4096)`` at true length,
    ``ref (1,128,h',w')`` an optional clean reference latent (never noised)."""
    noise = torch.randn(
        latent.shape, generator=gen, device=latent.device, dtype=latent.dtype
    )
    noisy = (1.0 - sigma) * latent + sigma * noise
    target = noise - latent
    timestep = torch.tensor([NUM_TRAIN_TIMESTEPS * sigma], device=latent.device)
    pred = transformer(noisy, timestep, embeds, return_dict=False, ref=ref)[0]
    return F.mse_loss(pred.float(), target.float())


def training_loss_packed(
    transformer, latents, embeds, sigmas, gen=None, max_batch=None
):
    """Packed flow-matching loss over samples with different shapes.

    ``latents`` / ``embeds`` are per-sample ``(1,C,h,w)`` / ``(1,S,4096)`` on
    device; ``sigmas`` one σ per sample. Returns the mean loss and the per-sample
    values, so the optimizer update weights every sample equally.
    """
    noises = [
        torch.randn(
            latent.shape, generator=gen, device=latent.device, dtype=latent.dtype
        )
        for latent in latents
    ]
    noisy = [
        (1.0 - sigma) * latent + sigma * noise
        for latent, noise, sigma in zip(latents, noises, sigmas)
    ]
    targets = [noise - latent for latent, noise in zip(latents, noises)]
    timesteps = torch.tensor(
        [NUM_TRAIN_TIMESTEPS * sigma for sigma in sigmas], device=latents[0].device
    )
    preds = transformer.forward_packed(noisy, timesteps, embeds, max_batch=max_batch)
    per_sample = [
        F.mse_loss(pred.float(), target.float()) for pred, target in zip(preds, targets)
    ]
    return torch.stack(per_sample).mean(), per_sample


def sample_joint(s: Sample) -> int:
    """Joint image + reference + text tokens one sample contributes."""
    return s.n_img + s.n_ref + s.n_text


def make_batches(
    samples: list[Sample], batch_size: int, token_batch: int
) -> list[list[Sample]]:
    """Split a shuffled sample list into packed forwards.

    ``batch_size`` caps the sample count; ``token_batch`` > 0 caps the joint token
    count. The first sample of a batch always fits alone, so a single oversized
    sample cannot deadlock the loop.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if token_batch <= 0:
        return [
            samples[start : start + batch_size]
            for start in range(0, len(samples), batch_size)
        ]
    batches, current, tokens = [], [], 0
    for sample in samples:
        joint = sample_joint(sample)
        if current and (len(current) >= batch_size or tokens + joint > token_batch):
            batches.append(current)
            current, tokens = [], 0
        current.append(sample)
        tokens += joint
    if current:
        batches.append(current)
    return batches


def max_packed_joint(samples: list[Sample], batch_size: int, token_batch: int) -> int:
    """Upper bound on one packed batch's joint tokens, for the VRAM reserve."""
    joints = sorted((sample_joint(s) for s in samples), reverse=True)
    worst = sum(joints[:batch_size])
    if token_batch > 0:
        worst = min(worst, max(joints[0], token_batch))
    return worst


def checkpoint_blocks(blocks) -> None:
    """Per-block activation checkpointing around ``block.forward``."""
    for block in blocks:
        inner = block.forward

        @functools.wraps(inner)
        def forward(*args, _inner=inner, **kwargs):
            if not torch.is_grad_enabled():
                return _inner(*args, **kwargs)
            return checkpoint(_inner, *args, use_reentrant=False, **kwargs)

        block.forward = forward


# Uncheckpointed, the compiled saved set is ~5.25 MB per joint token at budget
# 1.0 (128 KiB per block x 42); the constant covers AdamW state and allocator
# growth (fit to budgets 0.15-0.25 at 4.4k tokens, ~1 GiB above measured).
_BUDGET_BASE_GB = 3.5
_BUDGET_MB_PER_TOKEN = 6.0


def step_reserve_gb(
    joint: int, *, grad_checkpointing: bool, budget: float = 1.0
) -> float:
    """VRAM one training step needs beyond the placed weights, at ``joint``
    (image + text) tokens."""
    if grad_checkpointing:
        # ~0.6 MB per joint token (qwen21's measured slope, same hidden width
        # class) plus one recomputed block.
        return 1.5 + 0.6 * joint / 1024
    return _BUDGET_BASE_GB + budget * _BUDGET_MB_PER_TOKEN * joint / 1024


def budget_for_free(free_gb: float, joint: int, step: float = 0.05) -> float:
    """The largest budget on a ``step`` grid whose :func:`step_reserve_gb` fits
    ``free_gb``; 0.0 when not even one ``step`` fits."""
    room = (free_gb - _BUDGET_BASE_GB) / (_BUDGET_MB_PER_TOKEN * joint / 1024)
    return min(1.0, math.floor(max(0.0, room) / step + 1e-9) * step)


def set_activation_memory_budget(budget: float) -> None:
    """Inductor's partitioner reads this at compile time: set it before
    ``compile_blocks`` (and reset ``torch._dynamo`` to change it later)."""
    import torch._functorch.config as functorch_config

    functorch_config.activation_memory_budget = budget


def place(transformer, device, blocks_to_swap: int, reserve_gb: float):
    """Everything on ``device``, or the block swapper for ``blocks_to_swap`` > 0
    (``-1`` sizes it from ``mem_get_info`` free). Returns the attachment or None."""
    from library.runtime.blockswap import place as place_blocks

    return place_blocks(
        transformer,
        "layers",
        device,
        blocks_to_swap=None if blocks_to_swap < 0 else blocks_to_swap,
        supports_backward=True,
        activation_reserve_gb=reserve_gb,
        label="transformer",
    )


def report_fit(transformer, attached, n_img: int, n_text: int) -> dict:
    """After step 1 (the largest sample): peak against what is actually free."""
    import library.runtime.blockswap as blockswap

    allocated = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.max_memory_reserved() / 1024**3
    free = free_vram_gb()
    per = blockswap.block_size_gb(transformer.layers)
    swapped = attached.offloader.blocks_to_swap if attached else 0
    fit = {
        "image_tokens": n_img,
        "text_tokens": n_text,
        "peak_allocated_gb": round(allocated, 3),
        "peak_reserved_gb": round(reserved, 3),
        "free_gb": round(free, 3),
        "blocks_to_swap": swapped,
    }
    print(
        f"fit @ step 1 ({n_img} image + {n_text} text tokens, the largest sample): "
        f"peak {allocated:.2f} GB allocated / {reserved:.2f} GB reserved, "
        f"{free:.2f} GB actually free, swapping {swapped} at {per:.3f} GB each",
        flush=True,
    )
    if swapped:
        spare = max(0, int((free - per) // per))
        if spare:
            print(
                f"  -> rerun with --blocks_to_swap {max(0, swapped - spare)}",
                flush=True,
            )
    return fit


def empty_prompt_embeds(req: TrainRequest) -> torch.Tensor:
    """``(S, 4096)`` bf16 embedding of ``""`` (10 chat-template tokens)."""
    from safetensors.torch import load_file, save_file

    path = resolve_under_home(EMPTY_PROMPT_PATH)
    if not path.exists():
        from library.moss.loader import encode_prompts

        print(f"encoding the empty prompt -> {path}", flush=True)
        embeds = encode_prompts(req.model_dir, [""])[""].to(torch.bfloat16)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({"embeds": embeds.contiguous()}, str(path))
    return load_file(str(path))["embeds"]


RESUME_FORMAT = 1
_RESUME_KEYS = (
    "model",
    "algo",
    "rank",
    "alpha",
    "factor",
    "targets",
    "down_init",
    "channel_scaling_alpha",
    "dc_isolate",
    "svd_slice",
    "svd_slice_count",
    "optimizer",
    "lr",
    "warmup_ratio",
    "batch_size",
    "token_batch",
    "grad_accum",
    "precision",
    "fp16_fix",
    "fp16_fix_c",
    "ref_timestep",
    "ref_isolate",
    "caption_dropout",
    "plain_caption_rate",
    "caption_variant_rate",
    "repa_weight",
    "repa_layer",
    "max_grad_norm",
    "sigma_mu",
    "sigma_s",
    "sigma_cutoff",
    "sigma_uniform",
    "seed",
)


def resume_signature(req: TrainRequest) -> dict:
    """Fields that must match for a state file to describe this run."""
    return {name: getattr(req, name) for name in _RESUME_KEYS}


def _cpu_tree(value):
    """Move tensors in a nested state dict to CPU for a portable ``torch.save``."""
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def save_resume_state(
    path: Path,
    *,
    req: TrainRequest,
    network,
    optimizer,
    lr_sched,
    scaler,
    step: int,
    micro_step: int,
    next_epoch: int,
    rates: dict,
    sigma_gen,
    noise_gen,
) -> None:
    """Write the adapter's optimizer/RNG state beside the LoRA file.

    The epoch boundary is the checkpoint boundary: a resumed run replays the
    last completed epoch if the process died inside one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format": RESUME_FORMAT,
        "signature": resume_signature(req),
        "network": network.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_sched": lr_sched.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "micro_step": micro_step,
        "next_epoch": next_epoch,
        "random": random.getstate(),
        "rates": {
            name: rng.getstate()
            for name, rng in rates.items()
            if hasattr(rng, "getstate")
        },
        "sigma_gen": sigma_gen.get_state(),
        "noise_gen": noise_gen.get_state(),
    }
    torch.save(_cpu_tree(state), path)


def load_resume_state(
    path: Path,
    *,
    req: TrainRequest,
    network,
    optimizer,
    lr_sched,
    scaler,
    rates: dict,
    sigma_gen,
    noise_gen,
) -> tuple[int, int, int]:
    """Restore a state written by :func:`save_resume_state`; returns the counters."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format") != RESUME_FORMAT:
        raise SystemExit(f"unsupported MOSS resume format in {path}")
    saved = state["signature"]
    current = resume_signature(req)
    mismatch = {
        key: (saved.get(key), current.get(key))
        for key in current
        if saved.get(key) != current.get(key)
    }
    if mismatch:
        raise SystemExit(
            "resume state does not match this run: "
            + ", ".join(
                f"{key} {have!r} != {want!r}" for key, (have, want) in mismatch.items()
            )
        )
    network.load_state_dict(state["network"])
    optimizer.load_state_dict(state["optimizer"])
    lr_sched.load_state_dict(state["lr_sched"])
    scaler.load_state_dict(state["scaler"])
    random.setstate(state["random"])
    for name, rng in rates.items():
        if hasattr(rng, "setstate"):
            rng.setstate(state["rates"][name])
    sigma_gen.set_state(state["sigma_gen"])
    # A CUDA generator's state is a CPU ByteTensor; moving it to the device is
    # rejected by ``set_state``.
    noise_gen.set_state(state["noise_gen"])
    return int(state["step"]), int(state["micro_step"]), int(state["next_epoch"])


def run_train(req: TrainRequest) -> Path:
    validate_request(req)
    cache_dir = resolve_under_home(req.cache)
    samples = samples_from_cache(cache_dir)
    if not samples:
        raise SystemExit(f"no cached pairs under {cache_dir}")
    empty = empty_prompt_embeds(req) if req.caption_dropout > 0 else None
    return train(req, samples, empty)


def train(
    req: TrainRequest, samples: list[Sample], empty: torch.Tensor | None = None
) -> Path:
    """The loop, over already-built ``samples`` (``run_train`` reads the cache;
    a fit smoke passes synthetic ones)."""
    validate_request(req)
    torch.manual_seed(req.seed)
    random.seed(req.seed)
    device = torch.device("cuda")
    # Sigma draws are CPU scalars; noise must be drawn by a generator on the
    # tensor's device (a CPU generator cannot seed a CUDA randn).
    sigma_gen = torch.Generator().manual_seed(req.seed)
    noise_gen = torch.Generator(device=device).manual_seed(req.seed)

    img = sorted(s.n_img for s in samples)
    txt = sorted(s.n_text for s in samples)
    print(
        f"{len(samples)} samples, image {img[0]}-{img[-1]} tokens, text {txt[0]}-{txt[-1]} tokens",
        flush=True,
    )
    n_refs = sum(s.load_ref is not None for s in samples)
    if n_refs:
        ref_tok = sorted(s.n_ref for s in samples if s.load_ref is not None)
        print(
            f"reference latents: {n_refs}/{len(samples)} samples, "
            f"{ref_tok[0]}-{ref_tok[-1]} tokens",
            flush=True,
        )
    packed = req.batch_size > 1
    if packed and n_refs:
        raise SystemExit(
            "packed batching does not support reference latents (i2i); use --batch_size 1"
        )
    if packed and req.repa_weight > 0:
        raise SystemExit("packed batching does not support REPA; use --batch_size 1")
    joint = (
        max_packed_joint(samples, req.batch_size, req.token_batch)
        if packed
        else max(sample_joint(s) for s in samples)
    )
    if packed:
        print(
            f"packed batches: up to {req.batch_size} samples"
            + (f" / ~{req.token_batch} joint tokens" if req.token_batch > 0 else "")
            + f", worst case {joint} joint tokens",
            flush=True,
        )
    use_plain = req.plain_caption_rate > 0
    n_plain = sum(s.load_plain is not None for s in samples) if use_plain else 0
    if n_plain:
        print(
            f"plain Anima captions: {n_plain}/{len(samples)} samples, "
            f"rate {req.plain_caption_rate}",
            flush=True,
        )

    def fetch(s):
        """``(latent, embeds, plain or None, ref or None)`` on CPU."""
        latent, embeds = s.load()
        alt = s.load_plain() if use_plain and s.load_plain is not None else None
        ref = s.load_ref() if s.load_ref is not None else None
        return latent, embeds, alt, ref

    # bf16 TE rows (S×4096×2 bytes) + fp32 latents (128×4 bytes per token).
    preload_bytes = sum(
        s.n_text * 8192 * (2 if use_plain and s.load_plain else 1)
        + (s.n_img + s.n_ref) * 512
        for s in samples
    )
    if preload_bytes <= PRELOAD_LIMIT_GB * 1024**3:
        resident = {s.stem: fetch(s) for s in samples}

        def get(s):
            return resident[s.stem]
    else:
        get = fetch
        print(
            f"data ~{preload_bytes / 1024**3:.0f} GB > {PRELOAD_LIMIT_GB} GB: "
            "read from disk each step",
            flush=True,
        )
    use_variants = req.caption_variant_rate > 0
    n_variant_samples = sum(1 for s in samples if s.n_variants) if use_variants else 0
    if n_variant_samples:
        print(
            f"caption variants: {n_variant_samples}/{len(samples)} samples, "
            f"rate {req.caption_variant_rate}",
            flush=True,
        )
    rates = dict(
        plain_rate=req.plain_caption_rate,
        variant_rate=req.caption_variant_rate,
        dropout=req.caption_dropout,
        plain_rng=random.Random(req.seed + 1),
        drop_rng=random.Random(req.seed + 2),
        variant_rng=random.Random(req.seed + 3),
    )
    kinds = {"plain": 0, "variant": 0, "empty": 0}

    compute = resolve_compute_dtype(req.precision, device)
    transformer = load_transformer(
        req.model_dir,
        req.model,
        dtype=compute,
        fp16_fix=req.fp16_fix,
        fp16_fix_c=req.fp16_fix_c,
        dit=req.dit,
    )
    transformer.requires_grad_(False)
    transformer.ref_timestep = req.ref_timestep if req.ref_timestep >= 0 else None
    transformer.ref_isolate = req.ref_isolate
    channel_alpha = req.channel_scaling_alpha if req.algo == "lora" else 0.0
    dc_isolate = req.dc_isolate and req.algo == "lora"
    network = LoRANetwork(
        transformer,
        rank=req.rank,
        alpha=req.alpha if req.alpha is not None or req.algo != "lora" else LORA_ALPHA,
        targets=req.targets,
        dtype=torch.float32,
        down_init=req.down_init,
        algo=req.algo,
        factor=req.factor,
        channel_stats=load_channel_stats(resolve_under_home(req.channel_stats))
        if channel_alpha > 0
        else None,
        channel_scaling_alpha=channel_alpha,
        channel_mean=load_channel_stats(resolve_under_home(req.channel_mean))
        if dc_isolate
        else None,
        dc_isolate=dc_isolate,
        svd_slice=req.svd_slice,
        svd_slice_count=req.svd_slice_count,
        fp32_compute=compute == torch.float16,
    )
    patched = network.apply_to()
    network.to(device)
    if compute == torch.float16:
        fix = (
            f"the stream fold c={req.fp16_fix_c:g} (online)"
            if req.fp16_fix == "online"
            else "the fp32 residual stream (fp32_residual)"
        )
        print(
            f"precision fp16 with {fix} + fp32 adapter math",
            flush=True,
        )
    else:
        print(f"precision {str(compute).removeprefix('torch.')}", flush=True)
    detail = {
        "lora": f"{req.down_init} down"
        + (
            f" slice {req.svd_slice}"
            + (f"/{req.svd_slice_count} interleaved" if req.svd_slice_count else "")
            if req.svd_slice or req.svd_slice_count
            else ""
        )
        + (f", channel scaling α={channel_alpha:g}" if channel_alpha > 0 else "")
        + (", dc isolate" if dc_isolate else ""),
        "lokr": f"factor {req.factor}",
        "loha": "hadamard",
    }[req.algo]
    print(
        f"{req.algo}: rank {req.rank} alpha {network.alpha} ({detail}) on {patched} linears, "
        f"{network.num_parameters / 1e6:.1f}M params (fp32 masters, saved {req.lora_dtype})",
        flush=True,
    )

    budgeted = req.compile and not req.grad_checkpointing
    budget = req.activation_memory_budget if budgeted else 1.0
    reserve_gb = step_reserve_gb(
        joint, grad_checkpointing=req.grad_checkpointing, budget=budget
    )
    attached = place(transformer, device, req.blocks_to_swap, reserve_gb)
    if budgeted and budget < 1.0:
        set_activation_memory_budget(budget)
        print(f"activation memory budget {budget}", flush=True)
    elif not req.compile and not req.grad_checkpointing:
        print(
            "warning: eager without checkpointing saves ~10 MB per joint token "
            "per step; add --grad_checkpointing unless the card has room",
            flush=True,
        )
    repa, repa_targets = None, {}
    if req.repa_weight > 0:
        from library.moss.repa import REPATerm, load_target, pe_sidecar

        pe_root = resolve_under_home(req.repa_pe_root)
        for s in samples:
            if s.latent_path is None or s.latent_hw is None:
                continue
            h, w = s.latent_hw
            path = pe_sidecar(s.latent_path, s.stem, w * 16, h * 16, pe_root)
            if path is not None:
                repa_targets[s.stem] = load_target(path, h, w)
        if not repa_targets:
            raise SystemExit(
                f"repa_weight {req.repa_weight} but no sample has a PE-Spatial sidecar "
                f"under {pe_root} (make preprocess-pe-spatial)"
            )
        repa = REPATerm(transformer, req.repa_layer)
        print(
            f"REPA: block {req.repa_layer}, weight {req.repa_weight:g}, DoG-Gram target "
            f"on {len(repa_targets)}/{len(samples)} samples",
            flush=True,
        )
    fused = fuse(transformer, fp32_adapters=getattr(network, "fp32_compute", False))
    if fused:
        print(f"[kernels] fused block ops on {fused} blocks", flush=True)
    if req.compile:
        compile_blocks(transformer.layers)
    if req.grad_checkpointing:
        checkpoint_blocks(transformer.layers)
    empty_cache()
    print(f"free VRAM before training: {free_vram_gb():.2f} GB", flush=True)

    params = list(network.parameters())
    if req.optimizer == "adamw16":
        from library.moss.optim import AdamW16

        optimizer = AdamW16(params, lr=req.lr)
    elif req.optimizer == "muon":
        from library.moss.optim import MuonHybrid

        optimizer = MuonHybrid([{"params": params, "use_muon": True}], lr=req.lr)
    else:
        optimizer = torch.optim.AdamW(params, lr=req.lr)
    # fp16 gradients underflow (the loss is fp32 but W_grad is not); the scaler
    # is the same dynamic-loss-scale contract upstream's DynamicScaler uses.
    # bf16 needs none — its exponent matches fp32.
    scaler = torch.amp.GradScaler("cuda", enabled=compute == torch.float16)
    total = req.epochs * len(samples)
    if req.max_steps:
        total = min(total, req.max_steps) if req.epochs else req.max_steps
    if req.grad_accum < 1:
        raise SystemExit(f"grad_accum must be >= 1, got {req.grad_accum}")
    accum = req.grad_accum
    # One update per `accum` forwarded batches; a packed batch carries several
    # samples, so the estimate uses batch_size (token_batch only narrows it).
    updates = max(1, math.ceil(total / max(1, req.batch_size) / accum))
    warmup = max(1, int(updates * req.warmup_ratio))
    lr_sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: min(1.0, (update + 1) / warmup)
    )
    epochs = math.ceil(total / len(samples))
    print(
        f"{total} samples ({epochs} epochs x {len(samples)}), "
        + (f"batch {req.batch_size}, " if packed else "")
        + (f"{updates} updates of {accum} batches, " if accum > 1 else "")
        + f"warmup {warmup}",
        flush=True,
    )

    out_path = resolve_under_home(req.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "base_variant": req.model,
        "lr": req.lr,
        "optimizer": req.optimizer,
        "samples": len(samples),
        "steps": total,
        "image_tokens": f"{img[0]}-{img[-1]}",
        "sigma": f"mu={req.sigma_mu} s={req.sigma_s} cutoff={req.sigma_cutoff} "
        f"uniform={req.sigma_uniform}",
        "caption_dropout": req.caption_dropout,
        "plain_caption_rate": req.plain_caption_rate if n_plain else 0.0,
        "caption_variant_rate": req.caption_variant_rate if n_variant_samples else 0.0,
        "reference_latents": n_refs,
        "precision": str(compute).removeprefix("torch."),
        "batch_size": req.batch_size,
    }
    if req.token_batch > 0:
        meta["token_batch"] = req.token_batch
    if req.dit:
        # Provenance only: an fp16-fix fold in the file was undone on load, so the
        # LoRA is in the official parameterisation like every folder-trained run.
        meta["base_dit"] = Path(req.dit).name
        if getattr(transformer, "base_fold", None) is not None:
            meta["base_fold_c"] = f"{transformer.base_fold['c']:g}"
    if compute == torch.float16:
        # A sample is only the trained model when the fix (and c) match.
        meta["fp16_fix"] = req.fp16_fix
        if req.fp16_fix == "online":
            meta["fp16_fix_c"] = f"{req.fp16_fix_c:g}"
    if accum > 1:
        meta["grad_accum"] = accum
    if n_refs:
        meta["ref_timestep"] = req.ref_timestep
        if req.ref_isolate:
            meta["ref_isolate"] = "1"
    if repa is not None:
        meta.update(repa_weight=req.repa_weight, repa_layer=req.repa_layer)

    def save(tag: str | None = None) -> Path:
        """The final file at ``out_path``; an epoch save under ``<dir>/<stem>/``."""
        if tag is None:
            path = out_path
        else:
            folder = out_path.parent / out_path.stem
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{out_path.stem}_{tag}{out_path.suffix}"
        network.save(path, req.lora_dtype, **meta)
        return path

    state_path = out_path.with_name(out_path.name + ".state.pt")
    step, micro_step, start_epoch = 0, 0, 0
    grad_norm = float("nan")
    if req.resume:
        step, micro_step, start_epoch = load_resume_state(
            resolve_under_home(req.resume),
            req=req,
            network=network,
            optimizer=optimizer,
            lr_sched=lr_sched,
            scaler=scaler,
            rates=rates,
            sigma_gen=sigma_gen,
            noise_gen=noise_gen,
        )
        if step > total:
            raise SystemExit(
                f"resume step {step} is past this run's {total} samples; "
                "raise --epochs or --max_steps"
            )
        print(
            f"resumed {req.resume} at sample {step}, epoch {start_epoch + 1}",
            flush=True,
        )

    def save_state(next_epoch: int) -> None:
        save_resume_state(
            state_path,
            req=req,
            network=network,
            optimizer=optimizer,
            lr_sched=lr_sched,
            scaler=scaler,
            step=step,
            micro_step=micro_step,
            next_epoch=next_epoch,
            rates=rates,
            sigma_gen=sigma_gen,
            noise_gen=noise_gen,
        )

    sink_path = command_job_path(
        str(out_path.parent.parent / "logs" / f"{out_path.stem}.progress.jsonl")
    )
    sink = ProgressSink(sink_path, run=out_path.stem, method="moss", preset=None)
    sink.run_start(total_steps=total, total_epochs=epochs, pid=os.getpid())
    print(f"progress sink: {sink_path}", flush=True)

    largest = max(samples, key=lambda s: s.n_img + s.n_ref + s.n_text)
    history, losses_all, fit = [], [], None
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    with run_scope(sink, final_step=lambda: step):
        t0_step = step
        per_step = None
        t_window, step_window = t_start, step
        next_epoch = start_epoch
        for epoch in range(start_epoch, epochs):
            order = random.sample(samples, len(samples))
            if epoch == 0 and req.fit_report:
                # Only the largest sample moves to step 1, the worst case the fit
                # report bounds; the rest of epoch 1 stays shuffled.
                order.remove(largest)
                order.insert(0, largest)
            losses, aligns, t_epoch = [], [], time.time()
            if packed:
                batches = make_batches(order, req.batch_size, req.token_batch)
                if epoch == 0 and req.fit_report:
                    batches = [[largest]] + make_batches(
                        [s for s in order if s is not largest],
                        req.batch_size,
                        req.token_batch,
                    )
            else:
                batches = [[s] for s in order]
            for batch in batches:
                if step >= total:
                    break
                if len(batch) > total - step:
                    batch = batch[: total - step]
                loaded = [get(s) for s in batch]
                chosen = []
                for s, (latent, embeds, alt, ref) in zip(batch, loaded):
                    embeds, kind = choose_embeds(
                        embeds,
                        alt,
                        s.n_variants if use_variants else 0,
                        s.load_variant,
                        empty,
                        worst_case=step == 0 and req.fit_report,
                        **rates,
                    )
                    kind = "variant" if kind.startswith("v") else kind
                    if kind in kinds:
                        kinds[kind] += 1
                    chosen.append((latent, embeds, ref))
                latents = [
                    latent.unsqueeze(0).to(device, torch.float32)
                    for latent, _, _ in chosen
                ]
                embeds = [embeds.unsqueeze(0).to(device) for _, embeds, _ in chosen]
                refs = [
                    None if ref is None else ref.unsqueeze(0).to(device, torch.float32)
                    for _, _, ref in chosen
                ]
                sigmas = [
                    sample_sigma(
                        sigma_gen,
                        req.sigma_mu,
                        req.sigma_s,
                        req.sigma_cutoff,
                        req.sigma_uniform,
                    )
                    for _ in batch
                ]
                if packed:
                    loss, per_sample = training_loss_packed(
                        transformer,
                        latents,
                        embeds,
                        sigmas,
                        gen=noise_gen,
                        max_batch=req.batch_size,
                    )
                    align = None
                else:
                    s = batch[0]
                    if repa is not None:
                        target, grid = repa_targets.get(s.stem, (None, None))
                        repa.prime(
                            None if target is None else target.to(device),
                            grid,
                            tuple(latents[0].shape[-2:]),
                        )
                    loss = training_loss(
                        transformer,
                        latents[0],
                        embeds[0],
                        sigmas[0],
                        gen=noise_gen,
                        ref=refs[0],
                    )
                    per_sample = [loss]
                    align = repa.loss() if repa is not None else None
                if align is not None:
                    scaler.scale((loss + req.repa_weight * align) / accum).backward()
                    aligns.append(align.item())
                else:
                    scaler.scale(loss / accum).backward()
                started = step
                step += len(batch)
                micro_step += 1
                updated = micro_step % accum == 0 or step >= total
                if updated:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        params, req.max_grad_norm
                    ).item()
                    finite = math.isfinite(grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    # A non-finite gradient makes GradScaler skip the step; the DC
                    # re-projection and the schedule must not advance on a skipped
                    # update either. Without a scaler (bf16/fp32) the step always
                    # runs, as before.
                    if finite or not scaler.is_enabled():
                        network.reproject_dc()
                        lr_sched.step()
                    optimizer.zero_grad(set_to_none=True)
                value = loss.item()
                per_values = [sample.item() for sample in per_sample]
                mean_sigma = sum(sigmas) / len(sigmas)
                sink.log(
                    {
                        "loss": value,
                        "sigma": mean_sigma,
                        "lr": lr_sched.get_last_lr()[0],
                        "samples": len(batch),
                    }
                    | ({"grad_norm": grad_norm} if updated else {})
                    | ({"repa": aligns[-1]} if align is not None else {}),
                    global_step=step,
                    epoch=epoch + 1,
                )
                losses.extend(per_values)
                for offset, (sample_loss, sample_sigma_value) in enumerate(
                    zip(per_values, sigmas)
                ):
                    losses_all.append(
                        {
                            "step": started + offset + 1,
                            "loss": sample_loss,
                            "sigma": sample_sigma_value,
                        }
                        | ({"repa": aligns[-1]} if align is not None else {})
                    )
                if started == 0:
                    if req.fit_report:
                        fit = report_fit(
                            transformer,
                            attached,
                            max(s.n_img for s in batch),
                            max(embed.shape[1] for embed in embeds),
                        )
                    t_start = time.time()  # excludes step 1's allocator warm-up
                    t0_step = step
                    t_window, step_window, per_step = t_start, step, None
                eta_rate = (
                    (time.time() - t_start) / max(1, step - t0_step)
                    if per_step is None
                    else per_step
                )
                eta = int(eta_rate * (total - step))
                if step % PROGRESS_EVERY == 0 or step == total:
                    print(
                        f"  progress {step}/{total} epoch {epoch + 1}/{epochs} "
                        f"loss {sum(losses) / len(losses):.4f} "
                        f"eta {eta // 3600}:{eta % 3600 // 60:02d}:{eta % 60:02d}",
                        flush=True,
                    )
                if step == total or step // req.log_every != started // req.log_every:
                    # s/sample over the samples since the last log line, so a
                    # recompile stays in the window it landed in. The first batch
                    # (compile) has none.
                    now = time.time()
                    if step > step_window:
                        per_step = (now - t_window) / (step - step_window)
                        t_window, step_window = now, step
                    rate = f" {per_step:.3f}s/sample" if per_step is not None else ""
                    if align is not None:
                        rate += f" repa {aligns[-1]:.4f}"
                    print(
                        f"  step {step}/{total} loss {value:.4f} |g| {grad_norm:.3f} "
                        f"sigma {mean_sigma:.3f} lr {lr_sched.get_last_lr()[0]:.2e} "
                        f"batch {len(batch)} "
                        f"peak {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
                        f"{rate}",
                        flush=True,
                    )
            if not losses:
                break
            mean = sum(losses) / len(losses)
            history.append(
                {"epoch": epoch + 1, "loss": mean, "seconds": time.time() - t_epoch}
                | ({"repa": sum(aligns) / len(aligns)} if aligns else {})
            )
            repa_mean = f" repa {sum(aligns) / len(aligns):.4f}" if aligns else ""
            print(
                f"epoch {epoch + 1}/{epochs}: loss {mean:.4f}{repa_mean} {time.time() - t_epoch:.0f}s "
                f"peak {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB",
                flush=True,
            )
            if (
                req.save_every_epochs
                and (epoch + 1) % req.save_every_epochs == 0
                and epoch + 1 < epochs
            ):
                saved = save(f"e{epoch + 1}")
                sink.ckpt(global_step=step, path=str(saved))
                print(f"  saved {saved}", flush=True)
            next_epoch = epochs if step >= total else epoch + 1
            save_state(next_epoch)
            print(f"  resume state {state_path}", flush=True)

        final = save()
        save_state(next_epoch)
        sink.ckpt(global_step=step, path=str(final))
        seconds = time.time() - t_start
        summary = {
            "output": str(final),
            "request": {k: getattr(req, k) for k in req.__dataclass_fields__},
            "fit": fit,
            "steps": step,
            "plain_steps": kinds["plain"],
            "variant_steps": kinds["variant"],
            "dropout_steps": kinds["empty"],
            "minutes": seconds / 60,
            "s_per_step": seconds / max(1, step - 1),
            "samples_per_s": step / max(seconds, 1e-9),
            "batch_size": req.batch_size,
            "token_batch": req.token_batch,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
            "history": history,
            "losses": losses_all,
        }
        report = out_path.with_suffix(".json")
        report.write_text(json.dumps(summary, indent=2))
        print(f"wrote {final} and {report} — {seconds / 60:.1f} min", flush=True)
    if attached is not None:
        attached.detach()
    return final
