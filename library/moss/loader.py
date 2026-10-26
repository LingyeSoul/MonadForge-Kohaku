"""MOSS-Image loading under a 16 GB budget: TE → encode → free → DiT → (DiT off) → VAE.

    text_encoder  Qwen3.5-9B (+ unused vision tower)  16.8 GB  <- larger than the card
    transformer   5B, 42 single-stream blocks         10.2 GB  (9.7 GiB resident, bf16)
    vae           FLUX.2 AutoencoderKLFlux2            0.25 GB (decoded in fp32)

The TE's 248k-row token embedding (1.9 GB) stays on the CPU, and the activation
reserve is sized to the longest prompt, so at caption lengths (< ~1k tokens) all
32 decoder layers (12.9 GB) fit on the card. Layers that still do not fit are
paged in whole, one at a time, from pinned memory; the encode is PCIe-bound while
any are. Leaf-level offload hooks break on Qwen3.5's gated-DeltaNet layers, which read
``conv1d.weight`` functionally instead of calling the module. The quantized TEs
(``precision="int8"``, the default request setting, W8A8 ``library/moss/int8.py``;
``precision="fp8"``, ``library/moss/fp8.py``) put the decoder layers at 6.5 GiB,
so none page at any prompt length.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from accelerate import init_empty_weights

from library.env import resolve_under_home
from library.moss.model import MOSSImageTransformer
from library.moss.requests import variant_dir
from library.runtime.device import clean_memory as empty_cache
from library.runtime.device import free_vram_gb

MAX_SEQUENCE_LENGTH = 4096
TE_PRECISIONS = ("bf16", "fp8", "int8")

logger = logging.getLogger(__name__)

# MONADFORGE_MOSS_COMPILE_PROBE=0 trusts the stack without testing it.
PROBE_ENV = "MONADFORGE_MOSS_COMPILE_PROBE"
_OFF = frozenset({"0", "false", "off", "no"})
_PROBE: bool | None = None


def te_reserve_gb(max_tokens: int) -> float:
    """VRAM left free for the TE's activations: 2.5 GB at 4096 tokens, linear in
    length above a fixed floor."""
    return 0.75 + 1.75 * max_tokens / MAX_SEQUENCE_LENGTH


def _embed_on_cpu(embed: torch.nn.Module, device) -> None:
    """Look tokens up on the CPU and hand the rows to ``device``."""
    embed.register_forward_pre_hook(lambda m, a: (a[0].cpu(),))
    embed.register_forward_hook(lambda m, a, o: o.to(device))


def _page_layer(layer: torch.nn.Module, device) -> None:
    """Copy ``layer`` onto ``device`` for its own forward from a pinned CPU copy,
    and drop the device copy after it (no copy back: the weights are unchanged)."""
    tensors = [*layer.parameters(), *layer.buffers()]
    host = [t.data.pin_memory() for t in tensors]
    for t, h in zip(tensors, host):
        t.data = h

    def load(_m, _a):
        for t, h in zip(tensors, host):
            t.data = h.to(device, non_blocking=True)

    def drop(_m, _a, _o):
        for t, h in zip(tensors, host):
            t.data = h

    layer.register_forward_pre_hook(load)
    layer.register_forward_hook(drop)


def offload_by_layer(te, device, reserve_gb):
    """Non-layer modules but the token embedding, and as many leading decoder layers
    as fit in free VRAM minus ``reserve_gb``, stay resident; the rest page in for
    their own forward. The vision tower stays on the CPU (unused for T2I)."""
    lm = te.language_model
    for name, child in lm.named_children():
        if name == "embed_tokens":
            _embed_on_cpu(child, device)
        elif name != "layers":
            child.to(device)
    budget = free_vram_gb(device.index or 0) - reserve_gb
    paged = 0
    for layer in lm.layers:
        size = sum(p.numel() * p.element_size() for p in layer.parameters()) / 1024**3
        if size <= budget:
            layer.to(device)
            budget -= size
            continue
        paged += 1
        _page_layer(layer, device)
    return paged


def load_tokenizer(model_dir: str | Path | None):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(variant_dir(model_dir, "base") / "tokenizer")


def load_text_encoder(
    model_dir: str | Path | None,
    device="cuda",
    max_tokens=MAX_SEQUENCE_LENGTH,
    log=print,
    tok=None,
    precision="bf16",
):
    """``(tokenizer, text_encoder)`` on ``device``, layers that do not fit paged in
    per forward. ``max_tokens`` is the longest prompt the caller will encode (it
    sizes the activation reserve). TE and tokenizer are byte-identical between
    Base and SFT; Base's are used. ``precision="int8"`` / ``"fp8"`` loads the
    catalog's quantized file (``library/moss/fp8.py``, fetched on first use)
    instead of the bf16 shards; the tokenizer still comes from Base."""
    from transformers import AutoModel

    if precision not in TE_PRECISIONS:
        raise ValueError(
            f"text encoder precision {precision!r}, not one of {TE_PRECISIONS}"
        )
    tok = tok or load_tokenizer(model_dir)
    if precision != "bf16":
        from library.moss.fp8 import load_fp8_text_encoder

        te = load_fp8_text_encoder(log=log, precision=precision)
    else:
        root = variant_dir(model_dir, "base")
        te = AutoModel.from_pretrained(
            root / "text_encoder", dtype=torch.bfloat16
        ).eval()
    reserve = te_reserve_gb(max_tokens)
    paged = offload_by_layer(te, torch.device(device), reserve)
    if log:
        log(
            f"[te] {precision}: {len(te.language_model.layers) - paged} decoder layers resident, "
            f"{paged} paged (reserve {reserve:.2f} GB for {max_tokens} tokens)"
        )
    return tok, te


def prompt_ids(tok, text: str, max_length=MAX_SEQUENCE_LENGTH):
    """The pipeline's tokenisation of one prompt (chat template, no padding)."""
    formatted = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return tok(formatted, return_tensors="pt", truncation=True, max_length=max_length)


@torch.no_grad()
def encode_prompt(tok, te, text: str, device="cuda", max_length=MAX_SEQUENCE_LENGTH):
    """One prompt → ``last_hidden_state`` ``(S, 4096)`` bf16 at its true length.
    Exactly ``MOSSImagePipeline.encode_prompt``: chat template with
    ``add_generation_prompt=True, enable_thinking=True``, one prompt per forward
    (no padding: the GDN layers misbehave on it)."""
    inputs = prompt_ids(tok, text, max_length).to(device)
    from library.moss import text_kernels

    if text_kernels.usable(te, device):
        return text_kernels.encode(te, inputs["input_ids"])
    return te(**inputs, use_cache=False, return_dict=True).last_hidden_state[0]


def encode_prompts(
    model_dir,
    texts,
    device="cuda",
    max_length=MAX_SEQUENCE_LENGTH,
    log=print,
    precision="bf16",
):
    """Load the TE, encode each distinct text once, free it → ``{text: (S,4096) cpu}``."""
    texts = list(dict.fromkeys(texts))
    tok = load_tokenizer(model_dir)
    longest = max(prompt_ids(tok, t, max_length).input_ids.shape[1] for t in texts)
    tok, te = load_text_encoder(
        model_dir, device, longest, log=log, tok=tok, precision=precision
    )
    out = {}
    for text in texts:
        out[text] = encode_prompt(tok, te, text, device, max_length).cpu()
        if log:
            log(f"[te] {out[text].shape[0]:5d} tokens | {text[:60]!r}")
    del tok, te
    empty_cache()
    return out


def pad_embeds(rows, device=None):
    """List of ``(S_i, D)`` → ``embeds (B, S_max, D)``, bool ``mask (B, S_max)``."""
    length = max(r.shape[0] for r in rows)
    embeds = rows[0].new_zeros(len(rows), length, rows[0].shape[-1])
    mask = torch.zeros(len(rows), length, dtype=torch.bool)
    for i, r in enumerate(rows):
        embeds[i, : r.shape[0]] = r
        mask[i, : r.shape[0]] = True
    if device is not None:
        embeds, mask = embeds.to(device), mask.to(device)
    return embeds, mask


def native_bf16(device=None) -> bool:
    """Whether this CUDA device has native bf16 arithmetic (sm_80+/Ampere).

    ``torch.cuda.is_bf16_supported()`` is not the question: with its default
    ``including_emulation=True`` it returns True on a V100, where a bf16 tensor
    can be *created* but every bf16 GEMM is unsupported (cuBLAS wants compute
    capability 8.0+). Ask the capability instead — and of ``device``, which
    ``is_bf16_supported`` cannot take.
    """
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 8


def resolve_compute_dtype(precision="auto", device=None) -> torch.dtype:
    """``auto | bf16 | fp16`` → the DiT's compute dtype for this device.

    ``auto`` is bf16 on a card with native bf16 and fp16 elsewhere (V100 sm_70,
    T4 sm_75), as upstream KohakuMossTrainer resolves ``PRECISION='auto'``. An
    explicit ``bf16`` on a pre-Ampere card fails instead of running a bf16 GEMM
    that cuBLAS rejects. Without a GPU, ``auto`` keeps bf16 — the historical
    default, and nothing here runs a GEMM on a tensor core.
    """
    if precision == "auto":
        if not torch.cuda.is_available():
            return torch.bfloat16
        return torch.bfloat16 if native_bf16(device) else torch.float16
    if precision == "bf16":
        if torch.cuda.is_available() and not native_bf16(device):
            major, minor = torch.cuda.get_device_capability(device)
            raise ValueError(
                f"precision='bf16' needs a card with native bf16 (sm_80+), and "
                f"this GPU is sm_{major}{minor}: use precision='fp16' or 'auto'"
            )
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    raise ValueError(f"precision must be auto, bf16 or fp16, not {precision!r}")


def _load_folder(model_dir, variant, dtype):
    """The published diffusers `transformer/` folder of one variant."""
    tr = MOSSImageTransformer.from_pretrained(
        variant_dir(model_dir, variant) / "transformer", torch_dtype=dtype
    ).eval()
    tr.base_file = None
    tr.base_fold = None
    return tr


def _load_single_file(dit, dtype):
    """A ComfyUI single file: header check, inferred arch, one tensor at a time.

    The file holds only the DiT (fused `proj.qkv`, optionally a key prefix and an
    fp16-fix fold in its weights); `library.moss.checkpoint` splits, strips and
    un-folds it, so the result is interchangeable with the diffusers folder and
    the fp16 guards below apply exactly as they do there. The host peak is the
    model plus one weight: each tensor is assigned as it is read.
    """
    from library.moss import checkpoint

    path = resolve_under_home(dit)
    if path.is_dir():
        raise ValueError(
            f"dit={dit!r} is a folder: a single-file DiT is one .safetensors "
            "(keep --model/--model_dir for a diffusers folder)"
        )
    if not path.is_file():
        raise ValueError(f"dit={dit!r} is not a file")
    checkpoint.assert_compute_weights(path)
    fold = checkpoint.fold_record(path)
    cfg = checkpoint.arch_config(path)
    with init_empty_weights():
        tr = MOSSImageTransformer(**cfg)
    plan = checkpoint.inspect(
        path, {key: tuple(t.shape) for key, t in tr.state_dict().items()}
    )
    for name, tensor in checkpoint.read_state(path, plan, dtype, fold):
        checkpoint.assign(tr, name, tensor)
    tr.eval().requires_grad_(False)
    tr.base_file = str(path)
    tr.base_fold = fold
    print(
        f"[dit] {path.name}: {len(plan)} tensors, "
        f"{cfg['num_layers']} blocks x {cfg['dim']}, "
        f"{str(dtype).removeprefix('torch.')}"
        + (
            f", un-folded fp16-fix c={fold['c']:g} from the weights"
            if fold is not None
            else ""
        ),
        flush=True,
    )
    return tr


def load_transformer(
    model_dir,
    variant="base",
    dtype=None,
    device=None,
    quant="",
    fp16_fix=None,
    fp16_fix_c=None,
    dit="",
):
    """The DiT in its compute dtype (``dtype=None``: resolved for this device).

    ``dit`` loads a **ComfyUI single file** (one ``.safetensors`` holding only
    the DiT) instead of ``variant_dir(model_dir, variant) / "transformer"``: fused
    ``proj.qkv`` rows are split into q / k / v, key prefixes stripped, the
    architecture inferred from the file's shapes, and its fp16-fix fold undone on
    load (``library.moss.checkpoint``). The text encoder and VAE still come from
    ``model_dir``.

    fp16 compute turns on the residual-stream guard: the 42-block stream exceeds
    fp16's 65504 ceiling and overflows to NaN without one. ``fp16_fix`` picks it
    (upstream KohakuMossTrainer's ``FP16_FIX``):

    - ``"online"`` (default; ``fp16_fix_c``, default 1/16): fold a scale ``c``
      into the joint and text streams (``enable_stream_scale``). The stream stays
      16-bit, so the fused kernels still run; the function is exact, and nothing
      is written into the weights.
    - ``"fp32_residual"``: the stream is carried in fp32 with fp16 matmuls
      (``enable_fp32_residual``); the residual GEMM degrades to the split path.

    Both must happen before ``compile_blocks`` — dynamo specializes on the flags.
    ``dtype`` other than fp16 (bf16 / fp32) ignores the fix.
    """
    if dtype is None:
        dtype = resolve_compute_dtype("auto", device)
    tr = (
        _load_single_file(dit, dtype)
        if dit
        else _load_folder(model_dir, variant, dtype)
    )
    if dtype == torch.float16:
        from library.moss.requests import DEFAULT_FP16_FIX, DEFAULT_FP16_FIX_C

        fix = DEFAULT_FP16_FIX if fp16_fix is None else fp16_fix
        if fix == "fp32_residual":
            tr.enable_fp32_residual()
        elif fix == "online":
            tr.enable_stream_scale(
                DEFAULT_FP16_FIX_C if fp16_fix_c is None else fp16_fix_c
            )
        else:
            raise ValueError(
                f"fp16_fix must be 'online' or 'fp32_residual', not {fix!r}"
            )
    if device is not None:
        tr = tr.to(device)
    if quant:
        from library.moss.quant import quantize_transformer

        quantize_transformer(tr, quant)
    return tr


def triton_compile_ok() -> bool:
    """Whether ``torch.compile`` can trace this Triton stack (probed once, cached).

    Dynamo derives a Triton kernel's mutated-argument list by generating TTIR from
    the call's example arguments, where a runtime scalar such as a norm ``eps``
    arrives as a Python ``float``. The vendored kernels do ``eps.to(tl.float32)``,
    so on a torch/triton pair that hands scalars through untraced the probe raises
    ``AttributeError``; torch then assumes *every* input is mutated and the
    compiled graph reinterprets a buffer at the wrong dtype (``float != Half``
    inside the first ``mm``). That is a hard crash on step 1 of a fused run, so
    the compatibility is measured rather than assumed.
    """
    global _PROBE
    if _PROBE is not None:
        return _PROBE
    if os.environ.get(PROBE_ENV, "1").strip().lower() in _OFF:
        _PROBE = True
        return True
    if not torch.cuda.is_available():
        _PROBE = True  # nothing to fuse without a card; fuse() declines anyway
        return True
    healthy, transient = _probe_triton_compile()
    # A transient failure (the probe could not run, e.g. OOM) says nothing about
    # this torch/triton pair, so it must not be cached as a permanent verdict.
    if not transient:
        _PROBE = healthy
    return healthy


def _probe_matches(reference: torch.Tensor, got: torch.Tensor) -> bool:
    """Whether a compiled probe reproduced the eager result, dtype included.

    The mismatch this guards against is a *dtype* one (a fp32 buffer reaching the
    fp16 projection), so the dtype is compared before the values: a fp32 answer
    that happens to hold the right numbers is still a broken graph.
    """
    if got.dtype != reference.dtype or got.shape != reference.shape:
        return False
    return bool(torch.allclose(got, reference, atol=2e-2, rtol=2e-2))


def _probe_triton_compile() -> bool:
    """Compile the real fused norm plus a projection, and check what comes out.

    The incompatibility is not a trace-time exception: dynamo *warns* that it
    cannot introspect the kernel and assumes every argument is mutated, so the
    first real step feeds a fp32 buffer to a fp16 GEMM (``expected mat1 and mat2
    to have the same dtype ... float != c10::Half``). A synthetic kernel cannot
    be trusted to reproduce that, so the probe runs the vendored op at a small
    training shape and compares against the eager result. ~5 s, once per process.
    """
    from library.moss import ops

    def forward(x, weight, shift, scale, seg, proj):
        y = ops.rms_norm_mod(x, weight, shift, scale, seg, 1e-6)
        return y @ proj

    rows, dim = 16, 256
    transient = False
    try:
        device = torch.device("cuda")
        x = torch.randn(rows, dim, device=device, dtype=torch.float16)
        weight = torch.ones(dim, device=device, dtype=torch.float16)
        shift = torch.zeros(1, dim, device=device, dtype=torch.float16)
        scale = torch.zeros(1, dim, device=device, dtype=torch.float16)
        seg = torch.zeros(rows, device=device, dtype=torch.int32)
        proj = torch.randn(dim, dim, device=device, dtype=torch.float16)
        args = (x, weight, shift, scale, seg, proj)
        reference = forward(*args)
        got = torch.compile(forward, dynamic=True)(*args)
        healthy = _probe_matches(reference, got)
        reason = "compiled result differs from the eager one" if not healthy else ""
    except torch.cuda.OutOfMemoryError as err:
        # Not a torch/triton incompatibility: the card could not run the probe.
        # Report it, but do not poison the once-per-process cache with a verdict
        # that says nothing about this stack.
        healthy = False
        transient = True
        reason = f"{type(err).__name__}: {str(err).splitlines()[0][:200]}"
    except Exception as err:  # noqa: BLE001 — any failure means do not compile
        healthy = False
        reason = f"{type(err).__name__}: {str(err).splitlines()[0][:200]}"
    if not healthy:
        logger.warning(
            "torch.compile cannot run the fused MOSS Triton kernels correctly on "
            "this torch/triton pair (%s); compiling is skipped and the fused "
            "kernels run eagerly. Set %s=0 to compile anyway.",
            reason,
            PROBE_ENV,
        )
    return healthy, transient


def compile_blocks(blocks) -> None:
    """``torch.compile(dynamic=True)`` on each block's ``forward`` (instance
    attribute, so swap hooks stay outside the graph). One graph for every joint
    length; must run after ``LoRANetwork.apply_to`` / ``load_for_inference``.

    Compiling a *fused* block first checks that dynamo can trace the Triton
    kernels (:func:`triton_compile_ok`); when it cannot, the blocks keep running
    their fused ops eagerly instead of crashing inside the graph. The reference
    branch has no Triton kernel to trace and always compiles.
    """
    if any(getattr(block, "fused", False) for block in blocks) and not (
        triton_compile_ok()
    ):
        return
    limit = max(torch._dynamo.config.recompile_limit, 16)
    torch._dynamo.config.recompile_limit = limit
    # dynamic=True otherwise lifts float attributes (the LoRA's scale ·
    # multiplier) into 0-dim CPU float64 inputs, and Inductor emits a C++ CPU
    # kernel for them, which needs cl.exe on Windows (InvalidCxxCompiler).
    torch._dynamo.config.specialize_float = True
    for block in blocks:
        block.forward = torch.compile(block.forward, dynamic=True)


def load_vae(model_dir, dtype=torch.float32, device=None, compile=False, vae=""):
    """The FLUX.2 VAE in `dtype`: `vae=` reads one ComfyUI / diffusers
    `.safetensors` (either key layout, config inferred; `library.moss.checkpoint`),
    otherwise `<base>/vae/`."""
    if vae:
        from library.moss.checkpoint import load_vae_file

        model = load_vae_file(resolve_under_home(vae), dtype)
    else:
        from diffusers import AutoencoderKLFlux2

        model = AutoencoderKLFlux2.from_pretrained(
            variant_dir(model_dir, "base") / "vae", torch_dtype=dtype
        ).eval()
    if compile:
        model.to(memory_format=torch.channels_last)
        model.decode = torch.compile(model.decode, dynamic=True)
    return model.to(device) if device is not None else model
