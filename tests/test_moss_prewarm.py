"""MOSS kernel warm-up: cache-length enumeration, problem keys and the busy guard.

Everything here runs on fakes -- the plan is pure Python, so no CUDA and no Triton
is needed (tests/conftest.py hides the card for the whole suite anyway).
"""

from __future__ import annotations

import importlib.util
import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.moss import prewarm_kernels as prewarm


@pytest.mark.parametrize("capability", [(8, 0), (8, 9), (9, 0), (12, 0)])
def test_runtime_bf16_accepts_native_cards(capability):
    fake_torch = SimpleNamespace(
        bfloat16="bf16",
        cuda=SimpleNamespace(get_device_capability=lambda device: capability),
    )
    assert prewarm._runtime_dtype("bf16", "cuda", fake_torch) == "bf16"


@pytest.mark.parametrize("capability", [(7, 0), (7, 5)])
def test_runtime_bf16_rejects_older_cards_but_allows_fp16(capability):
    fake_torch = SimpleNamespace(
        float16="fp16",
        cuda=SimpleNamespace(get_device_capability=lambda device: capability),
    )
    with pytest.raises(SystemExit, match="bf16 needs sm80"):
        prewarm._runtime_dtype("bf16", "cuda", fake_torch)
    assert prewarm._runtime_dtype("fp16", "cuda", fake_torch) == "fp16"


def _write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, tuple[int, ...]]],
    metadata: dict[str, str] | None = None,
) -> None:
    """A header-only safetensors file: the reader never touches tensor data."""
    header: dict[str, object] = {}
    if metadata:
        header["__metadata__"] = metadata
    for name, (dtype, shape) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [0, 0],
        }
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


def _fake_cache(tmp_path: Path) -> Path:
    """Three complete pairs + one latent without text embeddings.

    a: 30x40 latent grid -> 1200 image tokens, 300 text tokens   (joint 1500)
    b: 20x20 -> 400 image, 250 text (plain beats embeds; variants 50+100) (650)
    c: 10x10 -> 100 image, 120 text, 8x6 reference -> 48 ref      (268)
    orphan: a latent with no TE file, must be reported and skipped.
    """
    cache = tmp_path / "moss"
    cache.mkdir()
    _write_safetensors(
        cache / "a_64x64.moss_latent.safetensors",
        {"latent": ("F32", (128, 30, 40))},
    )
    _write_safetensors(
        cache / "a.moss_te.safetensors", {"embeds": ("BF16", (300, 4096))}
    )
    _write_safetensors(
        cache / "b_64x64.moss_latent.safetensors",
        {"latent": ("F32", (128, 20, 20))},
    )
    _write_safetensors(
        cache / "b.moss_te.safetensors",
        {
            "embeds": ("BF16", (200, 4096)),
            "embeds_plain": ("BF16", (250, 4096)),
            "embeds_v1": ("BF16", (100, 4096)),
        },
        metadata={"variant_prefix": "50"},
    )
    _write_safetensors(
        cache / "c_64x64.moss_latent.safetensors",
        {"latent": ("F32", (128, 10, 10))},
    )
    _write_safetensors(
        cache / "c.moss_te.safetensors", {"embeds": ("BF16", (120, 4096))}
    )
    _write_safetensors(
        cache / "c.moss_ref.safetensors", {"latent": ("F32", (128, 8, 6))}
    )
    _write_safetensors(
        cache / "orphan_64x64.moss_latent.safetensors",
        {"latent": ("F32", (128, 10, 10))},
    )
    return cache


def _plan(cache: Path, **overrides):
    samples, warnings = prewarm.read_cache_samples(cache)
    args = {
        "cache_dir": cache,
        "samples": samples,
        "geometry": prewarm.MOSS_IMAGE_GEOMETRY,
        "geometry_source": "test",
        "precision": "fp16",
        "batch_size": 1,
        "from_cache": False,
    }
    args.update(overrides)
    return prewarm.build_plan(**args), samples, warnings


# --------------------------------------------------------------------------- #
# Cache headers
# --------------------------------------------------------------------------- #
def test_read_cache_samples_uses_headers(tmp_path):
    cache = _fake_cache(tmp_path)
    samples, warnings = prewarm.read_cache_samples(cache)

    assert [sample.stem for sample in samples] == ["a", "b", "c"]
    by_stem = {sample.stem: sample for sample in samples}
    # Image tokens come from the latent header, not from the file name (64x64).
    assert by_stem["a"].image_tokens == 1200
    assert by_stem["a"].text_tokens == 300
    assert by_stem["b"].text_tokens == 250  # embeds_plain is longest
    assert by_stem["c"].ref_tokens == 48
    assert by_stem["c"].stream_tokens(include_ref=True) == 268
    assert by_stem["c"].stream_tokens(include_ref=False) == 220
    assert len(warnings) == 1 and "orphan" in warnings[0]


def test_read_cache_samples_newest_latent_wins(tmp_path):
    cache = _fake_cache(tmp_path)
    stale = cache / "a_64x64.moss_latent.safetensors"
    fresh = cache / "a_999x999.moss_latent.safetensors"
    _write_safetensors(fresh, {"latent": ("F32", (128, 16, 16))})
    stale.touch()
    os.utime(stale, (1, 1))
    samples, _ = prewarm.read_cache_samples(cache)
    assert {sample.stem: sample.image_tokens for sample in samples}["a"] == 256


def test_read_header_rejects_garbage(tmp_path):
    path = tmp_path / "broken.moss_te.safetensors"
    path.write_bytes(b"nope")
    with pytest.raises(ValueError, match="header"):
        prewarm.read_header(path)


# --------------------------------------------------------------------------- #
# Plan enumeration
# --------------------------------------------------------------------------- #
def test_plan_default_buckets(tmp_path):
    cache = _fake_cache(tmp_path)
    plan, _, _ = _plan(cache)

    # lengths reported for the power-of-two cascade up to the longest joint
    assert plan.lengths == (64, 128, 256, 268, 650, 1500)
    # batch 1 runs torch SDPA, so no attention_varlen table is warmed
    assert not [p for p in plan.problems if p.op == "attention_varlen"]

    # blocks: one run per (op, row bucket); 650 and 268 share the 1024 bucket
    blocks = [p for p in plan.problems if p.op != "attention_varlen"]
    assert len(blocks) == 4 * 2 * 2  # 4 ops x 2 buckets x fwd+bwd
    keys = {(p.kernel, p.key) for p in blocks}
    assert ("triton_rms_norm_fwd", "d3072-torch.float16-TrueTrueTrue") in keys
    assert (
        "triton_rms_norm_bwd",
        "d3072-torch.float16-TrueTrueTrue-False-n2048",
    ) in keys
    assert ("triton_qk_rope_bwd", "dh128-h24-torch.float16-False") in keys
    assert ("triton_situ_fwd", "f8192-torch.float16") in keys
    assert (
        "triton_gated_residual_bwd",
        "d3072-torch.float16-False-n2048",
    ) in keys
    assert len(plan.runs) == 4 * 2


def test_plan_from_cache_uses_observed_lengths_only(tmp_path):
    cache = _fake_cache(tmp_path)
    plan, _, _ = _plan(cache, from_cache=True)

    # only the observed lengths: 268 -> L512, 650 -> L1024, 1500 -> L2048
    assert plan.lengths == (268, 650, 1500)
    assert len(plan.runs) == 4 * 2  # blocks only at batch 1


def test_plan_max_len_drops_long_samples(tmp_path):
    cache = _fake_cache(tmp_path)
    plan, _, _ = _plan(cache, max_len=1000)

    assert plan.skipped == (("a", 1500),)
    assert [sample.stem for sample in plan.samples] == ["b", "c"]
    assert plan.lengths == (64, 128, 256, 268, 650)


def test_plan_max_len_rejects_everything(tmp_path):
    cache = _fake_cache(tmp_path)
    with pytest.raises(ValueError, match="no sample fits"):
        _plan(cache, max_len=100)


def test_plan_packed_refuses_reference_latents(tmp_path):
    """`train.py` exits on refs at batch > 1, so the plan must not warm them."""
    cache = _fake_cache(tmp_path)
    with pytest.raises(ValueError, match="reference latents need --batch-size 1"):
        _plan(cache, batch_size=8)


def test_plan_packed_segment_counts_follow_make_batches(tmp_path):
    """Segments are the batch sizes `make_batches` builds: a full batch + tail."""
    from dataclasses import replace

    cache = _fake_cache(tmp_path)
    samples, _ = prewarm.read_cache_samples(cache)
    ref_free = [replace(sample, ref_tokens=0) for sample in samples]
    assert prewarm.reachable_segments(3, 1) == [1]
    assert prewarm.reachable_segments(3, 2) == [1, 2]
    assert prewarm.reachable_segments(4, 2) == [2]
    assert prewarm.reachable_segments(1, 8) == [1]
    assert prewarm.reachable_segments(2, 8) == [2]
    # a token cap ends a batch early; the warmed set is its upper bound
    assert prewarm.reachable_segments(8, 8, token_batch=250, longest=100) == [1, 2]

    plan = prewarm.build_plan(
        cache,
        ref_free,
        prewarm.MOSS_IMAGE_GEOMETRY,
        "test",
        "fp16",
        2,
        False,
    )
    attention = [p for p in plan.problems if p.op == "attention_varlen"]
    # both segment counts occur, so both keys must be warmed at every length
    assert {p.key for p in attention} == {
        f"torch.float16-h24-d128-L{bucket}-S{segments}"
        for bucket in (64, 128, 256, 512, 1024, 2048)
        for segments in (1, 2)
    }
    assert {p.kernel for p in attention} == {
        "attn_varlen_impl_triton",
        "attn_varlen_fwd_triton",
        "attn_varlen_dkdv_triton",
        "attn_varlen_dq_triton",
        "attn_varlen_fused_triton",
        "attn_varlen_bwd_strategy_triton",
    }
    assert any("fp16 lane" in note for note in plan.notes)


def test_reachable_segments_match_train_make_batches():
    """The planner and the trainer must agree on how many segments a batch has."""
    from library.moss.train import make_batches

    class Sample:
        n_img = n_ref = n_text = 1

    def built(count, batch_size, token_batch):
        return sorted(
            {
                len(batch)
                for batch in make_batches([Sample()] * count, batch_size, token_batch)
            }
        )

    for count in range(1, 13):
        for batch_size in (1, 2, 3, 4, 8):
            assert prewarm.reachable_segments(count, batch_size) == built(
                count, batch_size, 0
            )
    # a token cap can only shrink batches, so its counts are a superset
    for count in range(2, 13):
        for batch_size in (2, 4, 8):
            planned = set(prewarm.reachable_segments(count, batch_size, 400, 1, 1))
            assert {
                prewarm._segment_bucket(size) for size in built(count, batch_size, 400)
            } <= planned


def test_plan_bf16_keys(tmp_path):
    cache = _fake_cache(tmp_path)
    plan, _, _ = _plan(cache, precision="bf16")
    assert plan.dtype == "torch.bfloat16"
    assert (
        "triton_rms_norm_fwd",
        "d3072-torch.bfloat16-TrueTrueTrue",
    ) in {(p.kernel, p.key) for p in plan.problems}


def test_bucket_helpers_match_vendor(tmp_path):
    path = (
        Path(prewarm.__file__).resolve().parents[2]
        / "library/moss/vendor/kohakumoss_kernels/tuning/buckets.py"
    )
    spec = importlib.util.spec_from_file_location("vendor_buckets", path)
    vendor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vendor)

    for n in (1, 63, 64, 65, 300, 1500, 4998, 8192):
        for floor in (1, 64, 256, 1024):
            assert prewarm._pow2(n, floor) == vendor.pow2(n, floor)
    for segments in range(1, 20):
        assert prewarm._segment_bucket(segments) == vendor.segment_bucket(segments)
    for top in (64, 100, 300, 1500, 4998, 8192):
        assert prewarm._pow2_buckets(top, 64) == list(
            vendor.pow2_buckets(top, floor=64)
        )


# --------------------------------------------------------------------------- #
# Busy guard
# --------------------------------------------------------------------------- #
def _runner(stdout, returncode=0, stderr=""):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    run.calls = calls
    return run


def test_busy_guard_allows_a_roomy_card():
    run = _runner("12235, 16384\n")
    memory = prewarm.check_gpu_free(3000, False, runner=run)
    assert memory == prewarm.GpuMemory(used_mib=12235, total_mib=16384)
    assert memory.free_mib == 4149
    command, kwargs = run.calls[0]
    assert command == [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    assert kwargs["capture_output"] is True and kwargs["text"] is True


def test_busy_guard_refuses_a_full_card():
    run = _runner("15000, 16384\n")
    with pytest.raises(SystemExit, match="GPU is busy"):
        prewarm.check_gpu_free(3000, False, runner=run)


def test_busy_guard_allow_busy_overrides():
    run = _runner("15000, 16384\n")
    memory = prewarm.check_gpu_free(3000, True, runner=run)
    assert memory.free_mib == 1384


def test_busy_guard_uses_the_first_gpu_row():
    run = _runner("100, 16000\n9000, 16000\n")
    assert prewarm.query_gpu_memory(run).used_mib == 100


@pytest.mark.parametrize(
    "stdout, stderr, match",
    [
        ("", "", "no GPU rows"),
        ("100, 200, 300", "", "expected 'used, total'"),
        ("abc, 16384", "", "as MiB"),
        ("100, 16384", "driver exploded", "nvidia-smi failed"),
    ],
)
def test_busy_guard_rejects_unreadable_output(stdout, stderr, match):
    code = 1 if stderr else 0
    run = _runner(stdout, returncode=code, stderr=stderr)
    with pytest.raises(ValueError, match=match):
        prewarm.query_gpu_memory(run)


# --------------------------------------------------------------------------- #
# CLI wiring
# --------------------------------------------------------------------------- #
def test_main_dry_run_needs_no_gpu(tmp_path, monkeypatch, capsys):
    cache = _fake_cache(tmp_path)

    def bomb(*args, **kwargs):
        raise AssertionError("--dry-run must not touch the GPU")

    monkeypatch.setattr(prewarm, "check_gpu_free", bomb)
    monkeypatch.setenv("KOHAKUMOSS_KERNEL_CACHE", str(tmp_path / "kernel-cache"))
    assert prewarm.main(["--cache", str(cache), "--dry-run", "--json-only"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["dry_run"] is True
    assert summary["counts"]["problems"] == 16
    assert summary["counts"]["runs"] == 8
    assert summary["counts"]["would_tune"] == 16
    # the vendor's real tables live elsewhere; the fake cache has no card dir
    assert summary["tuning_cache"]["cards"] == []


def test_main_checks_the_gpu_before_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(
        prewarm,
        "check_gpu_free",
        lambda *args, **kwargs: (_ for _ in ()).throw(SystemExit("GPU is busy")),
    )
    with pytest.raises(SystemExit, match="GPU is busy"):
        prewarm.main(["--cache", str(tmp_path / "missing")])


def test_dry_run_as_a_module(tmp_path, repo_root):
    cache = _fake_cache(tmp_path)
    env = {
        "CUDA_VISIBLE_DEVICES": "",
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path),
        "KOHAKUMOSS_KERNEL_CACHE": str(tmp_path / "kernel-cache"),
    }
    proc = subprocess.run(
        check=False,
        args=[
            sys.executable,
            "-m",
            "scripts.moss.prewarm_kernels",
            "--cache",
            str(cache),
            "--dry-run",
            "--json-only",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["counts"]["problems"] == 16
    assert summary["counts"]["would_tune"] == 16
    assert summary["lengths"]["joint_max"] == 1500
