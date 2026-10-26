import pytest
import torch

from library.moss.requests import TuneSwapRequest
from library.moss.train import budget_for_free, step_reserve_gb
from library.moss.tune import flags, pick


def test_budget_for_free_inverts_reserve():
    joint = 4396
    for free in (5.0, 7.24, 9.64, 13.0):
        budget = budget_for_free(free, joint)
        assert step_reserve_gb(joint, grad_checkpointing=False, budget=budget) <= free
        if budget < 1.0:
            over = round(budget + 0.05, 6)
            assert step_reserve_gb(joint, grad_checkpointing=False, budget=over) > free


def test_budget_for_free_measured_points():
    # 2026-10-17 arms at 4096 + 300 tokens: swap 24 left 9.64 GB free before
    # training and ran budget 0.2 with 2.26 GB to spare.
    assert budget_for_free(9.64, 4396) == pytest.approx(0.2)
    assert budget_for_free(3.0, 4396) == 0.0
    assert budget_for_free(100.0, 4396) == 1.0


def test_pick_respects_margin():
    arms = [
        {
            "blocks_to_swap": 0,
            "grad_checkpointing": True,
            "budget": 1.0,
            "s_per_step": 2.09,
            "free_gb": 1.6,
        },
        {
            "blocks_to_swap": 24,
            "grad_checkpointing": False,
            "budget": 0.2,
            "s_per_step": 1.78,
            "free_gb": 2.26,
        },
        {
            "blocks_to_swap": 28,
            "grad_checkpointing": False,
            "budget": 0.25,
            "s_per_step": 1.73,
            "free_gb": 0.6,
        },
        {"blocks_to_swap": 4, "grad_checkpointing": False, "skipped": "does not fit"},
    ]
    chosen = pick(arms, margin_gb=1.0)
    assert chosen["blocks_to_swap"] == 24
    # 28 is faster but leaves too little; with room, 1.73 vs 1.78 is not a tie.
    arms[2]["free_gb"] = 2.0
    assert pick(arms, margin_gb=1.0)["blocks_to_swap"] == 28
    assert pick(arms, margin_gb=1.0, tolerance=0.05)["blocks_to_swap"] == 24
    arms[2]["free_gb"] = 0.6
    assert flags(chosen) == "--blocks_to_swap 24 --activation_memory_budget 0.2"
    assert flags(arms[0]) == "--grad_checkpointing --blocks_to_swap 0"
    assert flags(chosen, 4, 12000) == (
        "--blocks_to_swap 24 --activation_memory_budget 0.2 "
        "--batch_size 4 --token_batch 12000"
    )
    assert pick(arms, margin_gb=5.0) is None


def test_request_round_trip():
    req = TuneSwapRequest(sizes="1024x1024:300", swap_grid="16,24", reference=False)
    assert TuneSwapRequest.from_argv(req.to_argv()) == req


def test_an_oom_replaces_the_model_before_the_next_budget(monkeypatch):
    """An aborted forward can leave swapped blocks on the CPU; a plain retry then
    reads them on the wrong device, so the arm must place the model again."""
    from types import SimpleNamespace

    from library.moss import tune

    class Attached:
        def __init__(self, tag: int):
            self.tag = tag
            self.detached = False

        def detach(self) -> None:
            self.detached = True

    placements: list[int] = []

    def fake_place(_model, _device, swap, **_kwargs):
        placements.append(swap)
        return Attached(len(placements))

    monkeypatch.setattr(tune, "place", fake_place)
    monkeypatch.setattr(tune, "empty_cache", lambda: None)
    monkeypatch.setattr(tune, "free_vram_gb", lambda: 10.0)
    monkeypatch.setattr(tune, "budget_for_free", lambda *_args, **_kwargs: 0.5)

    bench = tune._Bench.__new__(tune._Bench)
    bench.transformer = SimpleNamespace(layers=[], to=lambda *_a, **_k: None)
    bench.device = "cuda"
    bench.attached = None
    bench.params = []
    bench.joint = 4000
    bench.fp32_adapters = False
    bench.req = TuneSwapRequest(steps=1)

    attempts: list[float] = []

    def fake_time(budget, _grad_checkpointing):
        attempts.append(budget)
        if len(attempts) == 1:
            raise torch.OutOfMemoryError("fixture: out of memory")
        return {"s_per_step": 1.0, "peak_allocated_gb": 1.0, "peak_reserved_gb": 1.0}

    monkeypatch.setattr(bench, "_time", fake_time, raising=False)

    result = bench.arm(8)
    assert result["blocks_to_swap"] == 8
    assert result["oom_budgets"] == [0.5]
    assert result["s_per_step"] == 1.0
    # Placed once for the arm, then again after the OOM; the retry ran lower.
    assert placements == [8, 8]
    assert attempts[1] < attempts[0]
    assert bench.attached is None  # _restore took the last placement back


def test_tune_request_carries_the_compile_switch():
    """The tuning form can turn compilation off, like the training form can."""
    req = TuneSwapRequest(compile=False)
    assert req.to_argv() == ["--no-compile"]
    assert TuneSwapRequest.from_argv(req.to_argv()) == req
    assert TuneSwapRequest().compile is True
