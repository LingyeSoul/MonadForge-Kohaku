"""MOSS sidecars; independent typed requests and model family."""

from scripts.tasks._common import PY, queue_command, run


def _run_job(command: str, module: str, extra: list[str]) -> None:
    argv = ["-m", module, *[arg for arg in extra if arg != "--queue"]]
    if "--queue" in extra:
        queue_command(command, argv, stall_timeout=900.0)
    else:
        run([PY, *argv])


def cmd_cache(extra: list[str]) -> None:
    """Cache MOSS text embeddings and FLUX.2 VAE latents; --queue uses daemon."""
    _run_job("moss-cache", "scripts.moss.cache", extra)


def cmd_train(extra: list[str]) -> None:
    """Train MOSS LoRA/LoKr/LoHa from cache; --queue uses daemon."""
    _run_job("moss-train", "scripts.moss.train", extra)


def cmd_generate(extra: list[str]) -> None:
    """Generate with MOSS Base/SFT; --queue uses daemon."""
    _run_job("moss-generate", "scripts.moss.generate", extra)


def cmd_tune(extra: list[str]) -> None:
    """Explicitly run the MOSS swap/activation training benchmark; --queue uses daemon."""
    _run_job("moss-tune", "scripts.moss.tune_swap", extra)


def cmd_cache_train(extra: list[str]) -> None:
    """Cache then train in one daemon job; cache failure prevents training."""
    _run_job("moss-cache-train", "scripts.moss.workflow", extra)


def cmd_prewarm(extra: list[str]) -> None:
    """Tune the vendored MOSS kernels on the cache's real token lengths; --dry-run lists the plan."""
    _run_job("moss-prewarm", "scripts.moss.prewarm_kernels", extra)


def cmd_report(extra: list[str]) -> None:
    """Summarize a MOSS run from its progress.jsonl plus trainer log."""
    _run_job("moss-report", "scripts.moss.train_report", extra)
