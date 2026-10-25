#!/usr/bin/env python3
"""Update MonadForge-Kohaku from GitHub.

Resolves the newest published release tag, falling back to the head commit of
``main`` when the repository has no releases (its state until the first tag
ships), downloads that tarball from LingyeSoul/MonadForge-Kohaku, extracts it to
a temp dir, then merges over the working tree using a 3-way reconciliation
of (baseline / user / new) sha256 hashes:

  - Datasets, outputs, models, caches, .venv: never touched.
  - Configs in configs/methods/, configs/gui-methods/,
    configs/preprocess.toml, configs/presets.toml, configs/sam_mask.yaml,
    configs/clause_vocabulary.yaml: prompt on conflict
    (keep yours / overwrite / backup-and-overwrite / show diff).
  - Code files (library/, scripts/, train.py, etc.) AND configs/base.toml /
    configs/model.toml:
    overwritten silently when unmodified; user-modified versions are copied to
    .anima-update-backups/<timestamp>/ before being overwritten. These files are
    deliberately treated as code — user model paths live under configs/custom/,
    so upstream defaults and shared keys must always reach the user (even under
    --keep-conflicts), with the user's prior copy preserved in the backup dir.

The baseline manifest lives at .anima_release.json. If it doesn't exist
(first run after upgrading from a release that predates this script), every
file that differs from upstream is treated as user-modified — so configs
will prompt, code files will be backed up. Use --yes-overwrite to default
all conflicts to backup-and-overwrite.

After file merge, runs `uv sync` (skip with --no-sync).
"""

from __future__ import annotations

import argparse
import difflib
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

# Non-UTF-8 Windows code pages (cp949, cp1252, …) can't encode the em-dash /
# arrow / ellipsis below — force UTF-8 so the GUI subprocess matches a terminal.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

ROOT = Path(__file__).resolve().parent.parent


class UpdateSourceError(RuntimeError):
    """GitHub could not supply an update target (missing ref, network, API)."""


def _detect_windows_gpu_vendor() -> str | None:
    """Return 'nvidia' or 'amd' from the installed display adapters, or None."""
    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | "
                "ForEach-Object { $_.Name + ' ' + $_.AdapterCompatibility }",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    adapters = result.stdout.lower()
    # Prefer NVIDIA on mixed-vendor systems (iGPU + dGPU), matching install.ps1.
    if "nvidia" in adapters:
        return "nvidia"
    if (
        "amd" in adapters
        or "advanced micro devices" in adapters
        or "radeon" in adapters
    ):
        return "amd"
    return None


def _selected_windows_backend() -> str:
    """Keep the accelerator chosen at install time across release updates."""
    override = os.environ.get("ANIMA_BACKEND", "").strip().lower()
    if override in {"cuda", "rocm"}:
        return override

    marker = ROOT / ".anima_backend"
    if marker.is_file():
        saved = marker.read_text(encoding="ascii").strip().lower()
        if saved in {"cuda", "rocm"}:
            return saved

    # No marker yet: decide from the hardware, NOT from the venv's torch build —
    # an older extra-unaware `uv sync` installed ROCm torch on every Windows
    # machine (GH #92), so trusting `torch.version.hip` would lock NVIDIA users
    # into that state.
    vendor = _detect_windows_gpu_vendor()
    if vendor == "amd":
        return "rocm"
    if vendor == "nvidia":
        return "cuda"

    # No adapter identified (headless/VM): the venv build is the last resort.
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    if python.is_file():
        try:
            result = subprocess.run(
                [str(python), "-c", "import torch; print(bool(torch.version.hip))"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            if result.stdout.strip() == "True":
                return "rocm"
        except (OSError, subprocess.CalledProcessError):
            pass
    return "cuda"


# Opt-in CUDA torch stacks (pyproject `torch-213` / `torch-212` groups); the
# default 2.14 stack needs no flags.
TORCH_STACKS = {"2.12": "torch-212", "2.13": "torch-213", "2.14": None}


def _selected_torch_group() -> str | None:
    """Return the opt-in torch group to sync, or None for the default stack.

    ``ANIMA_TORCH=2.13`` picks one (and is saved to ``.anima_torch`` so later
    updates keep it); otherwise the saved marker is reused.
    """
    marker = ROOT / ".anima_torch"
    choice = os.environ.get("ANIMA_TORCH", "").strip()
    if choice:
        if choice not in TORCH_STACKS:
            print(
                f"  ignoring ANIMA_TORCH={choice!r} "
                f"(expected one of {', '.join(TORCH_STACKS)})"
            )
            return None
        try:
            marker.write_text(choice + "\n", encoding="ascii")
        except OSError:
            pass
    elif marker.is_file():
        choice = marker.read_text(encoding="ascii").strip()
    return TORCH_STACKS.get(choice)


def _uv_sync_command() -> tuple[list[str], str | None]:
    """Return the sync command and the Windows backend it targets (None off-Windows)."""
    torch_group = _selected_torch_group()
    command = ["uv", "sync"]
    # An opt-in stack carries every platform's torch, so it replaces both
    # default groups (torch-214 + cuda-windows), which it conflicts with.
    cuda_command = (
        command + ["--no-default-groups", "--group", torch_group]
        if torch_group
        else command
    )
    if sys.platform != "win32":
        return cuda_command, None
    backend = _selected_windows_backend()
    # cuda-windows is a DEFAULT dependency group (GH #92), so the CUDA stack
    # needs no flags; ROCm swaps the default group out explicitly.
    if backend == "rocm":
        if torch_group:
            print(f"  ROCm ships its own torch; ignoring the {torch_group} choice")
        command.extend(["--no-group", "cuda-windows", "--group", "rocm-windows"])
    else:
        command = cuda_command
    # Persist the decision so later updates (and manual `make update` runs)
    # stop re-deriving it — install.ps1 writes the same marker.
    try:
        (ROOT / ".anima_backend").write_text(backend + "\n", encoding="ascii")
    except OSError:
        pass
    return command, backend


def _verify_windows_backend(backend: str) -> None:
    """Warn loudly when the synced venv's torch build contradicts the backend."""
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    if not python.is_file():
        return
    try:
        result = subprocess.run(
            [
                str(python),
                "-c",
                "import torch; print('rocm' if torch.version.hip else "
                "('cuda' if torch.version.cuda else 'cpu'))",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return
    build = result.stdout.strip()
    if build != backend:
        torch_group = _selected_torch_group()
        if backend == "rocm":
            fix = "uv sync --no-group cuda-windows --group rocm-windows"
        elif torch_group:
            fix = f"uv sync --no-default-groups --group {torch_group}"
        else:
            fix = "uv sync"
        print(
            f"\nWARNING: the installed PyTorch is a {build} build but the "
            f"selected backend is {backend} — training/caching would fall back "
            f"to CPU.\n  Fix: {fix}\n"
            f"  (backend override: set ANIMA_BACKEND=cuda|rocm or edit "
            f".anima_backend)"
        )


REPO = "LingyeSoul/MonadForge-Kohaku"
API = f"https://api.github.com/repos/{REPO}"
MANIFEST_FILE = ROOT / ".anima_release.json"
BACKUP_ROOT = ROOT / ".anima-update-backups"

# Directories never touched by update (user data, caches, env, downloads).
# Matched by leading path segments, so "archive/graft/runtime" matches that
# prefix while leaving the rest of archive/ updatable. "_archive/" is
# preserved wholesale (never overwritten or pruned).
PRESERVE_DIRS: tuple[str, ...] = (
    "_archive",
    "workspace",
    "image_dataset",
    "post_image_dataset",
    "ip-adapter-dataset",
    "easycontrol-dataset",
    "output",
    "models",
    "masks",
    "masks_sam",
    "masks_mit",
    "bench",
    "logs",
    "results",
    ".venv",
    ".git",
    ".claude",
    "test_output",
    "output_temp",
    "workflows",
    "archive/graft/runtime",
    "configs/custom",
    "__pycache__",
    "anima_lora.egg-info",
    ".anima-update-backups",
)
PRESERVE_FILES: tuple[str, ...] = (
    ".env",
    ".anima_release.json",
)

# Files that prompt on conflict instead of silent overwrite (globs relative to
# ROOT). configs/base.toml is not here: it takes the code-file path (see the
# module docstring).
CONFLICT_GLOBS: tuple[str, ...] = (
    "configs/methods/*.toml",
    "configs/gui-methods/*.toml",
    "configs/preprocess.toml",
    "configs/presets.toml",
    "configs/sam_mask.yaml",
    "configs/clause_vocabulary.yaml",
    "configs/datasets/*",
    # Self-contained per-method dirs (EasyControl).
    "configs/easycontrol/*.toml",
)


def _is_preserved(rel: str) -> bool:
    if rel in PRESERVE_FILES:
        return True
    parts = rel.split("/")
    for pres in PRESERVE_DIRS:
        pres_parts = pres.split("/")
        if parts[: len(pres_parts)] == pres_parts:
            return True
    return False


def _is_conflict_path(rel: str) -> bool:
    return any(fnmatch.fnmatchcase(rel, g) for g in CONFLICT_GLOBS)


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _walk_tree(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and not p.is_symlink():
            yield p


def _github_headers() -> dict[str, str]:
    """API headers; GITHUB_TOKEN/GH_TOKEN lifts the 60 requests/hour limit."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "anima-update",
    }
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _github_json(url: str, *, missing_ok: bool) -> dict:
    """GET one GitHub API object; ``{}`` when ``missing_ok`` and it is a 404.

    Transient 5xx/network failures are retried twice with a warning; the last
    error is raised so the CLI/GUI reports the real reason.
    """
    attempts = 3
    last_error = ""
    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(url, headers=_github_headers())
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if missing_ok and e.code == 404:
                return {}
            last_error = f"GitHub API {url} returned {e.code} {e.reason}"
            if e.code < 500 or attempt == attempts:
                raise UpdateSourceError(last_error) from e
        except urllib.error.URLError as e:
            last_error = f"Could not reach GitHub: {e.reason}"
            if attempt == attempts:
                raise UpdateSourceError(last_error) from e
        except ValueError as e:
            raise UpdateSourceError(
                f"GitHub API {url} returned invalid JSON: {e}"
            ) from e
        else:
            if not isinstance(payload, dict):
                raise UpdateSourceError(
                    f"GitHub API {url} returned unexpected JSON (expected an object)"
                )
            return payload
        print(f"  {last_error}; retrying ({attempt}/{attempts})")
        time.sleep(attempt)
    raise UpdateSourceError(last_error)


def _release_target(
    payload: dict, fallback_tag: str | None = None
) -> tuple[str, str, str, str]:
    """Return (tag, tarball_url, notes, "release") for a release payload."""
    tag = payload.get("tag_name") or fallback_tag
    if not tag:
        raise UpdateSourceError("GitHub release payload has no tag_name")
    body = (payload.get("body") or "").strip()
    tarball = f"https://github.com/{REPO}/archive/refs/tags/{tag}.tar.gz"
    return tag, tarball, body, "release"


def _commit_target(query: str, label: str) -> tuple[str, str, str, str]:
    """Return (label, tarball_url, notes, "commit") for a branch or commit sha."""
    payload = _github_json(f"{API}/commits/{query}", missing_ok=False)
    sha = payload.get("sha")
    if not sha:
        raise UpdateSourceError(f"GitHub returned no commit sha for {query!r}")
    commit = payload.get("commit") or {}
    message = (commit.get("message") or "").strip()
    date = (commit.get("committer") or {}).get("date") or ""
    header = f"Latest commit on {label} — {date}" if date else f"Latest commit on {label}"
    body = f"{header}\n\n{message}" if message else header
    short = sha[:7]
    resolved = f"{label}@{short}" if label else short
    return resolved, f"https://github.com/{REPO}/archive/{sha}.tar.gz", body, "commit"


def _resolve_target(version: str | None) -> tuple[str, str, str, str]:
    """Return (label, tarball_url, notes, kind).

    version=None prefers the newest published release and falls back to the
    ``main`` head commit when the repository has no releases. "main" pins that
    head commit, a 7–40 char hex string pins that commit, and anything else
    names a release tag (a bare tag without a release object still works).
    """
    if version is None:
        release = _github_json(f"{API}/releases/latest", missing_ok=True)
        return _release_target(release) if release else _commit_target("main", "main")
    if version == "main":
        return _commit_target("main", "main")
    if re.fullmatch(r"[0-9a-fA-F]{7,40}", version):
        return _commit_target(version.lower(), "")
    release = _github_json(f"{API}/releases/tags/{version}", missing_ok=True)
    if release:
        return _release_target(release, fallback_tag=version)
    if _github_json(f"{API}/git/ref/tags/{version}", missing_ok=True):
        return (
            version,
            f"https://github.com/{REPO}/archive/refs/tags/{version}.tar.gz",
            "",
            "release",
        )
    raise UpdateSourceError(
        f"{REPO} has no release, tag, or commit named {version!r}; "
        f"see https://github.com/{REPO}/tags or pass --version main"
    )


def _update_id(label: str | None) -> str | None:
    """Comparable identity of a version label (``main@abc1234`` → ``abc1234``)."""
    if not label:
        return None
    return label.rsplit("@", 1)[1] if "@" in label else label


def _print_update_notes(label: str, body: str, kind: str) -> None:
    rule = "─" * 64
    if kind == "release":
        heading = f"Release notes — {label}"
    else:
        heading = f"Latest commit — {label}"
    print()
    print(rule)
    print(f"  {heading}")
    print(rule)
    if body:
        for line in body.splitlines():
            print(f"  {line}" if line else "")
    else:
        print("  (no description)")
    print(rule)


def _confirm_update(from_tag: str | None, to_tag: str) -> bool:
    """Prompt the user before applying an update.

    Returns True if we should proceed, False to abort. When stdin isn't a
    TTY (e.g. invoked from the GUI's QProcess) we auto-proceed silently —
    the caller is expected to have shown the changelog already.
    """
    if not sys.stdin.isatty():
        print("  (stdin not a TTY — proceeding without prompt)")
        return True
    from_label = from_tag if from_tag else "(no baseline)"
    try:
        ans = input(f"\nProceed with update {from_label} → {to_tag}? [Y/n]: ")
    except EOFError:
        return True
    ans = ans.strip().lower()
    if ans in ("", "y", "yes"):
        return True
    print("aborted.")
    return False


def _download(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest``, retrying transient network failures.

    codeload.github.com intermittently answers 502 on large archives; each
    attempt reopens ``dest`` with "wb", so a partial file is never kept.
    """
    attempts = 3
    for attempt in range(1, attempts + 1):
        suffix = f" (attempt {attempt}/{attempts})" if attempt > 1 else ""
        print(f"  downloading {url}{suffix}")
        req = urllib.request.Request(url, headers={"User-Agent": "anima-update"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp, dest.open(
                "wb"
            ) as f:
                shutil.copyfileobj(resp, f, length=1 << 20)
            return
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == attempts:
                raise UpdateSourceError(f"could not download {url}: {e}") from e
            print(f"  download failed ({e}); retrying")
            time.sleep(attempt)


def _skip_links_filter(member, path):
    """The data filter, but symlinks/hardlinks are dropped instead of fatal.

    A committed symlink in the release tarball (absolute target →
    LinkOutsideDestinationError) would otherwise abort the update; release
    content never depends on links, so skipping is safe.
    """
    if member.islnk() or member.issym():
        print(f"  skipping link in tarball: {member.name}")
        return None
    return tarfile.data_filter(member, path)


def _extract_tarball(tar: Path, dest: Path) -> Path:
    """Extract tarball to dest/, return path to single top-level dir inside."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar, "r:gz") as tf:
        tf.extractall(dest, filter=_skip_links_filter)
    children = [p for p in dest.iterdir() if p.is_dir()]
    if len(children) != 1:
        sys.exit(f"unexpected tarball layout: {[p.name for p in children]}")
    return children[0]


def _load_baseline() -> tuple[str | None, dict[str, str]]:
    if not MANIFEST_FILE.exists():
        return None, {}
    data = json.loads(MANIFEST_FILE.read_text())
    return data.get("version"), data.get("files", {})


def _save_manifest(version: str, files: dict[str, str]) -> None:
    MANIFEST_FILE.write_text(
        json.dumps(
            {
                "version": version,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "files": dict(sorted(files.items())),
            },
            indent=2,
        )
    )


def _git_head_sha() -> str | None:
    """Short HEAD of the checkout; None for tarball installs without ``.git``."""
    if not (ROOT / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short=7", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def describe_installed() -> dict:
    """Installed build from the update baseline, else the checkout's HEAD."""
    version, _hashes = _load_baseline()
    if version:
        return {"label": version, "sha": _update_id(version)}
    sha = _git_head_sha()
    return {"label": f"git@{sha}" if sha else None, "sha": sha}


def describe_available() -> dict:
    """Newest GitHub target the updater would install right now."""
    label, _tarball_url, body, kind = _resolve_target(None)
    detail = next((line.strip() for line in body.splitlines() if line.strip()), "")
    return {"label": label, "sha": _update_id(label), "kind": kind, "detail": detail}


def _print_diff(user: Path, new: Path, rel: str) -> None:
    try:
        a = user.read_text().splitlines()
        b = new.read_text().splitlines()
    except UnicodeDecodeError:
        print("    (binary file — diff skipped)")
        return
    out = "\n".join(
        difflib.unified_diff(
            a,
            b,
            fromfile=f"a/{rel} (yours)",
            tofile=f"b/{rel} (new)",
            lineterm="",
        )
    )
    print(out or "    (files differ but unified diff is empty)")


def _conflict_hint(rel: str) -> str:
    return (
        f"\nconflict: {rel} (you modified it AND upstream changed it)\n"
        "  stdin is not interactive — cannot prompt. Re-run with one of:\n"
        "    --keep-conflicts   keep all your modified configs as-is\n"
        "    --yes-overwrite    back up your configs, then overwrite with upstream"
    )


def _prompt_conflict(rel: str, user_path: Path, new_path: Path) -> str:
    """Return action: 'keep' | 'overwrite' | 'backup'."""
    # Without a real TTY (e.g. the GUI's QProcess), input() would block forever
    # on stdin nobody can write to — bail with an actionable message instead.
    if not sys.stdin.isatty():
        sys.exit(_conflict_hint(rel))
    while True:
        try:
            choice = (
                input(
                    f"\n  conflict: {rel} (you modified it AND upstream changed it)\n"
                    f"    [k]eep yours / [o]verwrite with new / [b]ackup yours then overwrite / [d]iff: "
                )
                .strip()
                .lower()
            )
        except EOFError:
            sys.exit(_conflict_hint(rel))
        if choice in ("k", "keep"):
            return "keep"
        if choice in ("o", "overwrite"):
            return "overwrite"
        if choice in ("b", "backup", ""):
            return "backup"
        if choice in ("d", "diff"):
            _print_diff(user_path, new_path, rel)
            continue
        print("    invalid choice; please pick k/o/b/d")


def seed_manifest(version: str | None) -> int:
    """Write .anima_release.json for the current tree without downloading.

    Used by the bootstrap installer (install.sh / install.ps1) right after it
    extracts a release tarball, so the *first* `make update` has a correct
    baseline and treats nothing as user-modified. The non-preserved file set
    hashed here must match exactly what `_apply` records, otherwise a later
    update would see preserved files as "upstream-removed" and delete them —
    so this reuses `_is_preserved` / `_sha256_file` rather than reimplementing
    the walk in shell. Run before `uv sync` so `.venv` doesn't exist yet.
    """
    if version is None:
        version, _url, _body, _kind = _resolve_target(None)
    hashes: dict[str, str] = {}
    for p in _walk_tree(ROOT):
        rel = p.relative_to(ROOT).as_posix()
        if _is_preserved(rel):
            continue
        hashes[rel] = _sha256_file(p)
    _save_manifest(version, hashes)
    print(f"seeded {MANIFEST_FILE.name} → {version} ({len(hashes)} files)")
    return 0


def update(
    version: str | None,
    dry_run: bool,
    yes_overwrite: bool,
    keep_conflicts: bool,
    no_sync: bool,
    assume_yes: bool,
) -> int:
    if yes_overwrite and keep_conflicts:
        sys.exit("--yes-overwrite and --keep-conflicts are mutually exclusive")
    print(f"MonadForge-Kohaku update — repo {REPO}")
    tag, tarball_url, body, kind = _resolve_target(version)
    print(f"  target: {tag} ({kind})")

    baseline_version, baseline_hashes = _load_baseline()
    if baseline_version:
        print(f"  current baseline: {baseline_version}")
    else:
        print("  no baseline manifest — first run; conflicts will be prompted")

    if version is None and _update_id(baseline_version) == _update_id(tag):
        print(f"already on {tag}; nothing to do")
        return 0

    # Show the release notes / commit message; the GUI passes --yes because
    # its own dialog replaced this prompt.
    if not assume_yes:
        _print_update_notes(tag, body, kind)

    if not dry_run and not assume_yes and not _confirm_update(baseline_version, tag):
        return 0

    with tempfile.TemporaryDirectory(prefix="anima-update-") as td:
        tdir = Path(td)
        tar_path = tdir / "release.tar.gz"
        _download(tarball_url, tar_path)
        extracted_root = _extract_tarball(tar_path, tdir / "extracted")
        return _apply(
            extracted_root,
            tag,
            baseline_hashes,
            dry_run=dry_run,
            yes_overwrite=yes_overwrite,
            keep_conflicts=keep_conflicts,
            no_sync=no_sync,
        )


def _restart_daemon_if_idle() -> None:
    """Bring the training daemon onto the freshly-updated code.

    The daemon supervisor runs detached and has already imported the OLD
    ``anima_daemon/*`` into memory; swapping the files on disk doesn't touch
    it, so a new-code client would otherwise be talking to a stale-code daemon.
    Restarting fixes that — but training jobs run in their own detached
    processes (they imported everything at launch), so an *active* run is not
    disrupted by a supervisor restart and must not be killed:

    - idle → graceful shutdown (``kill_jobs=False``); the next ``ensure_daemon``
      relaunches it on the new code.
    - busy → leave it; warn that it stays on old code until the job finishes.

    Best-effort: any daemon hiccup is reported, never fatal to the update.
    """
    try:
        from anima_daemon import client as _client
    except Exception:  # noqa: BLE001 — daemon module optional / mid-swap
        return
    if not _client.is_running():
        return
    cl = _client.DaemonClient()
    active = (cl.health() or {}).get("active_job")
    if active:
        print(
            f"\n⚠ training daemon left running on the OLD code: job {active} is "
            "active.\n"
            "  It keeps using the pre-update daemon until that job finishes.\n"
            "  Restart when convenient — `make daemon-terminate` (it relaunches\n"
            "  on the new code the next time it's used)."
        )
        return
    try:
        cl.shutdown(kill_jobs=False)
        print(
            "\ntraining daemon stopped (was idle) — it will relaunch on the new "
            "code next time it's used."
        )
    except Exception as e:  # noqa: BLE001
        print(f"\ncould not stop the idle training daemon: {e} (restart manually)")


def _v100_sync_protected() -> bool:
    """Never replace a dedicated V100 stack with the default uv groups.

    Probe distribution metadata, not CUDA. Check the project venv as well as
    this process: users may invoke the updater with a system Python.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from library.runtime.profiles import is_v100_runtime, resolve_runtime

    default = resolve_runtime("default", configs_dir=ROOT / "configs")
    candidates = {Path(sys.executable), default.python}
    project_python = ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if project_python.is_file():
        candidates.add(project_python)
    return os.environ.get("ANIMA_RUNTIME_PROFILE") == "v100" or any(
        is_v100_runtime(python) for python in candidates
    )


def _apply(
    new_root: Path,
    new_tag: str,
    baseline_hashes: dict[str, str],
    *,
    dry_run: bool,
    yes_overwrite: bool,
    keep_conflicts: bool,
    no_sync: bool,
) -> int:
    if not no_sync and not dry_run:
        try:
            protected = _v100_sync_protected()
        except ValueError as exc:
            print(f"Cannot safely select the dependency runtime: {exc}")
            return 2
        if protected:
            no_sync = True
            print(
                "V100 Torch 2.10/CUDA 12.9 runtime detected: skipping ordinary uv sync. "
                "Code will update; preserve the dedicated requirements-v100.txt stack "
                "and re-run the V100 FlashAttention validation separately."
            )
    new_files: dict[str, Path] = {}
    new_hashes: dict[str, str] = {}
    for p in _walk_tree(new_root):
        rel = p.relative_to(new_root).as_posix()
        if _is_preserved(rel):
            continue
        new_files[rel] = p
        new_hashes[rel] = _sha256_file(p)

    summary = {
        "wrote_new": 0,
        "overwrote_unchanged": 0,
        "no_change": 0,
        "config_kept": 0,
        "config_overwrote": 0,
        "config_backed_up": 0,
        "code_backed_up": 0,
        "deleted": 0,
        "kept_user_added": 0,
    }

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    backup_session = BACKUP_ROOT / timestamp

    def _backup(dest: Path) -> None:
        rel = dest.relative_to(ROOT).as_posix()
        bak = backup_session / rel
        if dry_run:
            return
        bak.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dest, bak)

    def _do_copy(src: Path, dest: Path) -> None:
        rel = dest.relative_to(ROOT).as_posix()
        print(f"  write  {rel}")
        if dry_run:
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    for rel, src in new_files.items():
        dest = ROOT / rel
        if not dest.exists():
            _do_copy(src, dest)
            summary["wrote_new"] += 1
            continue

        user_hash = _sha256_file(dest)
        new_hash = new_hashes[rel]
        if user_hash == new_hash:
            summary["no_change"] += 1
            continue

        baseline_hash = baseline_hashes.get(rel)
        user_modified = (baseline_hash is None) or (user_hash != baseline_hash)

        if not user_modified:
            _do_copy(src, dest)
            summary["overwrote_unchanged"] += 1
            continue

        if _is_conflict_path(rel):
            if yes_overwrite:
                action = "backup"
            elif keep_conflicts:
                action = "keep"
            else:
                action = _prompt_conflict(rel, dest, src)
            if action == "keep":
                summary["config_kept"] += 1
                continue
            if action == "backup":
                _backup(dest)
                summary["config_backed_up"] += 1
            else:
                summary["config_overwrote"] += 1
            _do_copy(src, dest)
        else:
            _backup(dest)
            summary["code_backed_up"] += 1
            _do_copy(src, dest)

    for rel, baseline_hash in baseline_hashes.items():
        if rel in new_files:
            continue
        dest = ROOT / rel
        if not dest.exists():
            continue
        user_hash = _sha256_file(dest)
        if user_hash == baseline_hash:
            print(f"  delete {rel}")
            if not dry_run:
                dest.unlink()
            summary["deleted"] += 1
        else:
            print(f"  upstream removed but you modified: {rel} (kept)")
            summary["kept_user_added"] += 1

    print("\nsummary:")
    for k, v in summary.items():
        if v:
            print(f"  {k}: {v}")
    if backup_session.exists():
        print(f"\n  backups in {backup_session.relative_to(ROOT)}/")

    if dry_run:
        print("\n(dry run — no files written, manifest not updated)")
        return 0

    _save_manifest(new_tag, new_hashes)
    print(f"\nmanifest updated: {MANIFEST_FILE.name} → {new_tag}")

    if not no_sync:
        sync_command, backend = _uv_sync_command()
        print(f"\nrunning {' '.join(sync_command)} …")
        env = os.environ.copy()
        venv_scripts = ROOT / ".venv" / ("Scripts" if sys.platform == "win32" else "bin")
        if venv_scripts.exists():
            env["PATH"] = str(venv_scripts) + os.pathsep + env.get("PATH", "")
        try:
            subprocess.run(sync_command, cwd=ROOT, check=True, env=env)
        except FileNotFoundError:
            print("  uv not found on PATH; skip or run `uv sync` manually")
        except subprocess.CalledProcessError as e:
            print(f"  uv sync failed (exit {e.returncode}); rerun manually")
            return e.returncode
        else:
            if backend is not None:
                _verify_windows_backend(backend)

    # Code on disk changed → the running daemon supervisor is now stale (skip
    # if nothing was written — a no-op update leaves the daemon correct).
    changed = (
        summary["wrote_new"]
        + summary["overwrote_unchanged"]
        + summary["code_backed_up"]
        + summary["config_overwrote"]
        + summary["config_backed_up"]
        + summary["deleted"]
    )
    if changed:
        _restart_daemon_if_idle()

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Update MonadForge-Kohaku from GitHub release")
    ap.add_argument(
        "--version",
        help="Release tag, branch, or commit sha to install. Default: the "
        "newest release tag, falling back to the latest commit on main.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without touching files",
    )
    ap.add_argument(
        "--yes-overwrite",
        action="store_true",
        help="Non-interactive: on config conflicts, back up user file and overwrite",
    )
    ap.add_argument(
        "--keep-conflicts",
        action="store_true",
        help="Non-interactive: on config conflicts, keep the user's version",
    )
    ap.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip the trailing `uv sync`",
    )
    ap.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the changelog confirmation prompt (e.g. when invoked from a GUI)",
    )
    ap.add_argument(
        "--seed-manifest",
        action="store_true",
        help="Write .anima_release.json for the current tree (no download) and exit; "
        "used by the bootstrap installer. Records --version, else resolves latest tag.",
    )
    args = ap.parse_args()
    try:
        if args.seed_manifest:
            return seed_manifest(args.version)
        return update(
            version=args.version,
            dry_run=args.dry_run,
            yes_overwrite=args.yes_overwrite,
            keep_conflicts=args.keep_conflicts,
            no_sync=args.no_sync,
            assume_yes=args.yes,
        )
    except UpdateSourceError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
