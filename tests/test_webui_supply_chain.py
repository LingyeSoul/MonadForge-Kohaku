"""Regression guards for the frontend dependency trust boundary."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "webui" / "frontend"


def test_frontend_lock_has_no_keyv_and_pinned_integrity():
    lock = json.loads((FRONTEND / "package-lock.json").read_text(encoding="utf-8"))
    packages = lock["packages"]
    for name, info in packages.items():
        if not name:
            continue
        assert "keyv" not in name.lower()
        assert str(info.get("resolved", "")).startswith("https://")
        assert str(info.get("integrity", "")).startswith("sha512-")


def test_frontend_lock_checker_passes_without_installing():
    checker = ROOT / "scripts" / "webui" / "ensure_npm_deps.mjs"
    result = subprocess.run(
        ["node", str(checker)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "lock verified" in result.stdout


def test_build_entrypoints_use_clean_ignore_scripts_install():
    linux = (ROOT / "setup-linux.sh").read_text(encoding="utf-8")
    windows = (ROOT / "build-webui-win.bat").read_text(encoding="utf-8")
    # The shared checker owns the actual npm ci invocation; these entrypoints
    # must request a clean repair instead of trusting node_modules existence.
    assert "ensure_npm_deps.mjs" in linux and "--clean" in linux
    assert "ensure_npm_deps.mjs" in windows and "--clean" in windows
    checker = (ROOT / "scripts" / "webui" / "ensure_npm_deps.mjs").read_text(
        encoding="utf-8"
    )
    assert 'from "./npm_args.mjs"' in checker
    script = """
import { installArgs } from "./scripts/webui/npm_args.mjs";
for (const major of [10, 11, 12, 13]) {
  for (const offline of [false, true]) {
    const args = installArgs(major, offline);
    if (args[0] !== "ci") throw new Error("not a ci install: " + args);
    if (!args.includes("--ignore-scripts")) throw new Error("lifecycle scripts allowed: " + args);
    if (args.includes("--offline") === args.includes("--prefer-offline")) {
      throw new Error("offline mode is ambiguous: " + args);
    }
  }
}
console.log("ok");
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip() == "ok"


def test_linux_launcher_builds_frontend_when_dist_is_missing():
    """Startup must not open a tab that 404s: build dist/ before launching."""
    launcher = (ROOT / "start-webui-linux.sh").read_text(encoding="utf-8")
    assert "webui/frontend/dist/index.html" in launcher
    assert "ensure_npm_deps.mjs" in launcher
    assert "npm run build" in launcher


def test_lock_checker_allows_locked_tarball_hosts_on_npm_12():
    """npm >= 12 refuses URL tarballs unless allow-remote is raised."""
    script = """
import { installArgs } from "./scripts/webui/npm_args.mjs";
const modern = installArgs(12, false);
const legacy = installArgs(11, false);
if (!modern.includes("--allow-remote=all")) throw new Error("npm 12 lost allow-remote");
if (legacy.includes("--allow-remote=all")) throw new Error("npm 11 must not get allow-remote");
if (!installArgs(12, true).includes("--offline")) throw new Error("offline mode lost");
if (installArgs(12, false).includes("--offline")) throw new Error("online mode forced offline");
console.log("ok");
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip() == "ok"


def test_lock_checker_enforces_keyv_alias_and_native_probe_boundaries():
    checker = (ROOT / "scripts" / "webui" / "ensure_npm_deps.mjs").read_text(
        encoding="utf-8"
    )
    assert "lock.lockfileVersion !== 3" in checker
    assert "allowedLifecyclePackages" in checker
    assert "npm:" in checker and "workspace:" in checker
    assert "@keyv" in checker
    assert "transformSync" in checker
    assert "r.rollup" in checker
    assert "installArgs(" in checker


def test_lock_checker_resolves_windows_npm_without_command_shim():
    checker = ROOT / "scripts" / "webui" / "ensure_npm_deps.mjs"
    source = checker.read_text(encoding="utf-8")
    assert "process.execPath" in source
    assert "node_modules/npm/bin/npm-cli.js" in source
    assert "shell: true" in source
