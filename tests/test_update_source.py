from __future__ import annotations

import json

import pytest

from scripts import update

MAIN_SHA = "20a788b751ef1a7f590c2a7f08bed13d14b63457"


def fake_api(monkeypatch, routes):
    """Replace the GitHub client with a {url: payload} lookup."""

    def get(url, *, missing_ok):
        return routes.get(url, {})

    monkeypatch.setattr(update, "_github_json", get)


def main_commit_payload():
    return {
        "sha": MAIN_SHA,
        "commit": {
            "message": "Unify the Anima config sections\n\nSecond line",
            "committer": {"date": "2026-10-24T16:44:53Z"},
        },
    }


def test_published_release_is_preferred(monkeypatch):
    fake_api(
        monkeypatch,
        {
            f"{update.API}/releases/latest": {
                "tag_name": "v1.2.3",
                "body": "  fixes\n",
            }
        },
    )

    label, url, body, kind = update._resolve_target(None)

    assert label == "v1.2.3"
    assert url == f"https://github.com/{update.REPO}/archive/refs/tags/v1.2.3.tar.gz"
    assert body == "fixes"
    assert kind == "release"


def test_missing_release_falls_back_to_main_head_commit(monkeypatch):
    fake_api(monkeypatch, {f"{update.API}/commits/main": main_commit_payload()})

    label, url, body, kind = update._resolve_target(None)

    assert label == "main@20a788b"
    assert url == f"https://github.com/{update.REPO}/archive/{MAIN_SHA}.tar.gz"
    assert body.startswith("Latest commit on main — 2026-10-24T16:44:53Z")
    assert "Second line" in body
    assert kind == "commit"


def test_explicit_main_resolves_the_head_commit(monkeypatch):
    fake_api(monkeypatch, {f"{update.API}/commits/main": main_commit_payload()})

    label, _url, _body, kind = update._resolve_target("main")

    assert label == "main@20a788b"
    assert kind == "commit"


def test_explicit_commit_sha_resolves_that_commit(monkeypatch):
    fake_api(monkeypatch, {f"{update.API}/commits/{MAIN_SHA}": main_commit_payload()})

    label, url, _body, kind = update._resolve_target(MAIN_SHA)

    assert label == "20a788b"
    assert url.endswith(f"/archive/{MAIN_SHA}.tar.gz")
    assert kind == "commit"


def test_bare_tag_without_a_release_object_still_updates(monkeypatch):
    fake_api(
        monkeypatch,
        {f"{update.API}/git/ref/tags/v1.0": {"ref": "refs/tags/v1.0"}},
    )

    label, url, body, kind = update._resolve_target("v1.0")

    assert label == "v1.0"
    assert url.endswith("/archive/refs/tags/v1.0.tar.gz")
    assert body == ""
    assert kind == "release"


def test_unknown_version_reports_the_available_refs(monkeypatch):
    fake_api(monkeypatch, {})

    with pytest.raises(update.UpdateSourceError) as excinfo:
        update._resolve_target("v9.9.9")

    message = str(excinfo.value)
    assert "no release, tag, or commit named 'v9.9.9'" in message
    assert "--version main" in message


def test_installed_label_matches_the_commit_identity(monkeypatch, tmp_path):
    manifest = tmp_path / ".anima_release.json"
    manifest.write_text(json.dumps({"version": "main@20a788b", "files": {}}))
    monkeypatch.setattr(update, "MANIFEST_FILE", manifest)

    installed = update.describe_installed()

    assert installed == {"label": "main@20a788b", "sha": "20a788b"}


def test_installed_falls_back_to_git_head_without_a_baseline(monkeypatch, tmp_path):
    monkeypatch.setattr(update, "MANIFEST_FILE", tmp_path / ".anima_release.json")
    monkeypatch.setattr(update, "_git_head_sha", lambda: "20a788b")

    assert update.describe_installed() == {"label": "git@20a788b", "sha": "20a788b"}


def test_describe_available_exposes_label_kind_and_summary(monkeypatch):
    fake_api(monkeypatch, {f"{update.API}/commits/main": main_commit_payload()})

    available = update.describe_available()

    assert available["label"] == "main@20a788b"
    assert available["sha"] == "20a788b"
    assert available["kind"] == "commit"
    assert available["detail"] == "Latest commit on main — 2026-10-24T16:44:53Z"


def test_update_is_a_noop_when_the_baseline_matches_the_target(monkeypatch, capsys):
    monkeypatch.setattr(
        update,
        "_resolve_target",
        lambda version: ("main@20a788b", "unused-url", "notes", "commit"),
    )
    monkeypatch.setattr(update, "_load_baseline", lambda: ("main@20a788b", {}))

    def no_download(*_args):
        raise AssertionError("an up-to-date tree must not download the tarball")

    monkeypatch.setattr(update, "_download", no_download)

    assert update.update(None, False, False, False, True, False) == 0
    assert "already on main@20a788b" in capsys.readouterr().out
