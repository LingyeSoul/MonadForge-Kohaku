"""Documentation API endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

_ROOT = Path(__file__).resolve().parent.parent.parent
_GUIDELINES_DIR = _ROOT / "docs" / "guidelines"

# Language → guidebook filename mapping. The localized files are the ones the
# docs tree actually ships; ``guidebook.md`` is the legacy English copy kept as
# a last-resort fallback.
_GUIDEBOOK_FILES: dict[str, str] = {
    "cn": "指南书.md",
    "en": "guidebook_en.md",
    "ja": "ガイドブック.md",
    "ko": "가이드북.md",
}
_LEGACY_GUIDEBOOK = "guidebook.md"

# The MOSS line ships its own guidebook in the same four languages.
_MOSS_GUIDEBOOK_FILES: dict[str, str] = {
    "cn": "moss-指南书.md",
    "en": "moss-guidebook.md",
    "ja": "moss-ガイドブック.md",
    "ko": "moss-가이드북.md",
}


def _guidebook_path(lang: str, book: str) -> Path:
    """Resolve the localized guidebook, falling back to English then legacy."""
    if book == "moss":
        files = _MOSS_GUIDEBOOK_FILES
        candidates = (files.get(lang, files["en"]), files["en"])
    else:
        files = _GUIDEBOOK_FILES
        candidates = (
            files.get(lang, files["en"]),
            files["en"],
            _LEGACY_GUIDEBOOK,
        )
    for name in dict.fromkeys(candidates):
        path = _GUIDELINES_DIR / name
        if path.is_file():
            return path
    raise HTTPException(
        status_code=404,
        detail=f"Guidebook not found for book={book!r} lang={lang!r}",
    )


@router.get("/guidebook")
def get_guidebook(lang: str = Query("cn"), book: str = Query("anima")):
    """Return the guidebook markdown for the given language and model line."""
    if book not in ("anima", "moss"):
        raise HTTPException(status_code=400, detail=f"Unknown guidebook: {book}")
    guidebook_path = _guidebook_path(lang, book)
    return {
        "content": guidebook_path.read_text(encoding="utf-8"),
        "filename": guidebook_path.name,
        "lang": lang,
        "book": book,
    }
