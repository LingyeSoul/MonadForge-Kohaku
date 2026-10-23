"""Language state for the server-side field-help and guide resolvers.

The Vue app owns every UI string (``webui/frontend/src/i18n/``); this module
only carries the active language code for the few Python helpers that render
text without an explicit ``lang`` argument. There is no server-side string
catalog to load any more.
"""

from __future__ import annotations

LANGUAGES = ("en", "cn", "ko", "ja")

_current_lang = "en"


def current_language() -> str:
    """Return the active language code."""
    return _current_lang


def set_language(lang: str) -> None:
    """Set the active language (no persistence — caller handles that)."""
    global _current_lang
    _current_lang = lang if lang in LANGUAGES else "en"
