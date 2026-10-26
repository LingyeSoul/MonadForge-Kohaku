#!/usr/bin/env python3
"""Build a MOSS colorize cache (vision-tower page + VAE reference) — flags in
``library.moss.requests.ColorizeCacheRequest``, work in ``library.moss.colorize``."""

from library.moss.requests import ColorizeCacheRequest
from library.moss.validation import validate_request


def main() -> None:
    req = ColorizeCacheRequest.from_argv()
    validate_request(req)
    from library.moss.colorize import run_colorize_cache

    run_colorize_cache(req)


if __name__ == "__main__":
    main()
