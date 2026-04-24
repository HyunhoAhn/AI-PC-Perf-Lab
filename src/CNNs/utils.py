#!/usr/bin/env python3
"""Small utility entrypoints for CNN test helpers."""

from __future__ import annotations

import os
import sys
import time


def _get_sleep_seconds() -> float:
    raw_value = os.environ.get("CNN_SLEEP_SECONDS")
    if raw_value is None:
        raise SystemExit("CNN_SLEEP_SECONDS is required.")
    try:
        seconds = float(raw_value)
    except ValueError as exc:
        raise SystemExit(f"Invalid CNN_SLEEP_SECONDS: {raw_value!r}") from exc
    if seconds < 0:
        raise SystemExit("CNN_SLEEP_SECONDS must be zero or positive.")
    return seconds


def sleep_entrypoint() -> int:
    time.sleep(_get_sleep_seconds())
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit("A utility command is required.")

    command = args[0]
    if command == "sleep":
        return sleep_entrypoint()

    raise SystemExit(f"Unsupported utility command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
