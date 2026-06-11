"""User-facing CLI error formatting."""

from __future__ import annotations

import re

from rich.console import Console

from heta.cli.branding import MUTED, WARN

_MAX_REASON_CHARS = 2000


def error_reason(exc: BaseException) -> str:
    """Return a compact reason suitable for terminal output."""
    reason = str(exc).strip() or exc.__class__.__name__
    reason = re.sub(r"\s+", " ", reason)
    if len(reason) > _MAX_REASON_CHARS:
        reason = reason[: _MAX_REASON_CHARS - 1].rstrip() + "…"
    return reason


def print_error(console: Console, title: str, exc: BaseException) -> None:
    console.print(f"[{WARN}]?[/] {title}")
    console.print(f"[{MUTED}]  Reason:[/] {error_reason(exc)}")
