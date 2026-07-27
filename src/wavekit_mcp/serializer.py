from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import Config


def serialize_result(value: Any, config: Config) -> str | None:
    """Return the last expression as truncated REPL-like display text."""
    if value is None:
        return None
    return _truncate_str(repr(value), config.limits.result_str_max)


def _truncate_str(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    omitted = len(s) - max_len
    return s[:max_len] + f"...[{omitted} chars omitted]"
