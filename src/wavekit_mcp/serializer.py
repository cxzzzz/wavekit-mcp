from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .config import Config


def serialize_result(value: Any, config: Config) -> Any:
    lim = config.limits

    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, str):
        return _truncate_str(value, lim.result_str_max)

    if isinstance(value, np.ndarray):
        return _serialize_ndarray(value, lim)

    # Import wavekit types lazily to avoid hard dependency at module load
    try:
        from wavekit import MatchResult, Waveform
    except ImportError:
        Waveform = None  # type: ignore[assignment]
        MatchResult = None  # type: ignore[assignment]

    if Waveform is not None and isinstance(value, Waveform):
        return _serialize_waveform(value, lim)

    if MatchResult is not None and isinstance(value, MatchResult):
        return _serialize_match_result(value, lim)

    if isinstance(value, dict):
        return _serialize_dict(value, config)

    if isinstance(value, (list, tuple)):
        return _serialize_list(list(value), config)

    # Fallback
    return {
        "type": type(value).__name__,
        "repr": _truncate_str(repr(value), lim.result_str_max),
    }


# ── type-specific serializers ─────────────────────────────────────────────────

def _serialize_ndarray(arr: np.ndarray, lim) -> dict:
    preview = [_py_scalar(x) for x in arr.flat[: lim.result_preview_items]]
    return {
        "type": "ndarray",
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "preview": preview,
    }


def _serialize_waveform(wave, lim) -> dict:
    n = len(wave.value)
    preview = [_py_scalar(x) for x in wave.value[: lim.result_preview_items]]
    out: dict[str, Any] = {
        "type": "Waveform",
        "name": wave.signal.full_name or wave.signal.name,
        "width": wave.width,
        "signed": wave.signed,
        "length": n,
        "value_preview": preview,
    }
    if n > 0:
        out["clock_range"] = [int(wave.clock[0]), int(wave.clock[-1])]
        out["time_range"] = [int(wave.time[0]), int(wave.time[-1])]
    return out


def _serialize_match_result(result, lim) -> dict:
    from wavekit import MatchStatus

    total = len(result)
    ok_count = int(np.sum(result.status.value == MatchStatus.OK.value))
    timeout_count = int(np.sum(result.status.value == MatchStatus.TIMEOUT.value))
    req_viol_count = int(
        np.sum(result.status.value == MatchStatus.REQUIRE_VIOLATED.value)
    )

    out: dict[str, Any] = {
        "type": "MatchResult",
        "total": total,
        "ok": ok_count,
        "timeout": timeout_count,
        "require_violated": req_viol_count,
    }
    if ok_count > 0:
        valid = result.filter_valid()
        out["duration_preview"] = [
            int(x) for x in valid.duration.value[: lim.result_preview_items]
        ]
        if valid.captures:
            out["captures_keys"] = list(valid.captures.keys())
    return out


def _serialize_dict(d: dict, config) -> dict:
    lim = config.limits
    items = list(d.items())

    # dict[tuple, Waveform] — pattern match output
    try:
        from wavekit import Waveform

        if items and isinstance(items[0][0], tuple) and isinstance(
            items[0][1], Waveform
        ):
            return _serialize_waveform_dict(d, lim)
    except ImportError:
        pass

    truncated = items[: lim.result_list_max]
    out = {str(k): serialize_result(v, config) for k, v in truncated}
    if len(items) > lim.result_list_max:
        out["__truncated__"] = f"{len(items) - lim.result_list_max} more items omitted"
    return out


def _serialize_list(lst: list, config) -> list:
    lim = config.limits
    truncated = lst[: lim.result_list_max]
    out = [serialize_result(v, config) for v in truncated]
    if len(lst) > lim.result_list_max:
        out.append(f"...[{len(lst) - lim.result_list_max} more items omitted]")
    return out


def _serialize_waveform_dict(d: dict, lim) -> dict:
    from wavekit import Waveform

    items = list(d.items())[: lim.result_list_max]
    entries = {}
    for k, v in items:
        key_str = str(k[0]) if len(k) == 1 else str(k)
        entries[key_str] = (
            _serialize_waveform(v, lim) if isinstance(v, Waveform) else repr(v)[:200]
        )
    out: dict[str, Any] = {
        "type": "dict[tuple, Waveform]",
        "count": len(d),
        "entries": entries,
    }
    if len(d) > lim.result_list_max:
        out["truncated"] = f"{len(d) - lim.result_list_max} more entries omitted"
    return out


# ── helpers ───────────────────────────────────────────────────────────────────

def _truncate_str(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    omitted = len(s) - max_len
    return s[:max_len] + f"...[{omitted} chars omitted]"


def _py_scalar(x: Any) -> Any:
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x
