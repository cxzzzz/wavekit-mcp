from __future__ import annotations

import ast
import contextlib
import io
import logging
import operator
import threading
import time
import traceback
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from RestrictedPython import compile_restricted
from RestrictedPython.Guards import (
    guarded_iter_unpack_sequence,
    guarded_unpack_sequence,
    safe_builtins,
    safer_getattr,
)

from .config import Config
from .serializer import serialize_result


# ── RestrictedPython print bridge ─────────────────────────────────────────────

class _StdoutPrinter:
    """RestrictedPython _print_ factory that forwards to the current sys.stdout.

    RestrictedPython transforms ``print(x)`` into:
        _print_(_getattr_)._call_print(x)
    Forwarding _call_print to the real print() means redirect_stdout captures it.
    """

    def __init__(self, _getattr_=None):
        pass

    def _call_print(self, *args, **kwargs):
        print(*args, **kwargs)  # writes to sys.stdout → captured by redirect_stdout


# ── result types ──────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    result: Any
    output: str
    error: str | None
    duration_ms: int


@dataclass
class HistoryEntry:
    code: str
    error: str | None
    duration_ms: int


# ── RestrictedPython guards ───────────────────────────────────────────────────

_INPLACE_OPS: dict[str, Any] = {
    "+=": operator.iadd,
    "-=": operator.isub,
    "*=": operator.imul,
    "/=": operator.itruediv,
    "//=": operator.ifloordiv,
    "%=": operator.imod,
    "**=": operator.ipow,
    "&=": operator.iand,
    "|=": operator.ior,
    "^=": operator.ixor,
    "<<=": operator.ilshift,
    ">>=": operator.irshift,
}


def _guarded_inplacevar(op: str, x: Any, y: Any) -> Any:
    fn = _INPLACE_OPS.get(op)
    if fn is None:
        raise ValueError(f"Unsupported inplace operator: {op}")
    return fn(x, y)


import builtins as _builtins

# Start from safe_builtins and add commonly needed functions that are safe.
# Notably absent: open, __import__, exec, eval, compile.
_ALLOWED_BUILTINS: dict[str, Any] = {
    **safe_builtins,
    # I/O
    "print": print,
    "input": None,  # explicitly block interactive input
    # containers
    "list": list,
    "dict": dict,
    "set": set,
    "frozenset": frozenset,
    "bytearray": bytearray,
    # iteration / functional
    "iter": iter,
    "next": next,
    "enumerate": enumerate,
    "map": map,
    "filter": filter,
    "reversed": reversed,
    "sum": sum,
    "min": min,
    "max": max,
    "any": any,
    "all": all,
    "zip": zip,
    "sorted": sorted,
    # introspection
    "dir": dir,
    "vars": vars,
    "type": type,
    "id": id,
    "len": len,
    "repr": repr,
    "hasattr": hasattr,
    "getattr": getattr,
    # conversions
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "bytes": bytes,
    "hex": hex,
    "oct": oct,
    "bin": _builtins.bin,
    "chr": chr,
    "ord": ord,
    "format": format,
    # math
    "abs": abs,
    "round": round,
    "pow": pow,
    "divmod": divmod,
    # misc
    "range": range,
    "slice": slice,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "callable": callable,
    "hash": hash,
    "object": object,
    "staticmethod": staticmethod,
    "classmethod": classmethod,
    "property": property,
    "super": super,
    "NotImplemented": NotImplemented,
    "Ellipsis": ...,
}

_BASE_GUARDS: dict[str, Any] = {
    "__builtins__": _ALLOWED_BUILTINS,
    "_getattr_": safer_getattr,
    "_getitem_": lambda obj, idx: obj[idx],
    "_getiter_": iter,
    "_write_": lambda x: x,
    "_inplacevar_": _guarded_inplacevar,
    # RestrictedPython 8.x transforms `print(...)` → `_print_(_getattr_)._call_print(...)`
    "_print_": _StdoutPrinter,
    # sequence unpacking: `a, b = func()` and `for a, b in items:`
    "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
    "_unpack_sequence_": guarded_unpack_sequence,
}


# ── Session ───────────────────────────────────────────────────────────────────

class Session:
    def __init__(self, session_id: str, config: Config):
        self.session_id = session_id
        self.config = config
        self.managed_readers: list[Any] = []
        self.history: list[HistoryEntry] = []
        self.namespace: dict[str, Any] = {}
        self._reset_namespace()

    # ── namespace management ──────────────────────────────────────────────────

    def _reset_namespace(self) -> None:
        """Close open readers, clear user variables, re-inject base objects."""
        self._close_readers()

        import wavekit

        ns: dict[str, Any] = {
            **_BASE_GUARDS,
            "np": np,
            "Pattern": wavekit.Pattern,
            "MatchStatus": wavekit.MatchStatus,
            "open_reader": self._make_open_reader(),
            "VcdReader": self._make_reader_class(wavekit.VcdReader),
            "FsdbReader": self._make_reader_class(wavekit.FsdbReader),
        }

        if self.config.file_access.read_enabled or self.config.file_access.write_enabled:
            ns["open"] = self._make_safe_open()

        self.namespace = ns

    def _close_readers(self) -> None:
        for r in self.managed_readers:
            try:
                r.close()
            except Exception:
                pass
        self.managed_readers = []

    def close(self) -> None:
        self._close_readers()
        self.namespace = {}

    # ── injected helpers ──────────────────────────────────────────────────────

    def _make_open_reader(self):
        def open_reader(path: str):
            """Open a VCD or FSDB waveform file.

            File format is auto-detected by extension (.vcd → VcdReader,
            anything else → FsdbReader).  The reader is automatically closed
            when the session is reset or closed.

            Returns a Reader with: load_waveform(), load_matched_waveforms(),
            get_matched_signals(), get_matched_scopes(), eval(), top_scope_list()
            """
            import wavekit

            ext = Path(path).suffix.lower()
            if ext == ".vcd":
                r = wavekit.VcdReader(path)
            else:
                r = wavekit.FsdbReader(path)
            r.__enter__()
            self.managed_readers.append(r)
            return r

        return open_reader

    def _make_reader_class(self, cls):
        """Return a wrapper that instantiates cls, enters its context, and registers it for auto-close."""
        managed_readers = self.managed_readers

        class _ManagedReader:
            def __new__(new_cls, path: str, *args, **kwargs):
                r = cls(path, *args, **kwargs)
                r.__enter__()
                managed_readers.append(r)
                return r

        _ManagedReader.__name__ = cls.__name__
        _ManagedReader.__qualname__ = cls.__qualname__
        return _ManagedReader

    def _make_safe_open(self):
        cfg = self.config.file_access
        real_open = open

        def safe_open(path, mode="r", **kwargs):
            is_write = any(c in mode for c in ("w", "a", "x", "+"))

            if is_write:
                if not cfg.write_enabled:
                    raise PermissionError(
                        "File write access is disabled in this session."
                    )
                allowed = cfg.write_allowed_paths
            else:
                if not cfg.read_enabled:
                    raise PermissionError(
                        "File read access is disabled in this session."
                    )
                allowed = cfg.read_allowed_paths

            resolved = Path(path).expanduser().resolve()
            if not any(resolved.match(p) for p in allowed):
                raise PermissionError(
                    f"Path '{resolved}' is not in the allowed list.\n"
                    f"Allowed patterns: {allowed}"
                )
            return real_open(resolved, mode, **kwargs)

        return safe_open

    # ── execution ─────────────────────────────────────────────────────────────

    def execute(self, code: str) -> RunResult:
        start = time.monotonic()

        # Container shared between threads: [result, output, error]
        out: list[Any] = [None, "", None]

        def target():
            try:
                out[0], out[1] = self._exec(code)
            except Exception:
                out[2] = traceback.format_exc()

        t = threading.Thread(target=target, daemon=True)
        t.start()
        t.join(timeout=self.config.limits.run_timeout_sec)

        duration_ms = int((time.monotonic() - start) * 1000)

        if t.is_alive():
            error = (
                f"Execution timed out after {self.config.limits.run_timeout_sec}s. "
                "The session may be in a partial state — consider reset_session()."
            )
            self._add_history(HistoryEntry(code=code, error=error, duration_ms=duration_ms))
            return RunResult(result=None, output="", error=error, duration_ms=duration_ms)

        raw_result, raw_output, error = out[0], out[1], out[2]

        output = _truncate_output(raw_output, self.config.limits.output_max_chars)
        result = serialize_result(raw_result, self.config) if error is None else None

        self._add_history(HistoryEntry(code=code, error=error, duration_ms=duration_ms))
        return RunResult(result=result, output=output, error=error, duration_ms=duration_ms)

    def _exec(self, code: str) -> tuple[Any, str]:
        """Compile with RestrictedPython and execute; return (last_expr_value, stdout)."""
        body_src, expr_src = _split_last_expr(code)

        buf = io.StringIO()
        result = None

        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=SyntaxWarning, module="RestrictedPython")
                if body_src.strip():
                    byte_code = compile_restricted(body_src, "<session>", "exec")
                    exec(byte_code, self.namespace)  # noqa: S102

                if expr_src:
                    # Try eval first (captures return value of expressions).
                    # Falls back to exec for statements that look like expressions
                    # but aren't valid in eval context.
                    try:
                        expr_code = compile_restricted(expr_src, "<session>", "eval")
                        result = eval(expr_code, self.namespace)  # noqa: S307
                    except SyntaxError:
                        stmt_code = compile_restricted(expr_src, "<session>", "exec")
                        exec(stmt_code, self.namespace)  # noqa: S102

        # Restore guards in case user code overwrote them
        self.namespace.update(_BASE_GUARDS)

        return result, buf.getvalue()

    def _add_history(self, entry: HistoryEntry) -> None:
        self.history.append(entry)
        max_h = self.config.limits.history_max
        if len(self.history) > max_h:
            self.history = self.history[-max_h:]


# ── SessionManager ────────────────────────────────────────────────────────────

class SessionManager:
    def __init__(self, config: Config):
        self.config = config
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._log = logging.getLogger("wavekit_mcp")

    def open_session(self) -> str:
        with self._lock:
            max_s = self.config.limits.max_sessions
            if len(self._sessions) >= max_s:
                raise RuntimeError(
                    f"Maximum sessions ({max_s}) reached. "
                    "Call close_session() to free one first."
                )
            sid = uuid.uuid4().hex[:8]
            self._sessions[sid] = Session(sid, self.config)
            self._log.info("session_open sid=%s total=%d", sid, len(self._sessions))
            return sid

    def close_session(self, session_id: str) -> None:
        with self._lock:
            self._get(session_id).close()
            del self._sessions[session_id]
            self._log.info("session_close sid=%s total=%d", session_id, len(self._sessions))

    def reset_session(self, session_id: str) -> None:
        self._get(session_id)._reset_namespace()
        self._log.info("session_reset sid=%s", session_id)

    def run(self, session_id: str, code: str) -> RunResult:
        result = self._get(session_id).execute(code)

        # INFO: lightweight event record (no payload)
        self._log.info(
            "run sid=%s duration_ms=%d status=%s",
            session_id,
            result.duration_ms,
            "error" if result.error else "ok",
        )

        # DEBUG: full code + full result/error (what AI sent and what AI received)
        if self._log.isEnabledFor(logging.DEBUG):
            self._log.debug(
                "run sid=%s\n--- code ---\n%s\n--- output ---\n%s\n--- result ---\n%s\n--- error ---\n%s",
                session_id,
                code,
                result.output or "(none)",
                result.result if result.result is not None else "(none)",
                result.error or "(none)",
            )

        return result

    def get_history(self, session_id: str, last_n: int = 10) -> list[dict]:
        entries = self._get(session_id).history[-last_n:]
        return [
            {"code": e.code, "error": e.error, "duration_ms": e.duration_ms}
            for e in entries
        ]

    def _get(self, session_id: str) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(
                f"Session '{session_id}' not found. Call open_session() first."
            )
        return session


# ── helpers ───────────────────────────────────────────────────────────────────

def _split_last_expr(source: str) -> tuple[str, str | None]:
    """Split source into (body_source, last_expr_source_or_None).

    Separates the last statement when it's a value-producing expression, so
    it can be eval()'d and its return value captured as the run result.

    Exception: ``print(...)`` calls must stay in the exec body because
    RestrictedPython only applies the _print_ → _call_print transformation
    in exec mode, not eval mode.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, None

    if not tree.body:
        return source, None

    last = tree.body[-1]

    if not isinstance(last, ast.Expr):
        return source, None

    # Keep print() calls in exec body so RestrictedPython's _print_ guard works.
    if (
        isinstance(last.value, ast.Call)
        and isinstance(last.value.func, ast.Name)
        and last.value.func.id == "print"
    ):
        return source, None

    lines = source.splitlines(keepends=True)
    last_line = last.lineno - 1  # 0-indexed
    body_src = "".join(lines[:last_line])
    expr_src = "".join(lines[last_line:]).strip()
    return body_src, expr_src


def _truncate_output(output: str, max_chars: int) -> str:
    if len(output) <= max_chars:
        return output
    omitted = len(output) - max_chars
    return (
        output[:max_chars]
        + f"\n...[truncated: {omitted} chars omitted. Use smaller print() calls.]"
    )
