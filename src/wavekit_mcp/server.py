from __future__ import annotations

import argparse
import inspect
import logging
import pydoc as _pydoc
from importlib.metadata import version, PackageNotFoundError
from typing import Any

from fastmcp import FastMCP

from .config import Config
from .session import SessionManager

# Get version from package metadata
try:
    __version__ = version("wavekit-mcp")
except PackageNotFoundError:
    __version__ = "unknown"

mcp = FastMCP("wavekit-mcp", version=__version__)

_manager: SessionManager | None = None


def _get_manager() -> SessionManager:
    """Get the global SessionManager instance."""
    if _manager is None:
        raise RuntimeError("Server not initialised.")
    return _manager


# ── tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def open_session(description: str | None = None) -> str:
    """Create a new persistent Python execution session for waveform analysis.

    Args:
        description: Optional description to identify this session (shown in list_sessions)

    Returns a session_id used by all other tools.

    Pre-injected objects:
      wavekit            — wavekit module (wavekit.VcdReader, wavekit.Waveform, etc.)
      Viewer             — waveform visualization (call Viewer() to create instance)

    Available via default allowed_imports:
      import numpy as np

    Typical workflow:
      1. sid = open_session()
      2. run(sid, "import wavekit\nr = wavekit.VcdReader('/data/sim.vcd')")
      3. run(sid, "data = r.load_waveform('tb.data[7:0]', clock='tb.clk')")
      4. run(sid, "print(len(data.value))")
      5. close_session(sid)
    """
    return _get_manager().open_session(description)


@mcp.tool()
def close_session(session_id: str) -> str:
    """Close a session and release all resources (open readers, viewer, memory).

    IMPORTANT: If the session has a viewer open, closing the session will also
    close the viewer. Do not close sessions if the user may still be viewing
    waveforms. Wait for explicit user confirmation before closing.

    Args:
        session_id: The session ID to close
    """
    _get_manager().close_session(session_id)
    return f"Session '{session_id}' closed."


@mcp.tool()
def list_sessions() -> list[dict]:
    """List all active sessions.

    Returns a list with each entry containing:
      session_id   — the session identifier
      description  — user-provided description (may be None)
      created_at   — creation timestamp (ISO format string)
    """
    return _get_manager().list_sessions()


@mcp.tool()
def run(session_id: str, code: str) -> dict[str, Any]:
    """Execute Python code in a persistent session. State persists across calls.

    PRE-INJECTED: wavekit, Viewer. Import numpy and wavekit.pattern symbols as needed.
    UNFAMILIAR WITH THE API? Call get_api_docs(topic='Waveform') first.

    OPEN FILES:
        import wavekit
        r = wavekit.VcdReader("/path/to/sim.vcd")      # open VCD file
        r = wavekit.FstReader("/path/to/sim.fst")      # open FST file
        r = wavekit.FsdbReader("/path/to/sim.fsdb")    # open FSDB file

    WAVEFORM PROCESSING FOR VIEWER DISPLAY:
        # Use Waveform methods to preserve time/clock arrays:
        filtered = data.filter(lambda v: v > 0)   # ✓ keeps time/clock arrays
        viewer.waveforms.append(filtered)          # works

        # numpy operations on .value lose time/clock arrays:
        arr = data.value[data.value > 0]          # ✗ only values, no time info
        viewer.waveforms.append(???)               # can't use

        # Rule: for Viewer display → Waveform methods; for statistics → numpy

    VIEWER:
        viewer = Viewer()
        viewer.waveforms.append(wf)
        viewer.markers.append(time=1000, name="event")
        viewer.push_state()
        print(viewer.url)  # "gui://surfer" or "file:///path/to/viewer.vcd"

    MULTI-CALL WORKFLOW:
        # call 1 — load
        import wavekit
        r = wavekit.VcdReader("sim.vcd")
        data = r.load_waveform("tb.dut.data[7:0]", clock="tb.clk")
        # call 2 — data is still in namespace
        print(f"mean={np.mean(data.value):.2f}  n={len(data.value)}")

    OUTPUT LIMITS: output is capped; result shows a preview only.
    For large datasets, print computed scalars — NOT raw arrays:
        ✓  print(np.mean(data.value))
        ✓  print(np.histogram(data.value, bins=8))
        ✗  print(data.value.tolist())   # will be truncated, unhelpful

    To reduce a large Waveform before inspecting:
        data.compress()                  # remove consecutive duplicate values (RLE)
        data.downsample(500, np.mean)    # aggregate to N representative points
        data.cycle_slice(0, 500)         # first 500 clock cycles only
        data.filter(lambda v: v != 0)    # keep only interesting samples

    RETURNS:
        result      — last expression value (structured summary for large objects)
        output      — captured stdout / stderr from print() calls
        error       — exception traceback, or null on success
        duration_ms — wall-clock execution time in milliseconds
    """
    r = _get_manager().run(session_id, code)
    return {
        "result": r.result,
        "output": r.output,
        "error": r.error,
        "duration_ms": r.duration_ms,
    }


@mcp.tool()
def get_history(session_id: str, last_n: int = 10) -> list[dict]:
    """Return the last N execution records for a session.

    Each record contains:
      code        — the code that was executed
      error       — exception traceback, or null on success
      duration_ms — wall-clock execution time

    Output and result values are not stored to keep history compact.
    """
    return _get_manager().get_history(session_id, last_n)


@mcp.tool()
def get_api_docs(topic: str = "") -> str:
    """Get wavekit API documentation.

    Call with no arguments to list topics. Current topics are generated from
    wavekit's public Reader/Waveform classes and wavekit.pattern exports.
    """
    import wavekit
    import wavekit.pattern as pattern_api
    from wavekit.readers.base import Reader

    topic_map: dict[str, Any] = {
        "Waveform": wavekit.Waveform,
        "Reader": Reader,
        "Pattern": pattern_api,
        "MatchStatus": pattern_api.MatchStatus,
        "MatchPoint": pattern_api.MatchPoint,
        "MatchRecord": pattern_api.MatchRecord,
        "MatchRecords": pattern_api.MatchRecords,
        "PatternError": pattern_api.PatternError,
        "Channel": pattern_api.Channel,
        "Signal": wavekit.Signal,
        "Scope": wavekit.Scope,
        "VcdReader": wavekit.VcdReader,
        "FstReader": wavekit.FstReader,
        "FsdbReader": wavekit.FsdbReader,
    }

    if not topic:
        lines = ["Available topics (pass as topic= argument):\n"]
        lines += [f"  {name}" for name in topic_map]
        lines += [
            "",
            "Example: get_api_docs(topic='Pattern')",
        ]
        return "\n".join(lines)

    if topic not in topic_map:
        return (
            f"Unknown topic '{topic}'.\n"
            f"Available: {list(topic_map.keys())}"
        )

    return _render_api_docs(topic, topic_map[topic])


def _render_api_docs(topic: str, obj: Any) -> str:
    lines = [f"# {topic}", ""]
    doc = inspect.getdoc(obj)
    if doc:
        lines += [doc, ""]

    exports = getattr(obj, "__all__", None)
    if exports:
        lines += ["## Exports", ""]
        for name in exports:
            exported = getattr(obj, name, None)
            lines.append(_signature_line(name, exported))
        lines.append("")

    if inspect.isclass(obj):
        members = [
            (name, member)
            for name, member in inspect.getmembers(obj)
            if not name.startswith("_") and callable(member)
        ]
        if members:
            lines += ["## Public methods", ""]
            for name, member in members:
                lines.append(_signature_line(name, member))
            lines.append("")

    # pydoc keeps detailed method docs without us hand-maintaining API text.
    rendered = _pydoc.render_doc(obj, renderer=_pydoc.plaintext)
    lines += ["## pydoc", "", rendered[:12000]]
    if len(rendered) > 12000:
        lines.append("\n...[truncated]")
    return "\n".join(lines)


def _signature_line(name: str, obj: Any) -> str:
    try:
        sig = str(inspect.signature(obj)) if callable(obj) else ""
    except (TypeError, ValueError):
        sig = ""
    doc = inspect.getdoc(obj) or ""
    summary = doc.splitlines()[0] if doc else ""
    return f"- `{name}{sig}`" + (f" — {summary}" if summary else "")


# ── resources ─────────────────────────────────────────────────────────────────

@mcp.resource("wavekit://guide")
def wavekit_guide() -> str:
    """Wavekit analysis guide: typical workflows and task patterns.

    Read this resource at the start of a waveform analysis task to understand
    common patterns before writing code.
    """
    return """\
# wavekit Analysis Guide

## Session workflow

1. `open_session()` returns a `session_id`.
2. `run(session_id, code)` executes Python; state persists between calls.
3. Keep sessions open while a `Viewer` is visible; closing the session closes it.

Pre-injected names are intentionally small: `wavekit` and `Viewer` only.
Import everything else explicitly:

```python
import numpy as np
import wavekit
from wavekit.pattern import Pattern, match, collect, Channel, MatchStatus
```

---

## Open waveform files

```python
r = wavekit.VcdReader("/path/to/sim.vcd")
r = wavekit.FstReader("/path/to/sim.fst")
r = wavekit.FsdbReader("/path/to/sim.fsdb")
```

Do not use `with wavekit.VcdReader(...)` inside a persistent MCP session unless
you intend to close the reader at the end of that same `run()` call.

---

## Load waveforms

```python
clk = "tb.clk"
data = r.load_waveform("tb.dut.data[7:0]", clock=clk)

# Aligned numpy arrays
data.value   # sampled values
data.clock   # absolute clock cycles
data.time    # simulation timestamps
```

Useful options: `signed=True`, `xz_value=0`, `sample_on_posedge=True`,
`begin_time=...`, `end_time=...`, `begin_cycle=...`, `end_cycle=...`.

Unknown/X/Z mask loading is available when source unknown bits matter:

```python
value = r.load_waveform("tb.bus[7:0]", clock=clk, xz_value=0)
unknown = r.load_unknown_mask("tb.bus[7:0]", clock=clk)
known_value = value.mask(unknown == 0)
```

---

## Matched loading and query syntax

Use `signal_path` / `clock_path` terminology for matched loads:

```python
waves = r.load_matched_waveforms(
    signal_path="tb.dut.fifo_{0..3}.w_ptr[2:0]",
    clock_path="tb.clk",
)
for key, wave in waves.items():
    print(key, np.mean(wave.value))
```

Matched APIs return `dict[CaptureKey, ...]`. A `CaptureKey` is a tuple of typed
capture objects such as `BraceCapture`, `RegexCapture`, or `WildcardCapture`.
Use `str(key)` for reporting unless you need capture internals.

Query syntax:

| Syntax | Example | Meaning |
|--------|---------|---------|
| plain | `tb.dut.valid` | exact path |
| brace | `fifo_{0..3}.ptr` | alternatives/ranges |
| `/regex/` | `tb./lane_(\\d+)/.valid` | canonical regex |
| `@regex` | `tb.@(req|ack)` | legacy regex spelling |
| `*` / `**` | `tb.*.valid`, `tb.**.valid` | one-level / recursive wildcard |
| `$` / `$$` | `tb.$fifo.data`, `tb.$$fifo.data` | FSDB module-definition match |

Explore hierarchy with `r.top_scopes`, `r.get_matched_signals(path)`, and
`r.get_matched_scopes(path)`.

---

## Waveform operations

Use Waveform methods when you need to preserve time/clock arrays, especially for
Viewer display:

```python
active = data.mask(valid == 1)
window = data.cycle_slice(100, 500)
changes = state.compress()
small = data.downsample(500, np.mean)
```

Use numpy on `.value` for scalar statistics:

```python
print(np.mean(active.value))
print(np.histogram(data.value, bins=8))
```

Avoid printing full arrays; return counts, previews, histograms, or first failing
cycle.

---

## Declarative temporal matching

`Pattern` objects describe transaction shapes. Execute them with module-level
`match(...)`; timeout is also an argument to `match(...)`.

```python
from wavekit.pattern import Pattern, match, MatchStatus

ar_fire = arvalid & arready
r_fire = rvalid & rready

records = match(
    Pattern()
    .wait(ar_fire)
    .wait(r_fire)
    .capture("rdata", rdata),
    timeout=256,
)

ok = records.filter_ok()
failed = records.filter_failed()
print(f"ok={len(ok)} failed={len(failed)}")
print(f"latency cycles: {(ok.end.clock - ok.start.clock)[:8]}")
print(ok.captures["rdata"].value[:8])
```

Status values are objects. Use status classes, not enum constants:

```python
timeouts = records.filter_status(MatchStatus.Timeout)
require_failures = records.filter_status(MatchStatus.RequireViolated)
```

`wait(...)` observes an event. `consume(..., channel=...)` claims an event so
other matches cannot reuse it. Successful blocking steps continue in the same
cycle; add `.delay(1)` when next-cycle behavior is required.

---

## Programmable temporal extraction

Use `collect(body)` when transaction shape depends on values or branches.

```python
from wavekit.pattern import collect

cmd_fire = cmd_valid & cmd_ready
rsp_fire = rsp_valid & rsp_ready


def read_cmd(ctx):
    if not ctx.value(cmd_fire):
        return None
    addr = int(ctx.value(cmd_addr))
    ctx.consume(rsp_fire, channel="rsp")
    return {"addr": addr, "status": int(ctx.value(rsp_status))}

commands = collect(read_cmd, timeout=128)
print(f"commands={len(commands)}")
```

---

## Viewer

```python
viewer = Viewer()
viewer.waveforms.append(data)
viewer.markers.append(time=int(data.time[0]), name="start")
viewer.zoom_to_fit()
viewer.push_state()
print(viewer.url)
```

Keep the session open while the user is viewing.
"""


# ── entry point ───────────────────────────────────────────────────────────────

def _setup_logging(config: Config) -> None:
    log_cfg = config.log
    level = getattr(logging, log_cfg.level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_cfg.file:
        handlers.append(logging.FileHandler(log_cfg.file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=handlers,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="wavekit MCP server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Config file: ~/.config/wavekit-mcp/settings.toml (auto-created on first run)
        """,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"wavekit-mcp {__version__}",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to settings.toml (default: ~/.config/wavekit-mcp/settings.toml)",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default=None,
        help="Transport protocol (overrides config file)",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Host for streamable-http mode (overrides config file, default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for streamable-http mode (overrides config file, default: 8080)",
    )
    args = parser.parse_args()

    # Load config
    config = Config.load(args.config)
    _setup_logging(config)

    log = logging.getLogger("wavekit_mcp")
    log.info(
        "server_start config=%s",
        args.config or "<default>",
    )

    # Apply CLI overrides
    srv = config.server
    if args.transport:
        srv.transport = args.transport
    if args.host:
        srv.host = args.host
    if args.port:
        srv.port = args.port

    log.info("transport=%s", srv.transport)

    global _manager
    _manager = SessionManager(config)

    if srv.transport == "stdio":
        mcp.run(transport="stdio")
    elif srv.transport == "streamable-http":
        log.info("listening on %s:%d", srv.host, srv.port)
        mcp.run(transport="streamable-http", host=srv.host, port=srv.port)
    else:
        raise ValueError(
            f"Unknown transport '{srv.transport}'. "
            "Supported: stdio, streamable-http"
        )


if __name__ == "__main__":
    main()
