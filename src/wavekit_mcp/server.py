from __future__ import annotations

import argparse
import logging
import pydoc as _pydoc
import tempfile
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse

from .config import Config, get_default_config_path
from .session import SessionManager

# Get version from package metadata
try:
    __version__ = version("wavekit-mcp")
except PackageNotFoundError:
    __version__ = "unknown"

mcp = FastMCP("wavekit-mcp", version=__version__)

_manager: SessionManager | None = None
_plots_dir: Path | None = None


def _get_manager() -> SessionManager:
    """Get the global SessionManager instance."""
    if _manager is None:
        raise RuntimeError("Server not initialised.")
    return _manager


# ── tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def open_session() -> str:
    """Create a new persistent Python execution session for waveform analysis.

    Returns a session_id used by all other tools.

    Pre-injected objects available in every session:
      open_reader(path)  — open a .vcd or .fsdb file; auto-closed on reset/close
      VcdReader(path)    — open a VCD file directly; auto-closed on reset/close
      FsdbReader(path)   — open an FSDB file directly; auto-closed on reset/close
      np                 — numpy
      Pattern            — wavekit.Pattern  (temporal pattern matching DSL)
      MatchStatus        — wavekit.MatchStatus
      go                 — plotly.graph_objects (for creating figures)
      px                 — plotly.express (for quick plots)
      open(path, mode)   — standard file I/O (only if enabled in server config,
                           restricted to configured allowed paths)

    Typical workflow:
      1. sid = open_session()
      2. run(sid, "r = VcdReader('/data/sim.vcd')")
      3. run(sid, "data = r.load_waveform('tb.data[7:0]', clock='tb.clk')")
      4. run(sid, "print(np.mean(data.value))")
      5. close_session(sid)

    IMPORTANT: Do NOT use `import wavekit` — all wavekit objects are pre-injected.
    """
    return _get_manager().open_session()


@mcp.tool()
def close_session(session_id: str) -> str:
    """Close a session and release all resources (open readers, memory)."""
    _get_manager().close_session(session_id)
    return f"Session '{session_id}' closed."


@mcp.tool()
def reset_session(session_id: str) -> str:
    """Reset a session: close open readers and clear all user-defined variables.

    Pre-injected objects (open_reader, np, Pattern, MatchStatus) are restored.
    Use this to start a fresh analysis without creating a new session.
    """
    _get_manager().reset_session(session_id)
    return f"Session '{session_id}' reset."


@mcp.tool()
def run(session_id: str, code: str) -> dict[str, Any]:
    """Execute Python code in a persistent session. State persists across calls.

    PRE-INJECTED: VcdReader(path), FsdbReader(path), open_reader(path), np, Pattern, MatchStatus, go, px
    Do NOT use `import wavekit` — all objects are already available.
    UNFAMILIAR WITH THE API? Call get_api_docs(session_id) before writing code.

    OPEN FILES:
        r  = VcdReader("/path/to/sim.vcd")      # open VCD file
        r2 = FsdbReader("/path/to/ref.fsdb")    # open FSDB file
        # or use open_reader() for auto-detection by extension
        r3 = open_reader("/path/to/sim.vcd")    # .vcd → VcdReader, other → FsdbReader
        # multiple readers allowed; all auto-closed on reset/close

    VISUALIZATION (plotly):
        import plotly.graph_objects as go  # → use go directly (pre-injected)
        import plotly.express as px        # → use px directly (pre-injected)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1,2,3], y=[4,5,6]))
        # Then call save_plot(session_id, "fig") to get a viewable URL

    MULTI-CALL WORKFLOW:
        # call 1 — load
        r = VcdReader("sim.vcd")
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

    Call with no arguments to list all available topics.
    Call with a topic name for detailed docs:

      topic="Waveform"    — signal operations: filter, slice, bit ops, arithmetic
      topic="Reader"      — file loading: load_waveform, load_matched_waveforms, eval
      topic="Pattern"     — temporal pattern matching DSL
      topic="MatchResult" — pattern match output structure
      topic="Signal"      — signal metadata dataclass
      topic="Scope"       — hierarchy tree node
    """
    import wavekit
    from wavekit.readers.base import Reader

    topic_map: dict[str, Any] = {
        "Waveform": wavekit.Waveform,
        "Reader": Reader,
        "Pattern": wavekit.Pattern,
        "MatchResult": wavekit.MatchResult,
        "MatchStatus": wavekit.MatchStatus,
        "Signal": wavekit.Signal,
        "Scope": wavekit.Scope,
    }

    if not topic:
        lines = ["Available topics (pass as topic= argument):\n"]
        lines += [f"  {name}" for name in topic_map]
        lines += [
            "",
            "Example: get_api_docs(topic='Waveform')",
        ]
        return "\n".join(lines)

    if topic not in topic_map:
        return (
            f"Unknown topic '{topic}'.\n"
            f"Available: {list(topic_map.keys())}"
        )

    return _pydoc.render_doc(topic_map[topic], renderer=_pydoc.plaintext)


@mcp.tool()
def save_plot(
    session_id: str,
    figure_var: str,
    base_url: str = "http://localhost:8080",
) -> dict[str, str]:
    """Save a plotly Figure to interactive HTML and static PNG.

    Args:
        session_id: Session ID from open_session()
        figure_var: Name of the plotly Figure variable in the session
        base_url: Base URL for generating clickable links (default: http://localhost:8080)

    Returns:
        {
            "html_url": "http://localhost:8080/plots/plot_a1b2c3.html",
            "png_url": "http://localhost:8080/plots/plot_a1b2c3.png"  # or null if PNG failed
        }

    html_url: Interactive plot for viewing in browser. Download if you need long-term access.
    png_url: Static image for embedding in documents. Download if you need long-term access.

    Example:
        # First create a figure in the session
        run(sid, "fig = go.Figure()")
        run(sid, "fig.add_trace(go.Scatter(x=data.clock, y=data.value))")

        # Then save it
        result = save_plot(sid, "fig")
        # Tell user to open result["html_url"] in browser
    """
    return _get_manager().save_plot(session_id, figure_var, base_url)


# ── HTTP routes for plot serving ───────────────────────────────────────────────

@mcp.custom_route("/plots/{filename:path}", methods=["GET"])
async def serve_plot(request: Request) -> FileResponse:
    """Serve plot files (HTML and PNG)."""
    global _plots_dir
    if _plots_dir is None:
        return JSONResponse({"error": "Plots directory not initialized"}, status_code=500)

    filename = request.path_params.get("filename", "")
    if not filename:
        return JSONResponse({"error": "Filename required"}, status_code=400)

    # Security: prevent path traversal
    if ".." in filename or "/" in filename:
        return JSONResponse({"error": "Invalid filename"}, status_code=400)

    filepath = _plots_dir / filename
    if not filepath.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)

    # Determine content type
    if filename.endswith(".html"):
        media_type = "text/html"
    elif filename.endswith(".png"):
        media_type = "image/png"
    else:
        return JSONResponse({"error": "Unsupported file type"}, status_code=400)

    return FileResponse(filepath, media_type=media_type)


# ── resources ─────────────────────────────────────────────────────────────────

@mcp.resource("wavekit://guide")
def wavekit_guide() -> str:
    """Wavekit analysis guide: typical workflows and task patterns.

    Read this resource at the start of a waveform analysis task to understand
    common patterns before writing code.
    """
    return """\
# wavekit Analysis Guide

## Session Workflow

Always follow this structure:
  1. open_session() → sid
  2. run(sid, ...) — one or more calls, state persists between them
  3. close_session(sid) when done

Use reset_session(sid) to clear variables without reopening files.

---

## Opening Files

```python
# Single file — use VcdReader or FsdbReader directly (both are pre-injected)
r = VcdReader("/path/to/sim.vcd")
r = FsdbReader("/path/to/sim.fsdb")

# Or use open_reader() for auto-detection by file extension
r = open_reader("/path/to/sim.vcd")    # .vcd → VcdReader, other → FsdbReader

# Multiple files (e.g. golden vs actual comparison)
r_gold = VcdReader("/data/golden.vcd")
r_act  = VcdReader("/data/actual.vcd")
```

All readers are auto-closed on reset_session() / close_session().
Do NOT use `import wavekit` or `with VcdReader(...)` — just assign directly.

---

## Loading Waveforms

```python
# Single signal — sampled on negedge of clock by default
data = r.load_waveform("tb.dut.data[7:0]", clock="tb.clk")

# Access the three arrays
data.value   # np.ndarray of signal values (uint64 or int64)
data.clock   # np.ndarray of absolute clock cycle numbers
data.time    # np.ndarray of simulation timestamps

# Batch load — returns dict[tuple, Waveform]; key is tuple of captured values
waves = r.load_matched_waveforms("tb.dut.fifo_{0..3}.w_ptr[2:0]", clock_pattern="tb.clk")
for (idx,), wave in waves.items():
    print(f"fifo_{idx}: mean={np.mean(wave.value):.2f}")

# Computed expression — arithmetic on signal paths
occupancy = r.eval("tb.dut.w_ptr[3:0] - tb.dut.r_ptr[3:0]", clock="tb.clk")
```

### Signal path pattern syntax

| Syntax | Description | Example |
|--------|-------------|---------|
| `tb.dut.sig[7:0]` | Literal dotted path + bit range | `"tb.dut.data[7:0]"` |
| `{a,b,c}` | String alternatives → one key element each | `"tb.dut.J_{state,next}[3:0]"` |
| `{start..end}` | Integer range | `"tb.lane_{0..3}.valid"` |
| `{start..end..step}` | Integer range with step | `"tb.lane_{0..6..2}.valid"` |
| Multiple `{}` | Cartesian product of all expansions | `"tb.u{0,1}.fifo_{a,b}.cnt"` |
| `@<regex>` | Python `re.fullmatch()`; `(...)` groups captured | `r"tb.dut.@(req\|ack\|valid)"` |
| `$$ModName` | Any-depth scope by module def name (FSDB only) | `"$$axi_slave.rdata[31:0]"` |
| `$ModName` | Direct-child scope by module def name (FSDB only) | `"tb.dut.$pipe_stage"` |

---

## Common Analysis Patterns

### Basic statistics
```python
data = r.load_waveform("tb.dut.out[7:0]", clock="tb.clk")
print(f"min={np.min(data.value)}  max={np.max(data.value)}  mean={np.mean(data.value):.2f}")
print(np.histogram(data.value, bins=8))
```

### Filter to interesting cycles
```python
valid = r.load_waveform("tb.dut.valid", clock="tb.clk")
data  = r.load_waveform("tb.dut.data[7:0]", clock="tb.clk")

# Keep only cycles where valid=1
active = data.mask(valid == 1)
print(f"active cycles: {len(active.value)}  mean: {np.mean(active.value):.2f}")
```

### Detect transitions
```python
state = r.load_waveform("tb.dut.state[2:0]", clock="tb.clk")
changes = state.compress()   # one entry per distinct value run
print(f"state transitions: {len(changes.value) - 1}")
print(f"unique states: {np.unique(changes.value)}")
```

### Time-window analysis
```python
# By simulation time
window = data.time_slice(begin=1000, end=5000)

# By clock cycle
window = data.cycle_slice(begin=100, end=500)
```

### Compare two simulations
```python
r1 = VcdReader("/data/golden.vcd")
r2 = VcdReader("/data/actual.vcd")
gold = r1.load_waveform("tb.data[7:0]", clock="tb.clk")
act  = r2.load_waveform("tb.data[7:0]", clock="tb.clk")

match = (gold == act)
mismatch_cycles = match.mask(match == 0)
print(f"mismatches: {len(mismatch_cycles.value)}")
if len(mismatch_cycles.value) > 0:
    print(f"first mismatch at clock cycle: {mismatch_cycles.clock[0]}")
```

---

## Temporal Pattern Matching

Use `Pattern` to find transaction sequences (handshakes, latencies, bursts).

### AXI read latency
```python
arvalid = r.load_waveform("tb.arvalid",     clock="tb.clk")
arready = r.load_waveform("tb.arready",     clock="tb.clk")
rvalid  = r.load_waveform("tb.rvalid",      clock="tb.clk")
rready  = r.load_waveform("tb.rready",      clock="tb.clk")
rdata   = r.load_waveform("tb.rdata[31:0]", clock="tb.clk")

result = (
    Pattern()
    .wait(arvalid & arready)   # start: AR handshake
    .wait(rvalid  & rready)    # end:   R handshake
    .capture("data", rdata)
    .timeout(256)
    .match()
)

valid = result.filter_valid()
print(f"transactions : {len(valid.duration.value)}")
print(f"latency mean : {np.mean(valid.duration.value):.1f} cycles")
print(f"latency max  : {np.max(valid.duration.value)} cycles")
print(f"captured data preview: {valid.captures['data'].value[:8]}")
```

### AXI write burst (variable-length)
```python
awvalid = r.load_waveform("tb.awvalid", clock="tb.clk")
awready = r.load_waveform("tb.awready", clock="tb.clk")
wvalid  = r.load_waveform("tb.wvalid",  clock="tb.clk")
wready  = r.load_waveform("tb.wready",  clock="tb.clk")
wlast   = r.load_waveform("tb.wlast",   clock="tb.clk")
wdata   = r.load_waveform("tb.wdata[31:0]", clock="tb.clk")

beat = Pattern().wait(wvalid & wready).capture("beats[]", wdata)

result = (
    Pattern()
    .wait(awvalid & awready)     # AW handshake
    .loop(beat, until=wlast)     # collect beats until wlast
    .timeout(512)
    .match()
)

valid = result.filter_valid()
for i, beats in enumerate(valid.captures["beats"].value[:5]):
    print(f"burst {i}: {len(beats)} beats, data={beats}")
```

### Handshake with guard (require valid stays high)
```python
result = (
    Pattern()
    .wait(req, guard=enable)     # wait for req; fail if enable drops
    .wait(ack)
    .timeout(64)
    .match()
)

ok = result.filter_valid()
violated = result.status.mask(result.status == MatchStatus.REQUIRE_VIOLATED)
print(f"ok={len(ok.duration.value)}  guard_violations={len(violated.value)}")
```

---

## Working with Large Waveforms

Output and result previews are limited. Reduce data before inspecting:

```python
# Remove consecutive duplicates (good for state/control signals)
data.compress()

# Aggregate to N points (good for data/bus signals)
data.downsample(500, np.mean)

# Zoom into a region of interest
data.cycle_slice(1000, 2000)

# Print scalars, not arrays
print(np.mean(data.value))          # ✓
print(data.value.tolist())          # ✗ truncated
```

---

## Visualization

Use `go` (plotly.graph_objects) and `px` (plotly.express) for visualization.
Both are pre-injected — no import needed.

### Basic signal plot
```python
# Load waveform data
data = r.load_waveform("tb.dut.signal[7:0]", clock="tb.clk")

# Create figure
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=data.clock,
    y=data.value,
    mode='lines+markers',
    name='signal'
))
fig.update_layout(
    title="Signal Waveform",
    xaxis_title="Clock Cycle",
    yaxis_title="Value"
)

# Save and get URL for user
# Call save_plot(session_id, "fig", base_url="http://localhost:8080")
```

### Multiple signals comparison
```python
signal_a = r.load_waveform("tb.dut.a[7:0]", clock="tb.clk")
signal_b = r.load_waveform("tb.dut.b[7:0]", clock="tb.clk")

fig = go.Figure()
fig.add_trace(go.Scatter(x=signal_a.clock, y=signal_a.value, name='a'))
fig.add_trace(go.Scatter(x=signal_b.clock, y=signal_b.value, name='b'))
fig.update_layout(title="Signal Comparison")
```

### Histogram / distribution
```python
fig = px.histogram(x=data.value, nbins=20, title="Value Distribution")
```

### Pattern match results
```python
result = Pattern().wait(req).wait(ack).timeout(64).match()
valid = result.filter_valid()

fig = go.Figure()
fig.add_trace(go.Bar(
    x=list(range(len(valid.duration.value))),
    y=valid.duration.value
))
fig.update_layout(
    title="Transaction Latency",
    xaxis_title="Transaction #",
    yaxis_title="Cycles"
)
```

### Quick plots with px
```python
# Scatter plot
fig = px.scatter(x=data.clock, y=data.value, title="Signal Over Time")

# Line plot
fig = px.line(x=data.clock, y=data.value, title="Signal Trace")

# Bar chart
fig = px.bar(x=['a', 'b', 'c'], y=[10, 20, 15], title="Comparison")
```

### save_plot usage
```python
# After creating fig = go.Figure(...) or fig = px.line(...)
result = save_plot(sid, "fig", base_url="http://localhost:8080")

# Returns:
# {
#   "html_url": "http://localhost:8080/plots/plot_xxx.html",
#   "png_url": "http://localhost:8080/plots/plot_xxx.png"
# }

# Tell user to open html_url in browser for interactive viewing.
# Download png_url if they need to embed in documents.
```

> **Note:** URLs are temporary — valid only while the MCP server is running.
> Download files if long-term access is needed.

---

## Reading Log Files (if file access is enabled)

```python
with open("/data/sim/run.log") as f:
    log = f.read()

# Parse relevant lines
lines = [l for l in log.splitlines() if "ERROR" in l]
print(f"error lines: {len(lines)}")
for l in lines[:5]:
    print(l)
```

---

## Finding Signals by Module Name (FSDB only)

When analysing FSDB files you often know the *module type* but not the full
instance path. Use `$` / `$$` prefixes to search by module name:

```
$ModName    — direct-child scope whose module definition name is ModName
$$ModName   — any-depth descendant scope with that module name
```

This is the fastest way to locate signals when you only know the RTL module name.

### Get all instances of a module

```python
r = FsdbReader("/data/sim.fsdb")

# Find every instance of module "axi_slave" anywhere in the hierarchy
scopes = r.get_matched_scopes("$$axi_slave")
for key, scope in scopes.items():
    print(scope.full_name())   # prints the full instance path
```

### Load a signal from all instances of a module

```python
# Load "data_out[7:0]" from every instance of "fifo_unit" in the design
waves = r.load_matched_waveforms(
    "$$fifo_unit.data_out[7:0]",
    clock_pattern="tb.clk",
)
for key, wave in waves.items():
    print(f"instance key={key}  mean={np.mean(wave.value):.2f}")
```

### Narrow to direct children only

```python
# Only instances of "pipe_stage" that are direct children of "tb.dut"
scopes = r.get_matched_scopes("tb.dut.$pipe_stage")
```

### Combine with brace/regex patterns

```python
# All instances of either "fifo_a" or "fifo_b" anywhere in hierarchy
waves = r.load_matched_waveforms(
    "$$fifo_{a,b}.w_ptr[3:0]",
    clock_pattern="tb.clk",
)
```

> **Note:** `$` / `$$` rely on the module *definition* name stored in the FSDB
> (the `def_name` attribute). They are not available for VCD files, which do
> not record module type information.

---

## Tips

- **Signal path**: use full dotted path e.g. `"tb.dut.sub.signal[7:0]"`. If unsure,
  call `r.top_scope_list()` to traverse the hierarchy, or use `r.get_matched_signals("tb.@(.*)")`.
- **Signed values**: pass `signed=True` to `load_waveform()` for two's-complement signals.
- **X/Z values**: defaulted to 0; override with `xz_value=` parameter.
- **Clock edge**: default is negedge (stable value capture); use `sample_on_posedge=True` if needed.
- **Pattern result keys**: `captures["name"]` is a Waveform; for list captures (`"name[]"`),
  each element of `.value` is a Python list of beat values.
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

    # Create unique plots directory
    global _plots_dir
    if srv.plots_dir:
        _plots_dir = Path(srv.plots_dir)
        _plots_dir.mkdir(parents=True, exist_ok=True)
    else:
        _plots_dir = Path(tempfile.mkdtemp(prefix="wavekit_plots_"))
    srv.plots_dir = str(_plots_dir)  # Pass to workers via config
    log.info("plots_dir=%s", _plots_dir)

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
