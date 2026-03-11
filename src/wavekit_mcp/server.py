from __future__ import annotations

import argparse
import logging
import pydoc as _pydoc
from typing import Any

from fastmcp import FastMCP

from .config import Config
from .session import SessionManager

mcp = FastMCP("wavekit-mcp")

_manager: SessionManager | None = None


def _get() -> SessionManager:
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
      np                 — numpy
      Pattern            — wavekit.Pattern  (temporal pattern matching DSL)
      MatchStatus        — wavekit.MatchStatus
      open(path, mode)   — standard file I/O (only if enabled in server config,
                           restricted to configured allowed paths)

    Typical workflow:
      1. sid = open_session()
      2. run(sid, "r = open_reader('/data/sim.vcd')")
      3. run(sid, "data = r.load_waveform('tb.data[7:0]', clock='tb.clk')")
      4. run(sid, "print(np.mean(data.value))")
      5. close_session(sid)
    """
    return _get().open_session()


@mcp.tool()
def close_session(session_id: str) -> str:
    """Close a session and release all resources (open readers, memory)."""
    _get().close_session(session_id)
    return f"Session '{session_id}' closed."


@mcp.tool()
def reset_session(session_id: str) -> str:
    """Reset a session: close open readers and clear all user-defined variables.

    Pre-injected objects (open_reader, np, Pattern, MatchStatus) are restored.
    Use this to start a fresh analysis without creating a new session.
    """
    _get().reset_session(session_id)
    return f"Session '{session_id}' reset."


@mcp.tool()
def run(session_id: str, code: str) -> dict[str, Any]:
    """Execute Python code in a persistent session. State persists across calls.

    PRE-INJECTED: open_reader(path), np, Pattern, MatchStatus

    OPEN FILES:
        r  = open_reader("/path/to/sim.vcd")    # .vcd → VcdReader
        r2 = open_reader("/path/to/ref.fsdb")   # other → FsdbReader
        # multiple readers allowed; all auto-closed on reset/close

    MULTI-CALL WORKFLOW:
        # call 1 — load
        r = open_reader("sim.vcd")
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
    r = _get().run(session_id, code)
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
    return _get().get_history(session_id, last_n)


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
# Single file
r = open_reader("/path/to/sim.vcd")    # .vcd → VcdReader

# Multiple files (e.g. golden vs actual comparison)
r_gold = open_reader("/data/golden.vcd")
r_act  = open_reader("/data/actual.vcd")
```

All readers are auto-closed on reset_session() / close_session().

---

## Loading Waveforms

```python
# Single signal — sampled on negedge of clock by default
data = r.load_waveform("tb.dut.data[7:0]", clock="tb.clk")

# Access the three arrays
data.value   # np.ndarray of signal values (uint64 or int64)
data.clock   # np.ndarray of absolute clock cycle numbers
data.time    # np.ndarray of simulation timestamps

# Batch load with brace expansion — returns dict[tuple, Waveform]
waves = r.load_matched_waveforms(
    "tb.dut.fifo_{0..3}.w_ptr[2:0]",
    clock_pattern="tb.clk",
)
for (idx,), wave in waves.items():
    print(f"fifo_{idx}: mean={np.mean(wave.value):.2f}")

# Computed expression — arithmetic on signal paths
occupancy = r.eval(
    "tb.dut.w_ptr[3:0] - tb.dut.r_ptr[3:0]",
    clock="tb.clk",
)
```

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
r1 = open_reader("/data/golden.vcd")
r2 = open_reader("/data/actual.vcd")
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
r = open_reader("/data/sim.fsdb")

# Find every instance of module "axi_slave" anywhere in the hierarchy
scopes = r.get_matched_scope("$$axi_slave")
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
scopes = r.get_matched_scope("tb.dut.$pipe_stage")
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
    parser = argparse.ArgumentParser(description="wavekit MCP server")
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to a TOML config file (optional; defaults apply if omitted)",
    )
    args = parser.parse_args()

    config = Config.load(args.config)
    _setup_logging(config)

    log = logging.getLogger("wavekit_mcp")
    log.info("server_start config=%s", args.config or "<defaults>")

    global _manager
    _manager = SessionManager(config)

    mcp.run()


if __name__ == "__main__":
    main()
