# wavekit-mcp

English | [中文](./README_ZH.md)

An MCP server that gives AI assistants a persistent, sandboxed Python environment for waveform analysis using [wavekit](https://github.com/cxzzzz/wavekit).

The AI can open VCD/FST/FSDB files, load and manipulate waveforms, run temporal pattern matching, and iterate across multiple tool calls — all within a shared execution context that persists state between calls.

## Why wavekit-mcp?

**The problem:** Digital waveforms are huge. A single simulation can produce millions of transitions across thousands of signals. Sending this data to an LLM directly is both inefficient and ineffective — the AI sees noise, not insight.

**Our approach:** Give the AI tools, not data. wavekit-mcp exposes wavekit's full waveform analysis capabilities through a persistent Python session. The AI writes code to:
- Load signals from VCD/FST/FSDB files
- Apply temporal pattern matching
- Compute statistics, detect anomalies, extract events

The AI gets only the answers it asks for — a mean, a timing violation, a filtered subset — never the raw waveform. Output limits ensure the AI must think in terms of signal semantics, not value sequences.

## Installation

```bash
pip install wavekit-mcp
```

Start the server:

```bash
wavekit-mcp
wavekit-mcp --config /path/to/wavekit_mcp.toml
```

MCP client example:

```json
{
  "mcpServers": {
    "wavekit": {
      "command": "wavekit-mcp",
      "args": ["--config", "/path/to/wavekit_mcp.toml"]
    }
  }
}
```

## Configuration

Copy `wavekit_mcp.toml.example` and edit as needed. All fields are optional.

```toml
[limits]
max_sessions         = 8
run_timeout_sec      = 120
output_max_chars     = 4000

[file_access]
read_enabled         = false
write_enabled        = false
read_allowed_paths   = ["/tmp/**"]
write_allowed_paths  = ["/tmp/**"]

[log]
file  = ""      # empty = stderr only
level = "INFO"

[sandbox]
# Defaults already allow wavekit, wavekit.*, numpy, numpy.*
# allowed_imports = ["plotly", "matplotlib.*"]
```

Scalar fields can be overridden by environment variables:

```bash
WAVEKIT_MCP_RUN_TIMEOUT_SEC=300 wavekit-mcp
```

## Tools

| Tool | Description |
|------|-------------|
| `open_session(description?)` | Create a persistent Python execution session. |
| `close_session(session_id)` | Close a session and release worker resources. |
| `list_sessions()` | List active sessions. |
| `run(session_id, code)` | Execute Python and return `{result, output, error, duration_ms}`. |
| `get_history(session_id, last_n)` | Return recent execution records. |
| `get_api_docs(topic)` | Inspect wavekit Reader/Waveform/pattern API docs. |

Each session pre-injects only:

- `wavekit` — use `wavekit.VcdReader`, `wavekit.Waveform`, etc.
- `Viewer` — optional waveform visualization helper.

Import everything else explicitly:

```python
import numpy as np
import wavekit
from wavekit.pattern import Pattern, match, collect, Channel, MatchStatus
```

`run()` returns the last expression in a REPL-like form: the value is displayed as truncated `repr(...)` text; the real Python objects remain in the session namespace.

## Basic usage

```python
# call 1
import numpy as np
import wavekit

r = wavekit.VcdReader("/data/sim.vcd")
data = r.load_waveform("tb.dut.data[7:0]", clock="tb.clk")

# call 2 — state persists
print(f"samples={len(data.value)} mean={np.mean(data.value):.2f}")
```

## Reader examples

### Matched loading

```python
waves = r.load_matched_waveforms(
    signal_path="tb.dut.fifo_{0..3}.w_ptr[2:0]",
    clock_path="tb.clk",
)
for key, wave in waves.items():
    print(f"{key}: mean={np.mean(wave.value):.2f}")
```

Matched APIs return `dict[CaptureKey, ...]`. `CaptureKey` is a tuple of typed captures such as `BraceCapture`, `RegexCapture`, or `WildcardCapture`.

### Unknown/X/Z masks

```python
value = r.load_waveform("tb.bus[7:0]", clock="tb.clk", xz_value=0)
unknown = r.load_unknown_mask("tb.bus[7:0]", clock="tb.clk")
known_value = value.mask(unknown == 0)

unknowns = r.load_matched_unknown_masks(
    signal_path="tb.dut.fifo_{0..3}.data[7:0]",
    clock_path="tb.clk",
)
```

### Expressions

```python
occupancy = r.eval(
    "tb.dut.w_ptr[3:0] - tb.dut.r_ptr[3:0]",
    clock="tb.clk",
)

occupancies = r.eval(
    "tb.fifo_{0..3}.w_ptr[2:0] - tb.fifo_{0..3}.r_ptr[2:0]",
    clock="tb.clk",
    mode="zip",
)
```

### Query syntax

| Syntax | Example | Meaning |
|--------|---------|---------|
| Plain path | `tb.dut.valid` | Exact signal/scope path. |
| Brace | `fifo_{0..3}.ptr` | Alternatives or integer ranges. |
| `/regex/` | `tb./lane_(\d+)/.valid` | Canonical regex with captures. |
| `@regex` | `tb.@(req|ack)` | Legacy regex spelling. |
| `*` / `**` | `tb.*.valid`, `tb.**.valid` | One-level / recursive wildcard. |
| `$` / `$$` | `tb.$fifo.data`, `tb.$$fifo.data` | FSDB module-definition match. |

Use `r.top_scopes`, `r.get_matched_signals(path)`, and `r.get_matched_scopes(path)` to explore hierarchy.

## Pattern matching

`Pattern` builds declarative timing checks. Execute with module-level `match(...)`.

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
print(f"transactions={len(ok)}")
print(f"latencies={ok.end.clock - ok.start.clock}")
print(ok.captures["rdata"].value[:8])

timeouts = records.filter_status(MatchStatus.Timeout)
require_failures = records.filter_status(MatchStatus.RequireViolated)
```

Use `consume(..., channel=...)` when a match must claim an event exclusively. Successful blocking steps continue in the same cycle; use `.delay(1)` for next-cycle behavior.

For value-dependent flows, use programmable `collect(...)`:

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

## Viewer

```python
viewer = Viewer()
viewer.waveforms.append(data)
viewer.markers.append(time=int(data.time[0]), name="start")
viewer.zoom_to_fit()
viewer.push_state()
print(viewer.url)
```

Keep the session open while the user is viewing; closing the session closes the viewer.

## Security

User code runs under RestrictedPython in a worker process. Imports are restricted by `sandbox.allowed_imports`; file I/O is disabled unless explicitly enabled in `[file_access]`.

This prevents accidental operations and isolates crashes, but it is not a complete sandbox for hostile code.

## AI assistant skill

This repository includes a wavekit-mcp skill:

- [skills/wavekit-usage/SKILL.md](./skills/wavekit-usage/SKILL.md)
- [skills/wavekit-usage/references/cheatsheet.md](./skills/wavekit-usage/references/cheatsheet.md)
