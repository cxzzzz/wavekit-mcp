# wavekit-mcp

English | [中文](./README_ZH.md)

An MCP server that gives AI assistants a persistent, sandboxed Python environment for waveform analysis using [wavekit](https://github.com/cxzzzz/wavekit).

The AI can open VCD/FSDB files, load and manipulate waveforms, run temporal pattern matching, and iterate across multiple tool calls — all within a shared execution context that persists state between calls.

## Why wavekit-mcp?

**The problem:** Digital waveforms are huge. A single simulation can produce millions of transitions across thousands of signals. Sending this data to an LLM directly is both inefficient and ineffective — the AI sees noise, not insight.

**Our approach:** Give the AI tools, not data. wavekit-mcp exposes wavekit's full waveform analysis capabilities through a persistent Python session. The AI writes code to:
- Load signals from VCD/FSDB files
- Apply temporal pattern matching
- Compute statistics, detect anomalies, extract events

The AI gets only the answers it asks for — a mean, a timing violation, a filtered subset — never the raw waveform. Output limits ensure the AI must think in terms of signal semantics, not value sequences.

## Installation

```bash
pip install wavekit-mcp
```

Start the server:

```bash
wavekit-mcp                              # defaults
wavekit-mcp --config wavekit_mcp.toml   # custom config
```

Register with your MCP client (e.g. Claude Desktop):

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
max_sessions         = 5
run_timeout_sec      = 120
output_max_chars     = 500
result_preview_items = 30

[file_access]
read_enabled         = false
write_enabled        = false
read_allowed_paths   = ["/tmp/**"]
write_allowed_paths  = ["/tmp/**"]

[log]
file  = "/var/log/wavekit_mcp.log"   # empty = stderr only
level = "INFO"                        # DEBUG logs full code + result per run
```

Scalar fields can be overridden via environment variable:

```bash
WAVEKIT_MCP_RUN_TIMEOUT_SEC=300 wavekit-mcp
```

## Tools

| Tool | Description |
|------|-------------|
| `open_session(description?)` | Create a session; returns `session_id` |
| `close_session(sid)` | Release all resources |
| `list_sessions()` | List all active sessions with id, description, created_at |
| `run(sid, code)` | Execute Python; returns `{result, output, error, duration_ms}` |
| `get_history(sid, n)` | Last N execution records |
| `get_api_docs(topic)` | wavekit API reference |

Every session has these pre-injected: `wavekit`, `Pattern`, `VcdReader`, `FsdbReader`, `Viewer`.

Use `wavekit.MatchStatus`, `wavekit.Waveform`, etc. for other types.

`numpy` is available via default allowed_imports: `import numpy as np`.

`run()` returns structured summaries for large objects rather than raw data — the Waveform, ndarray, and MatchResult objects stay in the session namespace for further processing.

## Usage Examples

### Load and analyse

```python
# call 1
r = VcdReader("/data/sim.vcd")
data = r.load_waveform("tb.dut.data[7:0]", clock="tb.clk")

# call 2 — state persists
print(f"samples={len(data.value)}")
```

### Pattern matching (AXI read latency)

```python
arvalid = r.load_waveform("tb.arvalid",     clock="tb.clk")
arready = r.load_waveform("tb.arready",     clock="tb.clk")
rvalid  = r.load_waveform("tb.rvalid",      clock="tb.clk")
rready  = r.load_waveform("tb.rready",      clock="tb.clk")

result = (
    Pattern()
    .wait(arvalid & arready)
    .wait(rvalid  & rready)
    .timeout(256)
    .match()
)

valid = result.filter_valid()
print(f"transactions={len(valid.duration.value)}  mean={np.mean(valid.duration.value):.1f} cycles")
```

## Security

Code runs under [RestrictedPython](https://restrictedpython.readthedocs.io/): `import` is blocked by default, `__class__` / `__bases__` access is blocked, and file I/O is disabled by default. Designed to prevent accidental operations, not to sandbox fully untrusted code.

### Relaxing restrictions

To allow specific imports, add to your config:

```toml
[sandbox]
allowed_imports = ["plotly.*", "matplotlib.*"]  # glob patterns
# allowed_imports = ["*"]  # allow all imports
```

## More

See [SKILLS.md](./SKILLS.md) for a cheatsheet of common patterns.
