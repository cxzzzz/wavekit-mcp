# wavekit-mcp

An MCP server that gives AI assistants a persistent, sandboxed Python environment for waveform analysis using [wavekit](https://github.com/xxx/wavekit).

The AI can open VCD/FSDB files, load and manipulate waveforms, run temporal pattern matching, and iterate across multiple tool calls — all within a shared execution context that persists state between calls.

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
| `open_session()` | Create a session; returns `session_id` |
| `close_session(sid)` | Release all resources |
| `reset_session(sid)` | Clear variables, keep session |
| `run(sid, code)` | Execute Python; returns `{result, output, error, duration_ms}` |
| `get_history(sid, n)` | Last N execution records |
| `get_api_docs(topic)` | wavekit API reference |

Every session has these pre-injected: `open_reader(path)`, `np`, `Pattern`, `MatchStatus`.

`run()` returns structured summaries for large objects rather than raw data — the Waveform, ndarray, and MatchResult objects stay in the session namespace for further processing.

## Usage Examples

### Load and analyse

```python
# call 1
r = open_reader("/data/sim.vcd")
data = r.load_waveform("tb.dut.data[7:0]", clock="tb.clk")

# call 2 — state persists
print(f"samples={len(data.value)}  mean={np.mean(data.value):.2f}")
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

Code runs under [RestrictedPython](https://restrictedpython.readthedocs.io/): `import` is blocked, `__class__` / `__bases__` access is blocked, and file I/O is disabled by default. Designed to prevent accidental operations, not to sandbox fully untrusted code.
