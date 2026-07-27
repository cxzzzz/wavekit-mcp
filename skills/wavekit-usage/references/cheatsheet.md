# wavekit-mcp cheatsheet

Use this when writing Python for the `run(session_id, code)` tool. Keep outputs compact.

## Setup

```python
import numpy as np
import wavekit
from wavekit.pattern import Pattern, match, collect, Channel, MatchStatus
```

`wavekit` and `Viewer` are pre-injected, but explicit imports make snippets clearer.

## Minimal end-to-end

```python
r = wavekit.VcdReader("/path/to/sim.vcd")
print([s.full_name for s in r.top_scopes])

# Find the signals you care about first, then use the result to load them.
found = r.load_matched_waveforms(signal_path="tb./data_(i|o)/", clock_path="tb.clk")
print([str(k) for k in found])

valid = r.load_waveform("tb.valid", clock="tb.clk")
data_i = next(w for k, w in found.items() if "data_i" in str(k))
data_o = next(w for k, w in found.items() if "data_o" in str(k))

active_i = data_i.mask(valid == 1)
print(f"active={len(active_i.value)} mean={np.mean(active_i.value):.2f}")

records = match(Pattern().wait(valid == 1).capture("in", data_i).capture("out", data_o), timeout=64)
print(f"match_ok={len(records.filter_ok())} total={len(records)}")

# collect() returns a plain list.
def grab(ctx):
    if not ctx.value(valid):
        return None
    return {"in": int(ctx.value(data_i)), "out": int(ctx.value(data_o))}

vals = collect(grab, timeout=64)
print(vals[:4])
```

## Open files

```python
r = wavekit.VcdReader("/path/to/sim.vcd")
r = wavekit.FstReader("/path/to/sim.fst")
r = wavekit.FsdbReader("/path/to/sim.fsdb")
```

Avoid `with wavekit.VcdReader(...)` in a persistent session unless the reader should close in the same call.

## Reader API

| Method | Use |
|--------|-----|
| `r.load_waveform(signal, clock, ...)` | Load one signal as `Waveform`. |
| `r.load_unknown_mask(signal, clock, ...)` | Load X/Z source-bit mask as unsigned `Waveform`. |
| `r.load_matched_waveforms(signal_path, clock_path, ...)` | Batch-load matching signals. |
| `r.load_matched_unknown_masks(signal_path, clock_path, ...)` | Batch-load X/Z masks. |
| `r.eval(expr, clock, mode="single"|"zip", ...)` | Evaluate signal-path expression. |
| `r.get_matched_signals(path)` | Resolve matching signals without loading values. |
| `r.get_matched_scopes(path)` | Resolve matching scopes. |
| `r.top_scopes` | Root hierarchy scopes. |

## Query syntax

| Syntax | Example | Notes |
|--------|---------|-------|
| Plain | `tb.dut.valid` | Exact path. |
| Brace list | `sig_{read,write}` | Captures chosen group. |
| Brace range | `fifo_{0..3}.ptr` | Inclusive integer range. |
| Brace step | `lane_{0..6..2}` | Stepped range. |
| `/regex/` | `tb./lane_(\d+)/.valid` | Canonical regex. |
| `@regex` | `tb.@(req|ack)` | Legacy regex spelling. |
| `*` / `**` | `tb.*.valid`, `tb.**.valid` | One-level / recursive wildcard. |
| `$` / `$$` | `tb.$fifo.data`, `tb.$$fifo.data` | FSDB module-definition match. |

Matched APIs return `dict[CaptureKey, ...]`; keys are tuples of typed capture objects. Use `.items()` or `.values()`, and use `str(key)` for reports.

```python
waves = r.load_matched_waveforms(
    signal_path="tb.dut.fifo_{0..3}.w_ptr[2:0]",
    clock_path="tb.clk",
)
for key, wave in waves.items():
    print(f"{key}: n={len(wave.value)} mean={np.mean(wave.value):.2f}")
```

## Waveform basics

```python
data = r.load_waveform("tb.dut.data[7:0]", clock="tb.clk")
data.value   # np.ndarray values
data.clock   # sampled cycle numbers
data.time    # simulation timestamps
```

Useful options: `signed=True`, `xz_value=0`, `sample_on_posedge=True`, `begin_time=...`, `end_time=...`, `begin_cycle=...`, `end_cycle=...`.

Preserve alignment with Waveform methods:

```python
active = data.mask(valid == 1)
window = data.cycle_slice(100, 500)
changes = state.compress()
small = data.downsample(500, np.mean)
```

Use numpy for scalar reductions:

```python
print(f"min={np.min(data.value)} max={np.max(data.value)} mean={np.mean(data.value):.2f}")
print(np.histogram(data.value, bins=8))
```

## X/Z analysis

```python
value = r.load_waveform("tb.bus[7:0]", clock="tb.clk", xz_value=0)
unknown = r.load_unknown_mask("tb.bus[7:0]", clock="tb.clk")
known = value.mask(unknown == 0)
print(f"unknown_samples={np.count_nonzero(unknown.value)} known_samples={len(known.value)}")
```

## Compare traces

```python
r1 = wavekit.VcdReader("/data/golden.vcd")
r2 = wavekit.VcdReader("/data/actual.vcd")
gold = r1.load_waveform("tb.data[7:0]", clock="tb.clk")
act = r2.load_waveform("tb.data[7:0]", clock="tb.clk")
match_wf = gold == act
idx = np.nonzero(match_wf.value == 0)[0]
print(f"mismatches={len(idx)}")
if len(idx):
    i = int(idx[0])
    print(f"first idx={i} cycle={gold.clock[i]} time={gold.time[i]} gold={gold.value[i]} act={act.value[i]}")
```

## Declarative pattern matching

`match(pattern, timeout=...)` returns `MatchRecords`.

```python
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
print(f"total={len(records)} ok={len(ok)} failed={len(failed)}")
if len(ok):
    latency = ok.end.clock - ok.start.clock
    print(f"latency min={np.min(latency)} max={np.max(latency)} mean={np.mean(latency):.1f}")
    print(f"rdata_preview={ok.captures['rdata'].value[:8]}")
```

Status handling:

```python
timeouts = records.filter_status(MatchStatus.Timeout)
require_failures = records.filter_status(MatchStatus.RequireViolated)
print(f"timeouts={len(timeouts)} require_failures={len(require_failures)}")
```

Use `.consume(cond, channel=...)` to claim events exclusively. Use `.delay(1)` when the next step must start on the next sampled cycle.

```python
beat = Pattern().consume(wvalid & wready, channel="w").capture("beats", wdata, mode="list")
records = match(
    Pattern()
    .wait(awvalid & awready)
    .loop(beat, until=wlast),
    timeout=512,
)
for i, rec in enumerate(records.filter_ok()[:5]):
    print(f"burst {i}: beats={len(rec.captures['beats'])}")
```

## Programmable collect

Use `collect(body, timeout=...)` when control flow depends on waveform values.

```python
cmd_fire = cmd_valid & cmd_ready
rsp_fire = rsp_valid & rsp_ready


def read_cmd(ctx):
    if not ctx.value(cmd_fire):
        return None
    addr = int(ctx.value(cmd_addr))
    length = int(ctx.value(cmd_len))
    ctx.consume(rsp_fire, channel="rsp")
    return {"addr": addr, "length": length, "status": int(ctx.value(rsp_status))}

commands = collect(read_cmd, timeout=128)
print(f"commands={len(commands)}")
print(commands[:5])
```

## Viewer

```python
viewer = Viewer()
viewer.waveforms.append(data)
viewer.markers.append(time=int(data.time[0]), name="start")
viewer.zoom_to_fit()
viewer.push_state()
print(f"View at: {viewer.url}")
```

Keep the MCP session open while the user is viewing. Closing the session closes the viewer.

## Output hygiene

Do print:

```python
print(len(data.value))
print(np.mean(data.value))
print(data.value[:16])
```

Do not print huge arrays or `.tolist()` on full waveforms. Use `compress()`, `downsample(...)`, `cycle_slice(...)`, or masks first.
