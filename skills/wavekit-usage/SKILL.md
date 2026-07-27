---
name: wavekit-usage
description: Use this skill whenever the user wants to inspect, debug, analyze, compare, or visualize hardware simulation waveforms with wavekit-mcp, including VCD/FST/FSDB files, signal paths, waveform statistics, X/Z analysis, temporal pattern matching, handshakes, latency, bursts, protocol debugging, or viewer-based waveform inspection.
---

# Wavekit MCP Usage

Use the wavekit MCP tools to analyze large hardware simulation waveforms by running small Python snippets in a persistent session. Return compact answers: counts, ranges, histograms, first failing cycles, and short previews — not raw traces.

## Core workflow

1. `open_session(description=...)`.
2. `run(session_id, code)` to import APIs, open files, load signals, and compute summaries.
3. Use `Viewer` only when visual inspection is needed.
4. Keep the session open while a viewer is visible.
5. `close_session(session_id)` after analysis and viewer use are done.

## Session imports

Only `wavekit` and `Viewer` are pre-injected. Import what you need explicitly:

```python
import numpy as np
import wavekit
from wavekit.pattern import Pattern, match, collect, Channel, MatchStatus
```

Do not use wavekit 0.6-era chain execution, old valid filters, old matched-load clock parameter names, or old hierarchy traversal helpers.

## Defaults to prefer

- Open files as `wavekit.VcdReader(path)`, `wavekit.FstReader(path)`, or `wavekit.FsdbReader(path)`.
- Use `load_waveform(path, clock=...)` for single signals.
- Use `load_matched_waveforms(signal_path=..., clock_path=...)` for groups.
- Use `load_unknown_mask(...)` when X/Z source bits matter.
- Use Waveform methods (`mask`, `filter`, `cycle_slice`, `time_slice`, `compress`, `downsample`) when preserving time/clock alignment matters.
- Use numpy on `.value` only for statistics and reductions.

## Detailed examples

Read `references/cheatsheet.md` when you need concrete code for:

- Reader APIs and query syntax
- `CaptureKey` handling
- Waveform operations and output hygiene
- Unknown mask analysis
- Declarative `Pattern` + `match(...)`
- Programmable `collect(...)`
- Viewer usage

## Minimal real workflow

If you need a quick end-to-end path, use this shape:

```python
import numpy as np
import wavekit
from wavekit.pattern import Pattern, match, collect

r = wavekit.VcdReader("/path/to/sim.vcd")
# Find the signals you care about first, then use the result to load them.
found = r.load_matched_waveforms(signal_path="tb./data_(i|o)/", clock_path="tb.clk")
print([str(k) for k in found])

valid = r.load_waveform("tb.valid", clock="tb.clk")
data_i = next(w for k, w in found.items() if "data_i" in str(k))
data_o = next(w for k, w in found.items() if "data_o" in str(k))

active_i = data_i.mask(valid == 1)
print(f"active={len(active_i.value)} mean={np.mean(active_i.value):.2f}")

records = match(Pattern().wait(valid == 1).capture("in", data_i).capture("out", data_o), timeout=64)
print(f"ok={len(records.filter_ok())} total={len(records)}")

def grab(ctx):
    if not ctx.value(valid):
        return None
    return {"in": int(ctx.value(data_i)), "out": int(ctx.value(data_o))}

vals = collect(grab, timeout=64)
print(vals[:4])
```
