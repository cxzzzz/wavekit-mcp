---
name: wavekit-usage
description: Use this skill whenever the user wants to inspect, debug, analyze, compare, or visualize hardware simulation waveforms using wavekit-mcp, including VCD, FST, FSDB, waveform viewers, signal statistics, temporal pattern matching, handshakes, latency, bursts, or hardware simulation debugging. Use it even if the user only mentions waveform files, signal paths, RTL simulation traces, VCD/FST/FSDB, AXI timing, or wants to open signals in a viewer.
---

# Wavekit MCP Usage

Use the wavekit MCP tools to analyze hardware simulation waveforms through a persistent Python session. The goal is to ask focused questions of large waveform files and return compact, meaningful results instead of raw traces.

## Core workflow

1. Open a session with a useful description.
2. Open the waveform file with `VcdReader(path)`, `FstReader(path)`, or `FsdbReader(path)`.
3. Load only the signals needed for the user's question.
4. Compute scalar summaries, filtered events, compact tables, or temporal matches.
5. Use `Viewer` when the user needs visual inspection.
6. Keep the session open while the user is viewing waveforms.
7. Close the session only after the user is done with any viewer.

## Important constraints

- Do not use `import wavekit`; `wavekit`, `Pattern`, `Channel`, `VcdReader`, `FstReader`, `FsdbReader`, and `Viewer` are pre-injected.
- Do not use `with VcdReader(...)`, `with FstReader(...)`, or `with FsdbReader(...)`.
- Use Waveform methods when preserving time/clock alignment matters, especially before sending data to `Viewer`.
- Use numpy on `.value` only for statistics and reductions.
- Avoid printing large arrays. Print summaries, counts, histograms, ranges, first failing cycle, and small representative samples.
- Prefer `load_matched_waveforms`, `get_matched_signals`, and `get_matched_scopes` when the user asks about groups of signals or does not know exact paths.

## Common task patterns

For basic inspection, load the relevant signal and print scalar statistics:

```python
r = VcdReader("/path/to/sim.vcd")
data = r.load_waveform("tb.dut.data[7:0]", clock="tb.clk")
print(f"samples={len(data.value)} min={np.min(data.value)} max={np.max(data.value)} mean={np.mean(data.value):.2f}")
```

For active-cycle analysis, load the valid/enable signal and mask the data waveform:

```python
valid = r.load_waveform("tb.dut.valid", clock="tb.clk")
data = r.load_waveform("tb.dut.data[7:0]", clock="tb.clk")
active = data.mask(valid == 1)
print(f"active_cycles={len(active.value)} mean={np.mean(active.value):.2f}")
```

For visual inspection, create a viewer and keep the session open:

```python
viewer = Viewer()
viewer.waveforms.append(data)
viewer.zoom_to_fit()
viewer.push_state()
print(f"View at: {viewer.url}")
```

## Reference

Read `references/cheatsheet.md` for detailed examples covering:

- Session setup and viewer control
- VCD/FST/FSDB readers
- Signal path pattern syntax
- Waveform methods and arithmetic
- Batch loading and computed expressions
- Temporal pattern matching
- Large-data output hygiene
