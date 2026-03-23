# Switch Back to pyvcd for VCD Generation

## Problem

The custom `VcdWriter` class (~200 lines) has maintenance overhead and still has bugs. The original reason for replacing pyvcd was that it "skips duplicate values", but this is actually the correct behavior for VCD format.

The real issue was that pyvcd's value-change-only approach meant the final timestamp might not be recorded if the last value didn't change, causing the waveform to appear truncated.

## Solution

Switch back to pyvcd with a simple fix: pad `'x'` at both ends of the waveform to ensure the full time range is captured.

## Design

### Keep
- `transform_signal_name()` — Converts `[7:0]` to `_7_0_` for WCP compatibility
- `generate_merged_vcd()` — High-level API, returns `(path, name_mapping)`
- `SignalInfo` dataclass (simplified, for metadata only)

### Remove
- Entire custom `VcdWriter` class (~200 lines)
- Manual scope nesting, var_id generation, value formatting

### New Implementation

```python
from vcd import VCDWriter

def generate_merged_vcd(waveforms, output_path=None, timescale="1ps"):
    if not waveforms:
        raise ValueError("No waveforms to export")

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".vcd", prefix="wavekit_viewer_")
        os.close(fd)

    name_mapping = {}

    with open(output_path, 'w') as f:
        with VCDWriter(f, timescale=timescale) as writer:
            # Register all variables
            vars = {}
            for wf in waveforms:
                full_name = wf.signal.full_name
                wcp_name = transform_signal_name(full_name)
                name_mapping[full_name] = wcp_name

                scope, name = parse_scope_name(wcp_name)
                var = writer.register_var(scope, name, 'wire', wf.width or 1)
                vars[full_name] = var

            # Write values (use compress() for efficiency)
            for wf in waveforms:
                full_name = wf.signal.full_name
                var = vars[full_name]

                # Compress to get only value changes
                compressed = wf.compress()
                times = compressed.time
                values = compressed.value

                if len(times) == 0:
                    continue

                start_t = int(times[0])
                end_t = int(times[-1])

                # Pad x at start (ensures start time is recorded)
                writer.change(var, start_t - 1, 'x')

                # Write actual value changes
                for t, v in zip(times, values):
                    writer.change(var, int(t), int(v))

                # Pad x at end (ensures end time is recorded)
                writer.change(var, end_t + 1, 'x')

    return output_path, name_mapping
```

### Key Points

1. **`wf.compress()`** — Only write value changes, not every sample. This is what pyvcd expects and produces optimal VCD output.

2. **`'x'` padding** — pyvcd automatically formats `'x'` correctly for any width:
   - 1-bit: `x`
   - N-bit: `bx`

3. **Time range** — The `-1` and `+1` padding ensures pyvcd records both start and end timestamps.

### Code Size

- Current: ~380 lines
- After: ~80 lines

## Files Changed

- `src/wavekit_mcp/viewer/vcd_writer.py` — Major simplification, remove custom VcdWriter
- `tests/test_vcd_writer.py` — Update tests if needed

## Dependencies

- Add `pyvcd` back to dependencies (already in environment)
