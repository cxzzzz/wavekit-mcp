# wavekit-mcp cheatsheet

You are analyzing hardware simulation waveforms via the wavekit-mcp MCP server.
Follow the patterns below exactly. Do NOT use `import wavekit` or `with VcdReader(...)`.

---

## Session setup

```python
# All pre-injected — no imports needed:
# VcdReader(path), FsdbReader(path), open_reader(path), np, Pattern, MatchStatus
```

Always: open_session() → run() × N → close_session()

---

## Open a file

```python
r = VcdReader("/path/to/sim.vcd")     # VCD
r = FsdbReader("/path/to/sim.fsdb")   # FSDB
# auto-detection by extension:
r = open_reader("/path/to/sim.vcd")
```

Readers are auto-closed on reset_session() / close_session().

### Reader methods

| Method | Description |
|--------|-------------|
| `r.load_waveform(path, clock)` | Load one signal → Waveform |
| `r.load_matched_waveforms(pattern, clock_pattern)` | Batch load with pattern → dict[tuple, Waveform] |
| `r.eval(expr, clock)` | Arithmetic on signal paths → Waveform |
| `r.get_matched_signals(pattern)` | List matching signal paths → dict[tuple, Signal] |
| `r.get_matched_scopes(pattern)` | List matching scopes → dict[tuple, Scope] |
| `r.top_scope_list()` | Traverse top-level hierarchy |

### Signal path pattern syntax

Pattern methods return `dict[tuple, ...]` where the key tuple contains captured values from brace expansions and regex groups. Empty tuple `()` if no captures.

#### 1. Literal path
```python
r.load_waveform("tb.dut.data[7:0]", clock="tb.clk")
```

#### 2. Brace — string list `{a,b,...}`
```python
# keys: ('state',)  ('next',)
waves = r.load_matched_waveforms("tb.dut.J_{state,next}[3:0]", clock_pattern="tb.clk")
```

#### 3. Brace — integer range `{start..end}` or `{start..end..step}`
```python
# keys: (0,)  (1,)  (2,)  (3,)
waves = r.load_matched_waveforms("tb.dut.fifo_{0..3}.w_ptr[2:0]", clock_pattern="tb.clk")

# step: keys (0,)  (2,)  (4,)  (6,)
waves = r.load_matched_waveforms("tb.dut.lane_{0..6..2}.valid", clock_pattern="tb.clk")
```

#### 4. Multiple braces — cartesian product
```python
# keys: ('0','a')  ('0','b')  ('1','a')  ('1','b')
waves = r.load_matched_waveforms("tb.dut.u{0,1}.fifo_{a,b}.cnt[3:0]", clock_pattern="tb.clk")
```

#### 5. Regex `@<python_regex>`
`@` prefix triggers `re.fullmatch()` on each name. Capture groups `(...)` become key elements.
```python
# keys: ('state[3:0]',)  ('next[3:0]',)  ...
waves = r.load_matched_waveforms(r"tb.dut.@J_([a-z]+\[3:0\])", clock_pattern="tb.clk")

# keys: ('req',)  ('ack',)  ('valid',)
sigs = r.get_matched_signals(r"tb.dut.@(req|ack|valid)")
```

#### 6. Module-name search `$` / `$$` (FSDB only)
Use when you know the RTL module type but not the instance path.
```python
# $ModName  — direct-child scope whose module definition name is ModName
# $$ModName — any-depth descendant scope with that module name

scopes = r.get_matched_scopes("$$axi_slave")          # all instances anywhere
scopes = r.get_matched_scopes("tb.dut.$pipe_stage")   # direct children only

waves = r.load_matched_waveforms("$$fifo_unit.data_out[7:0]", clock_pattern="tb.clk")
waves = r.load_matched_waveforms("$$fifo_{a,b}.w_ptr[3:0]", clock_pattern="tb.clk")
```
> Uses the module *definition* name (`def_name`). Not available in VCD files.

---

## Waveform

`load_waveform` returns a `Waveform` with three aligned numpy arrays:

```python
data = r.load_waveform("tb.dut.data[7:0]", clock="tb.clk")
data.value   # np.ndarray — signal values (uint64 or int64)
data.clock   # np.ndarray — absolute clock cycle numbers
data.time    # np.ndarray — simulation timestamps
```

Options: `signed=True`, `xz_value=0`, `sample_on_posedge=True`

### Waveform methods

```python
data.compress()              # remove consecutive duplicates (good for control signals)
data.downsample(500, np.mean) # aggregate to N points (good for data/bus signals)
data.cycle_slice(100, 500)   # select clock cycles [100, 500)
data.time_slice(1000, 5000)  # select by simulation time
data.mask(cond)              # keep only cycles where cond Waveform != 0
data.filter(lambda v: v > 0) # keep samples matching predicate
```

Waveform arithmetic / comparison returns a new Waveform:
```python
diff = wf_a - wf_b
match = wf_a == wf_b   # element-wise; use .mask() to filter
```

### Batch load

```python
waves = r.load_matched_waveforms(
    "tb.dut.fifo_{0..3}.w_ptr[2:0]",
    clock_pattern="tb.clk",
)
for (idx,), wave in waves.items():
    print(f"fifo_{idx}: mean={np.mean(wave.value):.2f}")
```

### Computed expression

```python
occupancy = r.eval("tb.dut.w_ptr[3:0] - tb.dut.r_ptr[3:0]", clock="tb.clk")
```

---

## Common patterns

### Basic stats
```python
data = r.load_waveform("tb.dut.out[7:0]", clock="tb.clk")
print(f"min={np.min(data.value)}  max={np.max(data.value)}  mean={np.mean(data.value):.2f}")
print(np.histogram(data.value, bins=8))
```

### Filter to active cycles
```python
valid = r.load_waveform("tb.dut.valid", clock="tb.clk")
data  = r.load_waveform("tb.dut.data[7:0]", clock="tb.clk")
active = data.mask(valid == 1)
```

### Detect transitions
```python
state = r.load_waveform("tb.dut.state[2:0]", clock="tb.clk")
changes = state.compress()
print(f"transitions: {len(changes.value) - 1}  unique: {np.unique(changes.value)}")
```

### Compare two simulations
```python
r1 = VcdReader("/data/golden.vcd")
r2 = VcdReader("/data/actual.vcd")
gold = r1.load_waveform("tb.data[7:0]", clock="tb.clk")
act  = r2.load_waveform("tb.data[7:0]", clock="tb.clk")
mismatch = gold.mask((gold == act) == 0)
print(f"mismatches: {len(mismatch.value)}")
if len(mismatch.value):
    print(f"first at cycle: {mismatch.clock[0]}")
```

---

## Temporal Pattern Matching

Find transaction sequences (handshakes, latencies, bursts):

```python
# AXI read latency
arvalid = r.load_waveform("tb.arvalid", clock="tb.clk")
arready = r.load_waveform("tb.arready", clock="tb.clk")
rvalid  = r.load_waveform("tb.rvalid",  clock="tb.clk")
rready  = r.load_waveform("tb.rready",  clock="tb.clk")
rdata   = r.load_waveform("tb.rdata[31:0]", clock="tb.clk")

result = (
    Pattern()
    .wait(arvalid & arready)   # AR handshake
    .wait(rvalid  & rready)    # R handshake
    .capture("data", rdata)
    .timeout(256)
    .match()
)

valid = result.filter_valid()
print(f"count={len(valid.duration.value)}  mean={np.mean(valid.duration.value):.1f}cy  max={np.max(valid.duration.value)}cy")
captured_data = valid.captures["data"].value
```

```python
# Variable-length burst (collect all beats until wlast)
beat = Pattern().wait(wvalid & wready).capture("beats[]", wdata)
result = (
    Pattern()
    .wait(awvalid & awready)
    .loop(beat, until=wlast)
    .timeout(512)
    .match()
)
valid = result.filter_valid()
for i, beats in enumerate(valid.captures["beats"].value[:5]):
    print(f"burst {i}: {len(beats)} beats")
```

```python
# Guard condition (fail if enable drops during wait)
result = (
    Pattern()
    .wait(req, guard=enable)
    .wait(ack)
    .timeout(64)
    .match()
)
violated = result.status.mask(result.status == MatchStatus.REQUIRE_VIOLATED)
print(f"guard violations: {len(violated.value)}")
```

---

## Large data — print scalars not arrays

```python
print(np.mean(data.value))          # ✓
print(data.value.tolist())          # ✗ truncated, unhelpful
```

Reduce before inspecting: `compress()`, `downsample()`, `cycle_slice()`
