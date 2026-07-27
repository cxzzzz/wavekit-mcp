# wavekit-mcp

[English](./README.md) | 中文

基于 [wavekit](https://github.com/cxzzzz/wavekit) 的 MCP 服务器，为 AI 提供波形分析的 Python 执行环境。

## 为什么需要 wavekit-mcp？

**问题：** 波形数据量极大，一次仿真动辄包含数千信号、数百万次翻转。把原始数据喂给 LLM 既浪费 token，也难以得到有意义的分析结果。

**思路：** 与其给数据，不如给工具。wavekit-mcp 让 AI 通过编写代码来分析波形：
- 加载 VCD/FST/FSDB 信号
- 时序模式匹配
- 统计计算、异常检测、事件提取

AI 拿到的是分析结果——一个均值、一次时序违例、一段过滤后的数据——而不是原始波形。额外的输出限制迫使 AI 用"信号语义"思考，而非被无效的数值噪声填满上下文。

## 安装

```bash
pip install wavekit-mcp
```

启动：

```bash
wavekit-mcp
wavekit-mcp --config /path/to/wavekit_mcp.toml
```

MCP 客户端示例：

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

## 配置

复制 `wavekit_mcp.toml.example` 后按需修改。所有字段都是可选的。

```toml
[limits]
max_sessions         = 5
run_timeout_sec      = 120
output_max_chars     = 500

[file_access]
read_enabled         = false
write_enabled        = false
read_allowed_paths   = ["/tmp/**"]
write_allowed_paths  = ["/tmp/**"]

[log]
file  = ""
level = "INFO"

[sandbox]
# 默认已经允许 wavekit、wavekit.*、numpy、numpy.*
# allowed_imports = ["plotly", "matplotlib.*"]
```

标量字段也可以用环境变量覆盖：

```bash
WAVEKIT_MCP_RUN_TIMEOUT_SEC=300 wavekit-mcp
```

## 工具

| 工具 | 说明 |
|------|------|
| `open_session(description?)` | 创建持久 Python 会话。 |
| `close_session(session_id)` | 关闭会话并释放 worker 资源。 |
| `list_sessions()` | 列出当前会话。 |
| `run(session_id, code)` | 执行 Python，返回 `{result, output, error, duration_ms}`。 |
| `get_history(session_id, last_n)` | 返回最近的执行记录。 |
| `get_api_docs(topic)` | 查看 wavekit 的 Reader / Waveform / pattern 文档。 |

每个会话只预置：

- `wavekit` —— 直接访问 `wavekit.VcdReader`、`wavekit.Waveform` 等。
- `Viewer` —— 可选波形可视化。

其余符号请显式导入：

```python
import numpy as np
import wavekit
from wavekit.pattern import Pattern, match, collect, Channel, MatchStatus
```

`run()` 以接近 REPL 的形式返回最后一个表达式：结果显示为截断后的 `repr(...)` 文本；真实对象仍保留在会话命名空间里。

## 基本用法

```python
import numpy as np
import wavekit

r = wavekit.VcdReader("/data/sim.vcd")
data = r.load_waveform("tb.dut.data[7:0]", clock="tb.clk")

print(f"samples={len(data.value)} mean={np.mean(data.value):.2f}")
```

## Reader 示例

### 批量匹配加载

```python
waves = r.load_matched_waveforms(
    signal_path="tb.dut.fifo_{0..3}.w_ptr[2:0]",
    clock_path="tb.clk",
)
for key, wave in waves.items():
    print(f"{key}: mean={np.mean(wave.value):.2f}")
```

匹配 API 返回 `dict[CaptureKey, ...]`。`CaptureKey` 是由 `BraceCapture`、`RegexCapture`、`WildcardCapture` 等 typed capture 组成的 tuple。

### X/Z 掩码

```python
value = r.load_waveform("tb.bus[7:0]", clock="tb.clk", xz_value=0)
unknown = r.load_unknown_mask("tb.bus[7:0]", clock="tb.clk")
known_value = value.mask(unknown == 0)

unknowns = r.load_matched_unknown_masks(
    signal_path="tb.dut.fifo_{0..3}.data[7:0]",
    clock_path="tb.clk",
)
```

### 表达式

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

### 查询语法

| 语法 | 示例 | 含义 |
|------|------|------|
| 普通路径 | `tb.dut.valid` | 精确信号/层级路径 |
| 花括号 | `fifo_{0..3}.ptr` | 枚举或整数范围 |
| `/regex/` | `tb./lane_(\d+)/.valid` | 正则匹配 + 捕获 |
| `@regex` | `tb.@(req|ack)` | 兼容旧写法 |
| `*` / `**` | `tb.*.valid`, `tb.**.valid` | 单层 / 递归通配 |
| `$` / `$$` | `tb.$fifo.data`, `tb.$$fifo.data` | FSDB 按 module definition 名称匹配 |

用 `r.top_scopes`、`r.get_matched_signals(path)`、`r.get_matched_scopes(path)` 来探索层级。

## Pattern 匹配

`Pattern` 用来描述时序事务。通过模块级 `match(...)` 执行。

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

需要独占占用事件时用 `consume(..., channel=...)`。成功的 blocking step 会留在同一个 cycle；如果要下一拍行为，用 `.delay(1)`。

如果事务形状依赖数据值或分支，改用 `collect(...)`：

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

保持 session 打开，直到用户看完；关闭 session 会同时关闭 viewer。

## 安全

用户代码在独立 worker 进程里运行，受 RestrictedPython 和 `sandbox.allowed_imports` 限制。文件访问默认关闭，需在 `[file_access]` 中显式开启。

这能防止误操作并隔离崩溃，但不等于完整的恶意代码沙箱。

## AI skill

仓库内包含 wavekit-mcp skill：

- [skills/wavekit-usage/SKILL.md](./skills/wavekit-usage/SKILL.md)
- [skills/wavekit-usage/references/cheatsheet.md](./skills/wavekit-usage/references/cheatsheet.md)
