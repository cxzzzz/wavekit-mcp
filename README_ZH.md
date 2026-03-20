# wavekit-mcp

[English](./README.md) | 中文

基于 [wavekit](https://github.com/cxzzzz/wavekit) 的 MCP 服务器，为 AI 提供波形分析的 Python 执行环境。

## 为什么需要 wavekit-mcp？

**问题：** 波形数据量极大，一次仿真动辄包含数千信号、数百万次翻转。把原始数据喂给 LLM 既浪费 token，也难以得到有意义的分析结果。

**思路：** 与其给数据，不如给工具。wavekit-mcp 让 AI 通过编写代码来分析波形：
- 加载 VCD/FSDB 信号
- 时序模式匹配
- 统计计算、异常检测、事件提取

AI 拿到的是分析结果——一个均值、一次时序违例、一段过滤后的数据——而不是原始波形。输出限制迫使 AI 用"信号语义"思考，而非在数值序列中迷失。

## 安装

```bash
pip install wavekit-mcp
```

启动：

```bash
wavekit-mcp                              # 默认配置
wavekit-mcp --config wavekit_mcp.toml   # 指定配置
```

配置 MCP 客户端（以 Claude Desktop 为例）：

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

## 配置项

复制 `wavekit_mcp.toml.example` 按需修改：

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
file  = "/var/log/wavekit_mcp.log"   # 留空只输出到 stderr
level = "INFO"
```

环境变量覆盖：

```bash
WAVEKIT_MCP_RUN_TIMEOUT_SEC=300 wavekit-mcp
```

## 工具列表

| 工具 | 说明 |
|------|------|
| `open_session(description?)` | 创建会话 |
| `close_session(sid)` | 关闭会话 |
| `reset_session(sid)` | 重置会话（清空变量） |
| `list_sessions()` | 列出所有会话 |
| `run(sid, code)` | 执行 Python 代码 |
| `get_history(sid, n)` | 查看执行历史 |
| `get_api_docs(topic)` | 查看 wavekit API 文档 |
| `save_plot(sid, figure_var)` | 保存图表为 HTML/PNG |

每个会话预置：`open_reader()`、`np`、`Pattern`、`MatchStatus`

## 示例

### 基本用法

```python
# call 1
r = open_reader("/data/sim.vcd")
data = r.load_waveform("tb.dut.data[7:0]", clock="tb.clk")

# call 2 — state persists
print(f"samples={len(data.value)}  mean={np.mean(data.value):.2f}")
```

### AXI 读延迟分析

```python
arvalid = r.load_waveform("tb.arvalid", clock="tb.clk")
arready = r.load_waveform("tb.arready", clock="tb.clk")
rvalid  = r.load_waveform("tb.rvalid",  clock="tb.clk")
rready  = r.load_waveform("tb.rready",  clock="tb.clk")

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

## 安全限制

代码运行在 [RestrictedPython](https://restrictedpython.readthedocs.io/) 环境下：
- `import` 被禁用
- `__class__`、`__bases__` 等属性访问被禁用
- 文件 I/O 默认关闭

> 注意：设计目的是防止误操作，不能完全隔离恶意代码。

## 更多

常用模式速查见 [SKILLS.md](./SKILLS.md)
