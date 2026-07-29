# Effect 与 Flow

推荐将 Effect 定义为不可变、带 slots 的 dataclass：

```python
from dataclasses import dataclass
from purrcept_core import Effect


@dataclass(frozen=True, slots=True)
class ReadTemperature(Effect[float]):
    city: str
```

`Effect[float]` 让 `perform()` 把返回值推导为 `float`：

```python
from purrcept_core import AgentFlow, perform


def weather_flow() -> AgentFlow[str]:
    value = yield from perform(ReadTemperature("Shanghai"))
    return f"{value:.1f} °C"
```

直接 `yield effect` 也受支持，但 `yield from perform(effect)` 能保留更精确的结果类型。
嵌套 Flow 直接使用 `yield from child_flow()`；子 Flow 与父 Flow 属于同一 Run，共享
run ID，step index 连续。

Flow 可以捕获普通 Effect 异常。`asyncio.CancelledError`、`KeyboardInterrupt` 和
`SystemExit` 不会作为业务错误注入 Flow。

## 官方模型 Effect

`purrcept_core.models.Generate` 是官方模型能力层提供的自执行 Effect。Flow 负责描述
请求，Runtime 负责创建 Backend 并把它绑定到显式 `Model`：

```python
from purrcept_core import AgentFlow, perform
from purrcept_core.models import (
    Message,
    Model,
    ModelRequest,
    SystemInstruction,
)


def summarize(model: Model, text: str) -> AgentFlow[str]:
    response = yield from perform(
        model.generate(
            ModelRequest(
                (Message.user(text),),
                instructions=(SystemInstruction.from_text("Summarize the user's text."),),
            ),
        )
    )
    return response.text
```

`Generate` 的最终结果是 `ModelResponse`。当 `emit_stream_events=True`（默认值）时，
backend 发出的 `ModelStreamStarted`、`TextDelta`、`ToolCallDelta`、`UsageUpdate` 和
`ModelStreamCompleted` 会通过当前 Effect 的 `context.progress(...)` 进入
`EffectProgress.payload`。这不会改变 Flow 的恢复协议：Flow 仍只在 Effect 完成时收到
一次最终 `ModelResponse`。

`Generate` 不隐式重试、回退或切换模型。需要重试时，应由 Runtime 显式组合
`RetryMiddleware` 并只选择适合重试的 `ModelTransientError`；这样成本、幂等性以及已发出
的部分流事件仍由应用控制。

日常对话优先使用 `model.conversation()`。其自动工具循环仍逐个 yield 独立的
`Generate` 与 `InvokeTool`，因此同样遵守 Executor 唯一执行入口。
