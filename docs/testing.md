# 测试 Agent Flow

`ScriptedExecutor` 让 Flow 测试不依赖网络、模型或数据库：

```python
executor = ScriptedExecutor(
    [
        (Search("purrcept"), SearchResult(items=("a",))),
        (Summarize(("a",)), "answer"),
    ]
)

result = await AgentDriver(executor).run(agent(), host=None)
executor.assert_finished()
assert result == "answer"
```

脚本项可以返回结果，也可以放入异常让 Executor 抛出。执行器会按顺序比较 Effect；类型
或 dataclass 字段不匹配会立即给出明确断言。

`RecordingEventSink` 按接收顺序保存事件，适合断言事件类型、ID、step 和 progress。
Driver 的 clock、monotonic 与 ID factory 都可注入，从而避免依赖真实时间和随机值。

