# Executor

所有 Effect 都必须经过 Executor，即使 Effect 自带 `execute()`。

## InlineExecutor

调用 `effect.execute(context)`，同时接受同步结果和 Awaitable。没有可调用的
`execute` 时抛出 `UnsupportedEffectError`。

## DispatchExecutor

声明式 Effect 使用实例局部的 handler 映射：

```python
executor = DispatchExecutor(
    handlers={SendMessage: send_message},
    fallback=InlineExecutor(),
)
```

解析顺序遵循实际 Effect 类型的 MRO：具体类型优先，再查父类。handler 接收
`(effect, context)`，可以同步或异步。没有匹配 handler 和 fallback 时会显式失败。

映射只属于这个 Executor 实例。安装、导入或构造另一个扩展包不会改变其行为。

## ExecutionContext

Context 包含 `run_id`、`effect_id`、`step_index`、不透明的 `host` 和只读 metadata。
Effect 可以调用 `context.progress(payload)` 报告增量进度；这不会推进 Flow，也不会创建
新 step。

扩展包可以用 Protocol 约束 host：

```python
class FileHost(Protocol):
    files: FileService
```

模型 Backend 不需要放入 host；`Model` 已显式携带它。host 仍适合数据库、文件服务等
应用资源，并可由普通 Effect 或函数工具的 `ToolContext` 使用。Core 不提供字符串
Service Locator。
