# 核心概念

一次 Purrcept Run 是一个小型的、显式的反馈循环：

1. Flow 推进并产生一个 Effect。
2. Driver 为 Effect 创建上下文并交给 Executor。
3. Executor 返回结果或抛出异常。
4. Driver 通过 `send()` 或 `throw()` 把结果送回 Flow。
5. Flow 继续产生 Effect，或返回最终结果。

## Flow 决定顺序

Flow 是 `Generator[Effect[Any], Any, ResultT]`。条件、循环、局部变量、`try/except`、
`finally` 和子流程全部沿用 Python 语义，不存在第二套 Graph DSL。

## Effect 描述一步

`Effect[T]` 仅表达意图和成功结果类型。Effect 不保存 run ID、step ID 或可变执行状态，
也不能决定下一个 Effect。声明式 Effect 由 DispatchExecutor 解释；自执行 Effect 的
`execute()` 仍必须经过 InlineExecutor。

## Executor 执行一步

Executor 是 Effect 的唯一执行入口。它可以被 Runtime 替换，也可以通过 Middleware
组合超时、显式重试或自定义横切逻辑。Core 没有全局 handler 表。

## Driver 推进一次 Run

Driver 只依赖 `AgentRun` 协议。普通 Generator 由 `GeneratorRun` 适配；以后可以在不
改变 Effect 与 Executor 的前提下增加显式状态机 Run。

## Runtime 管理现实应用

Runtime 创建事件循环和真实资源，通过 `host` 传给 Effect，决定取消、调度、持久化、
审批、权限和最终结果去向。Purrcept Core 不接管这些职责。

## Models 是同发行包的标准能力层

`purrcept_core.models` 在通用 Effect/Run 内核之上提供两层 Provider 无关能力：
面向 Agent 的 `Model`、`Conversation`、函数工具与 Prompt 策略，以及面向 Provider 的
Message、ModelRequest、ModelResponse、流事件、错误分类与 `Generate`。依赖方向是
单向的：

```text
purrcept_core.models
        ↓
Effect / ExecutionContext / progress
```

通用内核不会反向导入 models。Runtime 创建实现 `ModelBackend` 的对象并绑定为显式
`Model`；Provider 包拥有 SDK、client、认证和网络传输。这样同一 Agent 可以组合多个
模型，而通用状态机不依赖任何供应商或单模型 host。

`Generate` 仍是普通 Effect：Flow 决定何时请求模型，Executor 是唯一执行入口，最终
ModelResponse 通过 `send()` 返回 Flow。Backend 的流式输出通过 progress 上报，不改变
Flow 控制权或 step index。`Conversation.ask()` 是官方高层 Flow；它自动维护历史和工具
循环，但每次模型与工具调用仍分别产生 `Generate` / `InvokeTool` Effect。
