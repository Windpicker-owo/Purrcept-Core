# ADR 0003：AgentRun 协议

- 状态：Accepted
- 日期：2026-07-26

## 决策

Driver 依赖 `AgentRun` 的 `start/send/throw/close` 协议，每次推进返回 `Yielded` 或
`Returned`。`GeneratorRun` 是 v0.1 的默认适配器。

## 结果

Generator 是默认语法而非唯一底层实现。未来的显式状态机或 Durable Run 无需推翻
Effect、Executor 和 Driver。v0.1 不尝试序列化 Generator。

