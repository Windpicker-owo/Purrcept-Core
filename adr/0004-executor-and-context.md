# ADR 0004：Executor 与 Context

- 状态：Accepted
- 日期：2026-07-26

## 决策

Executor 是 Effect 的唯一执行入口。Core 提供 Inline、Dispatch 和 Middleware 三种
组合方式。每次调用接收带标识、只读 metadata、不透明 host 和 progress 回调的
ExecutionContext。

## 结果

追踪、超时、显式重试、审批或测试替换都能围绕统一入口实现。Core 不提供字符串服务查找
或全局 handler registry。

观察器失败属于基础设施失败，不能被 Retry 中间件重试；progress 派发错误会在 Effect
边界锁存，即使 Effect 自身捕获也由 Driver 传播。Effect 结束后 progress 通道关闭。
取消仍保持为主异常；若 `RunCancelled` 派发失败，事件错误附在取消异常链上。
