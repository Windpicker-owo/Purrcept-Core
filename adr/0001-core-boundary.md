# ADR 0001：Core 与 Runtime 边界

- 状态：Accepted
- 日期：2026-07-26

## 决策

通用内核只描述并推进一次 Agent Run。平台会话、调度、持久化和现实资源由 Runtime 或
扩展包拥有，并通过显式 `host` 和 Executor 组合接入。

同一 distribution 中的 `purrcept_core.models` 可以在内核之上提供 Provider 无关的模型
Conversation、Prompt 语义和函数工具 Flow，但 Provider SDK、网络 client、认证、凭据与
传输必须留在 Runtime 或独立 Provider 包。Backend 通过普通 `Model` 对象显式绑定，不
放入全局容器或单模型 host。

## 结果

通用内核仍不创建事件循环或进程级资源，也不依赖 Provider。models 的函数工具内部使用
Pydantic v2，但不把其类型泄漏到公共协议；只有显式的线程工具策略会使用事件循环的工作
线程设施。Durable Execution、平台 Runtime 与 Provider 适配不能以内建便利功能进入
Core。
