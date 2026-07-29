# Core / Runtime 边界

`purrcept_core` distribution 同时包含通用运行内核和可选的官方 models 标准能力层。
判断一项能力属于哪里，应区分三层：

| 通用内核 | `purrcept_core.models` | Runtime 或 Provider 扩展 |
| --- | --- | --- |
| Flow、Effect、AgentRun | Model、Conversation、函数工具 | 应用会话、用户、频道 |
| Executor 组合 | Message、请求/响应、Generate / InvokeTool | Provider SDK 与网络 client |
| send/throw/cancel | Prompt、Reminder 状态、缓存、Continuation 语义 | 认证、凭据、传输与模型发现 |
| 进程内生命周期事件 | 模型流事件与 progress 桥接 | 日志、指标与审计存储 |
| 不透明 `host` | ToolContext 可显式访问 host | Backend client、数据库与平台适配 |

Core 不创建事件循环、后台任务、网络客户端、数据库连接或进程级单例，也不会读取用户
目录配置或修改全局日志。通用内核不导入 models；models 也不创建 Provider client。
Runtime 创建并管理 Backend，再把它显式绑定到普通 `Model` 对象；模型不再从 host
查找。只有工具作者明确选择 `sync_policy="thread"` 时，函数工具才会使用事件循环的工作
线程设施。

`provider_options` 只容纳 JSON 形态的 Provider 特有请求选项，并在构造时形成只读深快照。
模型层不解释这些键，也不使用它保存 client、认证对象或其他现实资源。

模型错误分类用于 Flow、Middleware 和 Runtime 做显式决策。Core 不会因为
`ModelTransientError`、`ModelRateLimitError` 或 `ModelUnavailableError` 而自动重试或
Fallback；若需要重试，必须显式配置 Middleware 或写入 Flow。

## 安全边界

Effect 与 handler 是宿主进程中的普通 Python 代码。Core 提供可观察、可包装的统一执行
入口，但不承诺沙箱、权限隔离、秘密脱敏或不可信代码执行安全。Runtime 必须在调用前完成
审批与权限判断，并决定哪些 Effect、结果和错误可以被记录。

模型请求可能包含提示词、工具参数、Provider 选项和二进制内容。Core 不自动脱敏，
Provider 凭据应由 Runtime 管理，不应放入 Message 或 `provider_options`。

## 持久化边界

v0.5 的 `ConversationCheckpoint` 是显式的会话可变状态边界，只包含已提交
`ConversationState` 与当前 `ReminderState`。它不包含 Model/Backend、工具 callable、
Prompt 配置、ReminderSource、Runtime 资源或活跃 Generator 栈；Core 也不提供序列化和
存储。需要 Durable Execution 时，应由外部 Runtime 重建配置、管理版本迁移，并与未来的
显式状态机 `AgentRun` 协议协作实现。
