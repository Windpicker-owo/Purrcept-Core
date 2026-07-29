# ADR 0006：模型标准能力层

- 状态：Accepted（v0.5 修订）
- 日期：2026-07-28

## 背景

通用 Effect/Run 内核不应知道任何模型供应商，但扩展包之间仍需要稳定、类型化的请求、
响应、流事件和错误语义。仅有这些低层传输对象时，Agent 开发者仍需手工维护历史、工具
Schema、callable 映射、tool-call loop、动态提醒和缓存位置。这会让标准能力停留在
“另一套模型 Client 类型”，不能形成一致的 Agent 开发体验。

模型层还需要保持 Purrcept 的 Effect 属性：模型调用、工具调用、重试、审批和取消都应
被 Runtime 看见，不能藏入一个不可观察的大型操作。

## 决策

在同一 `purrcept_core` distribution 中提供 `purrcept_core.models`，并明确分为两层：

```text
Agent 标准库
    PromptTemplate → Model → Conversation → FunctionTool

底层协议
    ModelBackend / ModelRequest / ModelResponse / Message / Generate
```

### 依赖与模型绑定

- 通用内核保持 Provider 无关，不导入 `purrcept_core.models`；
- models 可以依赖通用 `Effect`、`ExecutionContext` 和 Flow；
- Runtime 创建并拥有 Backend，`Model` 显式绑定 Backend 与模型名称；
- Agent 通过普通 Python 参数持有一个或多个 `Model`，不使用隐式单模型 service
  locator；
- Provider 实现 `backend.generate(request, *, model, emit)`；
- Provider SDK、网络 client、认证、传输、凭据和模型发现属于 Provider 包与 Runtime。

### Prompt 与请求语义

- `PromptTemplate` 在应用侧统一生成 instruction、message 和 reminder；
- `RenderPolicy` 采用显式同步组合，渲染默认严格，错误不得静默吞掉；
- `PromptLibrary` 与 `ToolSet` 一样是局部、不可变、拒绝重名的集合；
- Core 不提供全局 Prompt Manager、import-time 自动注册或全局构建事件；
- 模板在 Conversation 边界编译，不能进入 Provider 面向的 `ModelRequest`；
- 系统级内容从会话消息拆分为 `SystemInstruction` 和 `SystemReminder`；
- Reminder 拥有 key、scope、placement 和 priority，且不写入永久消息历史；
- Core 通过 `ReminderState`、内部 Turn Session 与 `PromptCompiler` 管理
  `next_request`、`turn` 和 `conversation` 生命周期，但不自动写入通用提醒；
- 同步 `ReminderSource` 在每份请求前投影当前 Runtime 状态，其结果不进入持久状态；
- `PromptCachePolicy` 和 `PromptStability` 表达跨 Provider 缓存意图；
- `ModelContinuation` 保存 Provider 不透明状态，默认采用 client-managed 策略；
- `provider_options` 保留为深冻结 JSON 逃生口，只承载无法通用化的 Provider 参数；
- `ModelSettings` 统一承载常用生成设置，`TokenUsage` 统一报告缓存与 reasoning token。

Reminder 是 Prompt 控制能力，不是安全边界。Adapter 决定供应商映射和必要的文本渲染；
权限必须由 Executor、Middleware 和 Runtime 强制执行。

### Python Function Tool

- `ToolSpec` 是 Provider 面向的低层 JSON Schema 值对象；
- `FunctionTool` 保存 callable、spec、参数适配器、结果编码和错误策略；
- `ToolSet` 是局部、不可变、拒绝重复名称的映射，不是全局 Registry；
- `InvokeTool` 是独立 Effect；
- Pydantic v2 用于内部 Schema 生成和参数验证，但其类型不进入公共协议；
- 未知工具和模型输入错误可返回给模型修正；
- Python 工具内部异常默认传播，只有显式 opt-in 才把异常文本返回给模型。

### Conversation 与上下文

- `Conversation` 是易用的可变外观，`ConversationState` 是不可变的已提交历史；
- `ConversationCheckpoint` 同时保存 history 与 `ReminderState`，但不保存配置或动态
  Source 结果；
- `conversation.ask()` 是官方 Flow，自动维护历史、工具映射、结果回送和循环；
- 每次模型调用产生独立 `Generate`，每次工具调用产生独立 `InvokeTool`；
- 同一 Conversation 不并发执行两个 turn，需要并发时显式 `fork()`；
- turn 只有完整成功后才提交内存状态；
- `ContextPolicy.prepare(state, model) -> PreparedContext` 是同步选择过程，不隐藏额外
  Effect；
- 内置 append-only、完整 turn 滑动窗口和 transcript-only token budget，并提供完整
  `ModelRequest` 的请求预算；
- Core 不默认执行 LLM 总结。

内存状态的事务提交不能回滚已经成功执行的外部工具副作用。工具仍需根据业务风险实现
幂等、去重或补偿。

### Streaming、错误与 Conformance

- Backend 通过同步 `ModelEventSink` 发出 `ModelStreamEvent`；
- `Generate` 校验顺序、类型和 Completed/最终响应一致性，再通过
  `ExecutionContext.progress()` 转发；
- 流式 token 不创建新的 Agent step；
- 模型错误采用官方分类，但 Core 不据此隐式重试、Fallback 或持久化；
- Provider 包使用统一 Conformance Kit 验证请求语义、Reminder 顺序/撤销/替换、流式
  生命周期、错误和取消。

## 边界

> `purrcept_core.models` 提供标准的模型函数工具、工具解析和工具循环 Flow。工具调用作为
> 独立 Effect 执行，因此可以被 Executor 和 Middleware 观察、审批或替换。Core 不提供
> 平台工具、业务权限系统、全局工具注册表或工具持久化 Runtime。

Core 也不负责聊天平台与用户会话、数据库、长期记忆、Worker、调度、Provider Client
实现、凭据或部署。

## 结果

日常 Agent 开发者不再手写消息历史、JSON Schema 或 tool-call loop；Provider 作者仍面向
稳定的低层协议。多模型通过普通对象组合，工具与模型步骤保留完整生命周期事件，Runtime
继续拥有现实资源和策略。

代价是 Core 新增 Pydantic v2 运行时依赖，并承担 Conversation、函数工具和 Prompt 语义
的兼容责任。通过不暴露 Pydantic 类型、让模板在 Provider 边界前编译、保留低层
`ModelRequest` / `Generate` 逃生口、拒绝全局 Registry 和隐藏模型调用，将这部分复杂度
限制在清晰边界内。
