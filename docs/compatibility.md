# 兼容性与迁移

## v0.5 状态

`0.5.0` 是当前 alpha 版本。它为 `SystemReminder` 增加完整 Prompt Control State、
Checkpoint、确定性编译、动态 Source、完整请求预算和 Provider 撤销契约。在 `1.0` 前，
真实 Provider 与 Runtime 集成仍可能推动小范围不兼容变更，所有变更都会记录在
CHANGELOG。

## 稳定边界

v0.5 希望保持的语义包括：

- `Effect[T]`、Generator Flow 的 send/throw/return 反馈循环；
- Driver 只依赖 `AgentRun`，所有 Effect 都经过 Executor；
- 取消与观察器错误不会作为普通业务失败送入 Flow；
- 扩展通过普通包与显式组合启用，不依赖全局 Registry；
- Runtime 拥有事件循环、现实资源、调度、持久化和 Provider client 生命周期；
- `purrcept_core.models` 提供 Provider 无关契约，通用内核不反向依赖它；
- `Model` 显式绑定 Backend 与模型名，可在同一 Agent 中组合多个模型；
- `Conversation` 的工具循环由多个独立 `Generate` / `InvokeTool` Effect 组成；
- 模型值对象不可变，可选字段使用 keyword-only 以保留扩展空间；
- Pydantic v2 只用于函数工具内部，不成为 Provider 或应用必须采用的公共类型。
- Prompt 模板在进入 Provider 协议前显式渲染，不依赖全局 Manager 或 import-time 注册。
- Reminder 是独立 Prompt Control State，永远不进入规范 Conversation History；
- `NEXT_REQUEST` 在请求成功编译并通过预算后消费，`TURN` 在 Flow 首次推进时获取，
  `CONVERSATION` 必须 keyed；
- Provider Adapter 不管理 Scope，且不得通过 continuation 保留已撤销或已替换的
  Reminder。

事件和模型对象是进程内 Python 结构；当前版本不承诺 JSON 或跨进程 wire format。以下划线
开头的模块、属性与类型属于实现细节。稳定入口以 `purrcept_core.__all__`、
`purrcept_core.models.__all__` 和公共 API 文档为准。

## 从 v0.4 迁移

v0.5 对 Conversation 状态 API 和无 key 的 conversation reminder 有意进行 alpha
breaking change；低层 `ModelRequest`、`ModelBackend` 与现有 PromptTemplate 构造方式
保持兼容。

### 1. Snapshot 与 Restore

旧代码把 `snapshot()` 当作纯历史：

```python
state = conversation.snapshot()
conversation.restore(state)
```

新 API 的完整状态包含历史与 reminder：

```python
checkpoint = conversation.snapshot()
conversation.restore(checkpoint)
branch = conversation.fork(checkpoint)
```

只读写已提交历史时显式使用：

```python
state = conversation.history_snapshot()
conversation.restore_history(state)
```

完整 snapshot、restore、fork 和所有状态修改都要求 Conversation idle。活动 turn 中仍可
读取 `state`、`messages`、`history_snapshot()` 与 `busy`。

### 2. Clear

`conversation.clear()` 暂时保留为发出 `DeprecationWarning` 的
`clear_history()` 别名。根据意图改用：

```python
conversation.clear_history()
conversation.clear_reminders()
conversation.clear_reminders(scope="turn")
conversation.reset()
```

这些操作分别只清历史、清 reminder、按 scope 清 reminder，以及同时清理两类可变状态；
Model、instructions、tools、Source 与其他配置都会保留。

### 3. Conversation Reminder 必须有 key

旧代码：

```python
conversation.remind("持续提示", scope="conversation")
```

新代码：

```python
conversation.remind(
    "持续提示",
    key="policy:persistent",
    scope="conversation",
)
```

这保证长期 reminder 可被确定替换、移除和恢复。`next_request` 与 `turn` 仍允许无 key。

### 4. 消费点

`NEXT_REQUEST` 不再按“首次 Generate 被尝试”消费，而是在不可变请求成功编译并通过完整
请求预算后消费。编译前 Context Policy、Source、Compiler 或预算失败会保留它；编译后
Provider 失败不会恢复。`TURN` 在 `ask()` Flow 首次推进时获取，任何结束路径都会过期。

### 5. Token Budget

`TokenBudgetContext` 的 Counter 签名没有改变，现在明确标记为 transcript-only。若预算
必须覆盖 instructions、tools 和 reminders，配置：

```python
conversation = model.conversation(
    request_budget=RequestTokenBudget(
        max_tokens=32_000,
        counter=provider_request_token_counter,
    ),
)
```

### 6. Provider Conformance

Provider 包应升级 Conformance Kit。新增场景会在同一 Backend 实例上连续发送“先有后无”
和“同 key 替换”的 reminder 请求。Adapter 必须保证服务端 continuation 不继续施加旧
控制值，并保持 Core 已确定的 reminder 顺序与 placement。

## 从 v0.3 迁移

v0.4 是向后兼容的功能新增。已有字符串、`SystemInstruction`、`Message` 和
`SystemReminder` 调用方式继续有效。需要复用 Prompt 时可以逐步替换：

```python
template = PromptTemplate(
    "researcher.system",
    "你是一名{style}助手。",
    values={"style": "严谨的研究"},
)

conversation = model.conversation(instructions=template)
```

与 Neo-MoFox 风格 builder 的主要差异是：模板不可变，`.with_values()` 返回新对象；
`.render()` / `.build()` 为同步调用并默认严格；模板不会自动注册到全局 Manager，也
不会触发全局构建事件。原有 `SystemReminder` scope / placement / priority 语义保持
不变。

## 从 v0.2 模型 API 迁移

### 1. 显式绑定 Model

旧代码从 host 上取得单一 Backend；新代码创建普通 Python 对象：

```python
model = Model(backend=backend, name="provider/model-name")
```

这不会转移 Backend 生命周期所有权。Runtime 仍创建和关闭 Backend，只是通过普通参数把
`Model` 传给 Agent。

### 2. 更新 Backend 签名

```python
async def generate(
    request: ModelRequest,
    *,
    model: str,
    emit: ModelEventSink | None = None,
) -> ModelResponse: ...
```

模型名称不再作为请求数据的一部分；Backend 必须接收显式 keyword-only `model` 参数。

### 3. 更新单次 Generate

```python
response = yield from perform(model.generate(request))
```

也可以直接构造 `Generate(model, request)`。这一低层路径仍只执行一次模型调用，不维护
历史或运行工具。

### 4. 分离系统提示

对话消息不再包含 system role。长期规则改为：

```python
instruction = SystemInstruction.from_text("你是一名严谨的助手。")
request = ModelRequest(
    (Message.user("Hello"),),
    instructions=(instruction,),
)
```

动态规则使用 `SystemReminder` 或 `conversation.remind()`，不要把它们追加为永久消息。
Chat Completions 适配器必须把 reminder 发成 `user` 消息；`system` 只属于
`SystemInstruction`。

### 5. 更新请求设置与工具描述

温度、最大输出 token、stop sequences 和 tool choice 统一放入 `ModelSettings`：

```python
request = ModelRequest(
    (Message.user("Hello"),),
    settings=ModelSettings(
        temperature=0.2,
        max_output_tokens=512,
    ),
)
```

底层 JSON Schema 工具描述现在使用 `ToolSpec`。日常 Agent 代码应优先传入 Python
callable 或 `FunctionTool`，由 Core 生成 Schema、验证参数、编码结果并维护局部映射。

### 6. 迁移到 Conversation

手写消息历史和 tool-call loop 可以替换为：

```python
conversation = model.conversation(
    instructions="你是一名严谨的助手。",
    tools=[search],
    cache="auto",
)
result = yield from conversation.ask("研究这个问题")
```

如果已有自定义 Loop，可以继续直接构造 `ModelRequest` 和执行 `Generate`；高层 API
不是强制封装。

### 7. 处理新的 Usage、Cache 与 Continuation

Provider 应映射缓存读取、缓存写入和 reasoning token，并按能力解释
`PromptCachePolicy`。Backend 返回的 `ModelContinuation` 由 Conversation 保存；默认
`client_managed` 不会在下一次请求中复用 Provider 状态。

### 8. 运行 Provider Conformance

Provider 扩展应使用 `run_backend_conformance()` 验证请求语义、流事件、错误和取消。
Conformance 场景使用 fake transport，不需要真实凭据或网络。

## 函数工具兼容性

函数工具运行时依赖 Pydantic v2，但公共工具协议不暴露 Pydantic model 或 validation
类型。工具定义需要可解析的类型提示，不支持 positional-only、`*args` 或 `**kwargs`。

Python 3.11 中将 `TypedDict` 用作工具参数时，应从 `typing_extensions` 导入，避免
Pydantic 对标准库旧实现的限制。同步工具默认 inline 执行；需要避免阻塞事件循环时显式
选择 `sync_policy="thread"`。

工具函数内部异常默认传播。只有确认异常文本适合发送给模型时，才选择
`error_policy="return_to_model"`。

## 从早期 Flow 原型迁移

若已有直接驱动 Generator 的原型：

1. 让 yielded 对象继承 `Effect[T]`；
2. 用 `yield from perform(effect)` 保留结果类型；
3. 把副作用移动到 Inline Effect 或显式 Dispatch handler；
4. 用 `AgentDriver(executor).run(flow(), host=...)` 推进；
5. 把平台会话、数据库、调度、Provider client 与凭据留在 Runtime。

Core 不提供旧包名兼容别名；唯一导入包名是 `purrcept_core`。模型错误分类只是显式决策
信息，不构成自动重试承诺；需要重试仍须配置 Middleware 或在 Flow 中处理。
