# 模型标准能力

`purrcept_core.models` 是与通用 Effect/Run 内核同发行的 Agent 模型标准库。它不包含
Provider SDK、网络 client、认证或传输实现，并同时服务两类使用者：

```text
Agent 标准库
    PromptTemplate / Model / Conversation / FunctionTool / ContextPolicy

Prompt 控制
    SystemReminder / ReminderState / PromptCompiler / ReminderSource

底层协议
    ModelBackend / ModelRequest / ModelResponse / Message / Generate
```

默认路径面向 Agent 开发者；底层协议是 Provider 作者、自定义 Agent Loop 和特殊单次调用
的逃生口。模型 API 显式从 `purrcept_core.models` 导入，通用 `purrcept_core` 内核不会
反向依赖它。

## 默认路径

```python
from typing import Annotated

from purrcept_core import AgentFlow
from purrcept_core.models import Model, ToolParameter, tool
from provider_package import ProviderBackend


@tool
async def search(
    query: Annotated[str, ToolParameter(description="需要搜索的内容")],
    limit: Annotated[int, ToolParameter(ge=1, le=10)] = 5,
) -> list[str]:
    """搜索公开信息。"""

    return [f"{query}: result {index}" for index in range(limit)]


model = Model(
    backend=ProviderBackend(...),
    name="provider/model-name",
)


def researcher(question: str) -> AgentFlow[str]:
    conversation = model.conversation(
        instructions="你是一名严谨的研究助手。",
        tools=[search],
        cache="auto",
    )
    conversation.remind(
        "在形成结论前，优先依据工具返回的信息。",
        key="evidence-policy",
        scope="turn",
    )
    result = yield from conversation.ask(question)
    return result.text
```

`@tool` 的结果是 `FunctionTool`。只有传给当前 `Conversation` 的工具才会暴露给模型；
没有全局注册，也没有 import-time 启用。

## Model 与 Backend

`Model` 把 Runtime 拥有的 Backend 与一个模型名称显式绑定：

```python
fast = Model(backend=backend, name="provider/fast")
smart = Model(backend=backend, name="provider/smart")
```

同一个 Agent 可以直接持有多个 `Model`。Backend 的 client、连接池、凭据、复用和关闭
仍由 Runtime 管理，不通过 host 上的隐式模型属性查找。

Provider 扩展实现：

```python
class ModelBackend(Protocol):
    async def generate(
        self,
        request: ModelRequest,
        *,
        model: str,
        emit: ModelEventSink | None = None,
    ) -> ModelResponse: ...
```

`model` 必须作为 keyword-only 参数接收。`ModelEventSink` 是同步
`Callable[[ModelStreamEvent], None]`；Backend 应按产生顺序立即调用它，不应创建后台
任务延迟发送。

## Conversation 与官方工具循环

`model.conversation(...)` 构造一个易用的可变外观，其已提交历史由不可变
`ConversationState` 表达。主要配置包括：

```python
conversation = model.conversation(
    instructions="长期稳定的应用规则",
    reminders=(),
    tools=(search,),
    settings=ModelSettings(max_output_tokens=1024),
    cache="auto",
    continuation_policy="client_managed",
    context_policy=AppendOnlyContext(),
    reminder_sources=(),
    prompt_compiler=PromptCompiler(),
    request_budget=None,
    prompt_trace_sink=None,
    max_model_rounds=8,
    metadata={},
    provider_options={},
)
```

`conversation.ask("...")` 返回普通 `AgentFlow[ModelTurnResult]`。每轮会自动：

1. 追加用户消息；
2. 通过 Context Policy 准备请求历史；
3. 重新解析动态 Reminder，并把完整 Prompt 编译为不可变 `ModelRequest`；
4. 产生一个 `Generate` Effect；
5. 追加完整 assistant 响应；
6. 为每个 tool call 产生一个独立 `InvokeTool` Effect；
7. 追加关联的 tool result；
8. 再次编译并产生 `Generate`，直到得到不含 tool call 的最终响应。

因此自动循环不是隐藏在一次大型模型 Effect 中。每次 `Generate` 和 `InvokeTool` 都有
独立 step/effect 标识，可被 Executor、Middleware 和事件观察器单独处理。模型重试不会
自动重试已经成功的工具调用。

`ModelTurnResult` 提供最终 `response`、新 `state`、`model_rounds` / `rounds`、本轮
`tool_results`、本轮 `messages`、`text` 和最新 `continuation`。当最后可用的模型轮次
仍要求调用工具时，循环会在执行这些工具前抛出 `ToolLoopLimitError`。

### 状态与并发

一次 `ask()` 对内存中的 `ConversationState` 是事务式的：只有完整成功后才提交用户、
assistant 和 tool 消息；中途失败不会留下部分历史。这里的“事务式”只覆盖内存状态。
已经成功执行的外部工具副作用不会被回滚，工具仍需按业务风险设计幂等键、去重或补偿。

同一个 `Conversation` 同时只能执行一个 `ask()`，否则抛出
`ConversationBusyError`。并发分支应使用：

```python
checkpoint = conversation.snapshot()
branch = conversation.fork(checkpoint)

conversation.restore(checkpoint)
history = conversation.history_snapshot()
conversation.restore_history(history)

conversation.clear_history()  # 只清历史与 continuation
conversation.clear_reminders()  # 只清存储的 Reminder
conversation.reset()  # 两者都清理
```

`snapshot()` 返回 `ConversationCheckpoint(history, reminders)`，而不是单独的
`ConversationState`。Checkpoint 只保存可变状态，不包含 Model/Backend、instructions、
工具 callable、Context Policy、Prompt Compiler、ReminderSource 或 Runtime 资源。
`history_snapshot()` 只读取已提交历史，并且可以在 busy 时使用；完整 snapshot、
restore、fork 与所有状态修改在 busy 时都会抛出 `ConversationBusyError`。

## 统一 Prompt 系统

`PromptTemplate` 负责应用侧 Prompt 的命名、占位符绑定和确定性渲染。同一套模板可以在
进入低层协议前编译为 `SystemInstruction`、`Message` 或 `SystemReminder`：

```python
from purrcept_core.models import (
    PromptTemplate,
    header,
    join_blocks,
    trim,
)


RESEARCHER_PROMPT = PromptTemplate(
    "researcher.system",
    "角色：{role}\n\n{context.blocks}",
    policies={
        "role": trim(),
        "context.blocks": join_blocks().then(header("# Context")),
    },
    values={"role": "严谨的研究助手"},
)

bound_prompt = RESEARCHER_PROMPT.with_values(
    {"context.blocks": ["只区分事实与推测", "引用可验证证据"]}
)

conversation = model.conversation(instructions=bound_prompt)
result = yield from conversation.ask(
    PromptTemplate(
        "researcher.question",
        "研究：{question}",
        values={"question": "Prompt Cache"},
    )
)
```

`render()` 是同步、无副作用且默认严格的；缺少任何值会抛出 `PromptRenderError`。显式
`strict=False` 时，缺失值会以 `None` 进入对应 policy，默认 `optional()` 会将其渲染为空
字符串。`render_partial()` 只替换已有值，保留尚未绑定的 `{field}`。`build()` /
`build_partial()` 是对应方法的同步别名。

占位符按完整名称精确查找，因此 `{context.blocks}` 是普通 key，不会进行 Python 属性
访问。模板不接受 `!r`、`:>10` 等 format conversion/specifier；格式化应由
`RenderPolicy` 显式完成。内置策略包括：

- `optional(default)`：空值回退；
- `trim()`：清理首尾空白；
- `header(title)` 与 `wrap(prefix, suffix)`：只装饰非空内容；
- `join_blocks(separator)`：过滤并连接块；
- `min_len(length)`：过滤过短内容。

策略可通过 `.then()` 从左到右组合。自定义 `RenderPolicy` 必须同步返回字符串；异常不会
被静默吞掉，而会作为 `PromptRenderError` 保留原始异常链。

`PromptLibrary` 是与 `ToolSet` 相同风格的局部、不可变、拒绝重名的命名集合：

```python
prompts = PromptLibrary((RESEARCHER_PROMPT,))
template = prompts.resolve("researcher.system")
```

Core 不提供全局 Prompt Manager、import-time 自动注册或全局 build event。需要动态修改
模板时，应显式创建新模板、组合 policy，或由应用在调用 `render()` 前准备 values。
模板进入 `Conversation` 后立即编译为现有协议值；Provider Adapter 仍只处理
instructions / messages / reminders，不依赖应用侧模板系统。

## System Instruction 与 System Reminder

系统级语义不属于会话消息。`MessageRole` 只有 `USER`、`ASSISTANT` 和 `TOOL`：

```python
instruction = SystemInstruction.from_text(
    "始终区分事实与推测。",
    key="epistemic-policy",
    stability=PromptStability.STABLE,
)
```

`SystemInstruction` 表达长期规则；`SystemReminder` 表达动态、高优先级提示。Reminder
不会写入永久消息历史，Core 也不会偷偷添加通用提醒内容。

```python
conversation.remind(
    "当前任务仍未完成，请继续处理。",
    key="continue-task",
    scope="turn",
    placement="tail",
    priority=10,
)
```

模板也可以保留现有 Reminder 生命周期语义后再注入：

```python
conversation.remind(
    reminder_template.with_values(state="仍有未完成步骤"),
    key="task-state",
    scope="turn",
    placement="tail",
)
```

生命周期如下：

| scope | 注入范围 | 结束后的状态 |
| --- | --- | --- |
| `next_request` | 下一份成功编译且通过预算的模型请求 | 编译接受后立即移除 |
| `turn` | 当前 `ask()` 内每次模型请求 | 成功、失败、取消或关闭后移除 |
| `conversation` | 后续每次模型请求 | 保留到同 key 替换或显式删除 |

同一非空 `key` 会原位替换旧 Reminder；没有 key 的 Reminder 会追加。每个请求按
`priority` 从高到低排列。`placement` 只是 Provider 映射提示：Adapter 可以映射到原生
instructions、受控尾部内容或其他等价结构，Core 不规定 XML 包装格式。

`conversation` scope 必须提供非空 key。`next_request` 在编译前失败时仍保留，编译后
即使 Provider 失败也不会恢复；Executor 对同一个 `Generate` 的重试会继续使用同一个
不可变请求。`turn` 在 Flow 首次推进时获取，所以 Context Policy 失败、取消或
`flow.close()` 也会使它过期。Reminder 始终不进入 `ConversationState.messages`。

`SystemReminder` 是 Prompt 控制能力，不是权限或安全边界。数据访问、工具审批和资源权限
必须由 Executor、Middleware 和 Runtime 强制执行。

### Prompt Control State 与 Compiler

高级 Runtime 可以直接使用不可变状态与纯编译边界：

```python
checkpoint = conversation.snapshot()
assert checkpoint.history is conversation.state
assert checkpoint.reminders is conversation.reminder_state

compiled = PromptCompiler().compile(
    PromptDraft(prepared_context, instructions=instructions),
    stored_reminders=checkpoint.reminders.reminders,
    model_round=1,
)
request = compiled.request
```

`PromptDraft` 保存除 reminder 外的 Provider-neutral 请求输入；`CompiledPrompt` 同时提供
不可变 request 与 `PromptDiagnostics`。Compiler 合并存储、turn 和动态 reminder，
检查 key 冲突并执行稳定 priority 排序，但不执行 I/O、Effect、生命周期消费或 Provider
wire formatting。生命周期由 Conversation 内部的 turn session 在成功编译后提交。

### 动态 ReminderSource

工具循环内的 Runtime 状态应通过同步 Source 投影，而不是在 busy Conversation 上调用
`remind()`：

```python
from purrcept_core.models import SystemReminder, TextBlock


class WorkspaceReminderSource:
    source_id = "workspace"

    def __init__(self, workspace):
        self._workspace = workspace

    def resolve(self, context):
        del context
        state = self._workspace.snapshot()
        return (
            SystemReminder(
                (TextBlock(f"workspace dirty={state.dirty}"),),
                key="workspace:state",
                placement="tail",
            ),
        )


conversation = model.conversation(
    reminder_sources=(WorkspaceReminderSource(workspace),),
)
```

Source 在每份模型请求前按注册顺序重新解析，必须同步、快速、无副作用，且不得返回
awaitable 或 Effect。其实例属于配置，解析结果只属于当前 request；两者都不进入
Checkpoint、`ReminderState` 或 History。Source 与任何其他有效 reminder 的非空 key
冲突会使编译明确失败。

完整的离线工具循环见
[`examples/08_prompt_control_state.py`](../examples/08_prompt_control_state.py)。

## Python Function Tool

`FunctionTool` 使用函数签名生成 `ToolSpec`，保存本地 callable 映射，并在执行时验证模型
参数。工具描述优先级是显式 `description`、docstring 首段、函数名的人类可读形式；
参数描述优先使用 `Annotated[..., ToolParameter(...)]`，其次使用 Google 风格
docstring 的 `Args:` / `Arguments:` / `Parameters:` 段。

```python
from typing import Annotated

from purrcept_core.models import ToolContext, ToolParameter, ToolSet, tool


@tool(sync_policy="thread")
def read_file(
    path: Annotated[str, ToolParameter(description="相对工作目录的路径")],
    context: ToolContext,
) -> str:
    """读取一个文本文件。"""

    context.progress({"path": path})
    return context.host.files.read_text(path)
```

精确标注为 `ToolContext` 的参数不会进入 JSON Schema，执行时会注入当前
`ExecutionContext` 和 `ToolCallBlock`。同步工具默认在当前执行线程运行；只有显式设置
`sync_policy="thread"` 才使用工作线程。

`ToolSet` 是局部、不可变、拒绝重复名称的映射：

```python
tools = ToolSet([search, read_file])  # 已由 @tool 转换
tool = tools.resolve("search")

# 未装饰的 callable 也可以批量转换：
raw_tools = ToolSet.from_callables([plain_python_function])
```

函数返回值按以下规则编码：

| Python 值 | 模型可见结果 |
| --- | --- |
| `str` | 一个 `TextBlock` |
| `None` | 空的成功 `ToolResult` |
| JSON primitive / list / dict | 紧凑 JSON 文本 |
| dataclass 实例 | 字段递归转换为 JSON 文本 |
| `ContentBlock` | 原样作为结果内容 |
| `ToolResult` | 原样使用 |

不支持的类型、非有限浮点数、非字符串 mapping key 或循环引用会抛出
`ToolResultEncodingError`。

未知工具和模型提供的参数验证错误会转换为 `is_error=True` 的 `ToolResult`，让模型有
机会修正。Python 工具内部异常默认让 `InvokeTool` Effect 失败，避免意外把路径、密钥或
内部状态发给模型。只有工具作者明确选择时才返回异常文本：

```python
@tool(error_policy="return_to_model")
async def recoverable_lookup(query: str) -> str: ...
```

函数工具内部使用 Pydantic v2 生成 JSON Schema 和转换参数，但公共 API 只暴露
`FunctionTool`、`ToolSpec`、`ToolParameter`、`ToolResult` 和 Purrcept 错误类型，不泄漏
Pydantic 类型。Python 3.11 项目若把 `TypedDict` 用作工具参数，建议从
`typing_extensions` 导入：

```python
from typing_extensions import TypedDict
```

## Cache、Continuation 与 Context Policy

`PromptCachePolicy` 表达跨 Provider 的缓存意图：

```python
from datetime import timedelta

from purrcept_core.models import CacheMode, PromptCachePolicy


PromptCachePolicy(
    mode=CacheMode.PREFER,
    key="research-agent",
    ttl=timedelta(minutes=10),
    strict=False,
)
```

`CacheMode` 包含 `AUTO`、`DISABLED`、`PREFER` 和 `EXPLICIT`。
`PromptStability` 包含 `STABLE`、`GROWING` 和 `VOLATILE`。Core 将 instructions、
messages 和 reminders 保持为分离语义，使 Provider Adapter 可以维持稳定前缀并把动态
Reminder 放到合适位置。Chat Completions 适配器必须把 reminder 发成 user 消息：只有
`SystemInstruction` 使用 system 角色。在供应商语义允许时，Adapter 应优先编译稳定
instructions、稳定 tool specs、不断增长的历史，最后再放当前输入和 volatile
reminders；具体缓存机制、TTL 支持与严格模式处理仍由 Provider 决定。

`ModelContinuation(provider, data)` 保存 Provider 的不透明续接状态。Conversation 总会
保留 Backend 返回的最新 continuation；默认 `CLIENT_MANAGED` 不把它放入下一次请求，
而是依赖标准消息历史。`PROVIDER_MANAGED` 与 `AUTO` 会把可用 continuation 传给
Backend，数据保留和 Provider 端持久化策略应由应用显式决定。

`ContextPolicy.prepare(state, model) -> PreparedContext` 是同步、无副作用的选择过程，
不会暗中产生模型调用或 Effect。内置策略：

- `AppendOnlyContext()`：发送完整 transcript，也是默认值；
- `SlidingWindowContext(max_turns)`：保留最近若干完整已提交 turn 和当前进行中的 turn；
- `TokenBudgetContext(max_tokens, counter)`：通过注入的同步 `TokenCounter`，在 transcript
  预算内保留最近完整 turn。

`TokenBudgetContext` 明确是 transcript-only。需要覆盖 instructions、tools、messages、
reminders 与 Provider-neutral 包装内容时，向 Conversation 传入：

```python
request_budget = RequestTokenBudget(
    max_tokens=32_000,
    counter=provider_request_token_counter,
)
```

完整预算在 Source 解析和 Prompt 编译后计数；超限时移除最旧完整 turn 并重新编译，绝不
拆散 tool call / tool result 所在的 turn。当前进行中的 turn 本身仍超限时抛出
`ContextBudgetExceededError`。Core 不默认进行 LLM 总结；Provider 包可以提供更精确的
同步 `RequestTokenCounter`。

### Prompt Trace

`prompt_trace_sink=` 可接收同步的 `PromptCompiled`、`ReminderConsumed`、
`ReminderSourceResolved` 与 `ReminderSourceFailed`。默认事件只携带 key、scope、
placement、priority、source ID、content hash、fingerprint、预算估算与裁剪数，不包含
原始 Prompt。Trace 用于调试或审计，不会成为模型历史或下一次编译输入。

## 底层逃生口：ModelRequest 与 Generate

一次完全显式的模型请求：

```python
from purrcept_core import AgentFlow, perform
from purrcept_core.models import Message, Model, ModelRequest, ModelResponse


def one_shot(model: Model) -> AgentFlow[ModelResponse]:
    request = ModelRequest((Message.user("Hello"),))
    return (yield from perform(model.generate(request)))
```

这只执行一次 `Generate`：不维护历史、不解析或执行工具、不自动循环，也不创建 client。
适合 Provider 测试、特殊请求、自定义 Agent Loop 和需要完全控制 Prompt 的代码。

`ModelRequest` 的结构是：

```python
ModelRequest(
    messages,
    *,
    instructions=(),
    reminders=(),
    tools=(),
    settings=ModelSettings(),
    cache=PromptCachePolicy(),
    continuation=None,
    metadata={},
    provider_options={},
)
```

`messages` 可以为空，但 messages / instructions / reminders 至少有一项。
`ToolSpec(name, *, description=None, parameters={})` 是面向 Backend 的低层 JSON Schema
描述。工具名、非空 instruction key 和非空 reminder key 在同一请求内必须唯一；
`ToolChoice.REQUIRED` 要求至少一个工具。`ModelSettings` 统一表达 temperature、最大输出
token、stop sequences、tool choice 和并行 tool-call 提示。`parallel_tool_calls` 是传给
模型/Provider 的生成能力提示；首版 Conversation 对同一响应中的多个 tool call 仍按
内容顺序逐个 yield `InvokeTool`，不会隐藏并发执行。`provider_options` 只保留确实无法
通用化的 Provider 参数；SDK client、凭据和回调不应放入请求。

所有 JSON mapping / sequence 都会在构造时复制并深冻结。`metadata` 是应用自有信息；
Provider 不应把它当作供应商请求参数。

## Message、Response 与 Usage

一个 `Message` 包含角色和一组不可变 `ContentBlock`：

```python
from purrcept_core.models import (
    ImageBlock,
    ImageUrl,
    Message,
    MessageRole,
    TextBlock,
)


message = Message(
    MessageRole.USER,
    (
        TextBlock("Describe this image"),
        ImageBlock(ImageUrl("https://example.invalid/image.png")),
    ),
)
```

公共 block 包括 `TextBlock`、`ReasoningBlock`、`ImageBlock`、`ToolCallBlock` 和
`ToolResultBlock`。`ReasoningBlock` 保存模型返回的隐藏推理文本，使后续请求可以按
供应商协议回传完整 assistant turn；它不会被 `Message.text` 投影为可见文本，也不应
由界面、世界投影或普通日志展示。
`Message.user()`、`Message.assistant()`、`Message.from_text()` 与 `Message.tool()` 提供
常用构造方式；`Message.text` 拼接可见文本。

`ModelResponse` 包含 assistant message、finish reason、模型/响应标识、usage、最新
continuation 和深冻结的 `provider_metadata`。`TokenUsage` 统一报告：

```python
TokenUsage(
    input_tokens,
    output_tokens,
    cached_input_tokens=0,
    cache_write_input_tokens=0,
    reasoning_tokens=0,
)
```

`total_tokens` 仍是 input + output。Core 不计算价格。

## 流式输出

官方流事件：

- `ModelStreamStarted(*, model=None, response_id=None, provider_metadata={})`；
- `TextDelta(delta, *, index=0)`；
- `ToolCallDelta(index, arguments_delta, *, tool_call_id=None, name=None)`；
- `UsageUpdate(usage)`；
- `ModelStreamCompleted(response)`。

`Generate(model, request, *, emit_stream_events=True)` 把同步校验 sink 传给 Backend，并把
合法事件桥接至当前 Effect 的 progress。设为 `False` 时传入的 `emit` 严格为 `None`。

Started 与 Completed 可选；若存在，Started 必须最先且只出现一次，Completed 必须最后且
匹配 Backend 最终返回的 `ModelResponse`。`Generate` 会校验事件类型、顺序和最终响应，
并在 Backend 成功返回前暂存 Completed。流事件不会创建 Agent step，也不会代替最终
响应。

## 错误分类与重试

```text
ModelError
├── ModelAuthenticationError
├── ModelRequestError
│   └── ModelContextWindowError
└── ModelTransientError
    ├── ModelRateLimitError
    └── ModelUnavailableError
```

Provider 应保留原异常链并转换为最具体的官方错误。Backend 必须让
`asyncio.CancelledError` 原样传播。分类只提供决策信息，不触发隐式重试、Fallback 或
模型切换；需要重试时由 Flow 或 Runtime 显式组合 `RetryMiddleware`，并考虑计费、部分
流输出和幂等风险。

## Provider Conformance Kit

Provider 包应使用 `run_backend_conformance()` 对 fake transport 运行统一场景。当前套件
覆盖基础响应、完整请求语义（包括 reminder 稳定顺序与三种 placement）、带结构化参数的
tool call / tool result 事务、多请求 reminder removal 与同 key replacement、
Started → TextDelta → Completed 流式生命周期、模型错误和取消传播：

```python
from purrcept_core.models import Model, run_backend_conformance


def scenario_factory(scenario):
    transport = FakeTransport.for_scenario(scenario)
    backend = ProviderBackend(transport=transport)
    return Model(backend=backend, name=scenario.model_name)


async def test_backend_conformance():
    report = await run_backend_conformance(scenario_factory)
    report.raise_for_failures()
```

多请求场景通过 `BackendConformanceStep` 复用同一个 Model/Backend。Provider 的 fake
transport 应断言后一份 wire request 不再包含已撤销旧值，Provider-managed continuation
也不得令其继续生效。Conformance Kit 不进行真实网络调用；认证、供应商特殊能力和可选
Schema 子集仍需在扩展包内补充测试。
