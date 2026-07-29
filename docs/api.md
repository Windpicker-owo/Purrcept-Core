# 公共 API 参考

通用内核从 `purrcept_core` 导入；模型标准库从 `purrcept_core.models` 导入；测试辅助从
`purrcept_core.testing` 导入。下列签名省略部分实现细节，所有可选模型值对象字段均为
keyword-only。

## Flow 与 Run

```python
class Effect(Generic[T]): ...


AgentFlow[T] = Generator[Effect[Any], Any, T]


def perform(effect: Effect[T]) -> Generator[Effect[T], T, T]: ...


@dataclass(frozen=True, slots=True)
class Yielded:
    effect: Effect[Any]


@dataclass(frozen=True, slots=True)
class Returned(Generic[T]):
    value: T


RunState[T] = Yielded | Returned[T]
```

`AgentRun[T]` 是 runtime-checkable Protocol，包含同步的 `start()`、`send(value)`、
`throw(error)` 和 `close()`。`GeneratorRun[T]` 适配同步 Generator；
`as_agent_run()` 保留已有 AgentRun 或包装 Generator。

## Context 与 Executor

```python
ExecutionContext(
    run_id: str,
    effect_id: str,
    step_index: int,
    host: HostT,
    metadata: Mapping[str, Any] = {},
)

await executor.execute(effect, context)
```

- `InlineExecutor()`：调用 `effect.execute(context)`；
- `DispatchExecutor(handlers=None, fallback=None)`：按实际类型 MRO 查找实例局部 handler；
- `MiddlewareExecutor(base, middlewares=())`：第一项 middleware 位于最外层。

`EffectExecutor[HostT]`、`EffectMiddleware` 和 `NextExecutor` 是组合扩展所需的 Protocol
与类型别名。

## Driver

```python
driver = AgentDriver(
    executor,
    event_sink=None,
    clock=...,
    monotonic=...,
    id_factory=...,
)

result = await driver.run(
    run_or_flow,
    host=host,
    metadata=None,
    run_id=None,
)
```

Driver 不创建事件循环。`clock`、`monotonic` 和 `id_factory` 是确定性测试注入点。

## Event

标准事件：

```text
RunStarted
EffectStarted
EffectProgress
EffectSucceeded
EffectFailed
RunSucceeded
RunFailed
RunCancelled
```

基础 sink：

```text
NullEventSink
CallbackEventSink
CompositeEventSink
SafeEventSink
```

事件 callback 必须同步并返回 `None`。传入 async callback 会显式抛出 `TypeError`。

## Middleware

```python
TimeoutMiddleware(timeout)

RetryMiddleware(
    max_attempts=3,
    retry_on=(),
    predicate=None,
    delay=0.0,
    sleep=asyncio.sleep,
)

FunctionMiddleware(function)
```

Retry 的 `max_attempts` 包含首次执行；没有 `retry_on` 和 predicate 时不会重试。

## 模型标准库

### Model、Conversation 与 turn

```python
Model(backend: ModelBackend, name: str)

model.generate(request: ModelRequest) -> Generate

model.conversation(
    *,
    instructions=(),
    reminders=(),
    tools=(),
    settings=None,
    cache=None,
    continuation_policy="client_managed",
    context_policy=None,
    reminder_sources=(),
    prompt_compiler=None,
    request_budget=None,
    prompt_trace_sink=None,
    max_model_rounds=8,
    metadata=None,
    provider_options=None,
    state=None,
) -> Conversation
```

`Conversation(model, ...)` 接受同一组配置。常用状态 API：

```python
conversation.ask(
    message: str | PromptTemplate | Message,
    *,
    max_model_rounds: int | None = None,
) -> AgentFlow[ModelTurnResult]

conversation.remind(
    reminder: str | PromptTemplate | SystemReminder,
    *,
    key=None,
    scope=ReminderScope.NEXT_REQUEST,
    placement=ReminderPlacement.AUTO,
    priority=0,
) -> SystemReminder

conversation.remove_reminder(key: str) -> bool
conversation.remove(key: str) -> bool
conversation.snapshot() -> ConversationCheckpoint
conversation.history_snapshot() -> ConversationState
conversation.restore(checkpoint: ConversationCheckpoint) -> None
conversation.restore_history(state: ConversationState) -> None
conversation.fork(
    checkpoint: ConversationCheckpoint | None = None,
) -> Conversation
conversation.clear_history() -> None
conversation.clear_reminders(scope: ReminderScope | str | None = None) -> None
conversation.reset() -> None
conversation.clear() -> None  # deprecated alias for clear_history()
```

```python
ConversationState(
    messages=(),
    *,
    continuation=None,
    turn_index=0,
    turn_boundaries=(),
)

ReminderState(reminders=())
state.upsert(reminder) -> ReminderState
state.remove(key) -> tuple[ReminderState, bool]
state.clear(scope=None) -> ReminderState

ConversationCheckpoint(
    history: ConversationState,
    reminders: ReminderState,
)

ModelTurnResult(
    response,
    state,
    model_rounds,
    *,
    tool_results=(),
)
```

`ModelTurnResult.text`、`rounds`、`messages` 和 `continuation` 是便捷只读属性。
`ConversationBusyError` 表示同一可变会话已有活动 turn；`ToolLoopLimitError` 表示模型
轮次耗尽；两者都继承 `PurrceptError`。完整 checkpoint、fork 和状态修改要求
Conversation idle；busy 时只读 `state`、`messages` 与 `history_snapshot()`。

### Context Policy

```python
class ContextPolicy(Protocol):
    def prepare(
        self,
        state: ConversationState,
        model: Model,
    ) -> PreparedContext: ...


PreparedContext(messages, *, continuation=None, turn_boundaries=())
AppendOnlyContext()
SlidingWindowContext(max_turns)
TokenBudgetContext(max_tokens, counter)
RequestTokenBudget(max_tokens, counter)
```

`TokenCounter(messages, model) -> int` 与
`RequestTokenCounter(request, model) -> int` 都是同步 Protocol。前者仅供
transcript-only `TokenBudgetContext` 使用；后者供完整请求预算使用。
`ContextBudgetExceededError` 表示当前进行中的 turn 本身已经超过相应预算。

### Function Tool

```python
FunctionTool(
    function,
    *,
    name=None,
    description=None,
    error_policy=ToolErrorPolicy.PROPAGATE,
    sync_policy=SyncToolPolicy.INLINE,
)

@tool
def lookup(query: str) -> str: ...

@tool(
    name=None,
    description=None,
    error_policy="propagate",
    sync_policy="inline",
)
def lookup(query: str) -> str: ...
```

`ToolParameter` 可通过 `Annotated` 提供 `description`、`ge`、`le`、`gt`、`lt`、
`min_length`、`max_length`、`pattern` 和 `strict`。

```python
ToolSet(tools=())
ToolSet.from_callables(callables)
tool_set.specs
tool_set.tools
tool_set.resolve(name)
tool_set.get(name)

InvokeTool(tool: FunctionTool | None, call: ToolCallBlock)

ToolContext(execution, call)
ToolResult(*, content=(), is_error=False)
```

工具错误：

```text
ToolDefinitionError
UnknownToolError
ToolArgumentsError
ToolResultEncodingError
```

未知工具和输入验证错误由 `InvokeTool` 转换为错误 Tool Result；函数内部异常默认原样
传播。`ToolErrorPolicy.RETURN_TO_MODEL` 是显式 opt-in。

### Prompt 语义

```python
PromptTemplate(
    name,
    template,
    *,
    policies={},
    values={},
)

template.fields
template.get(key, default=None)
template.has(key)
template.with_values(values=None, /, **overrides)
template.without_values(*keys)
template.clear_values()
template.render(values=None, /, *, strict=True, **overrides) -> str
template.render_partial(values=None, /, **overrides) -> str
template.build(...) -> str
template.build_partial(...) -> str
template.instruction(...) -> SystemInstruction
template.reminder(...) -> SystemReminder
template.message(role, ...) -> Message

RenderPolicy(transform)
policy.then(other)

PromptLibrary(templates=())
library.templates
library.resolve(name)
library.get(name)

PromptDraft(
    history: PreparedContext,
    *,
    instructions=(),
    tools=(),
    settings=ModelSettings(),
    cache=PromptCachePolicy(),
    continuation=None,
    metadata={},
    provider_options={},
)

PromptCompiler().compile(
    draft,
    *,
    stored_reminders=(),
    turn_reminders=(),
    dynamic_reminders=(),
    dynamic_source_ids=(),
    model_round: int,
) -> CompiledPrompt

CompiledPrompt(request, diagnostics, *, consumed_next_request=False)

SystemInstruction(
    content,
    *,
    key=None,
    stability=PromptStability.STABLE,
)

SystemInstruction.from_text(
    text,
    *,
    key=None,
    stability=PromptStability.STABLE,
)

SystemReminder(
    content,
    *,
    key=None,
    scope=ReminderScope.NEXT_REQUEST,
    placement=ReminderPlacement.AUTO,
    priority=0,
)

PromptCachePolicy(
    *,
    mode=CacheMode.AUTO,
    key=None,
    ttl=None,
    strict=False,
)

ModelContinuation(provider, data)
```

内置渲染策略：`optional`、`trim`、`header`、`wrap`、`join_blocks`、`min_len`。
`PromptLibrary` 与模板均为局部不可变对象；不存在全局注册或构建事件。

`PromptCompiler` 是同步、确定、Provider-neutral 的纯组装边界；Conversation 内部负责
scope 消费。`PromptDiagnostics` 只含 key、scope、placement、priority、hash、
fingerprint、source ID 与预算信息。编译 key 冲突会抛出 `PromptCompileError`。

动态 reminder 协议：

```python
class ReminderSource(Protocol):
    @property
    def source_id(self) -> str: ...

    def resolve(
        self,
        context: ReminderResolutionContext,
    ) -> Iterable[SystemReminder]: ...
```

Source 必须同步；每份模型请求前重新解析，结果不进入 `ReminderState`、Checkpoint 或
History。可选 `PromptTraceSink(event) -> None` 接收 `PromptCompiled`、
`ReminderConsumed`、`ReminderSourceResolved` 与 `ReminderSourceFailed`。

枚举值：

```text
PromptStability: stable / growing / volatile
ReminderScope: next_request / turn / conversation
ReminderPlacement: auto / instructions / tail
CacheMode: auto / disabled / prefer / explicit
ContinuationPolicy: client_managed / provider_managed / auto
```

### 请求、消息与内容

```python
ModelSettings(
    *,
    temperature=None,
    max_output_tokens=None,
    stop_sequences=(),
    tool_choice=ToolChoice.AUTO,
    parallel_tool_calls=None,
)

ToolSpec(name, *, description=None, parameters={})

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

`parallel_tool_calls` 是模型生成提示；当前 Conversation 仍按响应内容顺序逐个产生
`InvokeTool` Effect。

主要内容类型：

```text
ContentBlock
Message
MessageRole: user / assistant / tool
TextBlock
ImageBlock
ImageUrl
ImageBytes
ToolCallBlock
ToolResultBlock
JsonPrimitive
JsonValue
JsonObject
```

`Message.user()`、`Message.assistant()`、`Message.tool()` 与 `Message.from_text()` 是文本/
工具结果构造器。系统级内容使用 `SystemInstruction` 或 `SystemReminder`。

### Backend、Generate、Response 与 Streaming

```python
class ModelBackend(Protocol):
    async def generate(
        self,
        request: ModelRequest,
        *,
        model: str,
        emit: ModelEventSink | None = None,
    ) -> ModelResponse: ...


Generate(
    model: Model,
    request: ModelRequest,
    *,
    emit_stream_events=True,
)
```

`Generate` 返回 `ModelResponse`，并把 Backend 流事件同步桥接到
`ExecutionContext.progress`。`emit_stream_events=False` 时传给 Backend 的 `emit`
严格为 `None`。

```python
ModelResponse(
    message,
    *,
    usage=None,
    finish_reason=FinishReason.STOP,
    model=None,
    response_id=None,
    continuation=None,
    provider_metadata={},
)

TokenUsage(
    input_tokens,
    output_tokens,
    *,
    cached_input_tokens=0,
    cache_write_input_tokens=0,
    reasoning_tokens=0,
)
```

流事件：

```text
ModelStreamEvent
ModelStreamStarted
TextDelta
ToolCallDelta
UsageUpdate
ModelStreamCompleted
```

模型错误层级：

```text
ModelError
├── ModelAuthenticationError
├── ModelRequestError
│   └── ModelContextWindowError
└── ModelTransientError
    ├── ModelRateLimitError
    └── ModelUnavailableError
```

模型错误分类不会触发隐式重试。

### Provider Conformance

```python
async def check_provider(factory):
    report = await run_backend_conformance(
        factory,
        model_name="purrcept-conformance-model",
        include_streaming=True,
    )
    report.raise_for_failures()
```

公共辅助类型包括 `BackendConformanceCase`、`BackendConformanceScenario`、
`BackendConformanceStep`、`BackendScenarioFactory`、`BackendConformanceResult`、
`BackendConformanceReport` 和 `BackendConformanceError`。Scenario 的
`follow_up_steps` 会在同一个 Model/Backend 上顺序执行，用于验证 reminder removal、
replacement 与 continuation 撤销语义。

完整语义与示例见 [`models.md`](models.md)。

## 通用错误

```text
PurrceptError
InvalidRunStateError
UnsupportedEffectError
EventDispatchError
```

原始业务异常通常原样传播，不做无意义包装。取消不是普通业务异常。

## 测试辅助

```python
from purrcept_core.testing import RecordingEventSink, ScriptedExecutor
```

`ScriptedExecutor([(expected_effect, result_or_error), ...])` 按顺序校验 Effect；
`assert_finished()` 校验脚本已全部消费。
