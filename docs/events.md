# 生命周期事件

标准事件顺序为：

```text
RunStarted
EffectStarted
EffectProgress (零到多次)
EffectSucceeded | EffectFailed
...
RunSucceeded | RunFailed | RunCancelled
```

Effect 失败后 Flow 可以在 `try/except` 中恢复，因此 `EffectFailed` 不必然跟随
`RunFailed`。

事件 sink 是同步、快速、本地的接口。需要网络或持久化时，Runtime 应在 sink 中把事件
放入自己的队列。默认情况下 sink 错误会成为 `EventDispatchError`，关闭 Run 后向调用者
传播，且不会伪装成 Effect 失败。`SafeEventSink` 可显式隔离观察器错误并把错误交给
回调。

取消有一个明确的优先级例外：Driver 总是先关闭 Run；若 `RunCancelled` 本身派发失败，
原 `CancelledError` 仍作为主异常重新抛出，`EventDispatchError` 附在其 cause 与 note
中。这样 asyncio Task 仍保持 cancelled 状态，同时观察器故障不会丢失。

Effect 返回、失败或取消后，它收到的 Context 的 progress 通道会关闭。保留旧 Context
并继续调用 `progress()` 会抛出 `InvalidRunStateError`，不会在 Effect 或 Run 终态之后
追加乱序事件。即使 Effect 捕获了第一次 progress 派发错误，Driver 仍会锁存并传播该
`EventDispatchError`。

## 模型流事件

`purrcept_core.models.Generate` 将 Backend 产生的 `ModelStreamEvent` 作为
`EffectProgress.payload` 上报。典型顺序是：

```text
EffectStarted(effect=Generate(...))
EffectProgress(payload=ModelStreamStarted(...))
EffectProgress(payload=TextDelta(...))          # 零到多次
EffectProgress(payload=ToolCallDelta(...))      # 零到多次
EffectProgress(payload=UsageUpdate(...))        # 零到多次
EffectProgress(payload=ModelStreamCompleted(...))
EffectSucceeded(result=ModelResponse(...))
```

这些模型事件不是新的 Run 生命周期事件，也不会增加 step index。Runtime 可以在普通
EventSink 中检查 `EffectProgress.payload`，再桥接到 SSE、WebSocket、终端或自己的队列。
若 Backend 不流式输出，可以不发送任何模型流事件，直接返回 `ModelResponse`。

模型流事件不改变失败策略。Backend 抛出的模型错误按普通 Effect 异常送回 Flow；Core
不会根据错误分类隐式重试。若 progress 的 Event Sink 失败，仍遵循本页前述
`EventDispatchError` 规则。

## Prompt Trace

`PromptTraceSink` 是与 Driver 生命周期事件分离的同步、可选诊断接口。它可接收
`PromptCompiled`、`ReminderConsumed`、`ReminderSourceResolved` 和
`ReminderSourceFailed`，用于回答某次请求使用了哪些 reminder、来源、placement 与
预算裁剪。默认事件只携带 key、hash、fingerprint 和其他元数据，不保存原始 Prompt；
Trace 不会写回 Conversation History。

事件是进程内 Python 对象；v0.5 不保证 JSON 序列化、自动脱敏或跨进程格式稳定。
