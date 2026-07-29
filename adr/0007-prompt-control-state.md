# ADR 0007：Prompt Control State 与 SystemReminder 生命周期

- 状态：Accepted
- 日期：2026-07-28
- 目标版本：0.5.0

## 背景

`SystemReminder` 已与 `ConversationState.messages` 分离，但早期实现只快照历史，
Reminder 保存在 `Conversation` 的旁路可变字段中；`snapshot()`、`restore()` 与
`fork()` 因而无法表达同一份完整状态。`NEXT_REQUEST` 也按首次 Generate 尝试消费，
工具执行后的 Runtime 动态状态则无法在同一 turn 的下一份请求中刷新。

消息 token budget 只计算 transcript，Provider continuation 也缺少“后续请求撤销或替换
Reminder 后不得继续保留旧控制文本”的明确契约。

## 决策

### 状态边界

Conversation 的可变状态分为两个不可变值：

```text
ConversationState
    已提交的 USER / ASSISTANT / TOOL 历史、continuation 与 turn 边界

ReminderState
    idle Conversation 中仍等待消费或持续有效的 SystemReminder
```

`ConversationCheckpoint(history, reminders)` 是完整会话状态快照。它不包含
Model/Backend、工具 callable、instructions、Context Policy、PromptCompiler、
ReminderSource、Runtime 资源、动态 Source 上次解析值或活跃 Generator 栈。Runtime
负责存储、配置重建和版本迁移。

`snapshot()`、`restore()` 与 `fork()` 操作完整 checkpoint；只操作历史时使用
`history_snapshot()` 与 `restore_history()`。`clear_history()`、`clear_reminders()` 和
`reset()` 分别表达不同清理边界；`clear()` 暂时作为 `clear_history()` 的弃用别名。

### 生命周期

- `NEXT_REQUEST` 作用于下一份成功编译且通过请求预算的不可变 `ModelRequest`，随后立即
  从 `ReminderState` 消费。编译前失败时保留；编译后 Provider 失败时不恢复。
- `TURN` 在 `ask()` Flow 首次推进时由当前 turn 获取，作用于该 turn 的每次模型请求，
  并在成功、失败、取消或关闭时过期。
- `CONVERSATION` 作用于后续每份请求，直到同 key 原位替换或显式移除；它必须具有非空
  key。

相同 key 的更新保留原稳定槽位，新值追加到末尾。请求按 priority 降序稳定排列，相同
priority 保持槽位顺序。一次编译中的非空 key 必须唯一；存储值和动态值冲突时直接失败。

Reminder 永远不写入规范消息历史。Provider 可以临时渲染为供应商控制项，但不得把该
wire item 回写为 Purrcept `Message`。

### 编译边界

同步、确定、无 I/O 的 `PromptCompiler` 从 `PromptDraft` 与当前 reminder 集合构造唯一
的不可变 `ModelRequest`，并产生不含原始文本的 `PromptDiagnostics`。内部
`TurnPromptSession` 持有 scope 生命周期；模型工具循环只负责准备历史、请求编译、
Generate、InvokeTool 与事务提交。

Provider Adapter 接收已经排好序的 reminder，不得重新排序、消费或持久化它们。后一份
请求缺少某 reminder，或同 key 内容被替换时，Adapter 必须保证 Provider-managed
continuation 不继续施加旧值；无法保证时应回退 client-managed history 或明确报错。

### 动态状态

`ReminderSource.resolve(context)` 是同步、可重复调用的当前状态投影。Source 按注册顺序
在每份模型请求前重新解析，因此工具修改 Runtime 状态后，下一轮 Generate 能看到新状态。
Source 不得返回 awaitable、产生 Effect 或承担网络/慢 I/O；依赖由构造函数显式注入。

Source 实例是 Conversation 配置，解析结果是 request-local 值。两者都不进入
`ReminderState` 或 checkpoint，也不解除 Busy Guard。

### 预算与诊断

`TokenBudgetContext` 保持 transcript-only，避免悄然改变既有 Counter 签名。
`RequestTokenBudget` 在完整请求编译后通过同步 `RequestTokenCounter` 计数；超限时只删除
最旧完整 turn 并重新编译，当前 turn 本身仍超限则抛出
`ContextBudgetExceededError`。

可选同步 `PromptTraceSink` 接收编译、消费和 Source 解析事件。默认事件只包含 key、
scope、placement、priority、source ID、content hash 与 prompt fingerprint，不保留
原始敏感 Prompt。Trace 用于诊断，不作为下一次模型语义或 Conversation History。

### 并发

活动 turn 期间继续禁止 reminder 修改、完整 snapshot/restore/fork 与 clear/reset。
允许读取已提交 `state`、`messages`、`history_snapshot()` 和 `busy`。这避免已获取或已
消费 reminder 的归属歧义。

## 结果

完整 checkpoint 能恢复当前有效的 Prompt Control State；Reminder 消费点、失败路径和
工具循环内动态刷新具有确定语义。Provider Conformance 通过多请求 removal/replacement
场景锁定 continuation 撤销契约，并通过有序、多 placement fixture 锁定适配输入。

代价是 `snapshot()` / `restore()` / `fork()` 的 alpha API 发生不兼容变化，并新增若干
高级类型。日常 Agent 代码仍主要使用 `conversation.remind()` 与
`conversation.ask()`；复杂度集中在明确的状态、编译和 Provider 边界中，没有引入全局
Registry、Service Locator 或 import-time 副作用。
