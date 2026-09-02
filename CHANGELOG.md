# Changelog

All notable changes to this project are documented here.

## Unreleased

### Clarified

- Reminder placement is position relative to history, not wire role. Provider
  adapters must not serialize `SystemReminder` as the system role; that role is
  reserved for `SystemInstruction`. Chat Completions adapters emit reminders as
  user messages.

## 0.5.0 - 2026-07-28

### Added

- Immutable `ReminderState` and complete `ConversationCheckpoint` values so prompt-control state
  can be snapshotted, restored, forked, and cleared independently from committed history.
- A deterministic `PromptCompiler` boundary with `PromptDraft`, `CompiledPrompt`, metadata-only
  `PromptDiagnostics`, and optional synchronous prompt trace events.
- Synchronous, conversation-local `ReminderSource` projections that are resolved before every
  model request and naturally refresh after tool execution without weakening the busy guard.
- `RequestTokenBudget` and `RequestTokenCounter` for budgeting the complete `ModelRequest`,
  including instructions, tools, messages, reminders, and provider-neutral request overhead.
- Multi-request Provider Conformance scenarios for reminder removal and keyed replacement, plus
  explicit reminder order and placement fixtures.

### Changed

- `NEXT_REQUEST` reminders are consumed only after a request has compiled and passed its request
  budget; `TURN` reminders are acquired when the turn Flow first advances and expire on every
  ending path.
- `CONVERSATION` reminders now require a non-empty key.
- `Conversation.snapshot()` now returns `ConversationCheckpoint`. Use `history_snapshot()` and
  `restore_history()` when only committed transcript state is intended.
- `Conversation.fork()` copies complete mutable state while retaining the same reusable
  configuration. Full checkpoints and state mutations remain unavailable while a turn is busy.
- `TokenBudgetContext` is explicitly transcript-only; complete prompt budgeting uses
  `RequestTokenBudget`.

### Deprecated

- `Conversation.clear()` is retained as a warning-emitting alias for `clear_history()`. Use
  `reset()` to clear both committed history and stored reminders.

### Design

- Reminders remain prompt-control state and never become canonical conversation messages.
- Dynamic source output and reusable configuration are deliberately excluded from checkpoints.
- Provider continuations must not retain reminders that are absent or replaced in a later
  `ModelRequest`.

## 0.4.0 - 2026-07-28

### Added

- Immutable `PromptTemplate` values with strict and partial rendering, exact dotted placeholder
  keys, immutable value binding, and direct compilation to `SystemInstruction`, `Message`, or
  `SystemReminder`.
- Composable synchronous `RenderPolicy` values and built-in `optional`, `trim`, `header`, `wrap`,
  `join_blocks`, and `min_len` policies.
- `PromptLibrary`, an immutable local named template registry, plus structured prompt definition,
  lookup, and rendering errors.
- First-class template inputs for `model.conversation()`, `conversation.ask()`, and
  `conversation.remind()`.

### Design

- Prompt authoring is compiled before the Provider boundary, so `ModelRequest` remains a
  provider-neutral transport value.
- Core intentionally has no global prompt manager, import-time template registration, global
  build event, or hidden reminder content.
- Existing `SystemReminder` scope, placement, priority, replacement, and consumption behavior is
  unchanged.

## 0.3.0 - 2026-07-26

### Added

- `Model`, an explicit binding between a runtime-owned backend and one model name.
- `Conversation`, immutable `ConversationState`, transactional turns, snapshots, restore, fork,
  and the official observable tool-loop Flow.
- `FunctionTool`, `ToolSet`, `ToolContext`, `InvokeTool`, `ToolParameter`, tool policies, and
  provider-neutral tool results and errors.
- Pydantic v2-powered internal JSON Schema generation and argument validation without exposing
  Pydantic types in the public protocol.
- `SystemInstruction`, scoped `SystemReminder`, prompt cache/stability policies, model settings,
  and opaque Provider continuation state.
- Append-only, complete-turn sliding-window, and injected token-budget context policies.
- Expanded token usage for cache reads, cache writes, and reasoning tokens.
- A deterministic Provider Backend Conformance Kit covering request semantics, streaming,
  model errors, and cancellation.

### Changed

- The documented default path is now
  `Model → Conversation → FunctionTool → SystemReminder`; low-level requests and `Generate`
  remain the escape hatch.
- `Generate` now carries an explicit `Model`, and `model.generate(request)` is the preferred
  one-shot constructor.
- `ModelBackend.generate()` now receives the selected model name as a required keyword argument.
- System-level prompt content is represented separately from conversational messages.
- Common generation controls now live in `ModelSettings`; tools use `ToolSpec`.
- Each model generation and Python tool invocation in the automatic loop remains an independent
  Effect.

### Removed

- The implicit single-backend host binding used by model generation.
- The system role and system-message convenience constructor.
- Model-name and common generation-setting fields from the flat request surface.

### Compatibility

- This is an intentional alpha breaking release. See
  [`docs/compatibility.md`](docs/compatibility.md) for the migration guide.
- Purrcept Core now has a runtime dependency on Pydantic v2.

## 0.2.0 - 2026-07-26

### Added

- Provider-neutral model contracts in the `purrcept_core.models` capability layer.
- Immutable message, content block, request, response, usage, and streaming event types.
- A shared model error hierarchy for provider-independent retry and recovery policies.
- A self-executing one-shot model-generation Effect and runtime-owned Backend protocol.
- Runtime validation for Backend responses and ordered stream-event lifecycles.

### Changed

- Model contracts ship in the same distribution while remaining isolated from the generic
  Effect/Driver kernel.
- The external extension example implements the official model protocol instead of defining
  its own contract.

## 0.1.0 - 2026-07-26

### Added

- Effect and generator-flow primitives.
- Generator-backed and protocol-based agent runs.
- Inline, dispatch, and middleware executors.
- Deterministic lifecycle events and progress reporting.
- Async agent driver with explicit error and cancellation semantics.
- Testing helpers, runnable examples, and an external extension-package example.
