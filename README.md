<p align="center">
  <strong>English</strong> · <a href="./README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <img src="./assets/readme/hero.svg" width="100%"
       alt="Purrcept Core: composable agent foundations for Python">
</p>

<p align="center">
  <a href="./CHANGELOG.md"><img alt="Status: 0.5.0 alpha" src="https://img.shields.io/badge/status-0.5.0%20alpha-2F80FF"></a>
  <a href="./pyproject.toml"><img alt="Python 3.11 through 3.14" src="https://img.shields.io/badge/python-3.11%E2%80%933.14-10213F?logo=python&logoColor=white"></a>
  <a href="./src/purrcept_core/py.typed"><img alt="Typed package" src="https://img.shields.io/badge/typing-py.typed-FF6F91"></a>
  <a href="./pyproject.toml"><img alt="Coverage gate: 100 percent" src="https://img.shields.io/badge/coverage%20gate-100%25-10213F"></a>
</p>

<p align="center">
  <a href="#why-purrcept">Why Purrcept</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#add-the-model-layer">Model layer</a> ·
  <a href="#choose-the-boundary-you-need">Boundaries</a> ·
  <a href="#documentation">Documentation</a>
</p>

Purrcept Core is a Python-native, effect-driven foundation for composable agents. Plain
generator flows decide what happens next, typed `Effect` values describe one step, an
`Executor` performs it, and `AgentDriver` advances the run.

The optional `purrcept_core.models` standard library adds provider-neutral conversations,
prompts, function tools, streaming events, context policies, and backend conformance—without
taking ownership of provider clients, credentials, persistence, or the application runtime.

> [!IMPORTANT]
> `0.5.1` is an alpha release. The main development path is in place, but focused API changes
> may still follow real Provider and Runtime integration feedback before `1.0`.

## The execution model

<p align="center">
  <img src="./assets/readme/architecture.svg" width="100%"
       alt="A Flow yields an Effect to an Executor; AgentDriver advances the run, model operations use the same Effects, and the Runtime owns external resources">
</p>

Every yielded operation passes through one execution boundary. That gives cancellation,
middleware, progress, lifecycle events, retries, approvals, and testing a precise step to act on.
A model tool loop does not bypass this model: one `conversation.ask()` can yield
`Generate → InvokeTool → Generate`, with every generation and tool call remaining an independent
Effect.

## Why Purrcept

- **Compose with ordinary Python.** Use generators, `yield from`, conditions, loops, nested flows,
  return values, and normal exception handling.
- **Observe every side effect.** Run and Effect identifiers, step indexes, progress, success,
  failure, and cancellation all cross the same driver lifecycle.
- **Keep reality at the edge.** The Runtime owns provider clients, credentials, databases,
  scheduling, persistence, and platform policy.
- **Switch Providers without changing agent semantics.** The model layer defines immutable
  messages, requests, responses, tools, streaming events, cache intent, and error categories.
- **Extend by composition.** Extensions are ordinary Python packages and explicitly supplied
  objects—installing or importing a package does not silently enable it.

## Quick start

Purrcept Core is currently developed from source. From a checkout:

```bash
pip install -e .
```

Or install the full contributor environment and run the first offline example:

```bash
uv sync --all-groups
uv run python examples/01_inline_effect.py
```

Define one typed Effect, compose it in a Flow, and run it through the driver:

```python
import asyncio
from dataclasses import dataclass

from purrcept_core import (
    AgentDriver,
    AgentFlow,
    Effect,
    ExecutionContext,
    InlineExecutor,
    perform,
)


@dataclass(frozen=True, slots=True)
class Add(Effect[int]):
    left: int
    right: int

    def execute(self, context: ExecutionContext[None]) -> int:
        return self.left + self.right


def calculate() -> AgentFlow[int]:
    first = yield from perform(Add(1, 2))
    return (yield from perform(Add(first, 4)))


async def main() -> None:
    result = await AgentDriver(InlineExecutor()).run(calculate(), host=None)
    print(result)


asyncio.run(main())
```

```text
7
```

`perform()` preserves the result type of `Effect[T]`. The Runtime owns the event loop; Core never
calls `asyncio.run()` internally.

## Add the model layer

Bind a Runtime-owned backend to a model, then let `Conversation` run the observable tool loop:

```python
from purrcept_core import AgentFlow
from purrcept_core.models import Model, tool


@tool
async def search(query: str) -> list[str]:
    """Search a trusted application data source."""

    return [f"Result for {query}"]


def researcher(model: Model, question: str) -> AgentFlow[str]:
    conversation = model.conversation(
        instructions="Separate verified facts from assumptions.",
        tools=(search,),
        cache="auto",
    )
    result = yield from conversation.ask(question)
    return result.text
```

The concrete `ModelBackend` comes from a Provider package. It receives immutable,
provider-neutral requests and maps them to its SDK or transport. Core supplies a deterministic
[Backend Conformance Kit](./docs/models.md#provider-conformance-kit) so adapters can verify request
semantics, streaming order, errors, cancellation, tool transactions, and reminder removal without
making real network calls.

The separately developed
[`purrcept_litellm`](https://github.com/purrcept/purrcept_litellm) package is one concrete adapter
for LiteLLM Chat Completions.

## Choose the boundary you need

| Layer | Owns | Reach for it when |
| --- | --- | --- |
| Effect kernel | `Flow`, `Effect`, `Executor`, `AgentDriver` | Any operation needs explicit orchestration, observation, or policy. |
| Model standard library | `Model`, `Conversation`, prompts, tools, context, model events | An agent needs Provider-neutral model and tool semantics. |
| Provider package | `ModelBackend`, SDK mapping, transport errors | A concrete model service must implement the Core protocol. |
| Runtime / application | Clients, credentials, host resources, persistence, scheduling | Real resources and product policy must be owned and enforced. |

### Deliberately outside Core

Core does not create event loops, background tasks, provider clients, database connections, or
process-wide singletons. It does not provide platform sessions, Memory/RAG, task scheduling,
durable execution, HTTP/CLI services, a global tool registry, or hidden retries and fallbacks.

Effects and handlers are normal in-process Python code. The unified execution boundary makes them
observable and wrappable; it is not a sandbox or permission boundary. The Runtime must enforce
authorization and decide which inputs, outputs, and errors are safe to record.

## Model capabilities at a glance

- Immutable message, content, request, response, usage, continuation, and streaming-event values.
- `PromptTemplate`, composable render policies, local `PromptLibrary`, and deterministic
  `PromptCompiler`.
- Transactional `Conversation` turns, snapshots, restore, fork, scoped reminders, dynamic
  `ReminderSource`, prompt traces, and complete-request token budgets.
- Typed Python function tools with generated JSON Schema, argument validation, result encoding,
  and explicit error policy.
- Append-only, sliding-window, and injected token-budget context policies.
- Provider-neutral cache intent, continuation policy, structured model errors, and an offline
  backend conformance suite.

For the full contracts and examples, see [Model standard capabilities](./docs/models.md).

## Documentation

**Start here**

- [Core concepts](./docs/concepts.md)
- [Effects and flows](./docs/effects.md)
- [Model standard capabilities](./docs/models.md)
- [Core / Runtime boundary](./docs/runtime-boundary.md)

**Build and extend**

- [Executors](./docs/executors.md)
- [Lifecycle events](./docs/events.md)
- [Middleware](./docs/middleware.md)
- [Testing helpers](./docs/testing.md)
- [Extension packages](./docs/extension-packages.md)

<details>
<summary><strong>Reference, compatibility, and architecture decisions</strong></summary>

- [Public API reference](./docs/api.md)
- [Compatibility and migration](./docs/compatibility.md)
- [Changelog](./CHANGELOG.md)
- [Architecture decision records](./adr/)
- [Core boundary ADR](./adr/0001-core-boundary.md)
- [Model standard library ADR](./adr/0006-model-standard-library.md)
- [Prompt control state ADR](./adr/0007-prompt-control-state.md)

</details>

## Development

Purrcept Core targets Python `3.11` through `3.14`, publishes `py.typed`, uses strict Pyright, and
enforces 100% branch coverage.

```bash
uv sync --all-groups
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest --cov=purrcept_core --cov-report=term-missing
uv build
```

Read [CONTRIBUTING.md](./CONTRIBUTING.md) before changing a public contract. New behavior should
arrive with focused tests and matching narrative documentation.

## Project status

- Current version: `0.5.1` alpha.
- Supported Python versions: `3.11`, `3.12`, `3.13`, and `3.14`.
- Runtime dependency: Pydantic v2, used internally by function tools without entering the public
  model protocol.
- Published on PyPI as `purrcept_core`. See [Releasing](RELEASING.md) for the automated release process.
