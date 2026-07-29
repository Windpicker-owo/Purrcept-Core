# Contributing

Purrcept Core 保持两条清晰边界：

- 通用内核中，Flow 决定下一步，Effect 描述一次操作，Executor 执行，Driver 推进 Run；
- 模型标准库中，Conversation 编排独立的 Generate / InvokeTool Effect，并面向统一的
  Provider 协议。

平台会话、数据库、持久化、调度、网络、Provider client、凭据、平台工具和业务权限仍
属于 Runtime 或独立扩展包。不要通过全局 Registry、import-time 注册、隐藏重试或隐藏
模型调用跨越边界。

开发环境使用 Python 3.11+ 和 `uv`：

```bash
uv sync --all-groups
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
uv run python -m build
```

新增公共行为应包含聚焦测试和对应文档，并维持分支覆盖率门禁。模型值对象应保持不可变、
Provider 无关和 keyword-only 可扩展性。新的自动化步骤必须继续以可观察 Effect 表达。

函数工具可以在内部使用 Pydantic v2，但公共注解、Protocol、值对象和错误不能暴露
Pydantic 类型。涉及 Python 3.11 `TypedDict` 的工具测试应优先使用
`typing_extensions.TypedDict`。

Provider 适配变更应运行标准 Backend Conformance Kit，并在扩展包中补充供应商
wire-format、SDK 版本、认证和特殊能力测试。新增运行时依赖需要说明维护成本、替换边界
和为什么现有能力不足。

## Source Readability and Narrative Documentation

Repository-specific instructions take precedence over this section.

Treat source code as a technical document for future maintainers. Optimize for
guided reading, clear control flow, and explicit design intent rather than
minimum line count or maximum compactness.

Types describe data shape. Names describe local actions. Documentation and
comments must explain semantics, contracts, constraints, ordering, lifecycle,
ownership, and design rationale.

### General requirements

- Preserve the repository's established documentation language and style.
- If no documentation language is established, use English.
- Write comments and documentation as complete, precise sentences.
- Do not add comments merely to increase comment coverage.
- Do not assume that clean naming, type hints, or private visibility make
  non-obvious behavior self-explanatory.
- Documentation is part of correctness. Stale or misleading comments are
  defects and must be updated with the code.

### Files and modules

Every non-trivial source file or module must provide an opening documentation
comment or module docstring that gives the reader an initial mental model.

When relevant, explain:

- What the module owns.
- What it intentionally does not own.
- Its primary entry points.
- Its major processing stages.
- Its important invariants.
- Its relationship to neighboring modules.
- Where concrete implementations or extension points can be found.

Small data-only modules, generated files, re-export modules, and obvious
single-purpose wrappers may use shorter documentation.

### Public APIs and abstractions

Document all public APIs and all extension-facing abstractions.

Protocols, abstract classes, interfaces, hooks, policies, strategies, and
lifecycle components must document their shared contract, including relevant
details such as:

- Who creates the object.
- Who calls it.
- When it is called.
- Preconditions and postconditions.
- State ownership and mutation rules.
- Ordering guarantees.
- Concurrency and reuse assumptions.
- Error behavior.
- Built-in implementations.
- How an implementation is selected or registered.

The abstraction should own the shared contract. Concrete implementations
should document only behavior that differs from or extends that contract.

Do not write documentation that merely repeats a function signature, type
annotation, field name, or return type. Explain what the values mean and what
callers may rely on.

### Private functions

Private visibility does not remove the need for explanation.

A private function requires a docstring or a nearby rationale comment when it
contains non-obvious:

- State changes.
- Ordering dependencies.
- Cache behavior.
- Lifecycle behavior.
- Ownership or concurrency assumptions.
- Failure, retry, fallback, or suppression behavior.
- Algorithms or heuristics.
- Compatibility workarounds.
- Cross-module hand-offs.
- Semantics that cannot be inferred from its name, signature, and body at a
  glance.

Trivial private helpers whose complete behavior is obvious may omit a
docstring.

### Narrative control flow

The main orchestration path should read from top to bottom like an outline of
the operation.

- Keep major processing stages visible in the main function.
- Use meaningful intermediate variables instead of deeply nested expressions
  when they represent distinct concepts.
- Separate conceptual phases with blank lines.
- Add a short block comment before a phase when its purpose or position is not
  obvious.
- Explain why a phase exists or why it must occur at that point. Do not merely
  paraphrase the statements below it.
- Extract helpers only when they represent a meaningful concept, isolate real
  complexity, enable reuse, improve testing, or hide irrelevant detail.
- Do not split a readable workflow into many tiny private methods solely to
  reduce function length.
- Do not compress multiple semantic operations into one expression solely to
  reduce line count.

A reader should be able to understand the overall workflow before opening
every helper implementation.

### Required rationale comments

Explicitly document behavior involving:

- Invariants and state transitions.
- Steps whose order must not change.
- Stable prefixes, cache keys, invalidation, or cache boundaries.
- Object ownership, resource lifetime, or concurrency.
- Exception translation, ignored failures, retries, and fallbacks.
- Provider, platform, protocol, or version compatibility.
- Non-obvious performance or memory tradeoffs.
- Security-sensitive decisions.
- Deliberate rejection of a simpler-looking alternative.
- Code that appears redundant but must not be removed.
- Handoffs between subsystems.

Critical comments may begin with labels such as:

- `Invariant:`
- `Ordering:`
- `Cache:`
- `Lifecycle:`
- `Concurrency:`
- `Compatibility:`
- `Security:`

Use these labels selectively, only when they improve discoverability.

### File structure and visual rhythm

Long files may use restrained semantic section headings, for example:

    # ---------------------------------------------------------------------------
    # Context budget allocation
    # ---------------------------------------------------------------------------

Use section headings only for substantial conceptual regions. Prefer domain
headings such as `Reminder resolution` over structural headings such as
`Private helpers`.

Use whitespace to communicate changes in thought. Code should not appear as
an uninterrupted dense block when it contains multiple conceptual stages.

### Avoid

Do not write comments that:

- Restate obvious code.
- Repeat type annotations.
- Use vague verbs such as "handle", "process", or "manage" without explaining
  the actual semantics.
- Describe fragile line-by-line implementation details instead of stable
  intent.
- Compensate for poor names or unclear abstractions.
- Duplicate large amounts of documentation across an interface and all its
  implementations.
- Claim behavior that is not enforced by the implementation or tests.

### Completion check

Before completing a change, read every modified file from top to bottom as a
new maintainer.

Verify that:

- The file's responsibility and entry points are quickly understandable.
- The main control flow is visible without excessive navigation.
- Abstract contracts and extension points are documented.
- Important ordering, state, cache, lifecycle, and concurrency rules are
  explained.
- Dense expressions have not hidden distinct conceptual steps.
- Comments explain information that the code alone does not communicate.
- All affected documentation and comments still match the implementation.

Improve the narrative readability before considering the task complete.