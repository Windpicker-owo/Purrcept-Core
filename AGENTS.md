## Source narrative and commenting style

Purrcept Core is an abstraction-heavy SDK. Source code must be optimized for
guided reading and long-term maintenance, not for minimum line count.

### General principle

A reader should be able to understand the responsibility, control flow,
invariants, and extension points of a module without repeatedly searching
through unrelated files.

Comments must explain semantic intent, constraints, ordering, lifecycle, or
design rationale. Do not add comments that merely restate the code.

### Module documentation

Every non-trivial module must begin with a module docstring that explains:

- What the module owns.
- What the module intentionally does not own.
- Its primary entry points.
- Its major processing stages.
- Its important invariants.
- Its relationship to neighboring modules.

Small data-only or re-export modules may use a shorter docstring.

### Abstract interfaces

Every public Protocol, ABC, abstract class, and extension interface must
document:

- Who creates it.
- Who calls it.
- When it is called.
- Preconditions and postconditions.
- State ownership and lifecycle.
- Ordering and mutation rules.
- Error behavior.
- Built-in implementations.
- How an implementation is selected.

The abstraction owns the shared contract. Concrete implementations should
document only behavior that differs from that contract.

### Functions and methods

All public functions and methods require docstrings.

Private functions also require a docstring or an immediately adjacent block
comment when they involve non-obvious behavior, state changes, ordering,
caching, concurrency, lifecycle, error handling, or algorithmic decisions.

Trivial private helpers whose complete semantics are obvious from their name,
signature, and body do not require a docstring.

### Narrative flow

Functions containing multiple conceptual phases must separate those phases
with blank lines.

Add a short block comment before a phase when its purpose or ordering is not
obvious. Phase comments should explain why the phase exists or why it occurs
at that point, rather than paraphrasing its statements.

The main orchestration path must remain visible. Do not extract small private
helpers solely to reduce function length when doing so hides the overall flow.

### Section boundaries

Long modules should use restrained semantic section headings:

    # ---------------------------------------------------------------------------
    # Reminder resolution
    # ---------------------------------------------------------------------------

Use section headings only for substantial conceptual regions. Prefer domain
names such as "Context budget allocation" over structural names such as
"Private helpers".

### Required rationale comments

Explicitly document:

- Invariants and state transitions.
- Order-dependent behavior.
- Cache-key and stable-prefix decisions.
- Ownership and concurrency assumptions.
- Retry, suppression, and exception translation.
- Provider compatibility workarounds.
- Deliberate rejection of an obvious alternative.
- Cross-module hand-off points.
- Non-obvious performance tradeoffs.

Critical comments may begin with `Invariant:`, `Ordering:`, `Cache:`,
`Compatibility:`, or `Lifecycle:`.

### Style

Write comments as complete sentences. Keep them close to the behavior they
explain. Use paragraphs for multi-sentence explanations.

Do not repeat type hints, signatures, or obvious operations.

When changing behavior, update or remove affected comments in the same change.
A stale comment is a correctness defect.

Before completing a change, read every modified file from top to bottom as a
new maintainer. Add enough orientation that the main control flow can be
understood before opening helper implementations.