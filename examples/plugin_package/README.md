# Purrcept demo extension

This standalone 0.5 package demonstrates both supported extension styles:

- `Echo` owns a small inline `execute()` method.
- `DemoModelBackend` implements the official `purrcept_core.models.ModelBackend` Protocol.
- `RenderTemplate` is declarative and is paired with the exported `render_template` handler.

The package imports only public `purrcept_core` APIs. It has no entry points, decorators, global
registries, or import-time initialization. The default demo binds `DemoModelBackend` to a `Model`,
creates a `Conversation`, and supplies an ordinary annotated Python function as a tool. The
conversation automatically drives `Generate -> InvokeTool -> Generate`.

The Runtime host contains only application runtime data and does not own the backend. A real
provider extension would own its SDK client, authentication, transport, and backend adapter;
those concerns remain separate from both the universal kernel and the Runtime host.

The demo backend emits official model stream events through `ModelEventSink`. `Generate` bridges
them into `EffectProgress` for the active Effect while returning one final `ModelResponse` to the
Flow. Everything is deterministic and offline; provider calls are not implicitly retried.
