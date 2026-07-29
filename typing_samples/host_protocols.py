"""Exercise static contracts for models, hosts, prompt controls, and agent flows.

Pyright analyzes this module as a compile-time integration sample. Objects are
constructed only to assert inferred public types; the file is not a runtime
test or provider implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import assert_type

from purrcept_core import AgentFlow, ExecutionContext, perform
from purrcept_core.models import (
    Conversation,
    ConversationCheckpoint,
    Generate,
    Message,
    Model,
    ModelBackend,
    ModelEventSink,
    ModelRequest,
    ModelResponse,
    ModelTurnResult,
    PromptTemplate,
    ReminderResolutionContext,
    ReminderSource,
    RequestTokenBudget,
    RequestTokenCounter,
    SystemReminder,
    TextBlock,
    TextDelta,
)


class DemoBackend:
    """Minimal structurally compatible backend used for static assertions."""

    async def generate(
        self,
        request: ModelRequest,
        *,
        model: str,
        emit: ModelEventSink | None = None,
    ) -> ModelResponse:
        """Return a response while asserting protocol input types."""

        assert_type(request, ModelRequest)
        assert_type(model, str)
        if emit is not None:
            emit(TextDelta("ok"))
        return ModelResponse(Message.assistant(f"{model}: ok"), model=model)


@dataclass(slots=True)
class DemoRuntimeHost:
    """Typed application host exposed through ``ExecutionContext``."""

    tenant_id: str


def lookup(query: str) -> str:
    """Provide a plain callable whose inferred tool schema accepts text."""

    return query


def count_request_tokens(request: ModelRequest, model: Model) -> int:
    """Provide a synchronous complete-request token counter."""

    return (
        len(request.messages) + len(request.instructions) + len(request.reminders) + len(model.name)
    )


class StaticReminderSource:
    """Provide one statically typed dynamic reminder source."""

    source_id = "typing-sample"

    def resolve(
        self,
        context: ReminderResolutionContext,
    ) -> tuple[SystemReminder, ...]:
        """Project the current model round into request-local guidance."""

        return (
            SystemReminder(
                (TextBlock(f"round={context.model_round}"),),
                key="typing:round",
            ),
        )


backend: ModelBackend = DemoBackend()
fast_model = Model(backend, "demo-fast")
reasoning_model = Model(backend, "demo-reasoning")
conversation = reasoning_model.conversation(tools=(lookup,))
request_counter: RequestTokenCounter = count_request_tokens
reminder_source: ReminderSource = StaticReminderSource()
controlled_conversation = reasoning_model.conversation(
    reminder_sources=(reminder_source,),
    request_budget=RequestTokenBudget(100, request_counter),
)
system_prompt = PromptTemplate(
    "demo.system",
    "Act as {role}.",
    values={"role": "a researcher"},
)
prompt_conversation = reasoning_model.conversation(instructions=system_prompt)

assert_type(fast_model, Model)
assert_type(reasoning_model, Model)
assert_type(fast_model.backend, ModelBackend)
assert_type(conversation, Conversation)
assert_type(conversation.snapshot(), ConversationCheckpoint)
assert_type(controlled_conversation, Conversation)
assert_type(controlled_conversation.request_budget, RequestTokenBudget | None)
assert_type(system_prompt, PromptTemplate)
assert_type(system_prompt.render(), str)
assert_type(prompt_conversation, Conversation)


def inspect_context(context: ExecutionContext[DemoRuntimeHost]) -> None:
    """Assert generic host typing through the execution context."""

    assert_type(context.host, DemoRuntimeHost)
    assert_type(context.host.tenant_id, str)


def low_level_model_flow() -> AgentFlow[ModelResponse]:
    """Assert result inference for a low-level generation effect."""

    effect = fast_model.generate(
        ModelRequest(
            messages=(Message.user("hello"),),
        )
    )
    assert_type(effect, Generate)
    response = yield from perform(effect)
    assert_type(response, ModelResponse)
    return response


def conversational_model_flow() -> AgentFlow[ModelTurnResult]:
    """Assert result inference for the high-level conversation flow."""

    result = yield from conversation.ask("look this up")
    assert_type(result, ModelTurnResult)
    assert_type(result.response, ModelResponse)
    assert_type(result.text, str)
    return result
