"""Build reusable instructions, messages, and reminders from one prompt API.

The example keeps template definitions immutable, stores them in an explicit
local library, and renders the same authoring abstraction into three model
input channels.
"""

from purrcept_core.models import (
    MessageRole,
    PromptLibrary,
    PromptTemplate,
    ReminderScope,
    header,
    join_blocks,
    trim,
)

RESEARCHER = PromptTemplate(
    "researcher.system",
    "Role: {role}\n\n{context.blocks}",
    policies={
        "role": trim(),
        "context.blocks": join_blocks().then(header("# Context")),
    },
    values={"role": "careful research assistant"},
)

QUESTION = PromptTemplate("researcher.question", "Research: {question}")


def main() -> None:
    """Resolve and render templates into typed model values."""

    prompts = PromptLibrary((RESEARCHER, QUESTION))
    instruction = prompts.resolve("researcher.system").instruction(
        {"context.blocks": ["Separate facts from inference.", "Cite evidence."]}
    )
    message = prompts.resolve("researcher.question").message(
        MessageRole.USER,
        {"question": "provider-neutral prompt caching"},
    )
    reminder = PromptTemplate(
        "researcher.reminder",
        "Current constraint: {constraint}",
        values={"constraint": "finish the evidence check"},
    ).reminder(
        key="current-constraint",
        scope=ReminderScope.TURN,
    )

    print(instruction.content[0])
    print(message.text)
    print(reminder.content[0])


if __name__ == "__main__":
    main()
