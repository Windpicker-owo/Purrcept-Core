"""Verify the model package exports the complete provider-neutral public surface."""

from __future__ import annotations

from dataclasses import fields

import purrcept_core
import purrcept_core.models as models


def test_models_public_api_is_available_from_the_subpackage() -> None:
    required = {
        "AppendOnlyContext",
        "BackendConformanceScenario",
        "BackendConformanceStep",
        "CacheMode",
        "ContentBlock",
        "ContextPolicy",
        "ContinuationPolicy",
        "Conversation",
        "ConversationCheckpoint",
        "ConversationState",
        "FunctionTool",
        "Generate",
        "InvokeTool",
        "Message",
        "Model",
        "ModelBackend",
        "ModelContinuation",
        "ModelRequest",
        "ModelResponse",
        "ModelSettings",
        "ModelStreamEvent",
        "ModelTurnResult",
        "PromptCachePolicy",
        "PromptCompiler",
        "PromptLibrary",
        "PromptStability",
        "PromptTemplate",
        "ReminderSource",
        "ReminderState",
        "RenderPolicy",
        "RequestTokenBudget",
        "ReasoningBlock",
        "SlidingWindowContext",
        "SystemInstruction",
        "SystemReminder",
        "TextBlock",
        "TokenBudgetContext",
        "ToolChoice",
        "ToolParameter",
        "ToolSet",
        "ToolSpec",
        "run_backend_conformance",
    }

    assert required <= set(models.__all__)
    assert all(hasattr(models, name) for name in models.__all__)
    assert "ModelHost" not in models.__all__
    assert "ToolDefinition" not in models.__all__


def test_model_types_do_not_pollute_the_kernel_top_level() -> None:
    assert "Message" not in purrcept_core.__all__
    assert "ModelBackend" not in purrcept_core.__all__
    assert not hasattr(purrcept_core, "Generate")


def test_optional_dataclass_fields_are_keyword_only() -> None:
    expected = {
        models.ImageBlock: {"alt_text"},
        models.ToolCallBlock: {"arguments"},
        models.ToolResultBlock: {"content", "is_error"},
        models.Message: {"name", "metadata"},
        models.SystemInstruction: {"key", "stability"},
        models.SystemReminder: {"key", "scope", "placement", "priority"},
        models.ToolSpec: {"description", "parameters"},
        models.ModelSettings: {
            "temperature",
            "max_output_tokens",
            "stop_sequences",
            "tool_choice",
            "parallel_tool_calls",
        },
        models.PromptCachePolicy: {"mode", "key", "ttl", "strict"},
        models.PromptTemplate: {"policies", "values"},
        models.PromptDraft: {
            "instructions",
            "tools",
            "settings",
            "cache",
            "continuation",
            "metadata",
            "provider_options",
        },
        models.PromptDiagnostics: {
            "dynamic_source_ids",
            "estimated_tokens",
            "trimmed_turns",
        },
        models.CompiledPrompt: {"consumed_next_request"},
        models.PromptCompiled: {
            "dynamic_source_ids",
            "estimated_tokens",
            "trimmed_turns",
        },
        models.PreparedContext: {"continuation", "turn_boundaries"},
        models.ConversationState: {
            "continuation",
            "turn_index",
            "turn_boundaries",
        },
        models.ModelRequest: {
            "instructions",
            "reminders",
            "tools",
            "settings",
            "cache",
            "continuation",
            "metadata",
            "provider_options",
        },
        models.TokenUsage: {
            "cached_input_tokens",
            "cache_write_input_tokens",
            "reasoning_tokens",
        },
        models.ModelResponse: {
            "usage",
            "finish_reason",
            "model",
            "response_id",
            "continuation",
            "provider_metadata",
        },
        models.ToolResult: {"content", "is_error"},
        models.ModelTurnResult: {"tool_results"},
        models.BackendConformanceScenario: {
            "expected_response",
            "expected_error",
            "streaming",
            "follow_up_steps",
        },
        models.BackendConformanceResult: {"error"},
        models.ModelStreamStarted: {
            "model",
            "response_id",
            "provider_metadata",
        },
        models.TextDelta: {"index"},
        models.ToolCallDelta: {"tool_call_id", "name"},
        models.Generate: {"emit_stream_events"},
    }

    for value_type, expected_names in expected.items():
        actual_names = {item.name for item in fields(value_type) if item.kw_only}
        assert actual_names == expected_names
