"""Host-owned, request-bound native continuity maintenance transaction."""
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
import weakref

from agent.native_compaction import validate_persisted_native_compaction_history
from agent.native_incremental_handoff import (
    _continuity_note_arguments,
    _item_value,
    _note_fence,
    _projection_source_for_messages,
    _staged_note,
    create_native_incremental_note,
    record_native_incremental_note,
    record_native_incremental_note_from_tool_call,
    restore_native_incremental_note,
)


class NativeNoteRefreshFailure(RuntimeError):
    """Local maintenance failure; never a provider retry/fallback signal."""

    def __init__(self, reason):
        self.reason = reason
        super().__init__(f"Continuity-note refresh failed: {reason}")


_LIVE_CAPABILITIES = weakref.WeakSet()


class _NoteRefreshCapability:
    """Opaque host state, never a tool argument or persisted permission."""

    def __init__(self, agent: Any, messages):
        self.agent = agent
        self.session_id = agent.session_id
        self.request_id = getattr(agent, "_current_api_request_id", None)
        self.source_fence = _note_fence(messages)
        self.previous_note = deepcopy(_staged_note(agent, messages))
        source, _ = _projection_source_for_messages(agent, messages)
        if source is None:
            raise NativeNoteRefreshFailure("canonical maintenance source unauthenticated")
        self.canonical_source_fence = _note_fence(source)
        self.call_id: str | None = None
        self.arguments: dict | None = None
        self.consumed = False
        self.request: dict | None = None

    def check(self, agent):
        if (
            self not in _LIVE_CAPABILITIES
            or self.agent is not agent
            or getattr(agent, "_native_note_refresh_capability", None) is not self
            or self.session_id != agent.session_id
            or not self.request_id
            or self.request_id != getattr(agent, "_current_api_request_id", None)
        ):
            raise NativeNoteRefreshFailure("invalid or expired maintenance capability")

    def bind_request(self, request):
        self.check(self.agent)
        self.request = deepcopy(request)

    def check_request(self, agent, request):
        self.check(agent)
        if self.request is None or any(
            request.get(key) != self.request.get(key)
            for key in ("model", "input", "instructions", "tools", "tool_choice", "parallel_tool_calls")
        ):
            raise NativeNoteRefreshFailure("maintenance request changed after preparation")

    def close(self):
        _LIVE_CAPABILITIES.discard(self)
        if self.agent is not None and getattr(self.agent, "_native_note_refresh_capability", None) is self:
            self.agent._native_note_refresh_capability = None
        self.agent = None
        self.arguments = None
        self.previous_note = None
        self.request = None


def close_native_note_refresh(agent):
    capability = getattr(agent, "_native_note_refresh_capability", None)
    if type(capability) is _NoteRefreshCapability:
        capability.close()


def issue_native_note_refresh(agent, messages):
    close_native_note_refresh(agent)
    capability = _NoteRefreshCapability(agent, messages)
    _LIVE_CAPABILITIES.add(capability)
    agent._native_note_refresh_capability = capability
    return capability


def consume_native_note_refresh_capability(capability, agent, name, arguments, call_id):
    """Bind one dispatch; policy hooks still run and may deny it."""
    if type(capability) is not _NoteRefreshCapability:
        raise NativeNoteRefreshFailure("forged maintenance capability")
    capability.check(agent)
    if (
        capability.consumed or name != "continuity_note"
        or not capability.call_id or call_id != capability.call_id
        or _continuity_note_arguments(arguments) != capability.arguments
    ):
        raise NativeNoteRefreshFailure("maintenance call changed or duplicated")
    capability.consumed = True


def is_native_note_refresh_authorized(
    capability, *, tool_name, args, session_id, tool_call_id, api_request_id,
) -> bool:
    """Pure fail-closed proof for policy hooks, never a model-supplied flag."""
    if type(capability) is not _NoteRefreshCapability:
        return False
    try:
        agent = capability.agent
        if agent is None:
            return False
        capability.check(agent)
        return bool(
            capability.consumed
            and capability.request is not None
            and tool_name == "continuity_note"
            and session_id == capability.session_id == agent.session_id
            and api_request_id == capability.request_id
            and tool_call_id == capability.call_id
            and isinstance(tool_call_id, str) and tool_call_id
            and isinstance(args, dict) and args == capability.arguments
            and not getattr(agent, "_interrupt_requested", False)
        )
    except Exception:
        return False


def account_failed_native_note_refresh(agent, response):
    """Account a consumed response rejected before ordinary success accounting."""
    from agent.usage_pricing import normalize_usage, estimate_usage_cost

    usage = _item_value(response, "usage") if response is not None else None
    if usage is None:
        return
    canonical = normalize_usage(usage, provider=agent.provider, api_mode=agent.api_mode)
    cost = estimate_usage_cost(
        agent.model, canonical, provider=agent.provider, base_url=agent.base_url,
        api_key=getattr(agent, "api_key", ""),
    )
    for attr, value in (
        ("session_prompt_tokens", canonical.prompt_tokens),
        ("session_completion_tokens", canonical.output_tokens),
        ("session_total_tokens", canonical.total_tokens),
        ("session_input_tokens", canonical.input_tokens),
        ("session_output_tokens", canonical.output_tokens),
        ("session_cache_read_tokens", canonical.cache_read_tokens),
        ("session_cache_write_tokens", canonical.cache_write_tokens),
        ("session_reasoning_tokens", canonical.reasoning_tokens),
        ("session_api_calls", 1),
    ):
        setattr(agent, attr, getattr(agent, attr, 0) + value)
    if cost.amount_usd is not None:
        agent.session_estimated_cost_usd += float(cost.amount_usd)
    agent.session_cost_status, agent.session_cost_source = cost.status, cost.source
    if agent._session_db and agent.session_id:
        agent._session_db.queue_token_counts(
            agent.session_id, input_tokens=canonical.input_tokens,
            output_tokens=canonical.output_tokens,
            cache_read_tokens=canonical.cache_read_tokens,
            cache_write_tokens=canonical.cache_write_tokens,
            reasoning_tokens=canonical.reasoning_tokens,
            estimated_cost_usd=float(cost.amount_usd) if cost.amount_usd is not None else None,
            cost_status=cost.status, cost_source=cost.source,
            billing_provider=agent.provider, billing_base_url=agent.base_url,
            billing_mode="subscription_included" if cost.status == "included" else None,
            model=agent.model, api_call_count=1,
        )


def execute_native_note_refresh(agent, capability, response, messages, effective_task_id):
    """Publish one authenticated pair before allowing another request.

    Recording is isolated until durable readback succeeds, including when a
    timed-out middleware worker finishes late. Failures never replace history.
    """
    from agent.tool_executor import _run_sequential_tool_execution_middleware
    from agent.tool_dispatch_helpers import make_tool_result_message

    initial_message_count = len(messages)
    pair_persisted = False
    try:
        if type(capability) is not _NoteRefreshCapability:
            raise NativeNoteRefreshFailure("forged maintenance capability")
        capability.check(agent)
        if capability.call_id is not None or _note_fence(messages) != capability.source_fence:
            raise NativeNoteRefreshFailure("maintenance source changed or response duplicated")
        validate_persisted_native_compaction_history(messages)
        source, _ = _projection_source_for_messages(agent, messages)
        if source is None or _note_fence(source) != capability.canonical_source_fence:
            raise NativeNoteRefreshFailure("canonical maintenance source changed")
        if getattr(agent, "_interrupt_requested", False):
            raise NativeNoteRefreshFailure("maintenance cancelled")
        output = _item_value(response, "output")
        if _item_value(response, "status") != "completed" or not isinstance(output, list):
            raise NativeNoteRefreshFailure("maintenance response incomplete")
        calls = [item for item in output if _item_value(item, "type") == "function_call"]
        if len(calls) != 1 or any(
            _item_value(item, "type") not in {"function_call", "reasoning", "message"}
            for item in output
        ):
            raise NativeNoteRefreshFailure("expected exactly one direct continuity_note call")
        call = calls[0]
        raw_arguments = _item_value(call, "arguments")
        arguments = _continuity_note_arguments(raw_arguments)
        call_id = _item_value(call, "call_id")
        if (
            _item_value(call, "name") != "continuity_note" or arguments is None
            or not isinstance(raw_arguments, str)
            or not isinstance(call_id, str) or not call_id.strip()
            or any(row.get("tool_call_id") == call_id or any(
                prior.get("id") == call_id for prior in row.get("tool_calls", []) or []
            ) for row in messages)
        ):
            raise NativeNoteRefreshFailure("invalid, wrapped or reused maintenance call")
        if getattr(agent, "_persist_disabled", False) or getattr(agent, "_session_db", None) is None:
            raise NativeNoteRefreshFailure("durable maintenance persistence unavailable")
        capability.call_id, capability.arguments = call_id, arguments
        # Maintenance prose/reasoning is not ordinary assistant output. Stage
        # only the exact call and host-authenticated result, with no UI
        # emission. The pair is flushed once, atomically, after dispatch.
        messages.append({"role": "assistant", "content": "", "tool_calls": [{
            "id": call_id, "type": "function", "function": {
                "name": "continuity_note", "arguments": raw_arguments,
            },
        }]})
        if _note_fence(messages[:-1]) != capability.source_fence:
            raise NativeNoteRefreshFailure("maintenance source changed during persistence")
        source, _ = _projection_source_for_messages(agent, messages)
        if source is None:
            raise NativeNoteRefreshFailure("canonical maintenance source unauthenticated")
        validate_persisted_native_compaction_history(source)
        source_fence = _note_fence(source)
        expected = create_native_incremental_note(
            session_id=agent.session_id, source_messages=source, **arguments,
        )
        staged = SimpleNamespace(
            session_id=agent.session_id, native_incremental_handoff_enabled=True,
        )

        def record(final_args):
            capability.check(agent)
            if not capability.consumed or _continuity_note_arguments(final_args) != arguments:
                raise NativeNoteRefreshFailure("maintenance recorder arguments changed")
            return record_native_incremental_note_from_tool_call(staged, final_args, source)

        managed = _run_sequential_tool_execution_middleware(
            agent, function_name="continuity_note", function_args=arguments,
            effective_task_id=effective_task_id, tool_call_id=call_id,
            execute=record, maintenance_capability=capability,
        )
        capability.check(agent)
        if (
            managed.blocked or not managed.dispatched
            or getattr(staged, "_native_incremental_handoff_note", None) != expected
            or getattr(agent, "_interrupt_requested", False)
            or expected["source_cursor"] <= (capability.previous_note or {}).get("source_cursor", -1)
        ):
            raise NativeNoteRefreshFailure("blocked, failed or unchanged authenticated note")
        messages.append(make_tool_result_message(
            "continuity_note", managed.result, call_id, effect_disposition="none",
        ))
        if agent._flush_messages_to_session_db(messages) is False:
            raise NativeNoteRefreshFailure("maintenance pair persistence failed")
        pair_persisted = True
        # A successful flush alone is not proof: authenticate a fresh reader of
        # the canonical database, not the mutable replay or staged note.
        stored = agent._session_db.get_messages_as_conversation(
            agent.session_id, repair_alternation=False,
        )
        probe = SimpleNamespace(session_id=agent.session_id)
        if restore_native_incremental_note(probe, stored) != expected:
            raise NativeNoteRefreshFailure("durable note readback failed authentication")
        if not any(
            row.get("role") == "tool" and row.get("tool_call_id") == call_id
            and row.get("content") == managed.result for row in stored
        ):
            raise NativeNoteRefreshFailure("durable tool result missing")
        current_source, _ = _projection_source_for_messages(agent, messages[:-1])
        if current_source is None or _note_fence(current_source) != source_fence:
            raise NativeNoteRefreshFailure("maintenance source changed during dispatch")
        record_native_incremental_note(agent, expected, source)
        # The ordinary usage/accounting path still owns this physical request.
        # Give it only the validated call, not maintenance narration/refusals
        # that could trigger ordinary retry or user-facing delivery policies.
        return SimpleNamespace(
            output=[deepcopy(call)], status="completed",
            usage=_item_value(response, "usage"), model=_item_value(response, "model"),
            id=_item_value(response, "id"),
        )
    except Exception as exc:
        if not pair_persisted:
            del messages[initial_message_count:]
        if isinstance(exc, NativeNoteRefreshFailure):
            raise
        raise NativeNoteRefreshFailure("maintenance execution or persistence failed") from exc
    finally:
        if type(capability) is _NoteRefreshCapability:
            capability.close()
