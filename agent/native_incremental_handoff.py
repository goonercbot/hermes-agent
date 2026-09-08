"""Opt-in one-call native incremental compaction for OpenAI Codex.

This is intentionally separate from the legacy gpt-5.6 native path.  It sends
one bounded ``gpt-5.6-luna`` Responses request with inline compaction enabled;
it never asks the active agent model to write a handoff or summary.  The emitted
checkpoint, its post-checkpoint message suffix, an authenticated host-side note,
and the protected live tail are persisted through the existing v2 native
checkpoint carrier.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional

from agent.native_compaction import (
    NATIVE_COMPACTION_HANDOFF_ROLE,
    NATIVE_COMPACTION_METADATA_KEY,
    NATIVE_COMPACTION_VERSION,
    NativeCompactionAttempt,
    _latest_user_tail,
    _native_message_size,
    _set_native_attempt_state,
    native_continuity_boundary_fence,
    validate_persisted_native_compaction_history,
)

logger = logging.getLogger(__name__)

NATIVE_INCREMENTAL_NOTE_VERSION = 1
NATIVE_INCREMENTAL_MODEL = "gpt-5.6-luna"
NATIVE_INCREMENTAL_COMPACT_THRESHOLD = 32_000
NATIVE_INCREMENTAL_NOTE_MAX_CHARS = 8_000
NATIVE_INCREMENTAL_SUFFIX_MAX_ITEMS = 16

# Inline compaction is still a model turn: its temporary task instructions can
# survive in opaque state. Explicitly end that maintenance task on every normal
# replay, including old checkpoints minted with the unscoped no-answer prompt.
NATIVE_INCREMENTAL_RESUME_INSTRUCTIONS = (
    "The previous compaction-only operation is finished. Any prior instruction "
    "to avoid answering was scoped to that finished operation, not this turn. "
    "Answer the latest user message normally, following current instructions "
    "and tool permissions."
)


def native_incremental_resume_instructions(
    instructions: str, source_messages: List[Dict[str, Any]]
) -> str:
    """Scope maintenance instructions without rewriting checkpoint/history."""
    handoffs = validate_persisted_native_compaction_history(source_messages)
    if not any(text.startswith("NATIVE_INCREMENTAL_NOTE\n") for text in handoffs.values()):
        return instructions
    if instructions.endswith(NATIVE_INCREMENTAL_RESUME_INSTRUCTIONS):
        return instructions
    return instructions + "\n\n" + NATIVE_INCREMENTAL_RESUME_INSTRUCTIONS


def native_incremental_continuity_capable(
    agent: Any,
    *,
    is_codex_backend: bool,
    is_xai_responses: bool = False,
    is_github_responses: bool = False,
) -> bool:
    """Return true only for the explicit official-Codex luna route.

    ``agent.model`` is deliberately not consulted.  The active agent may be
    Astra/Sol; the compression request remains pinned to luna.
    """
    return bool(
        getattr(agent, "api_mode", None) == "codex_responses"
        and getattr(agent, "provider", "") == "openai-codex"
        and is_codex_backend
        and not is_xai_responses
        and not is_github_responses
        and bool(getattr(agent, "native_incremental_handoff_enabled", False))
        and bool(getattr(agent, "compression_enabled", True))
        and bool(getattr(agent, "_codex_reasoning_replay_enabled", True))
        and str(getattr(agent, "native_incremental_handoff_model", "")).strip().lower()
        == NATIVE_INCREMENTAL_MODEL
    )


def _note_fence(messages: List[Dict[str, Any]]) -> str:
    return native_continuity_boundary_fence(messages)


def _source_prefix_fence(messages: List[Dict[str, Any]], cursor: int) -> str:
    """Bind a note to an immutable exact prefix, not the growing transcript."""
    if not isinstance(cursor, int) or isinstance(cursor, bool) or cursor < 0:
        raise ValueError("native incremental note cursor must be a non-negative integer")
    if cursor > len(messages):
        raise ValueError("native incremental note cursor exceeds source")
    return _note_fence(messages[:cursor])


def bind_native_incremental_replay_projection(
    agent: Any,
    *,
    source_messages: List[Dict[str, Any]],
    replay_messages: List[Dict[str, Any]],
) -> bool:
    """Bind a gateway-cleaned request projection to its immutable DB source.

    Gateway replay cleanup deliberately replaces interrupted side-effect tool
    output with an ``UNKNOWN`` recovery record.  That is safe model input, but
    it is not the historical evidence that authenticated a continuity note.
    Keep both views in transient host state: persisted notes are validated only
    against ``source_messages``; the exact replay list is separately fenced so
    a caller cannot substitute an arbitrary transformed transcript.
    """
    if not isinstance(source_messages, list) or not isinstance(replay_messages, list):
        return False
    try:
        validate_persisted_native_compaction_history(source_messages)
        validate_persisted_native_compaction_history(replay_messages)
    except ValueError:
        return False
    agent._native_incremental_replay_projection = {
        "source": deepcopy(source_messages),
        "source_fence": _note_fence(source_messages),
        "replay": deepcopy(replay_messages),
        "replay_fence": _note_fence(replay_messages),
    }
    return True


def _projection_source_for_messages(
    agent: Any, messages: List[Dict[str, Any]]
) -> tuple[Optional[List[Dict[str, Any]]], Optional[int]]:
    """Return authenticated source plus the matching replay cursor.

    A projection is accepted only while its original replay prefix still has
    the host-recorded fence.  New rows appended during this turn are shared by
    source and projection, so a newly minted note can be authenticated against
    the canonical source and still compact the replay list on the next turn.
    """
    state = getattr(agent, "_native_incremental_replay_projection", None)
    if not isinstance(state, dict):
        return list(messages), None
    source = state.get("source")
    replay = state.get("replay")
    if not isinstance(source, list) or not isinstance(replay, list):
        return None, None
    if (
        state.get("source_fence") != _note_fence(source)
        or state.get("replay_fence") != _note_fence(replay)
        or len(messages) < len(replay)
        or _note_fence(messages[:len(replay)]) != state.get("replay_fence")
    ):
        return None, None
    return deepcopy(source) + deepcopy(messages[len(replay):]), len(replay)


def _projection_cursor_for_source_cursor(
    source_cursor: int,
    *,
    source_base_count: int,
    replay_base_count: int,
) -> Optional[int]:
    """Map a source cursor through a bounded host replay projection."""
    if source_cursor < 0:
        return None
    if source_cursor <= source_base_count:
        # Existing notes can be replayed only when cleanup preserved row
        # positions.  A deletion before a note is ambiguous; fail closed.
        return source_cursor if source_base_count == replay_base_count else None
    return replay_base_count + (source_cursor - source_base_count)


def create_native_incremental_note(
    *,
    session_id: Any,
    source_messages: List[Dict[str, Any]],
    objective: Any,
    current_plan: Any,
    next_action: Any,
    blockers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create bounded agent-authored state for ordinary-work continuity.

    The exact *prefix* at ``source_cursor`` is authenticated.  Later ordinary
    messages may append without staling the note; the compactor retains every
    row after that cursor plus the latest user correction as protected tail.
    """
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("native incremental note requires a session id")
    if not isinstance(source_messages, list):
        raise ValueError("native incremental note requires source messages")
    values = {
        "objective": objective,
        "current_plan": current_plan,
        "next_action": next_action,
    }
    if not all(isinstance(value, str) and value.strip() for value in values.values()):
        raise ValueError("native incremental note fields must be non-empty text")
    if blockers is None:
        blockers = []
    if not isinstance(blockers, list) or not all(
        isinstance(value, str) and value.strip() for value in blockers
    ):
        raise ValueError("native incremental note blockers must be text")
    cursor = len(source_messages)
    note = {
        "version": NATIVE_INCREMENTAL_NOTE_VERSION,
        "session_id": session_id,
        "source_cursor": cursor,
        "source_prefix_fence": _source_prefix_fence(source_messages, cursor),
        "claim_kind": "agent_authored",
        "instruction_precedence": (
            "Current and later user instructions supersede this historical "
            "agent-authored continuity note."
        ),
        **values,
        "blockers": list(blockers),
    }
    canonical = json.dumps(note, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(canonical) > NATIVE_INCREMENTAL_NOTE_MAX_CHARS:
        raise ValueError("native incremental note exceeds bounded size")
    return note


def record_native_incremental_note(
    agent: Any,
    note: Dict[str, Any],
    source_messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Validate and stage a note for the next native incremental operation.

    The note is not inferred from transcript text.  This prevents lookalike
    markers in user/tool output from becoming trusted continuity state.
    """
    expected = create_native_incremental_note(
        session_id=getattr(agent, "session_id", None),
        source_messages=source_messages,
        objective=note.get("objective") if isinstance(note, dict) else None,
        current_plan=note.get("current_plan") if isinstance(note, dict) else None,
        next_action=note.get("next_action") if isinstance(note, dict) else None,
        blockers=note.get("blockers") if isinstance(note, dict) else None,
    )
    if not isinstance(note, dict) or note != expected:
        raise ValueError("native incremental note has invalid identity or shape")
    agent._native_incremental_handoff_note = deepcopy(expected)
    return deepcopy(expected)


def _staged_note(agent: Any, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    note = getattr(agent, "_native_incremental_handoff_note", None)
    if not isinstance(note, dict):
        return None
    cursor = note.get("source_cursor")
    if not isinstance(cursor, int) or isinstance(cursor, bool):
        return None
    source_messages, replay_base_count = _projection_source_for_messages(agent, messages)
    if source_messages is None:
        return None
    try:
        expected = create_native_incremental_note(
            session_id=getattr(agent, "session_id", None),
            source_messages=source_messages[:cursor],
            objective=note.get("objective"),
            current_plan=note.get("current_plan"),
            next_action=note.get("next_action"),
            blockers=note.get("blockers"),
        )
        source_prefix_fence = _source_prefix_fence(source_messages, cursor)
    except (TypeError, ValueError):
        return None
    # ``create`` computes a cursor relative to its prefix; restore the exact
    # source cursor before structural comparison.
    expected["source_cursor"] = cursor
    expected["source_prefix_fence"] = source_prefix_fence
    if note != expected:
        return None
    if replay_base_count is None:
        projection_cursor = cursor
    else:
        projection_cursor = _projection_cursor_for_source_cursor(
            cursor,
            source_base_count=len(getattr(agent, "_native_incremental_replay_projection")["source"]),
            replay_base_count=replay_base_count,
        )
    if projection_cursor is None or projection_cursor > len(messages):
        return None
    agent._native_incremental_handoff_projection_cursor = projection_cursor
    return deepcopy(note)


def _protected_tail_since_note(
    messages: List[Dict[str, Any]], note: Dict[str, Any], *, projection_cursor: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Keep new operational evidence and the latest user correction verbatim."""
    cursor = note["source_cursor"] if projection_cursor is None else projection_cursor
    latest_user_tail = _latest_user_tail(messages)
    latest_user_start = len(messages) - len(latest_user_tail)
    start = min(cursor, latest_user_start)
    # Recording happens inside a tool group. Its result can be the first row
    # after the cursor, so keep the matching assistant call and sibling results.
    while start > 0 and start < len(messages) and messages[start].get("role") == "tool":
        start -= 1
    return deepcopy(messages[start:])


def _continuity_note_arguments(value: Any) -> Optional[Dict[str, Any]]:
    """Return schema-shaped continuity arguments, never a permissive wrapper."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    if not isinstance(value, dict):
        return None
    allowed = {"objective", "current_plan", "next_action", "blockers"}
    if set(value) - allowed or not all(
        isinstance(value.get(field), str) and value[field].strip()
        for field in ("objective", "current_plan", "next_action")
    ):
        return None
    blockers = value.get("blockers", [])
    if not isinstance(blockers, list) or not all(
        isinstance(blocker, str) and blocker.strip() for blocker in blockers
    ):
        return None
    return {
        "objective": value["objective"],
        "current_plan": value["current_plan"],
        "next_action": value["next_action"],
        "blockers": list(blockers),
    }


def _tool_call_authenticates_note(
    messages: List[Dict[str, Any]], index: int, tool_call_id: Any, note: Dict[str, Any]
) -> bool:
    """Bind an authenticated result to its exact direct or deferred model call."""
    if not isinstance(tool_call_id, str) or not tool_call_id:
        return False
    expected_arguments = _continuity_note_arguments({
        "objective": note.get("objective"),
        "current_plan": note.get("current_plan"),
        "next_action": note.get("next_action"),
        "blockers": note.get("blockers"),
    })
    if expected_arguments is None:
        return False
    for prior in reversed(messages[:index]):
        if not isinstance(prior, dict) or prior.get("role") != "assistant":
            continue
        for call in prior.get("tool_calls") or []:
            # ``tool_call_id`` is the provider's persisted call identity. Do
            # not accept nearby calls or alternate IDs from wrapper text.
            if not isinstance(call, dict) or call.get("id") != tool_call_id:
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else call
            if not isinstance(function, dict):
                continue
            if function.get("name") == "continuity_note":
                call_arguments = _continuity_note_arguments(function.get("arguments"))
            elif function.get("name") == "tool_call":
                wrapper = function.get("arguments")
                if isinstance(wrapper, str):
                    try:
                        wrapper = json.loads(wrapper)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue
                if not isinstance(wrapper, dict) or set(wrapper) != {"name", "arguments"}:
                    continue
                if wrapper.get("name") != "continuity_note":
                    continue
                call_arguments = _continuity_note_arguments(wrapper.get("arguments"))
            else:
                continue
            if call_arguments == expected_arguments:
                return True
    return False


def _note_from_authenticated_tool_history(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    from tools.continuity_note_tool import CONTINUITY_NOTE_AUTHENTICATOR

    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        if (message.get("name") or message.get("tool_name")) != "continuity_note":
            continue
        try:
            payload = json.loads(message.get("content", ""))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("authenticated_by") != CONTINUITY_NOTE_AUTHENTICATOR:
            continue
        note = payload.get("note")
        if not isinstance(note, dict) or note.get("version") != NATIVE_INCREMENTAL_NOTE_VERSION:
            continue
        if not _tool_call_authenticates_note(messages, index, message.get("tool_call_id"), note):
            continue
        return deepcopy(note)
    return None


def native_incremental_note_from_history(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Read only authenticated host-persisted notes, never transcript text."""
    try:
        validate_persisted_native_compaction_history(messages)
    except ValueError:
        return None
    # A note carried by a validated checkpoint is safe across session rotation;
    # direct tool output additionally requires exact tool-call pairing above.
    for message in messages:
        if not isinstance(message, dict):
            continue
        items = message.get("codex_reasoning_items")
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            metadata = item.get(NATIVE_COMPACTION_METADATA_KEY)
            if not isinstance(metadata, dict):
                continue
            handoff = metadata.get("handoff")
            if not isinstance(handoff, str) or not handoff.startswith("NATIVE_INCREMENTAL_NOTE\n"):
                continue
            try:
                note = json.loads(handoff.split("\n", 1)[1])
            except (TypeError, ValueError, json.JSONDecodeError):
                return None
            if (
                isinstance(note, dict)
                and note.get("version") == NATIVE_INCREMENTAL_NOTE_VERSION
                and note.get("claim_kind") == "agent_authored"
            ):
                return deepcopy(note)
    return _note_from_authenticated_tool_history(messages)


def restore_native_incremental_note(agent: Any, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Restore the recorded cursor; only a sealed checkpoint permits rebinding."""
    from types import SimpleNamespace

    agent._native_incremental_handoff_note = None
    agent._native_incremental_handoff_projection_cursor = None
    source_messages, _replay_base_count = _projection_source_for_messages(agent, messages)
    if source_messages is None:
        return None
    try:
        validate_persisted_native_compaction_history(source_messages)
    except ValueError:
        return None
    note = native_incremental_note_from_history(source_messages)
    tool_note = _note_from_authenticated_tool_history(source_messages)
    carrier_end = None
    for index, message in enumerate(source_messages):
        for item in message.get("codex_reasoning_items", []) or []:
            metadata = item.get(NATIVE_COMPACTION_METADATA_KEY) if isinstance(item, dict) else None
            if isinstance(metadata, dict) and str(metadata.get("handoff", "")).startswith("NATIVE_INCREMENTAL_NOTE\n"):
                carrier_end = index + 2
    # A newer ordinary note wins over the old checkpoint. The old tool result
    # may itself survive in a protected tail; equality identifies that replay.
    if tool_note is not None and (carrier_end is None or tool_note != note):
        probe = SimpleNamespace(session_id=getattr(agent, "session_id", None),
                                _native_incremental_handoff_note=tool_note)
        valid = _staged_note(probe, source_messages)
        if valid is None:
            return None
        agent._native_incremental_handoff_note = deepcopy(valid)
        return deepcopy(valid)
    if note is None or carrier_end is None:
        return None
    try:
        rebound = create_native_incremental_note(
            session_id=getattr(agent, "session_id", None),
            source_messages=source_messages[:carrier_end],
            objective=note.get("objective"), current_plan=note.get("current_plan"),
            next_action=note.get("next_action"), blockers=note.get("blockers"),
        )
    except (TypeError, ValueError):
        return None
    agent._native_incremental_handoff_note = deepcopy(rebound)
    return deepcopy(rebound)


def record_native_incremental_note_from_tool_call(
    agent: Any, arguments: Dict[str, Any], messages: List[Dict[str, Any]]
) -> str:
    """The sole ordinary-dispatch seam; caller-provided history is never accepted."""
    from tools.continuity_note_tool import CONTINUITY_NOTE_AUTHENTICATOR

    if not bool(getattr(agent, "native_incremental_handoff_enabled", False)):
        return json.dumps({"error": "native incremental handoff is not enabled"})
    source_messages, _replay_base_count = _projection_source_for_messages(agent, messages)
    if source_messages is None:
        return json.dumps({"error": "native incremental replay projection is not authenticated"})
    try:
        note = create_native_incremental_note(
            session_id=getattr(agent, "session_id", None),
            source_messages=source_messages,
            objective=arguments.get("objective") if isinstance(arguments, dict) else None,
            current_plan=arguments.get("current_plan") if isinstance(arguments, dict) else None,
            next_action=arguments.get("next_action") if isinstance(arguments, dict) else None,
            blockers=arguments.get("blockers") if isinstance(arguments, dict) else None,
        )
        record_native_incremental_note(agent, note, source_messages)
    except (TypeError, ValueError) as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)
    return json.dumps(
        {
            "ok": True,
            "authenticated_by": CONTINUITY_NOTE_AUTHENTICATOR,
            "note": note,
            "instruction_precedence": note["instruction_precedence"],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _wire_handoff(note: Dict[str, Any]) -> str:
    # Canonical JSON is deliberate: the persisted note is auditable, bounded,
    # and cannot be confused with free-form transcript injection.
    return "NATIVE_INCREMENTAL_NOTE\n" + json.dumps(
        note, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _item_value(item: Any, key: str) -> Any:
    return item.get(key) if isinstance(item, dict) else getattr(item, key, None)


def _message_suffix(item: Any) -> Optional[Dict[str, Any]]:
    if _item_value(item, "type") != "message":
        return None
    if _item_value(item, "role") not in {None, "assistant"}:
        raise ValueError("native incremental message suffix has unsupported role")
    parts = _item_value(item, "content")
    if not isinstance(parts, list):
        raise ValueError("native incremental message suffix has invalid content")
    retained = []
    text_parts = []
    for part in parts:
        part_type = _item_value(part, "type")
        text = _item_value(part, "text")
        if part_type not in {"output_text", "text"} or not isinstance(text, str):
            raise ValueError("native incremental message suffix contains unsupported content")
        retained.append({"type": "output_text", "text": text})
        text_parts.append(text)
    if not retained:
        return None
    raw = {
        "type": "message",
        "role": "assistant",
        "status": str(_item_value(item, "status") or "completed"),
        "content": retained,
    }
    item_id = _item_value(item, "id")
    if isinstance(item_id, str) and item_id:
        raw["id"] = item_id
    phase = _item_value(item, "phase")
    if isinstance(phase, str) and phase:
        raw["phase"] = phase
    return {"role": "assistant", "content": "".join(text_parts), "codex_message_items": [raw]}


def _latest_checkpoint_and_suffix(response: Any) -> tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    output = getattr(response, "output", None)
    if not isinstance(output, list):
        raise ValueError("native incremental response has no output list")
    seen_ids: Dict[str, Any] = {}
    distinct = []
    for item in output:
        item_id = _item_value(item, "id")
        plain = item if isinstance(item, dict) else (
            item.model_dump() if hasattr(item, "model_dump") else vars(item)
        )
        if isinstance(item_id, str) and item_id:
            if item_id in seen_ids:
                if seen_ids[item_id] != plain:
                    raise ValueError("native incremental conflicting duplicate output id")
                continue
            seen_ids[item_id] = deepcopy(plain)
        distinct.append(item)
    output = distinct
    checkpoint: Optional[Dict[str, Any]] = None
    suffix: List[Dict[str, Any]] = []
    # Validate all checkpoints, but only replay output AFTER the last one.
    # Real native streams can emit reasoning between successive checkpoints;
    # that intermediate output is already absorbed by the later checkpoint.
    last_checkpoint = max(
        (i for i, item in enumerate(output) if _item_value(item, "type") == "compaction"),
        default=-1,
    )
    for index, item in enumerate(output):
        item_type = _item_value(item, "type")
        if item_type == "compaction":
            encrypted = _item_value(item, "encrypted_content")
            if not isinstance(encrypted, str):
                raise ValueError("native incremental checkpoint encrypted content must be text")
            if not encrypted.strip():
                raise ValueError("native incremental checkpoint is blank")
            # Do not strip/normalize ciphertext; it is opaque provider state.
            checkpoint = {"type": "compaction", "encrypted_content": encrypted}
            suffix = []
            continue
        if checkpoint is not None and index > last_checkpoint:
            if item_type == "reasoning":
                encrypted = _item_value(item, "encrypted_content")
                if not isinstance(encrypted, str) or not encrypted.strip():
                    raise ValueError("native incremental reasoning suffix is not replayable")
                if len(suffix) >= NATIVE_INCREMENTAL_SUFFIX_MAX_ITEMS:
                    raise ValueError("native incremental output suffix exceeds bound")
                suffix.append({"role": "assistant", "content": "", "codex_reasoning_items": [{
                    "type": "reasoning", "encrypted_content": encrypted,
                    "summary": deepcopy(_item_value(item, "summary") or []),
                }]})
                continue
            message = _message_suffix(item)
            if message is not None:
                if len(suffix) >= NATIVE_INCREMENTAL_SUFFIX_MAX_ITEMS:
                    raise ValueError("native incremental output suffix exceeds bound")
                suffix.append(message)
            elif item_type != "message":
                # A new provider output type is not safe to silently discard
                # after a checkpoint. Keep the original transcript instead
                # until this consumer knows how to replay that exact item.
                raise ValueError(
                    "native incremental output suffix contains unsupported item"
                )
    return checkpoint, suffix


def _usage(response: Any) -> Dict[str, Optional[int]]:
    usage = getattr(response, "usage", None)
    def value(name: str) -> Optional[int]:
        raw = _item_value(usage, name) if usage is not None else None
        return raw if isinstance(raw, int) and not isinstance(raw, bool) else None
    details = _item_value(usage, "input_tokens_details") if usage is not None else None
    cached = _item_value(details, "cached_tokens") if details is not None else None
    return {
        "input_tokens": value("input_tokens"),
        "output_tokens": value("output_tokens"),
        "cached_tokens": cached if isinstance(cached, int) and not isinstance(cached, bool) else None,
    }


def native_incremental_compact_context(
    agent: Any, messages: List[Dict[str, Any]], system_message: str = ""
) -> List[Dict[str, Any]]:
    """Run exactly one luna inline-compaction request, or return source unchanged.

    A missing/stale note is an explicit not-ready outcome.  It never falls back
    to a full-history active-model handoff or a generic summarizer.
    """
    from agent.codex_responses_adapter import _chat_messages_to_responses_input, classify_responses_route

    route = classify_responses_route(agent)
    if not native_incremental_continuity_capable(
        agent,
        is_codex_backend=route.is_codex_backend,
        is_xai_responses=route.is_xai_responses,
        is_github_responses=route.is_github_responses,
    ):
        _set_native_attempt_state(agent, error="native incremental route configuration is invalid")
        agent._last_native_incremental_compaction = {"disposition": "invalid_route"}
        return messages
    note = _staged_note(agent, messages)
    if note is None:
        # Keep the cause visible; generic structural backoff otherwise reads as
        # ordinary no-progress even though no provider request was attempted.
        logger.info("Native incremental compaction not ready: authenticated continuity note missing or stale")
        _set_native_attempt_state(agent, error="native incremental handoff note not ready")
        agent._last_native_incremental_compaction = {"disposition": "not_ready"}
        return messages
    frozen = deepcopy(messages)
    validate_persisted_native_compaction_history(frozen)
    # Preserve every post-note operational row and, independently, the latest
    # user correction even when the note was written after that correction.
    tail = _protected_tail_since_note(
        frozen,
        note,
        projection_cursor=getattr(agent, "_native_incremental_handoff_projection_cursor", None),
    )
    if len(tail) == len(frozen):
        _set_native_attempt_state(agent)
        return messages
    fingerprint = hashlib.sha256(
        (_note_fence(frozen) + "\n" + _wire_handoff(note)).encode("utf-8")
    ).hexdigest()
    if getattr(agent, "_native_incremental_no_progress_fingerprint", None) == fingerprint:
        _set_native_attempt_state(agent, error="native incremental request unchanged since no progress")
        agent._last_native_incremental_compaction = {"disposition": "unchanged_no_progress"}
        return messages
    threshold = getattr(agent, "native_incremental_compact_threshold", NATIVE_INCREMENTAL_COMPACT_THRESHOLD)
    if not isinstance(threshold, int) or isinstance(threshold, bool) or threshold <= 0:
        raise ValueError("native incremental compact threshold unavailable")
    model = getattr(agent, "native_incremental_handoff_model", None)
    if str(model or "").strip().lower() != NATIVE_INCREMENTAL_MODEL:
        raise ValueError("native incremental model must be gpt-5.6-luna")
    try:
        wire = _chat_messages_to_responses_input(
            frozen, native_compaction_eligible=True, current_issuer_kind="openai_codex"
        )
        request = {
            "model": NATIVE_INCREMENTAL_MODEL,
            # Inline compaction is an ordinary Responses turn, not a separate
            # maintenance endpoint. Special no-answer/ack-only instructions
            # can survive in its opaque checkpoint and hijack later replies.
            # The API field below requests compaction; keep normal instructions.
            "instructions": (
                getattr(agent, "_cached_system_prompt", None)
                or system_message
                or "You are a helpful assistant. Follow the latest user request."
            ),
            "input": wire,
            "tools": [],
            "store": False,
            "reasoning": {"effort": "low"},
            "context_management": [{"type": "compaction", "compact_threshold": threshold}],
        }
        started = time.monotonic()
        response = agent._interruptible_api_call(request)
        elapsed = time.monotonic() - started
        status = getattr(response, "status", "completed")
        if status != "completed":
            raise ValueError(f"native incremental response failed: {status}")
        checkpoint, suffix = _latest_checkpoint_and_suffix(response)
    except BaseException as exc:
        _set_native_attempt_state(agent, error=str(exc), aborted=True)
        raise
    if checkpoint is None:
        agent._native_incremental_no_progress_fingerprint = fingerprint
        _set_native_attempt_state(agent)
        agent._last_native_incremental_compaction = {
            "model": NATIVE_INCREMENTAL_MODEL,
            "elapsed_seconds": elapsed,
            "disposition": "no_checkpoint",
            **_usage(response),
        }
        return messages

    handoff = _wire_handoff(note)
    identity = uuid.uuid4().hex
    carrier = {
        "role": "assistant",
        "content": "",
        "display_kind": "hidden",
        "codex_reasoning_items": [{
            **checkpoint,
            NATIVE_COMPACTION_METADATA_KEY: {
                "version": NATIVE_COMPACTION_VERSION,
                "identity": identity,
                "handoff": handoff,
                "tail_count": 0,
                "tail_fence": "",
            },
        }],
    }
    handoff_row = {"role": NATIVE_COMPACTION_HANDOFF_ROLE, "content": handoff, "_native_compaction_handoff": identity}
    result = [carrier, handoff_row, *suffix, *tail]
    metadata = carrier["codex_reasoning_items"][0][NATIVE_COMPACTION_METADATA_KEY]
    metadata["tail_count"] = len(result) - 2
    metadata["tail_fence"] = native_continuity_boundary_fence(result[2:])
    if _native_message_size(result) > _native_message_size(frozen):
        _set_native_attempt_state(agent, refused_would_grow=True)
        return messages
    before_size = _native_message_size(frozen)
    after_size = _native_message_size(result)
    _set_native_attempt_state(
        agent,
        dropped_count=len(frozen) - len(tail),
        made_progress=True,
        savings_pct=round((before_size - after_size) * 100.0 / before_size, 2) if before_size else 0.0,
    )
    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        compressor.compression_count = getattr(compressor, "compression_count", 0) + 1
    agent._native_incremental_no_progress_fingerprint = None
    agent._native_compaction_attempt = NativeCompactionAttempt(
        identity=identity,
        carrier=carrier,
        handoff_row=handoff_row,
        metadata=metadata,
        allow_turn_rebind=bool(getattr(agent, "_native_compaction_turn_active", False)),
    )
    usage = _usage(response)
    agent._native_incremental_compaction_usage = {
        "usage_kind": "native_incremental_compaction",
        "model": NATIVE_INCREMENTAL_MODEL,
        "elapsed_seconds": elapsed,
        **usage,
    }
    agent._last_native_incremental_compaction = {
        "model": NATIVE_INCREMENTAL_MODEL,
        "elapsed_seconds": elapsed,
        "disposition": "checkpoint",
        "suffix_count": len(suffix),
        "protected_tail_count": len(tail),
        **usage,
    }
    # The persisted checkpoint carries the original note; this live agent can
    # bind its next note check to the compacted prefix immediately. Fresh agents
    # perform the same rebind from the validated carrier during turn setup.
    agent._native_incremental_handoff_note = create_native_incremental_note(
        session_id=getattr(agent, "session_id", None),
        source_messages=result,
        objective=note["objective"],
        current_plan=note["current_plan"],
        next_action=note["next_action"],
        blockers=note["blockers"],
    )
    # A compacted transcript is the newly persisted canonical history.  It is
    # safe to replace the transient gateway source/projection pair only after
    # the validated checkpoint candidate has been constructed.
    bind_native_incremental_replay_projection(
        agent, source_messages=result, replay_messages=result
    )
    logger.info(
        "Native incremental compaction checkpoint: model=%s elapsed=%.3fs input=%s output=%s cached=%s suffix=%d",
        NATIVE_INCREMENTAL_MODEL, elapsed,
        agent._last_native_incremental_compaction["input_tokens"],
        agent._last_native_incremental_compaction["output_tokens"],
        agent._last_native_incremental_compaction["cached_tokens"], len(suffix),
    )
    return result
