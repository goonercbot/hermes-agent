"""Pure, deterministic evidence records for tool outcomes.

This module deliberately does not import ``hermes_state`` or call a model.  It
turns a small, closed set of tool result shapes into typed evidence and provides
a conservative reducer for consumers (including SessionDB) to use.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
import json
import re
import secrets
import shlex
from typing import Any

ISSUER = "hermes.evidence"
ADAPTER_VERSION = "1"
REDACTED = "[REDACTED]"
MAX_METADATA_DEPTH = 8
MAX_METADATA_ITEMS = 64
MAX_TEXT = 4096
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
_SENSITIVE_KEY_RE = re.compile(
    r"(?:^|[_\-.])(api[_\-.]?key|token|secret|password|passwd|pass|"
    r"credential|authorization|auth|private[_\-.]?key)(?:$|[_\-.])",
    re.IGNORECASE,
)

# An issuer string alone is not enough: the state must have been produced by
# one of these closed adapters.  Generic observations cannot enter this set.
_TRUSTED_ADAPTER_STATES = {
    "pytest": frozenset({"tested_pass", "tested_fail"}),
    "unittest": frozenset({"tested_pass", "tested_fail"}),
    "nox": frozenset({"tested_pass", "tested_fail"}),
    "git-rev-parse-head": frozenset({"commit_observed"}),
    "gh-pr-view": frozenset({"pr_open", "accepted", "acceptance_revoked", "pr_merged"}),
    "file-mutator": frozenset({"implemented"}),
}
_CURRENT_STATES = frozenset({"accepted", "pr_open", "pr_merged"})
_TRANSIENT_STATUS_STATES = frozenset({"timed_out", "interrupted", "cancelled", "retrying"})
_TERMINAL_TOOL_NAMES = frozenset({"terminal", "terminal_exec", "shell", "subprocess"})
_RUNTIME_CONTEXT_MARKER = secrets.token_hex(32)


def canonical_json(value: Any) -> str:
    """Return a stable JSON representation suitable for hashing and storage."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_event_json(event: Mapping[str, Any]) -> str:
    """Canonical representation excluding the self-referential event hash."""
    return canonical_json({key: value for key, value in event.items() if key != "event_hash"})


def event_hash(event: Mapping[str, Any]) -> str:
    """Hash an event without its ``event_hash`` field."""
    return sha256(canonical_event_json(event).encode("utf-8")).hexdigest()


def source_hash(value: Any) -> str:
    """Return the SHA-256 of canonical, already-redacted source material."""
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def is_valid_source_hash(value: Any) -> bool:
    """Accept only lower-case SHA-256 digests, avoiding ambiguous encodings."""
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _secret_variants(secret: str) -> tuple[str, ...]:
    # A secret can reach a text field either raw or JSON-escaped.  Replacing
    # both serializations before canonical serialization prevents quote and
    # backslash escapes from defeating configured-secret redaction.
    encoded = json.dumps(secret, ensure_ascii=False)
    return tuple(dict.fromkeys((secret, encoded, encoded[1:-1])))


def _redact_text(text: str, known_secrets: Sequence[str]) -> str:
    result = text
    for secret in known_secrets:
        if not isinstance(secret, str) or not secret:
            continue
        for variant in sorted(_secret_variants(secret), key=len, reverse=True):
            if variant:
                result = result.replace(variant, REDACTED)
    return result[:MAX_TEXT]


def _is_sensitive_key(key: Any) -> bool:
    text = str(key)
    if _SENSITIVE_KEY_RE.search(text):
        return True
    # JSON and API payloads commonly use camelCase (apiKey, accessToken).
    # Treat only complete suffixes as sensitive so ordinary words such as
    # ``tokenizer`` and ``secretary`` are not needlessly erased.
    normalized = re.sub(r"[^a-z0-9]", "", text.lower())
    return normalized.endswith((
        "apikey", "token", "secret", "password", "passwd", "pass",
        "credential", "authorization", "auth", "privatekey",
    ))


def redact_value(value: Any, *, known_secrets: Sequence[str] = (), _depth: int = 0) -> Any:
    """Recursively redact sensitive-key values and configured secret values.

    Unsupported values are represented by a constant rather than ``repr`` so
    an object's representation cannot leak credentials or make hashes unstable.
    """
    if _depth > MAX_METADATA_DEPTH:
        return "[TRUNCATED]"
    if isinstance(value, str):
        return _redact_text(value, known_secrets)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Mapping):
        clean: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_METADATA_ITEMS:
                clean["[TRUNCATED]"] = "[TRUNCATED]"
                break
            safe_key = _redact_text(str(key), known_secrets)
            clean[safe_key] = REDACTED if _is_sensitive_key(key) else redact_value(
                item, known_secrets=known_secrets, _depth=_depth + 1
            )
        return clean
    if isinstance(value, (list, tuple)):
        return [redact_value(item, known_secrets=known_secrets, _depth=_depth + 1) for item in value[:MAX_METADATA_ITEMS]]
    return "[UNSERIALIZABLE]"


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, Mapping) else {}
    return {}


def _result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, Mapping):
        for key in ("output", "stdout", "content", "result"):
            value = result.get(key)
            if isinstance(value, str):
                return value
    return ""


def _exit_code(result: Any) -> int | None:
    value = _as_mapping(result).get("exit_code")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _structured_status(result: Any) -> str | None:
    value = _as_mapping(result).get("status")
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return {
        "error": "execution_failed",
        "failed": "execution_failed",
        "failure": "execution_failed",
        "timeout": "timed_out", "timed_out": "timed_out",
        "interrupted": "interrupted", "interrupt": "interrupted",
        "cancelled": "cancelled", "canceled": "cancelled",
        "retry": "retrying", "retrying": "retrying",
    }.get(normalized)


def _generic_observation(result: Any) -> tuple[str, str]:
    """Classify only explicit machine-readable success/failure signals."""
    data = _as_mapping(result)
    code = _exit_code(result)
    if code is not None:
        return (
            ("observed_success", "success")
            if code == 0
            else ("observed_failure", "failure")
        )
    status = data.get("status")
    if isinstance(status, str):
        normalized = status.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {"error", "failed", "failure"}:
            return "observed_failure", "failure"
        if normalized in {"ok", "passed", "success", "succeeded", "completed"}:
            return "observed_success", "success"
    error = data.get("error")
    if data.get("success") is False or (
        error is not None and error != "" and error is not False
    ):
        return "observed_failure", "failure"
    if data.get("success") is True:
        return "observed_success", "success"
    return "observed", "observed"


def _command_from(tool_input: Any) -> str | None:
    if isinstance(tool_input, str):
        return tool_input
    if isinstance(tool_input, Mapping):
        for key in ("command", "cmd"):
            value = tool_input.get(key)
            if isinstance(value, str):
                return value
        args = tool_input.get("args")
        if isinstance(args, Mapping):
            return _command_from(args)
    return None


def _direct_tokens(command: str | None) -> list[str] | None:
    if not isinstance(command, str) or not command.strip():
        return None
    # Any shell grammar is rejected.  A trusted command must be the direct
    # executable invocation, not an arbitrary script that happens to mention it.
    if re.search(r"(?:^|\s)(?:&&|\|\||[;|&])(?:\s|$)|[`$<>\n\r]", command):
        return None
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return None


def _test_adapter(tokens: list[str] | None) -> str | None:
    if not tokens:
        return None
    if tokens[0] == "pytest":
        return "pytest"
    if len(tokens) >= 3 and tokens[0] in {"python", "python3"} and tokens[1:] and tokens[1] == "-m" and tokens[2] == "pytest":
        return "pytest"
    if tokens[0] == "unittest":
        return "unittest"
    if len(tokens) >= 3 and tokens[0] in {"python", "python3"} and tokens[1] == "-m" and tokens[2] == "unittest":
        return "unittest"
    if tokens[0] == "nox":
        return "nox"
    return None


def _test_target_identity(
    adapter: str,
    command: str | None,
    parent_lineage: Any,
    known_secrets: Sequence[str],
) -> tuple[str, str]:
    target = {
        "adapter": adapter,
        "command_sha256": sha256((command or "").encode("utf-8")).hexdigest(),
        "parent_lineage": redact_value(parent_lineage, known_secrets=known_secrets),
    }
    return (
        f"test:{adapter}",
        f"test-target:{sha256(canonical_json(target).encode('utf-8')).hexdigest()}",
    )


def _file_mutation_landed(tool_name: str, result: Any) -> bool:
    data = _as_mapping(result)
    if tool_name not in {"write_file", "patch"} or not data or data.get("error"):
        return False
    if data.get("landed") is True:
        return True
    if tool_name == "write_file":
        return isinstance(data.get("bytes_written"), int) and not isinstance(data.get("bytes_written"), bool)
    return data.get("success") is True


def _json_output(result: Any) -> Mapping[str, Any]:
    text = _result_text(result).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _gh_pr_adapter(tokens: list[str] | None, result: Any) -> tuple[str, dict[str, Any], str] | None:
    if not tokens or len(tokens) < 6 or tokens[:3] != ["gh", "pr", "view"]:
        return None
    # Require a literal PR number in the command and a --json field list that
    # asks GitHub for every identity component used below.
    if not tokens[3].isdigit():
        return None
    try:
        json_index = tokens.index("--json")
        fields = set(tokens[json_index + 1].split(","))
    except (ValueError, IndexError):
        return None
    if not {"number", "state", "headRefOid"}.issubset(fields):
        return None
    data = _json_output(result)
    number, state, oid = data.get("number"), data.get("state"), data.get("headRefOid")
    if number != int(tokens[3]) or state not in {"OPEN", "MERGED"} or not isinstance(oid, str) or not _SHA1_RE.fullmatch(oid):
        return None
    review_decision = data.get("reviewDecision")
    if state == "MERGED":
        state_name = "pr_merged"
    elif "reviewDecision" in fields and review_decision == "APPROVED":
        state_name = "accepted"
    elif "reviewDecision" in fields and review_decision == "CHANGES_REQUESTED":
        state_name = "acceptance_revoked"
    else:
        state_name = "pr_open"
    lineage = {"pr_number": number, "head_ref_oid": oid.lower()}
    return state_name, lineage, oid.lower()


def _bounded_metadata(
    tool_name: str,
    tool_input: Any,
    result: Any,
    known_secrets: Sequence[str],
    extra: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Store compact descriptors only; raw arguments/results stay in messages."""
    safe_input = redact_value(tool_input, known_secrets=known_secrets)
    safe_result = redact_value(result, known_secrets=known_secrets)
    input_json = canonical_json(safe_input)
    result_json = canonical_json(safe_result)
    metadata: dict[str, Any] = {
        "tool_name": _redact_text(tool_name, known_secrets),
        "input_sha256": sha256(input_json.encode("utf-8")).hexdigest(),
        "result_sha256": sha256(result_json.encode("utf-8")).hexdigest(),
        "result_bytes": len(result_json.encode("utf-8")),
    }
    code = _exit_code(result)
    if code is not None:
        metadata["exit_code"] = code
    status = _structured_status(result)
    if status is not None:
        metadata["structured_status"] = status
    if extra:
        metadata["extra"] = redact_value(extra, known_secrets=known_secrets)
    return metadata


def runtime_context(
    tool_name: str,
    tool_call_id: str | None,
    *,
    cwd: str | None = None,
) -> dict[str, Any]:
    """Return non-secret candidate identity attached before transcript flush.

    The tool-call id is the immutable identity of this exact execution. The
    working directory is represented only by a digest; exact arguments remain
    recoverable from the assistant tool-call source row.
    """
    call_id = str(tool_call_id or "").strip()
    context: dict[str, Any] = {
        "_marker": _RUNTIME_CONTEXT_MARKER,
        "subject": f"tool:{tool_name}",
        "candidate": f"tool-call:{call_id}" if call_id else None,
        "parent_lineage": None,
    }
    if cwd:
        context["parent_lineage"] = {
            "cwd_sha256": sha256(str(cwd).encode("utf-8")).hexdigest()
        }
    return context


def validate_runtime_context(
    context: Mapping[str, Any],
    *,
    tool_name: str,
    tool_call_id: str,
) -> dict[str, Any]:
    """Authenticate one process-local persistence context and bind its identity.

    The marker is a process-local random secret, so transcript input, replay,
    or copied JSON from another process cannot manufacture a trusted execution
    status or lineage. Persistence strips the private context after use.
    Subject and candidate are recomputed from authoritative persisted fields
    rather than accepted from the private mapping.
    """
    if context.get("_marker") != _RUNTIME_CONTEXT_MARKER:
        return {}
    call_id = str(tool_call_id or "").strip()
    parent = context.get("parent_lineage")
    valid_parent = None
    if isinstance(parent, Mapping):
        cwd_sha256 = parent.get("cwd_sha256")
        if isinstance(cwd_sha256, str) and _SHA256_RE.fullmatch(cwd_sha256):
            valid_parent = {"cwd_sha256": cwd_sha256}
    status = context.get("status")
    if status not in {*_TRANSIENT_STATUS_STATES, "error"}:
        status = None
    return {
        "subject": f"tool:{tool_name}",
        "candidate": f"tool-call:{call_id}" if call_id else None,
        "parent_lineage": valid_parent,
        "status": status,
    }


def adapt_tool_result(
    tool_name: str,
    tool_input: Any = None,
    result: Any = None,
    *,
    subject: str | None = None,
    candidate: Any = None,
    parent_lineage: Any = None,
    previous_event_hash: str | None = None,
    known_secrets: Sequence[str] = (),
    issuer: str = ISSUER,
    adapter_version: str = ADAPTER_VERSION,
    metadata: Mapping[str, Any] | None = None,
    declared_source_hash: str | None = None,
    trusted_execution: bool = False,
) -> dict[str, Any]:
    """Create one redacted event draft from a tool result.

    Only closed, exact command adapters from a process-authenticated execution
    emit trusted lifecycle evidence. Every other input remains a generic
    observation. Structured timeout-like statuses take precedence over
    success-looking payloads because their outcome is unknown.
    """
    tool_name = tool_name if isinstance(tool_name, str) else "unknown"
    command = _command_from(tool_input)
    tokens = _direct_tokens(command)
    status_state = _structured_status(result)
    state, outcome = _generic_observation(result)
    adapter, trusted = "generic", False
    derived_subject, derived_candidate, derived_parent = subject, candidate, parent_lineage
    lifecycle_dimension = None
    test_adapter = _test_adapter(tokens)

    if status_state:
        state, adapter = status_state, "structured-status"
        outcome = "failure" if status_state == "execution_failed" else "incomplete"
        if trusted_execution and tool_name in _TERMINAL_TOOL_NAMES and test_adapter:
            derived_subject, derived_candidate = _test_target_identity(
                test_adapter, command, derived_parent, known_secrets
            )
            lifecycle_dimension = "test"
    else:
        code = _exit_code(result)
        if (
            trusted_execution
            and tool_name in _TERMINAL_TOOL_NAMES
            and test_adapter
            and code is not None
        ):
            state = "tested_pass" if code == 0 else "tested_fail"
            adapter, trusted, outcome = test_adapter, True, "success" if code == 0 else "failure"
            derived_subject, derived_candidate = _test_target_identity(
                test_adapter, command, derived_parent, known_secrets
            )
            lifecycle_dimension = "test"
        elif (
            trusted_execution
            and tool_name in _TERMINAL_TOOL_NAMES
            and tokens == ["git", "rev-parse", "HEAD"]
            and _exit_code(result) == 0
        ):
            sha = _result_text(result).strip()
            if _SHA1_RE.fullmatch(sha):
                state, adapter, trusted, outcome = "commit_observed", "git-rev-parse-head", True, "success"
                derived_candidate = sha.lower()
                derived_subject = derived_subject or "git:HEAD"
        else:
            pr = (
                _gh_pr_adapter(tokens, result)
                if trusted_execution
                and tool_name in _TERMINAL_TOOL_NAMES
                and _exit_code(result) == 0
                else None
            )
            if pr is not None:
                state, derived_parent, derived_candidate = pr
                adapter, trusted = "gh-pr-view", True
                outcome = "failure" if state == "acceptance_revoked" else "success"
                derived_subject = derived_subject or f"pr:{derived_parent['pr_number']}"
            elif trusted_execution and _file_mutation_landed(tool_name, result):
                state, adapter, trusted, outcome = "implemented", "file-mutator", True, "success"

    safe_metadata = _bounded_metadata(tool_name, tool_input, result, known_secrets, metadata)
    source_material = {
        "tool_name": tool_name,
        "tool_input": redact_value(tool_input, known_secrets=known_secrets),
        "result": redact_value(result, known_secrets=known_secrets),
    }
    computed_source_hash = source_hash(source_material)
    event = {
        "issuer": issuer if isinstance(issuer, str) else ISSUER,
        "adapter_version": adapter_version if isinstance(adapter_version, str) else ADAPTER_VERSION,
        "adapter": adapter,
        "trusted": trusted,
        "state": state,
        "outcome": outcome,
        "lifecycle_dimension": lifecycle_dimension,
        "subject": redact_value(derived_subject, known_secrets=known_secrets),
        "candidate": redact_value(derived_candidate, known_secrets=known_secrets),
        "parent_lineage": redact_value(derived_parent, known_secrets=known_secrets),
        "metadata": safe_metadata,
        "source_hash": declared_source_hash if is_valid_source_hash(declared_source_hash) else computed_source_hash,
        "previous_event_hash": previous_event_hash if previous_event_hash is None or is_valid_source_hash(previous_event_hash) else None,
    }
    event["event_hash"] = event_hash(event)
    return event


# Intentional aliases make the pure adapter discoverable to callers that use
# either event-draft or tool-result vocabulary.
draft_event = adapt_tool_result
build_event = adapt_tool_result


def verify_hash_chain(events: Sequence[Mapping[str, Any]]) -> bool:
    """Verify event hashes and predecessor links in list order."""
    previous: str | None = None
    for event in events:
        if not isinstance(event, Mapping) or not is_valid_source_hash(event.get("event_hash")):
            return False
        if event.get("event_hash") != event_hash(event):
            return False
        if event.get("previous_event_hash") != previous:
            return False
        previous = event["event_hash"]
    return True


def _lifecycle_dimension(event: Mapping[str, Any]) -> str:
    explicit = event.get("lifecycle_dimension")
    if explicit in {
        "observation",
        "execution",
        "implementation",
        "test",
        "commit",
        "review",
        "pr",
    }:
        return str(explicit)
    state = event.get("state")
    if state in {"tested_pass", "tested_fail"}:
        return "test"
    if state == "commit_observed":
        return "commit"
    if state in {"pr_open", "pr_merged"}:
        return "pr"
    if state in {"accepted", "acceptance_revoked"}:
        return "review"
    if state == "implemented":
        return "implementation"
    if state in {*_TRANSIENT_STATUS_STATES, "execution_failed"}:
        return "execution"
    return "observation"


def _trusted_event(event: Mapping[str, Any]) -> bool:
    adapter = event.get("adapter")
    allowed_states = _TRUSTED_ADAPTER_STATES.get(adapter, frozenset()) if isinstance(adapter, str) else frozenset()
    return (
        event.get("issuer") == ISSUER
        and event.get("trusted") is True
        and event.get("state") in allowed_states
        and event.get("outcome") in {"success", "failure"}
    )


def _trusted_runtime_event(event: Mapping[str, Any]) -> bool:
    """Accept only a separately authenticated runtime-verifier record."""
    return (
        event.get("issuer") == ISSUER
        and event.get("trusted") is True
        and event.get("adapter") == "runtime-verifier"
        and event.get("state") == "runtime_verified"
        and event.get("outcome") == "success"
        and is_valid_source_hash(event.get("source_hash"))
    )


def _call_integrity(callback: Callable[..., bool], events: Sequence[Mapping[str, Any]]) -> bool:
    try:
        return bool(callback(events))
    except TypeError:
        # Compatibility for a common SessionDB-shaped verifier that accepts a
        # candidate event list by keyword only.
        try:
            return bool(callback(events=events))
        except (TypeError, ValueError):
            return False
    except (ValueError, KeyError):
        return False


def resolve_claim(
    events: Sequence[Mapping[str, Any]],
    *,
    state: str,
    subject: Any,
    candidate: Any,
    parent_lineage: Any,
    source_hash_value: str,
    integrity_chain: Callable[..., bool] | None,
) -> dict[str, Any]:
    """Fail closed when deciding whether a lifecycle claim is currently true.

    Resolution requires a caller-supplied integrity-chain callback *and* an
    exact source, candidate, and parent lineage match.  A later event in the
    same lifecycle dimension supersedes earlier evidence.  PR state is live,
    so it additionally needs a later runtime verification event.
    """
    if not callable(integrity_chain):
        return {"resolved": False, "reason": "missing_integrity_callback", "event": None}
    if not is_valid_source_hash(source_hash_value):
        return {"resolved": False, "reason": "invalid_source_hash", "event": None}
    if not _call_integrity(integrity_chain, events):
        return {"resolved": False, "reason": "integrity_chain_rejected", "event": None}
    matches = [event for event in events if isinstance(event, Mapping) and event.get("subject") == subject and event.get("candidate") == candidate and event.get("parent_lineage") == parent_lineage]
    if not matches:
        return {"resolved": False, "reason": "no_exact_lineage", "event": None}
    requested_dimension = _lifecycle_dimension({"state": state})
    dimension_events = [event for event in matches if _lifecycle_dimension(event) == requested_dimension]
    if not dimension_events:
        return {"resolved": False, "reason": "no_lifecycle_evidence", "event": None}
    latest = dimension_events[-1]
    if latest.get("state") != state:
        return {"resolved": False, "reason": "superseded", "event": latest}
    if latest.get("source_hash") != source_hash_value:
        return {"resolved": False, "reason": "source_hash_mismatch", "event": latest}
    if not _trusted_event(latest):
        return {"resolved": False, "reason": "untrusted_evidence", "event": latest}
    if state.endswith("_fail"):
        return {"resolved": False, "reason": "unsuccessful_outcome", "event": latest}
    if state in _CURRENT_STATES:
        runtime = [event for event in matches if _trusted_runtime_event(event)]
        if not runtime or events.index(runtime[-1]) <= events.index(latest):
            return {"resolved": False, "reason": "fresh_runtime_verification_required", "event": latest}
    return {"resolved": True, "reason": "resolved", "event": latest}


def claim_is_resolved(*args: Any, **kwargs: Any) -> bool:
    """Boolean convenience wrapper around :func:`resolve_claim`."""
    return bool(resolve_claim(*args, **kwargs)["resolved"])
