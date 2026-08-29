"""SessionDB persistence/read integration for prospective evidence events."""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List, Mapping, Optional, Tuple

from agent.evidence_ledger import (
    adapt_tool_result,
    canonical_json,
    event_hash,
    resolve_claim,
    source_hash,
    validate_runtime_context,
)

_MESSAGE_FIELDS = (
    "id",
    "session_id",
    "role",
    "content",
    "tool_call_id",
    "tool_calls",
    "tool_name",
    "effect_disposition",
    "timestamp",
)
_MAX_CALL_LOOKBACK = 100
_MAX_METADATA_BYTES = 16_384



def _message_payload(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {field: row[field] for field in _MESSAGE_FIELDS}


def _identity_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value[:1024]
    return canonical_json(value)[:1024]


def _decode_identity(value: Optional[str]) -> Any:
    if value is None:
        return None
    if value[:1] in {"{", "[", '"'}:
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            pass
    return value


def _dimension(state: str) -> str:
    if state in {"tested_pass", "tested_fail"}:
        return "test"
    if state == "commit_observed":
        return "commit"
    if state in {"pr_open", "pr_merged"}:
        return "pull_request"
    if state in {"accepted", "acceptance_revoked"}:
        return "review"
    if state == "implemented":
        return "implementation"
    if state in {"timed_out", "interrupted", "cancelled", "retrying", "execution_failed"}:
        return "execution"
    return "observation"


def _tool_call_parts(call: Mapping[str, Any]) -> Tuple[str, str, Any]:
    function = call.get("function")
    if isinstance(function, Mapping):
        name = str(function.get("name") or "")
        arguments = function.get("arguments")
    else:
        name = str(call.get("name") or "")
        arguments = call.get("arguments")
    call_id = str(call.get("id") or call.get("call_id") or "")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (TypeError, ValueError):
            pass
    return call_id, name, arguments


class SessionEvidenceMixin:
    """Host contract: ``_conn``, ``_lock``, ``_read_ctx`` and ``evidence_enabled``."""

    @staticmethod
    def _source_row(conn: sqlite3.Connection, message_id: int) -> sqlite3.Row:
        row = conn.execute(
            "SELECT id, session_id, role, content, tool_call_id, tool_calls, "
            "tool_name, effect_disposition, timestamp FROM messages WHERE id = ?",
            (int(message_id),),
        ).fetchone()
        if row is None:
            raise ValueError(f"evidence source message {message_id} is missing")
        return row

    @classmethod
    def _find_tool_call_source(
        cls,
        conn: sqlite3.Connection,
        *,
        session_id: str,
        result_message_id: int,
        tool_call_id: str,
    ) -> Tuple[Optional[sqlite3.Row], Any]:
        if not tool_call_id:
            return None, None
        rows = conn.execute(
            "SELECT id, session_id, role, content, tool_call_id, tool_calls, "
            "tool_name, effect_disposition, timestamp FROM messages "
            "WHERE session_id = ? AND role = 'assistant' AND tool_calls IS NOT NULL "
            "AND id < ? ORDER BY id DESC LIMIT ?",
            (session_id, int(result_message_id), _MAX_CALL_LOOKBACK),
        ).fetchall()
        for row in rows:
            try:
                calls = json.loads(row["tool_calls"] or "[]")
            except (TypeError, ValueError):
                continue
            if not isinstance(calls, list):
                continue
            for call in calls:
                if not isinstance(call, Mapping):
                    continue
                call_id, _name, arguments = _tool_call_parts(call)
                if call_id == tool_call_id:
                    return row, arguments
        return None, None

    def _insert_evidence_events(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> int:
        """Insert tool evidence inside the caller's transcript transaction."""
        if not getattr(self, "evidence_enabled", False):
            return 0
        inserted = 0
        for message in messages:
            if message.get("role") != "tool" or message.get("_row_id") is None:
                continue
            source_row = self._source_row(conn, int(message["_row_id"]))
            result_digest = source_hash(_message_payload(source_row))
            tool_call_id = str(message.get("tool_call_id") or "")
            tool_name = str(
                message.get("tool_name")
                or message.get("name")
                or source_row["tool_name"]
                or "unknown"
            )
            call_row, tool_input = self._find_tool_call_source(
                conn,
                session_id=session_id,
                result_message_id=int(message["_row_id"]),
                tool_call_id=tool_call_id,
            )
            call_id = int(call_row["id"]) if call_row is not None else None
            call_digest = (
                source_hash(_message_payload(call_row)) if call_row is not None else None
            )
            context = message.get("_evidence_context")
            if not isinstance(context, Mapping):
                context = {}
            context = validate_runtime_context(
                context,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
            )
            trusted_execution = bool(context)
            previous = conn.execute(
                "SELECT event_hash FROM evidence_events WHERE session_id = ? "
                "ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            previous_hash = previous[0] if previous else None
            adapter_result: Any = source_row["content"]
            if context.get("status") in {
                "timed_out", "interrupted", "cancelled", "retrying", "error"
            }:
                adapter_result = {
                    "status": context["status"],
                    "source_sha256": result_digest,
                }
            effect_disposition = message.get("effect_disposition")
            if effect_disposition not in {"none", "unknown"}:
                effect_disposition = None
            evidence_metadata: Dict[str, Any] = {
                "source_call_bound": call_row is not None,
            }
            if effect_disposition is not None:
                evidence_metadata.update({
                    "effect_disposition": effect_disposition,
                    "side_effect_attempted": effect_disposition == "unknown",
                    "side_effect_confirmed": False,
                })
            draft = adapt_tool_result(
                tool_name,
                tool_input,
                adapter_result,
                subject=context.get("subject"),
                candidate=context.get("candidate"),
                parent_lineage=context.get("parent_lineage"),
                previous_event_hash=previous_hash,
                declared_source_hash=result_digest,
                metadata=evidence_metadata,
                trusted_execution=trusted_execution,
            )
            state = str(draft["state"])
            draft_dimension = draft.get("lifecycle_dimension")
            dimension = (
                str(draft_dimension)
                if draft_dimension in {
                    "observation",
                    "execution",
                    "implementation",
                    "test",
                    "commit",
                    "review",
                    "pr",
                }
                else _dimension(state)
            )
            subject_id = _identity_text(draft.get("subject")) or f"tool:{tool_name}"
            candidate_id = _identity_text(draft.get("candidate"))
            parent_candidate_id = _identity_text(draft.get("parent_lineage"))
            if candidate_id is None and tool_call_id:
                candidate_id = f"tool-call:{tool_call_id}"
            superseded = None
            if candidate_id is not None:
                superseded = conn.execute(
                    "SELECT id FROM evidence_events WHERE session_id = ? "
                    "AND candidate_id = ? AND lifecycle_dimension = ? "
                    "ORDER BY id DESC LIMIT 1",
                    (session_id, candidate_id, dimension),
                ).fetchone()
            metadata_value = draft.get("metadata") or {}
            if draft.get("adapter") == "file-mutator" and draft.get("outcome") == "success":
                extra = dict(metadata_value.get("extra") or {})
                extra.update({
                    "side_effect_attempted": True,
                    "side_effect_confirmed": True,
                })
                metadata_value = {**metadata_value, "extra": extra}
            metadata_json = canonical_json(metadata_value)
            if len(metadata_json.encode("utf-8")) > _MAX_METADATA_BYTES:
                raise ValueError("evidence metadata exceeds bounded storage limit")
            record: Dict[str, Any] = {
                "session_id": session_id,
                "source_message_id": int(message["_row_id"]),
                "source_sha256": result_digest,
                "source_call_message_id": call_id,
                "source_call_sha256": call_digest,
                "tool_call_id": tool_call_id or None,
                "tool_name": tool_name,
                "event_type": (
                    "side_effect_uncertain"
                    if effect_disposition == "unknown"
                    else ("lifecycle" if draft.get("trusted") else "tool_result")
                ),
                "lifecycle_dimension": dimension,
                "lifecycle_state": state,
                "outcome": str(draft["outcome"]),
                "issuer": str(draft["issuer"]),
                "adapter": str(draft["adapter"]),
                "adapter_version": str(draft["adapter_version"]),
                "subject_type": subject_id.split(":", 1)[0],
                "subject_id": subject_id,
                "candidate_id": candidate_id,
                "parent_candidate_id": parent_candidate_id,
                "artifact_id": candidate_id if state in {
                    "tested_pass", "tested_fail", "commit_observed", "accepted", "acceptance_revoked", "pr_open", "pr_merged"
                } else None,
                "metadata_json": metadata_json,
                "occurred_at": float(source_row["timestamp"]),
                "supersedes_event_id": int(superseded[0]) if superseded else None,
                "previous_event_hash": previous_hash,
                "trusted": 1 if draft.get("trusted") else 0,
                "freshness_scope": "snapshot" if state in {"accepted", "acceptance_revoked", "pr_open", "pr_merged"} else "historical",
            }
            record["event_hash"] = event_hash(record)
            columns = tuple(record)
            conn.execute(
                f"INSERT INTO evidence_events ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' for _ in columns)})",
                tuple(record[column] for column in columns),
            )
            inserted += 1
        return inserted

    def get_evidence_events(self, session_id: str) -> List[Dict[str, Any]]:
        with self._read_ctx() as conn:
            rows = conn.execute(
                "SELECT * FROM evidence_events WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _verify_event_rows(conn: sqlite3.Connection, rows: List[Mapping[str, Any]]) -> bool:
        previous = None
        for row in rows:
            if row.get("previous_event_hash") != previous:
                return False
            result_source = SessionEvidenceMixin._source_row(
                conn, int(row["source_message_id"])
            )
            if source_hash(_message_payload(result_source)) != row.get("source_sha256"):
                return False
            call_id = row.get("source_call_message_id")
            if call_id is not None:
                call_source = SessionEvidenceMixin._source_row(conn, int(call_id))
                if source_hash(_message_payload(call_source)) != row.get("source_call_sha256"):
                    return False
            payload = dict(row)
            payload.pop("id", None)
            stored_hash = payload.pop("event_hash", None)
            if event_hash(payload) != stored_hash:
                return False
            try:
                metadata = json.loads(str(row.get("metadata_json") or "{}"))
            except (TypeError, ValueError):
                return False
            if not isinstance(metadata, dict):
                return False
            previous = stored_hash
        return True

    def verify_evidence_integrity(self, session_id: str) -> bool:
        with self._read_ctx() as conn:
            rows = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM evidence_events WHERE session_id = ? ORDER BY id",
                    (session_id,),
                ).fetchall()
            ]
            return self._verify_event_rows(conn, rows)

    def get_evidence_source(self, event_id: int) -> Dict[str, Any]:
        with self._read_ctx() as conn:
            event = conn.execute(
                "SELECT * FROM evidence_events WHERE id = ?", (int(event_id),)
            ).fetchone()
            if event is None:
                raise KeyError(event_id)
            session_rows = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM evidence_events WHERE session_id = ? ORDER BY id",
                    (event["session_id"],),
                ).fetchall()
            ]
            if not self._verify_event_rows(conn, session_rows):
                raise ValueError("evidence integrity check failed")
            source = self._source_row(conn, int(event["source_message_id"]))
            return dict(source)

    def get_compact_evidence_receipts(
        self,
        session_id: str,
        tool_call_ids: Optional[List[str]] = None,
    ) -> Dict[str, List[str]]:
        """Return verified, bounded receipts suitable for folded context.

        The authoritative result stays in ``messages``.  Receipts contain only
        typed state, lineage, exact source/event hashes, and side-effect
        disposition; integrity failure aborts the caller instead of emitting a
        plausible-looking partial ledger.
        """
        requested = set(tool_call_ids) if tool_call_ids is not None else None
        with self._read_ctx() as conn:
            rows = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM evidence_events WHERE session_id = ? ORDER BY id",
                    (session_id,),
                ).fetchall()
            ]
            if not self._verify_event_rows(conn, rows):
                raise ValueError("evidence integrity check failed")

        receipts: Dict[str, List[str]] = {}
        for row in rows:
            call_id = row.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                continue
            if requested is not None and call_id not in requested:
                continue
            try:
                metadata = json.loads(row.get("metadata_json") or "{}")
            except (TypeError, ValueError):
                raise ValueError("invalid evidence metadata") from None
            extra = metadata.get("extra") if isinstance(metadata, dict) else None
            effect = None
            if isinstance(extra, dict) and any(
                key in extra
                for key in (
                    "effect_disposition",
                    "side_effect_attempted",
                    "side_effect_confirmed",
                )
            ):
                effect = {
                    key: extra[key]
                    for key in (
                        "effect_disposition",
                        "side_effect_attempted",
                        "side_effect_confirmed",
                    )
                    if key in extra
                }
            payload: Dict[str, Any] = {
                "adapter": row["adapter"],
                "candidate": row["candidate_id"],
                "event_sha256": row["event_hash"],
                "event_type": row["event_type"],
                "freshness": row["freshness_scope"],
                "occurred_at": row["occurred_at"],
                "outcome": row["outcome"],
                "source_message_id": row["source_message_id"],
                "source_sha256": row["source_sha256"],
                "state": row["lifecycle_state"],
                "subject": row["subject_id"],
                "supersedes_event_id": row["supersedes_event_id"],
                "trusted": bool(row["trusted"]),
            }
            if effect is not None:
                payload["side_effect"] = effect
            receipts.setdefault(call_id, []).append(
                "[Hermes evidence v1] " + canonical_json(payload)
            )
        return receipts

    def resolve_evidence_claim(
        self,
        session_id: str,
        *,
        state: str,
        subject: str,
        candidate: Optional[str],
        parent_lineage: Any,
        source_hash_value: str,
    ) -> Dict[str, Any]:
        rows = self.get_evidence_events(session_id)
        if not self.verify_evidence_integrity(session_id):
            return {"resolved": False, "reason": "integrity_chain_rejected", "event": None}
        events = []
        for row in rows:
            events.append({
                "issuer": row["issuer"],
                "adapter_version": row["adapter_version"],
                "adapter": row["adapter"],
                "trusted": bool(row["trusted"]),
                "state": row["lifecycle_state"],
                "lifecycle_dimension": row["lifecycle_dimension"],
                "outcome": row["outcome"],
                "subject": row["subject_id"],
                "candidate": row["candidate_id"],
                "parent_lineage": _decode_identity(row["parent_candidate_id"]),
                "metadata": json.loads(row["metadata_json"] or "{}"),
                "source_hash": row["source_sha256"],
                "previous_event_hash": row["previous_event_hash"],
                "event_hash": row["event_hash"],
            })
        return resolve_claim(
            events,
            state=state,
            subject=subject,
            candidate=candidate,
            parent_lineage=parent_lineage,
            source_hash_value=source_hash_value,
            integrity_chain=lambda _events: True,
        )
