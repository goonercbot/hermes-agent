"""Deterministic route enforcement for the AJ fleet task-routing contract.

The plugin deliberately uses only the documented Hermes ``PluginContext``
registration APIs.  It stores no request, argument, result, path, command, or
credential content: routing inputs are enums and telemetry is an aggregate.
"""
from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping

PLUGIN_NAME = "fleet-task-router"
ROUTER_VERSION = "1.0.1"
RECEIPT_SCHEMA_VERSION = "1"
SYSTEM_PROMPT_SECTION_ID = "fleet-task-router.route-first"
SKILL_NAME = "policy"
SKILL_PATH = Path(__file__).resolve().parent / "skills" / "policy" / "SKILL.md"

# This is static by design.  Hermes freezes registered system-prompt sections
# per session, which keeps the shared prompt prefix cacheable across turns.
SYSTEM_PROMPT_SECTION = """## Fleet task routing
Before any nontrivial tool path, call `fleet_route_task` with independent work
shape and consequence. `Routine` is reversible local or read-only work; `material` is executable, behavioural, or reversible remote-metadata work needing stronger evidence; `consequential` changes protected data, credentials, schedules, messages, spending, merge, runtime, or production and needs exact authority plus readback. Tool counts are telemetry only and never block execution. Direct is prerequisite → action →
verification. Bounded permits one explicitly recorded changed-theory repair.
For durable work, hand off to a durable owner with bounded nodes; that is required
model guidance and the durable consumer independently accepts the opaque receipt,
not plugin enforcement. Change shape or consequence explicitly, retain accepted
evidence, and stop at the proof target. A tested restart is direct +
consequential; an unknown timed-out mutation is bounded + consequential until
reconciled. Existing Hermes approvals and protected boundaries remain in force."""

WORK_SHAPES = frozenset({"direct", "bounded", "durable"})
CONSEQUENCES = frozenset({"routine", "material", "consequential"})
REASON_CODES = frozenset({
    "known_short_path",
    "current_state_check",
    "named_local_change",
    "exact_effect",
    "focused_diagnosis",
    "changed_theory_repair",
    "uncertain_effect",
    "cross_owner_dependency",
    "delegation_required",
    "independent_review_required",
    "expensive_reconstruction",
    "multi_session_work",
    "durable_correction_loop",
})
PROOF_TARGETS = frozenset({
    "current_state",
    "destination_exists",
    "focused_test",
    "readback",
    "reviewed_tree",
    "durable_handoff",
    "operational_readback",
    "reconciliation",
})

# These names are deliberately a backstop, never a claim that a tool name makes
# an effect safe.  Prefix handling catches supported/current kanban variants.
DIRECT_RESTRICTED_TOOLS = frozenset({"todo", "delegate_task"})
DURABLE_ORCHESTRATION_TOOLS = frozenset({
    "delegate_task",
    "kanban",
    "kanban_create",
    "kanban_dispatch",
    "kanban_task_claim",
    "kanban_task_complete",
    "kanban_task_block",
    "task_graph",
    "create_durable_task",
})
ROUTE_SCHEMA = {
    "name": "fleet_route_task",
    "description": (
        "Record the current task route before nontrivial tool work. Inputs are "
        "controlled categories only; never include request text, paths, commands, "
        "arguments, results, or credentials. Use graduation fields only to broaden "
        "an already recorded direct or bounded route."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "work_shape": {"type": "string", "enum": sorted(WORK_SHAPES)},
            "consequence": {"type": "string", "enum": sorted(CONSEQUENCES)},
            "reason_codes": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(REASON_CODES)},
                "minItems": 1,
                "uniqueItems": True,
            },
            "proof_target": {"type": "string", "enum": sorted(PROOF_TARGETS)},
            "graduation_from": {
                "type": "string", "enum": ["direct", "bounded"],
                "description": "Prior route shape when explicitly broadening a route.",
            },
            "graduation_reason_code": {
                "type": "string",
                "enum": sorted(REASON_CODES),
                "description": "Controlled reason that justifies the graduation.",
            },
            "consequence_from": {
                "type": "string", "enum": sorted(CONSEQUENCES),
                "description": "Prior consequence when explicitly escalating it.",
            },
            "consequence_reason_code": {
                "type": "string", "enum": sorted(REASON_CODES),
                "description": "Controlled reason that justifies consequence escalation.",
            },
            "repair_reason_code": {
                "type": "string",
                "enum": ["changed_theory_repair"],
                "description": "Record the bounded route's single changed-theory repair.",
            },
        },
        "required": ["work_shape", "consequence", "reason_codes", "proof_target"],
    },
}


class RouteInputError(ValueError):
    """Raised when route data is absent, malformed, or would weaken a route."""


@dataclass(frozen=True)
class RouteRecord:
    profile: str
    session: str
    turn: str
    work_shape: str
    consequence: str
    reason_codes: tuple[str, ...]
    proof_target: str
    tool_count: int = 0
    model_call_count: int = 1
    api_model_call_count: int = 0
    started_at: float = 0.0
    graduated_from: str | None = None
    graduation_reason_code: str | None = None
    consequence_from: str | None = None
    consequence_reason_code: str | None = None
    repair_count: int = 0
    repair_reason_code: str | None = None


@dataclass(frozen=True)
class Receipt:
    path: Path
    sha256: str


@dataclass
class _RouteDispatchBinding:
    profile: str
    session: str
    turn: str
    dispatch_key: tuple[str, str]
    token: Token["_RouteDispatchBinding | None"] | None = None


class _RouteDispatchBridge:
    """Fail-closed handoff for v0.21's thread-isolated policy hooks."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._pending: dict[tuple[str, str], _RouteDispatchBinding] = {}

    def bind(self, binding: _RouteDispatchBinding) -> bool:
        with self._lock:
            if binding.dispatch_key in self._pending:
                return False
            self._pending[binding.dispatch_key] = binding
            return True

    def consume(self, kwargs: Mapping[str, Any]) -> _RouteDispatchBinding | None:
        key = _dispatch_key(kwargs)
        with self._lock:
            return self._pending.pop(key, None) if key is not None else None

    def discard(self, binding: _RouteDispatchBinding | None) -> None:
        if binding is None:
            return
        with self._lock:
            if self._pending.get(binding.dispatch_key) is binding:
                self._pending.pop(binding.dispatch_key, None)

    def discard_identity(self, kwargs: Mapping[str, Any]) -> None:
        key = _dispatch_key(kwargs)
        turn = kwargs.get("turn_id")
        if key is None or not isinstance(turn, str) or not turn.strip():
            return
        with self._lock:
            binding = self._pending.get(key)
            if binding is not None and binding.turn == turn:
                self._pending.pop(key, None)


_ACTIVE_ROUTE_DISPATCH: ContextVar[_RouteDispatchBinding | None] = ContextVar(
    "fleet_task_router_active_route_dispatch", default=None
)
_ROUTE_DISPATCH_BRIDGE = _RouteDispatchBridge()


class RouteStore:
    """Lock-protected active routes plus privacy-minimized JSONL telemetry."""

    def __init__(self, state_root: Path | None = None):
        self._state_root = _owner_only_directory(state_root or _default_state_root())
        self._lock = threading.RLock()
        self._routes: dict[tuple[str, str, str], RouteRecord] = {}

    @staticmethod
    def _key(profile: str, session: str, turn: str) -> tuple[str, str, str]:
        return (profile, session, turn)

    def set_route(self, route: RouteRecord) -> tuple[RouteRecord, Receipt | None]:
        key = self._key(route.profile, route.session, route.turn)
        with self._lock:
            previous = self._routes.get(key)
            _validate_transition(previous, route)
            route = _preserve_route_accounting(previous, route)
            receipt = self._write_receipt(route) if route.work_shape == "durable" else None
            self._routes[key] = route
            try:
                self._telemetry(route, event=_transition(previous, route), status="recorded")
            except OSError:
                # A partially functioning plugin must not claim a route whose
                # mandatory privacy-safe audit event was not recorded.
                if previous is None:
                    self._routes.pop(key, None)
                else:
                    self._routes[key] = previous
                if receipt is not None:
                    receipt.path.unlink(missing_ok=True)
                raise
            return route, receipt

    def get(self, profile: str, session: str, turn: str) -> RouteRecord | None:
        with self._lock:
            return self._routes.get(self._key(profile, session, turn))

    def note_tool(self, profile: str, session: str, turn: str, duration_ms: Any, status: Any) -> None:
        with self._lock:
            key = self._key(profile, session, turn)
            route = self._routes.get(key)
            if route is None:
                return
            route = replace(route, tool_count=route.tool_count + 1)
            self._routes[key] = route
            self._telemetry(route, event="tool", status=_status_enum(status), duration_ms=_duration(duration_ms))

    def note_api_model_call(self, profile: str, session: str, turn: str) -> None:
        """Count a post-route API request; the selection call starts at one."""
        with self._lock:
            key = self._key(profile, session, turn)
            route = self._routes.get(key)
            if route is None:
                return
            route = replace(
                route,
                model_call_count=route.model_call_count + 1,
                api_model_call_count=route.api_model_call_count + 1,
            )
            self._routes[key] = route
            self._telemetry(route, event="model", status="recorded")

    def note_llm_completion(self, profile: str, session: str, turn: str) -> None:
        """Fallback for runtimes that do not expose per-request API hooks."""
        with self._lock:
            key = self._key(profile, session, turn)
            route = self._routes.get(key)
            if route is None or route.api_model_call_count:
                return
            route = replace(route, model_call_count=route.model_call_count + 1)
            self._routes[key] = route
            self._telemetry(route, event="model", status="recorded")

    def block(self, route: RouteRecord | None, status: str = "blocked") -> None:
        if route is None:
            return
        with self._lock:
            self._telemetry(route, event="block", status=status)

    def finalize(self, profile: str, session: str, turn: str | None, status: str) -> int:
        """Finalize one turn, or every active turn in a session when unspecified."""
        with self._lock:
            keys = [
                key for key in self._routes
                if key[0] == profile and key[1] == session and (turn is None or key[2] == turn)
            ]
            for key in keys:
                route = self._routes.pop(key)
                self._telemetry(route, event="finalize", status=status, duration_ms=_duration((time.monotonic() - route.started_at) * 1000))
            return len(keys)

    def active_count(self) -> int:
        with self._lock:
            return len(self._routes)

    def _telemetry(self, route: RouteRecord, *, event: str, status: str, duration_ms: int = 0) -> None:
        # Exactly opaque identifiers, enums, counts, durations, and status.
        payload = {
            "profile_id": _opaque(route.profile),
            "session_id": _opaque(route.session),
            "turn_id": _opaque(route.turn),
            "event": event,
            "status": status,
            "work_shape": route.work_shape,
            "consequence": route.consequence,
            "reason_codes": list(route.reason_codes),
            "proof_target": route.proof_target,
            "graduated_from": route.graduated_from,
            "graduation_reason_code": route.graduation_reason_code,
            "consequence_from": route.consequence_from,
            "consequence_reason_code": route.consequence_reason_code,
            "repair_count": route.repair_count,
            "repair_reason_code": route.repair_reason_code,
            "tool_count": route.tool_count,
            "model_call_count": route.model_call_count,
            "duration_ms": duration_ms,
        }
        telemetry_directory = _owner_only_directory(self._state_root / "telemetry")
        profile_file = telemetry_directory / f"{_opaque(route.profile)}.jsonl"
        encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        # O_APPEND plus the in-process lock keeps concurrent callback records
        # intact without retaining raw hook payloads.
        fd = os.open(profile_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        start = os.lseek(fd, 0, os.SEEK_END)
        try:
            os.fchmod(fd, 0o600)
            _write_all(fd, encoded)
            os.fsync(fd)
        except BaseException:
            os.ftruncate(fd, start)
            raise
        finally:
            os.close(fd)

    def _write_receipt(self, route: RouteRecord) -> Receipt:
        """Atomically retain durable-route metadata for the route owner only."""
        body = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "router_version": ROUTER_VERSION,
            "profile_id": _opaque(route.profile),
            "session_id": _opaque(route.session),
            "turn_id": _opaque(route.turn),
            "work_shape": route.work_shape,
            "consequence": route.consequence,
            "reason_codes": list(route.reason_codes),
            "proof_target": route.proof_target,
            "graduated_from": route.graduated_from,
            "graduation_reason_code": route.graduation_reason_code,
            "consequence_from": route.consequence_from,
            "consequence_reason_code": route.consequence_reason_code,
            "repair_count": route.repair_count,
            "repair_reason_code": route.repair_reason_code,
        }
        canonical_body = _canonical_json(body)
        digest = sha256(canonical_body).hexdigest()
        receipt = {**body, "receipt_sha256": digest}
        receipts_root = _owner_only_directory(self._state_root / "receipts")
        directory = _owner_only_directory(receipts_root / _opaque(route.profile))
        path = directory / f"{_opaque(route.session)}-{_opaque(route.turn)}-{digest}.json"
        temporary = directory / f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        encoded_receipt = _canonical_json(receipt)
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            try:
                _write_all(fd, encoded_receipt)
                os.fsync(fd)
            finally:
                os.close(fd)
            # Link is atomic and refuses to overwrite a digest-qualified receipt.
            # This preserves a prior valid receipt across a failed new transition.
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        try:
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # The replacement already succeeded. Some filesystems do not allow
            # directory fsync, so preserve the durable receipt rather than fail.
            pass
        return Receipt(path=path, sha256=digest)


def _opaque(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()[:24]


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_all(fd: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(fd, data[offset:])
        if written <= 0:
            raise OSError("incomplete router state write")
        offset += written


def _owner_only_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)
    return path


def _default_state_root() -> Path:
    configured = os.environ.get("FLEET_TASK_ROUTER_STATE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    # HERMES_HOME resolves to the active profile home. It is profile-safe unlike
    # a hard-coded ~/.hermes path and needs no dependency on Hermes internals.
    home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")).expanduser()
    return home / "state" / PLUGIN_NAME


def _profile_local_identity() -> str:
    """Derive a stable opaque identity when v0.21 omits a profile field."""
    # Route records retain this digest, never the resolved HERMES_HOME/state path.
    root = _ROUTER._state_root.resolve()
    return "profile-" + sha256(os.fsencode(str(root))).hexdigest()


def _nonempty_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise RouteInputError(f"missing or invalid {label}")
    return value


def _dispatch_key(kwargs: Mapping[str, Any]) -> tuple[str, str] | None:
    session = kwargs.get("session_id")
    task = kwargs.get("task_id")
    if not isinstance(session, str) or not session.strip():
        return None
    if task is None:
        task = session
    if not isinstance(task, str) or not task.strip():
        return None
    return session, task


def _identity(
    kwargs: Mapping[str, Any], binding: _RouteDispatchBinding | None = None,
) -> tuple[str, str, str]:
    profile_value = kwargs.get("profile_name", kwargs.get("profile"))
    session_value = kwargs.get("session_id", kwargs.get("task_id"))
    turn_value = kwargs.get("turn_id")
    if binding is not None:
        profile = binding.profile if profile_value is None else _nonempty_id(profile_value, "profile identity")
        session = binding.session if session_value is None else _nonempty_id(session_value, "session identity")
        turn = binding.turn if turn_value is None else _nonempty_id(turn_value, "turn identity")
        if (profile, session, turn) != (binding.profile, binding.session, binding.turn):
            raise RouteInputError("route dispatch identity does not match its hook context")
        return profile, session, turn
    return (
        _nonempty_id(_profile_local_identity() if profile_value is None else profile_value, "profile identity"),
        _nonempty_id(session_value, "session identity"),
        _nonempty_id(turn_value, "turn identity"),
    )


def _parse_route(
    params: Any, kwargs: Mapping[str, Any], binding: _RouteDispatchBinding | None = None,
) -> RouteRecord:
    if not isinstance(params, Mapping):
        raise RouteInputError("route data must be an object")
    allowed = {
        "work_shape", "consequence", "reason_codes", "proof_target",
        "graduation_from", "graduation_reason_code", "consequence_from", "consequence_reason_code",
        "repair_reason_code",
    }
    if set(params) - allowed or not all(key in params for key in ("work_shape", "consequence", "reason_codes", "proof_target")):
        raise RouteInputError("route data has missing or unknown fields")
    shape = params["work_shape"]
    consequence = params["consequence"]
    proof_target = params["proof_target"]
    reasons = params["reason_codes"]
    if (
        not isinstance(shape, str) or shape not in WORK_SHAPES
        or not isinstance(consequence, str) or consequence not in CONSEQUENCES
        or not isinstance(proof_target, str) or proof_target not in PROOF_TARGETS
    ):
        raise RouteInputError("route data contains an unknown controlled value")
    if (
        not isinstance(reasons, list) or not reasons
        or any(not isinstance(reason, str) or reason not in REASON_CODES for reason in reasons)
        or len(reasons) != len(set(reasons))
    ):
        raise RouteInputError("reason_codes must be a unique non-empty controlled list")
    graduated_from = params.get("graduation_from")
    graduation_reason = params.get("graduation_reason_code")
    if (graduated_from is None) != (graduation_reason is None):
        raise RouteInputError("graduation identity and reason must be supplied together")
    if graduated_from is not None and (
        not isinstance(graduated_from, str) or graduated_from not in {"direct", "bounded"}
        or not isinstance(graduation_reason, str) or graduation_reason not in REASON_CODES
    ):
        raise RouteInputError("invalid graduation data")
    consequence_from = params.get("consequence_from")
    consequence_reason = params.get("consequence_reason_code")
    if (consequence_from is None) != (consequence_reason is None):
        raise RouteInputError("consequence identity and reason must be supplied together")
    if consequence_from is not None and (
        not isinstance(consequence_from, str) or consequence_from not in CONSEQUENCES
        or not isinstance(consequence_reason, str) or consequence_reason not in REASON_CODES
    ):
        raise RouteInputError("invalid consequence escalation data")
    repair_reason = params.get("repair_reason_code")
    if repair_reason is not None and repair_reason != "changed_theory_repair":
        raise RouteInputError("invalid repair data")
    if (repair_reason is not None) != ("changed_theory_repair" in reasons):
        raise RouteInputError("repair reason code and reason_codes must agree")
    profile, session, turn = _identity(kwargs, binding)
    return RouteRecord(
        profile=profile,
        session=session,
        turn=turn,
        work_shape=shape,
        consequence=consequence,
        reason_codes=tuple(sorted(reasons)),
        proof_target=proof_target,
        started_at=time.monotonic(),
        graduated_from=graduated_from,
        graduation_reason_code=graduation_reason,
        consequence_from=consequence_from,
        consequence_reason_code=consequence_reason,
        repair_reason_code=repair_reason,
    )


def _validate_transition(previous: RouteRecord | None, route: RouteRecord) -> None:
    if previous is None:
        if (
            route.graduated_from is not None
            or route.consequence_from is not None
            or route.repair_reason_code is not None
        ):
            raise RouteInputError("a transition requires an existing route in this turn")
        return
    work_shape_changed = route.work_shape != previous.work_shape
    consequence_changed = route.consequence != previous.consequence
    if route.repair_reason_code is not None:
        # This records a declared changed-theory repair. The controlled category
        # makes it auditable, but does not purport to prove theory semantics.
        if previous.work_shape != "bounded" or previous.repair_count:
            raise RouteInputError("bounded work permits exactly one declared repair")
        if work_shape_changed or consequence_changed:
            raise RouteInputError("a repair cannot change work shape or consequence")
        if route.graduated_from is not None or route.consequence_from is not None:
            raise RouteInputError("a repair cannot combine with another transition")
        return
    if work_shape_changed:
        if route.graduated_from != previous.work_shape:
            raise RouteInputError("a changed route requires explicit matching graduation")
        order = {"direct": 0, "bounded": 1, "durable": 2}
        if previous.work_shape not in {"direct", "bounded"} or order[route.work_shape] <= order[previous.work_shape]:
            raise RouteInputError("graduation may only broaden direct or bounded work")
    elif route.graduated_from is not None:
        raise RouteInputError("graduation metadata requires a work-shape change")
    consequence_order = {"routine": 0, "material": 1, "consequential": 2}
    if consequence_changed:
        if route.consequence_from != previous.consequence:
            raise RouteInputError("a changed consequence requires explicit matching escalation")
        if consequence_order[route.consequence] <= consequence_order[previous.consequence]:
            raise RouteInputError("consequence may only escalate monotonically")
    elif route.consequence_from is not None:
        raise RouteInputError("consequence escalation metadata requires a consequence change")
    if not work_shape_changed and not consequence_changed:
        raise RouteInputError("a repeated route must make an explicit transition")


def _preserve_route_accounting(previous: RouteRecord | None, route: RouteRecord) -> RouteRecord:
    if previous is None:
        return route
    return replace(
        route,
        tool_count=previous.tool_count,
        model_call_count=previous.model_call_count,
        api_model_call_count=previous.api_model_call_count,
        started_at=previous.started_at,
        repair_count=previous.repair_count + (1 if route.repair_reason_code is not None else 0),
        repair_reason_code=route.repair_reason_code or previous.repair_reason_code,
    )


def _transition(previous: RouteRecord | None, route: RouteRecord) -> str:
    if previous is None:
        return "route"
    if route.repair_count > previous.repair_count:
        return "repair"
    work_shape_changed = route.work_shape != previous.work_shape
    consequence_changed = route.consequence != previous.consequence
    if work_shape_changed and consequence_changed:
        return "graduation_and_consequence_escalation"
    if work_shape_changed:
        return "graduation"
    return "consequence_escalation"


def _duration(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, min(int(float(value)), 3_600_000))
    except (TypeError, ValueError):
        return 0


def _status_enum(value: Any) -> str:
    text = str(value).lower() if value is not None else "other"
    if text in {"success", "ok", "completed", "complete"}:
        return "complete"
    if text in {"blocked", "block"}:
        return "blocked"
    if text in {"interrupted", "cancelled", "canceled"}:
        return "interrupted"
    if text in {"error", "failed", "failure"}:
        return "error"
    return "other"


def _is_durable_orchestration(tool_name: str) -> bool:
    return tool_name in DURABLE_ORCHESTRATION_TOOLS or tool_name.startswith("kanban_")


def _restricted(tool_name: str) -> bool:
    return tool_name in DIRECT_RESTRICTED_TOOLS or _is_durable_orchestration(tool_name)


def _guidance(route: RouteRecord | None, tool_name: str) -> str:
    if route is None:
        return "Record fleet_route_task first; then complete the selected route, explicitly graduate it, or stop."
    if route.work_shape == "direct":
        return "Direct route: complete the prerequisite-action-verification path, explicitly graduate, or stop; TODO and delegation are unavailable."
    if route.work_shape == "bounded":
        return "Bounded route: complete the focused path or one changed-theory repair, explicitly graduate before delegation or durable orchestration, or stop."
    return "Durable route: use the durable owner and bounded nodes, or stop."


_ROUTER = RouteStore()


def _bind_route_dispatch(profile: str, session: str, turn: str, kwargs: Mapping[str, Any]) -> _RouteDispatchBinding:
    dispatch_key = _dispatch_key(kwargs)
    if dispatch_key is None:
        raise RouteInputError("route dispatch is missing its session/task key")
    binding = _RouteDispatchBinding(profile=profile, session=session, turn=turn, dispatch_key=dispatch_key)
    binding.token = _ACTIVE_ROUTE_DISPATCH.set(binding)
    if not _ROUTE_DISPATCH_BRIDGE.bind(binding):
        _clear_route_dispatch(binding)
        raise RouteInputError("route dispatch is already pending for this session/task")
    return binding


def _clear_route_dispatch(binding: _RouteDispatchBinding | None = None) -> None:
    current = _ACTIVE_ROUTE_DISPATCH.get()
    if current is None or (binding is not None and current is not binding):
        return
    token = current.token
    current.token = None
    try:
        if token is not None:
            _ACTIVE_ROUTE_DISPATCH.reset(token)
        else:
            _ACTIVE_ROUTE_DISPATCH.set(None)
    except (RuntimeError, ValueError):
        _ACTIVE_ROUTE_DISPATCH.set(None)


def _transition_for_route(route: RouteRecord) -> str:
    if route.graduated_from is not None and route.consequence_from is not None:
        return "graduation_and_consequence_escalation"
    if route.graduated_from is not None:
        return "graduation"
    if route.consequence_from is not None:
        return "consequence_escalation"
    if route.repair_reason_code is not None:
        return "repair"
    return "route"


def fleet_route_task(params: Any = None, **kwargs: Any) -> str:
    """Tool handler: accept only controlled route categories and return JSON."""
    context_binding = _ACTIVE_ROUTE_DISPATCH.get()
    bridge_binding = _ROUTE_DISPATCH_BRIDGE.consume(kwargs)
    binding = context_binding or bridge_binding
    if context_binding is not None and bridge_binding is not None and bridge_binding is not context_binding:
        _clear_route_dispatch(context_binding)
        _ROUTE_DISPATCH_BRIDGE.discard(bridge_binding)
        return json.dumps({
            "ok": False,
            "error": "invalid_or_unrecorded_route",
            "guidance": "Stop restricted orchestration. Record a complete controlled route or explicitly graduate the current route.",
        }, separators=(",", ":"))
    try:
        route = _parse_route(params, kwargs, binding)
        route, receipt = _ROUTER.set_route(route)
        result: dict[str, Any] = {
            "ok": True,
            "work_shape": route.work_shape,
            "consequence": route.consequence,
            "transition": _transition_for_route(route),
            "guidance": _guidance(route, "fleet_route_task"),
        }
        if receipt is not None:
            # The owner-only receipt remains internal. Its digest is the opaque
            # handle a durable consumer uses beneath its profile-local state root.
            result["receipt_sha256"] = receipt.sha256
        return json.dumps(result, separators=(",", ":"))
    except (RouteInputError, OSError):
        return json.dumps({
            "ok": False,
            "error": "invalid_or_unrecorded_route",
            "guidance": "Stop restricted orchestration. Record a complete controlled route or explicitly graduate the current route.",
        }, separators=(",", ":"))
    finally:
        _ROUTE_DISPATCH_BRIDGE.discard(binding)
        _clear_route_dispatch(binding)


def _authenticated_note_maintenance(tool_name: str, args: dict | None, kwargs: Mapping[str, Any]) -> bool:
    """Recognize host-owned maintenance, never model-supplied permission.

    The runtime owns capability identity and expiry. This plugin relaxes only
    its own work-routing prerequisite; every other hook and guardrail still runs.
    Older runtimes and unavailable validators retain the ordinary fail-closed path.
    """
    capability = kwargs.get("maintenance_context")
    if tool_name != "continuity_note" or capability is None:
        return False
    try:
        from importlib import import_module

        validator = import_module("agent.native_note_refresh").is_native_note_refresh_authorized
        return validator(
            capability,
            tool_name=tool_name,
            args=args,
            session_id=kwargs.get("session_id"),
            tool_call_id=kwargs.get("tool_call_id"),
            api_request_id=kwargs.get("api_request_id"),
        ) is True
    except Exception:
        return False


def pre_tool_call(tool_name: str, args: dict | None = None, **kwargs: Any) -> dict[str, str] | None:
    """Require a route and fail closed without approving or overriding Hermes."""
    # Discover only this prerequisite's schema. This neither records a route
    # nor permits execution of the described tool or any ordinary user work.
    if tool_name == "tool_describe" and args == {"names": ["fleet_route_task"]}:
        return None
    if _authenticated_note_maintenance(tool_name, args, kwargs):
        return None
    if tool_name == "fleet_route_task":
        try:
            _bind_route_dispatch(*_identity(kwargs), kwargs)
            return None
        except Exception:
            return {"action": "block", "message": "Route identity is unavailable: stop restricted orchestration."}
    try:
        profile, session, turn = _identity(kwargs)
        route = _ROUTER.get(profile, session, turn)
        if route is None:
            return {"action": "block", "message": _guidance(None, tool_name)}
        if route.work_shape == "direct" and _restricted(tool_name):
            _ROUTER.block(route)
            return {"action": "block", "message": _guidance(route, tool_name)}
        if route.work_shape == "bounded" and _is_durable_orchestration(tool_name):
            _ROUTER.block(route)
            return {"action": "block", "message": _guidance(route, tool_name)}
        return None
    except Exception:
        # Hook exceptions are isolated by Hermes. Returning a directive here is
        # essential: a state/telemetry failure must never authorize restricted
        # resources by accident. A route failure blocks ordinary tool work too,
        # because it cannot prove that the requested path is nontrivial.
        return {"action": "block", "message": "Route policy state is unavailable: stop restricted orchestration or record a new route."}


def post_tool_call(tool_name: str, args: dict | None = None, result: Any = None, duration_ms: Any = 0, status: Any = None, **kwargs: Any) -> None:
    """Count calls without retaining tool name, arguments, result, or errors."""
    del args, result
    try:
        if tool_name != "fleet_route_task":
            profile, session, turn = _identity(kwargs)
            _ROUTER.note_tool(profile, session, turn, duration_ms, status)
    except Exception:
        # Observer failures never change approval or normal tool semantics.
        return None
    finally:
        if tool_name == "fleet_route_task":
            _ROUTE_DISPATCH_BRIDGE.discard_identity(kwargs)
            _clear_route_dispatch()


def post_api_request(**kwargs: Any) -> None:
    """Observe every post-route model request without retaining model content."""
    try:
        profile, session, turn = _identity(kwargs)
        _ROUTER.note_api_model_call(profile, session, turn)
    except Exception:
        return None


def post_llm_call(**kwargs: Any) -> None:
    """Fallback turn-level model accounting for supported runtimes."""
    try:
        profile, session, turn = _identity(kwargs)
        _ROUTER.note_llm_completion(profile, session, turn)
    except Exception:
        return None


def transform_llm_output(response_text: Any = None, **kwargs: Any) -> None:
    """Finalize the active session route once the turn has its final output."""
    del response_text
    _finalize("complete", **kwargs)
    return None


def _finalize(status: str, **kwargs: Any) -> None:
    try:
        profile_value = kwargs.get("profile_name", kwargs.get("profile"))
        profile = _nonempty_id(_profile_local_identity() if profile_value is None else profile_value, "profile identity")
        session = _nonempty_id(kwargs.get("session_id", kwargs.get("task_id")), "session identity")
        turn_value = kwargs.get("turn_id")
        turn = _nonempty_id(turn_value, "turn identity") if turn_value is not None else None
        _ROUTER.finalize(profile, session, turn, status)
    except Exception:
        return None


def on_session_finalize(**kwargs: Any) -> None:
    _finalize("complete", **kwargs)


def on_session_reset(**kwargs: Any) -> None:
    _finalize("interrupted", **kwargs)


def on_session_end(**kwargs: Any) -> None:
    _finalize("interrupted", **kwargs)


def register(ctx: Any) -> None:
    """Register the standalone plugin through documented Hermes APIs only."""
    global _ROUTER
    configured_root = os.environ.get("FLEET_TASK_ROUTER_STATE_ROOT")
    get_config = getattr(ctx, "get_config", None)
    if callable(get_config):
        configured_root = configured_root or get_config("state_root", None)
    state_root = Path(configured_root).expanduser().resolve() if isinstance(configured_root, str) and configured_root else None
    _ROUTER = RouteStore(state_root=state_root)
    ctx.register_system_prompt_section(
        SYSTEM_PROMPT_SECTION_ID,
        SYSTEM_PROMPT_SECTION,
        position="after_memory",
        max_chars=1600,
    )
    ctx.register_tool(
        name="fleet_route_task",
        toolset="fleet_task_router",
        schema=ROUTE_SCHEMA,
        handler=fleet_route_task,
        description="Record controlled fleet task routing categories.",
    )
    ctx.register_skill(SKILL_NAME, SKILL_PATH, description="Fleet task-routing policy and counterexamples.")
    ctx.register_hook("pre_tool_call", pre_tool_call)
    ctx.register_hook("post_tool_call", post_tool_call)
    ctx.register_hook("post_api_request", post_api_request)
    ctx.register_hook("post_llm_call", post_llm_call)
    ctx.register_hook("transform_llm_output", transform_llm_output)
    ctx.register_hook("on_session_finalize", on_session_finalize)
    ctx.register_hook("on_session_reset", on_session_reset)
    ctx.register_hook("on_session_end", on_session_end)
