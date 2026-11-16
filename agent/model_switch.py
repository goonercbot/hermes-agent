"""Conversation-scoped selected routes and receipt-backed pending notices."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)
_BIND_LOCK = threading.RLock()


@dataclass
class PendingSwitch:
    text: str
    in_flight: bool = False


@dataclass
class ModelSwitchState:
    # Selected routes are not evidence of notification delivery.
    routes: dict = field(default_factory=dict)
    pending: list = field(default_factory=list)
    lock: Any = field(default_factory=threading.RLock, repr=False)


def bind_model_switch_state(agent, owner) -> None:
    if owner is None:
        return
    with _BIND_LOCK:
        state = getattr(owner, "_model_switch_state", None)
        if not isinstance(state, ModelSwitchState):
            state = getattr(agent, "_model_switch_state", None)
            if not isinstance(state, ModelSwitchState):
                state = ModelSwitchState()
            owner._model_switch_state = state
        agent._model_switch_state = state


def flush_model_switch_notices(agent) -> bool:
    """Try pending notices in order; failed/suppressed sends wait for the next eligible turn.

    Gateway sinks return futures whose result is an explicit delivery acknowledgement.
    Missing callbacks and a normal None return are not acknowledgements.
    """
    state = getattr(agent, "_model_switch_state", None)
    if not isinstance(state, ModelSwitchState):
        return False
    with state.lock:
        batch = [notice for notice in state.pending if not notice.in_flight]
        for notice in batch:
            notice.in_flight = True
    for notice in batch:
        _deliver_pending_switch(agent, state, notice)
    return bool(batch)


def _deliver_pending_switch(agent, state, notice):
    def settle(result):
        ok = result is True or getattr(result, "success", None) is True
        with state.lock:
            notice.in_flight = False
            if ok and notice in state.pending:
                state.pending.remove(notice)
        if not ok:
            logger.warning("Model-switch notice remains pending (no delivery acknowledgement)")

    try:
        sink = getattr(agent, "_model_switch_callback", None)
        if callable(sink):
            result = sink(notice.text)
        else:
            callback = getattr(agent, "status_callback", None)
            if callable(callback):
                result = callback("model_switch", notice.text)
            elif (getattr(agent, "platform", "cli") == "cli"
                  and not getattr(agent, "suppress_status_output", False)
                  and not getattr(agent, "_mute_notification_reply", False)):
                printer = getattr(agent, "_vprint", None)
                if not callable(printer):
                    settle(False)
                    return False
                printer(notice.text, force=True)
                result = True
            else:
                settle(False)
                return False
        if callable(getattr(result, "add_done_callback", None)):
            def completed(future):
                try:
                    receipt = future.result()
                except BaseException as exc:
                    logger.exception("Model-switch receipt failed (%s)", type(exc).__name__, exc_info=False)
                    receipt = False
                settle(receipt)
            result.add_done_callback(completed)
        else:
            settle(result)
        return True
    except Exception as exc:
        logger.exception("Model-switch notification sink failed (%s)", type(exc).__name__, exc_info=False)
        settle(False)
        return False


def initialize_main_model_notices(agent, owner, pending_fallback_notice):
    """Wire retained notices only after this turn's delivery callbacks are ready."""
    bind_model_switch_state(agent, owner)
    flush_model_switch_notices(agent)
    emit_pending_switch = getattr(agent, "_emit_pending_fallback_notice", None)
    if pending_fallback_notice and callable(emit_pending_switch):
        emit_pending_switch()
    elif not pending_fallback_notice:
        route = (str(getattr(agent, "provider", "") or "auto"), str(getattr(agent, "model", "") or "default"))
        emit_model_switch(agent, "main", *route, previous=route)


def emit_model_switch(agent, lane, provider, model, *, previous=None, text=None, force=False) -> bool:
    """Queue each selected-route transition; never imply a successful model response."""
    state = getattr(agent, "_model_switch_state", None)
    if not isinstance(state, ModelSwitchState):
        state = agent._model_switch_state = ModelSwitchState()
    route = (str(provider or "auto"), str(model or "default"))
    with state.lock:
        old = state.routes.get(lane, previous)
        changed = old != route or force
        state.routes[lane] = route
        if changed:
            label = "Main agent" if lane == "main" else "Compression"
            before = f"{old[1]} via {old[0]} → " if old else ""
            state.pending.append(PendingSwitch(text or f"🔔 {label} model selected: {before}{route[1]} via {route[0]}."))
    flush_model_switch_notices(agent)
    return changed
