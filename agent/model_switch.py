"""Conversation-scoped model route notices, outside the model's transcript."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)
_BIND_LOCK = threading.RLock()


@dataclass
class ModelSwitchState:
    routes: dict = field(default_factory=dict)
    lock: Any = field(default_factory=threading.RLock, repr=False)


def bind_model_switch_state(agent, owner) -> None:
    """Share ephemeral state across cached/recreated agents and gateway hygiene."""
    if owner is None:
        return
    with _BIND_LOCK:
        state = getattr(owner, "_model_switch_state", None)
        if not isinstance(state, ModelSwitchState):
            state = ModelSwitchState()
            owner._model_switch_state = state
        agent._model_switch_state = state


def emit_model_switch(agent, lane, provider, model, *, previous=None, text=None, force=False) -> bool:
    """Announce a selected route once per transition, never claim a successful provider call.

    Only provider/model labels enter the notice: no endpoint, credential or exception payload.
    An unknown initial compression route is disclosed once rather than guessing its predecessor.
    """
    state = getattr(agent, "_model_switch_state", None)
    if not isinstance(state, ModelSwitchState):
        state = agent._model_switch_state = ModelSwitchState()
    route = (str(provider or "auto"), str(model or "default"))
    with state.lock:
        old = state.routes.get(lane, previous)
        if old == route and not force:
            state.routes[lane] = route
            return False
        label = "Main agent" if lane == "main" else "Compression"
        if text is None:
            before = f"{old[1]} via {old[0]} → " if old else ""
            text = f"🔔 {label} model selected: {before}{route[1]} via {route[0]}."
        try:
            emit = getattr(agent, "_emit_status_kind", None)
            if callable(emit):
                emit("model_switch", text, origin="model_switch")
            else:
                callback = getattr(agent, "status_callback", None)
                if not callable(callback):
                    return False
                callback("model_switch", text)
        except Exception:
            logger.warning("Could not announce %s model selection", lane, exc_info=True)
            return False
        state.routes[lane] = route
        return True
