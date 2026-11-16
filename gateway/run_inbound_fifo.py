"""Keep task-owned FIFO transfer validation outside the admission coordinator."""
from typing import Any, cast
import asyncio


def take_fifo_handoff():
    task = asyncio.current_task()
    proof = getattr(task, "_hermes_fifo_handoff", None)
    if proof is not None:
        delattr(task, "_hermes_fifo_handoff")
    return task, proof


def rescue_fifo_handoff(runner, trusted, task, proof, event, source, is_internal, session_key):
    """Revalidate the exact task, adapter and routing key immediately before rescue."""
    if (trusted and asyncio.current_task() is task
            and runner._delivery_adapter_for(source) is proof[0]
            and runner._session_key_for_source(source) == session_key == proof[1]
            and proof[0]._session_tasks.get(session_key) is task):
        cast(Any, task)._hermes_fifo_handoff = (proof[0], session_key, event)
        try:
            return runner._hm_rescue_orphaned_fifo(event, source, is_internal, session_key)
        finally:
            delattr(task, "_hermes_fifo_handoff")
    return runner._hm_rescue_orphaned_fifo(event, source, is_internal, session_key)
