"""Call-time provenance, scoped to the admitted turn rather than a process-global agent.

Task IDs are Hermes logical turn IDs, not inferred project/business labels.
Capture before queueing or crossing a worker RPC; never consult ambient state at flush.
"""
from contextvars import ContextVar

ROLES = frozenset({'controller', 'implementation', 'independent_review', 'auxiliary', 'worker_unspecified', 'unknown'})
_context: ContextVar[dict | None] = ContextVar('usage_turn_attribution', default=None)


def current_usage_context():
    return dict(_context.get() or {})


def validate_usage_context(value):
    if not isinstance(value, dict) or set(value) - {'execution_role', 'task_id'}:
        raise ValueError('invalid usage attribution')
    if value.get('execution_role', 'unknown') not in ROLES:
        raise ValueError('invalid usage role')
    task_id = value.get('task_id')
    if task_id is not None and (not isinstance(task_id, str) or not task_id or len(task_id) > 4096):
        raise ValueError('invalid usage task id')
    return dict(value)


def set_usage_context(task_id, role):
    return _context.set(validate_usage_context({'task_id': task_id, 'execution_role': role}))


def reset_usage_context(token):
    _context.reset(token)


def role_for_agent(agent):
    explicit = getattr(agent, '_execution_role', None)
    if explicit in ROLES:
        return explicit
    if getattr(agent, '_turn_origin', None) == 'review':
        return 'independent_review'
    if getattr(agent, '_turn_origin', None) == 'background_review':
        return 'auxiliary'
    platform = getattr(agent, 'platform', None)
    if platform == 'subagent':
        return 'worker_unspecified'
    return 'controller' if platform else 'unknown'
