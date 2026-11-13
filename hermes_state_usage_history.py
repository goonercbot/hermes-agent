"""Compatibility helpers for the immutable logical usage history.

This ledger is distinct from physical-attempt accounting. All writes are made on
caller-owned transactions, including worker receipt transactions.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from urllib.parse import urlsplit

COUNTERS = ('api_call_count', 'input_tokens', 'output_tokens', 'cache_read_tokens',
            'cache_write_tokens', 'reasoning_tokens')
AMOUNTS = (*COUNTERS, 'estimated_cost_usd', 'actual_cost_usd')


def origin(value):
    if not value:
        return ''
    try:
        p = urlsplit(value)
        if p.scheme not in ('http', 'https') or not p.hostname:
            return ''
        host = f'[{p.hostname}]' if ':' in p.hostname else p.hostname
        return f'{p.scheme}://{host}' + (f':{p.port}' if p.port is not None else '')
    except (TypeError, ValueError):
        return ''


def cost_fields(status, source):
    # Native status is retained in cost_source when the old reader cannot accept it.
    # Never map an estimate to "reported" (which implies billed cost).
    if status and status.lower() not in ('unknown', 'included', 'reported'):
        source = json.dumps({'native_cost_status': status, 'native_cost_source': source},
                            sort_keys=True, separators=(',', ':'))
        status = 'unknown'
    return status, source


def new_part(values, *, occurred_at=None, attribution=None):
    from hermes_state_usage_context import current_usage_context, validate_usage_context
    context = current_usage_context() if attribution is None else validate_usage_context(attribution)
    return {**context, **values, 'event_id': 'delta:' + uuid.uuid4().hex, 'occurred_at': occurred_at}


def append_event(conn, session_id, part, *, task='', kind='delta', timing='point', interval=None, auxiliary=False):
    if not any(part.get(k) not in (None, 0) for k in AMOUNTS):
        return
    if kind == 'delta' and any((part.get(k) or 0) < 0 for k in AMOUNTS):
        timing, interval = 'unknown', None
    if timing == 'point' and part.get('occurred_at') is None:
        timing = 'unknown'
    row = conn.execute('SELECT parent_session_id, source, model, billing_provider, billing_base_url, billing_mode FROM sessions WHERE id=?', (session_id,)).fetchone()
    session = dict(row) if row else {}
    fallback = {} if task else session
    status, source = cost_fields(part.get('cost_status'), part.get('cost_source'))
    role = part.get('execution_role')
    if auxiliary:
        role = 'independent_review' if task == 'review' else 'auxiliary'
    if role not in ('controller', 'implementation', 'independent_review', 'auxiliary', 'worker_unspecified', 'unknown'):
        role = 'auxiliary' if auxiliary else 'unknown'
    start, end = interval if timing == 'interval' else (None, None)
    conn.execute('''INSERT INTO usage_events (
        event_id,schema_version,event_kind,session_id,parent_session_id,session_source,
        model,billing_provider,billing_base_url,billing_mode,task,execution_role,task_id,
        occurred_at,recorded_at,interval_start,interval_end,timing_kind,api_call_count,
        input_tokens,output_tokens,cache_read_tokens,cache_write_tokens,reasoning_tokens,
        estimated_cost_usd,actual_cost_usd,cost_status,cost_source)
        VALUES (?,1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
        (part['event_id'], kind, session_id, session.get('parent_session_id'), session.get('source') or 'unknown',
         part.get('model') or fallback.get('model') or 'unknown',
         part.get('billing_provider') or fallback.get('billing_provider') or '',
         origin(part.get('billing_base_url') or fallback.get('billing_base_url')),
         part.get('billing_mode') or fallback.get('billing_mode') or '', task or '', role, part.get('task_id'),
         part.get('occurred_at') if timing == 'point' else None, time.time(), start, end, timing,
         *(int(part.get(k) or 0) for k in COUNTERS), part.get('estimated_cost_usd'), part.get('actual_cost_usd'), status, source))


def initialize(conn):
    """Seed legacy per-route aggregates once; existing ledger markers stay authoritative."""
    conn.execute('BEGIN IMMEDIATE')
    try:
        import math
        markers = dict(conn.execute("SELECT key, value FROM usage_event_meta WHERE key IN ('schema_version','initialized_at')"))
        if 'schema_version' in markers and markers['schema_version'] != '1':
            raise ValueError('incompatible usage event schema marker')
        if 'initialized_at' in markers:
            try:
                stamp = float(markers['initialized_at'])
            except (TypeError, ValueError):
                raise ValueError('invalid usage event initialization timestamp') from None
            if not math.isfinite(stamp) or stamp < 0:
                raise ValueError('invalid usage event initialization timestamp')
        if len(markers) == 2:
            conn.commit()
            return
        if conn.execute('SELECT 1 FROM usage_events LIMIT 1').fetchone():
            # Existing event coverage is authoritative as-is. Aggregate overlap is
            # unknowable; adoption is not a claim that the ledger is complete.
            conn.execute("INSERT INTO usage_event_meta(key,value) VALUES ('legacy_adoption','existing_events_coverage_unknown') ON CONFLICT(key) DO NOTHING")
            conn.execute("INSERT INTO usage_event_meta(key,value) VALUES ('schema_version','1') ON CONFLICT(key) DO NOTHING")
            conn.execute("INSERT INTO usage_event_meta(key,value) VALUES ('initialized_at',?) ON CONFLICT(key) DO NOTHING", (str(time.time()),))
            conn.commit()
            return
        if markers:
            raise ValueError('incomplete usage event initialization markers on empty ledger')
        for row in conn.execute('SELECT * FROM session_model_usage').fetchall():
            part = dict(row)
            signed = any((part.get(k) or 0) < 0 for k in AMOUNTS)
            identity = [part[k] for k in ('session_id','model','billing_provider','billing_base_url','billing_mode','task')]
            part['event_id'] = 'baseline:' + hashlib.sha256(json.dumps(identity, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()
            first, last = part.get('first_seen'), part.get('last_seen')
            interval = (first, last) if not signed and first is not None and last is not None and 0 <= first <= last else None
            # The historical reader permits signed values only on unknown-timing
            # deltas. Preserve the aggregate exactly without inventing a period.
            append_event(conn, part['session_id'], part, task=part.get('task') or '', kind='delta' if signed else 'baseline', timing='interval' if interval else 'unknown', interval=interval, auxiliary=bool(part.get('task')))
        now = time.time()
        conn.executemany('INSERT INTO usage_event_meta(key,value) VALUES (?,?)', [('schema_version','1'), ('initialized_at',str(now))])
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
