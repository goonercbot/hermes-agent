"""Call metadata survives delayed writes, native worker RPC, and nested turns."""
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from hermes_state_runtime import _worker_usage, RuntimeStoreError
from hermes_state_usage_context import (current_usage_context, role_for_agent,
    set_usage_context, reset_usage_context)


def test_queued_calls_capture_their_own_context_before_flush(tmp_path):
    db = SessionDB(tmp_path/'state.db')
    for task,role in [('one','controller'),('two','independent_review')]:
        token = set_usage_context(task,role)
        try:
            db.queue_token_counts('s',model='m',api_call_count=1,input_tokens=11)
        finally:
            reset_usage_context(token)
    assert db.flush_token_counts()
    rows = list(db._conn.execute('select task_id,execution_role,occurred_at,input_tokens from usage_events order by rowid'))
    assert [(r[0],r[1]) for r in rows] == [('one','controller'),('two','independent_review')]
    assert all(r[2] is not None and r[3]==11 for r in rows)
    assert current_usage_context()=={}
    db.close()


def test_synchronous_and_auxiliary_calls_keep_task_but_distinct_role(tmp_path):
    db=SessionDB(tmp_path/'state.db')
    token=set_usage_context('admitted-turn','controller')
    try:
        db.update_token_counts('s',model='m',api_call_count=1)
        db.record_auxiliary_usage('s','vision',model='a',input_tokens=4)
    finally:
        reset_usage_context(token)
    rows=list(db._conn.execute('select task_id,execution_role from usage_events order by rowid'))
    assert [tuple(r) for r in rows]==[('admitted-turn','controller'),('admitted-turn','auxiliary')]
    db.close()


def test_worker_receives_explicit_context_not_server_ambient_context(tmp_path):
    db=SessionDB(tmp_path/'state.db');db.create_session('s','cli')
    token=set_usage_context('wrong-server-task','controller')
    try:
        db._execute_write(lambda c:_worker_usage(db,c,'s',{'model':'m','api_call_count':1,
            '_usage_occurred_at':1234,'_usage_attribution':{'task_id':'worker-task','execution_role':'implementation'}}))
        db._execute_write(lambda c:_worker_usage(db,c,'s',{'model':'m','api_call_count':1}))
    finally:
        reset_usage_context(token)
    rows=list(db._conn.execute('select task_id,execution_role,occurred_at from usage_events order by rowid'))
    assert tuple(rows[0])==('worker-task','implementation',1234)
    assert tuple(rows[1])==(None,'unknown',None)
    db.close()


@pytest.mark.parametrize('bad',[{'execution_role':'invented'},{'task_id':False},{'task_id':''},{'secret':'bad'}])
def test_worker_invalid_metadata_fails_before_counters_change(tmp_path,bad):
    db=SessionDB(tmp_path/'state.db');db.create_session('s','cli')
    with pytest.raises(RuntimeStoreError):
        db._execute_write(lambda c:_worker_usage(db,c,'s',{'model':'m','api_call_count':1,'_usage_attribution':bad}))
    assert db._conn.execute('select count(*) from usage_events').fetchone()[0]==0
    db.close()


def test_nested_context_and_unlabelled_workers_are_not_guessed():
    outer=set_usage_context('outer','controller')
    inner=set_usage_context('inner','independent_review')
    reset_usage_context(inner)
    assert current_usage_context()['task_id']=='outer'
    reset_usage_context(outer)
    assert role_for_agent(SimpleNamespace(platform='subagent'))=='worker_unspecified'
    assert role_for_agent(SimpleNamespace(platform='telegram'))=='controller'
    assert role_for_agent(SimpleNamespace(platform='cli',_turn_origin='review'))=='independent_review'
    assert role_for_agent(SimpleNamespace())=='unknown'
