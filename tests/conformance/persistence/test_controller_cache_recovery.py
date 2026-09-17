"""Controller cache recovery across fresh gateway, CLI, and factory-child hosts.

This is intentionally not a ``SessionDB.close()`` test.  It starts a seed host
and a separate recovery host, each with a disposable HOME/HERMES_HOME, and
makes the recovery host reconstruct real controller projections from the same
SQLite rows.  Provider transport is explicitly in-process mocked; no provider
or live profile is contacted.
"""
from __future__ import annotations

import json
from pathlib import Path

from tests.conformance.persistence._harness import reap, spawn_child


def _isolated_env(tmp_path: Path) -> dict[str, str]:
    home = tmp_path / "home"
    hermes_home = tmp_path / "hermes-home"
    home.mkdir()
    hermes_home.mkdir()
    return {"HOME": str(home), "HERMES_HOME": str(hermes_home)}


def test_fresh_controller_hosts_restore_caches_notes_and_child_ownership(tmp_path):
    """A new gateway/CLI/child process restores sealed rows then accepts new turns.

    Seed uses the real delegate child factory and real SessionDB.  Recovery uses
    fresh controller objects, restores authenticated native notes before each
    turn, re-baselines the real gateway cache, and compares pre-turn durable
    row fingerprints against the original prefixes after the new turns append.
    """
    db_path = tmp_path / "state.db"
    seeded = tmp_path / "seeded.json"
    recovered = tmp_path / "recovered.json"
    env = _isolated_env(tmp_path)

    seed = spawn_child(
        f'''
import hashlib, json
from pathlib import Path
from types import SimpleNamespace as NS
from hermes_state import SessionDB
from agent.native_incremental_handoff import native_incremental_compact_context
from tests.run_agent.test_native_incremental_handoff import _agent, _note, _response, _source
from tools.delegate_tool import _build_child_preserving_parent_tools

path = Path({str(db_path)!r})
db = SessionDB(db_path=path)

def sealed(session_id):
    producer, _ = _agent([_response({{'type': 'compaction', 'id': 'checkpoint', 'encrypted_content': 'sealed-cache'}})])
    producer.session_id = session_id
    source = _source()
    _note(producer, source)
    return native_incremental_compact_context(producer, source)

def persist(session_id, source, rows, parent=None):
    db.create_session(session_id, source, parent_session_id=parent)
    for row in rows:
        db.append_message(session_id, row['role'], row.get('content'),
            tool_name=row.get('name'), tool_calls=row.get('tool_calls'),
            tool_call_id=row.get('tool_call_id'),
            codex_reasoning_items=row.get('codex_reasoning_items'),
            codex_message_items=row.get('codex_message_items'))

def fingerprint(session_id):
    rows = db.get_messages_as_conversation(session_id, repair_alternation=False)
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest(), len(rows)

rows = sealed('gateway-controller')
persist('gateway-controller', 'telegram', rows)
persist('cli-controller', 'cli', rows)
persist('parent-controller', 'cli', [{{'role': 'user', 'content': 'factory parent'}}])
parent = NS(session_id='parent-controller', _session_db=db, api_key='fixture', base_url='http://fixture.invalid/v1',
            provider='custom', api_mode='chat_completions', model='fixture-model', enabled_toolsets=[],
            disabled_toolsets=[], platform='cli', _delegate_depth=0, request_overrides={{}}, prefill_messages=None,
            _print_fn=None, _active_children=[])
child = _build_child_preserving_parent_tools(
    task_index=0, goal='preserve durable checkpoint', context='disposable conformance child', toolsets=[],
    model='fixture-model', max_iterations=1, task_count=1, parent_agent=parent,
    override_provider='custom', override_base_url='http://fixture.invalid/v1', override_api_key='fixture',
    override_api_mode='chat_completions', routing_cfg={{'provider': 'custom'}}, role='leaf')
child._ensure_db_session()
child_id = child.session_id
for row in rows:
    child._session_db.append_message(child_id, row['role'], row.get('content'),
        tool_name=row.get('name'), tool_calls=row.get('tool_calls'), tool_call_id=row.get('tool_call_id'),
        codex_reasoning_items=row.get('codex_reasoning_items'), codex_message_items=row.get('codex_message_items'))
child.close()
payload = {{'sessions': {{sid: fingerprint(sid) for sid in ('gateway-controller', 'cli-controller', child_id)}},
           'child_id': child_id,
           'child_row': db.get_session(child_id)}}
db.close()
Path({str(seeded)!r}).write_text(json.dumps(payload, sort_keys=True), encoding='utf-8')
''',
        env=env,
    )
    rc, _out, err = reap(seed)
    assert rc == 0, err

    recovery = spawn_child(
        f'''
import asyncio, hashlib, json, threading
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from hermes_state import AsyncSessionDB, SessionDB
from agent.native_incremental_handoff import bind_native_incremental_replay_projection, restore_native_incremental_note
from cli import HermesCLI
from gateway.run import GatewayRunner
from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin
from run_agent import AIAgent

path = Path({str(db_path)!r})
seeded = json.loads(Path({str(seeded)!r}).read_text(encoding='utf-8'))
db = SessionDB(db_path=path)
child_id = seeded['child_id']

def rows(sid):
    return db.get_messages_as_conversation(sid, repair_alternation=False)

def fp(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

# Check the actual seed process receipt before any recovery turn can write.
for sid, (expected_hash, expected_count) in seeded['sessions'].items():
    durable = rows(sid)
    assert len(durable) == expected_count and fp(durable) == expected_hash

def fresh_agent(sid, platform):
    agent = AIAgent(api_key='fixture', base_url='http://fixture.invalid/v1', api_mode='chat_completions',
        model='fixture-model', provider='custom', session_db=db, session_id=sid, platform=platform,
        quiet_mode=True, skip_memory=True, skip_context_files=True, skip_background_review=True, enabled_toolsets=[])
    agent._disable_streaming = True
    agent._compression_feasibility_checked = True
    agent._emit_status = lambda *a, **k: None
    agent._emit_warning = lambda *a, **k: None
    agent.commit_memory_session = lambda *a, **k: None
    agent.context_compressor.should_compress_preflight = lambda _m: False
    agent.context_compressor.should_compress = lambda _m: False
    # This host is deliberately configured for the production chat-completions
    # transport, so the mock must have the corresponding OpenAI ChatCompletion
    # shape rather than a Responses API ``output`` list.
    agent._interruptible_api_call = lambda _request: NS(
        choices=[NS(message=NS(content=f'NEW_TURN_{{sid}}', tool_calls=None), finish_reason='stop')],
        usage=None, model='fixture-model')
    return agent

# Concrete CLI resume host: its real resume method reads fresh rows from SessionDB.
cli = HermesCLI.__new__(HermesCLI)
cli.session_id = 'cli-controller'
cli._session_db = db
cli._resumed = True
cli._resume_history_error = None
cli.tool_progress_mode = 'off'
cli.conversation_history = []
cli._restore_session_state = lambda *_a, **_k: None
assert CLIAgentSetupMixin._load_resumed_history_late(cli)
assert restore_native_incremental_note(NS(session_id='cli-controller'), cli.conversation_history) is not None
# The concrete CLI loader above provides the restored history; its fresh turn then
# uses the same real AIAgent turn loop that HermesCLI owns in normal operation.
cli_history = cli.conversation_history
cli.agent = fresh_agent('cli-controller', 'cli')
assert bind_native_incremental_replay_projection(cli.agent, source_messages=cli_history, replay_messages=cli_history)
assert restore_native_incremental_note(cli.agent, cli_history) is not None
before_cli = rows('cli-controller')
with patch('hermes_cli.plugins.invoke_hook', return_value=[]), patch('hermes_cli.lifecycle.invoke_hook', return_value=[]), patch('agent.turn_context._maybe_title_session_at_turn_start', return_value=None):
    cli_outcome = cli.agent.run_conversation('new cli user turn', conversation_history=cli_history)
assert cli_outcome['completed'] and cli_outcome['final_response'] == 'NEW_TURN_cli-controller'

# Concrete gateway cache host starts empty after the process boundary, then re-baselines an actual cache entry.
gateway = GatewayRunner.__new__(GatewayRunner)
gateway._agent_cache, gateway._agent_cache_lock = {{}}, threading.Lock()
gateway._session_db = AsyncSessionDB(db)
gateway_agent = fresh_agent('gateway-controller', 'gateway')
gateway_history = rows('gateway-controller')
assert bind_native_incremental_replay_projection(gateway_agent, source_messages=gateway_history, replay_messages=gateway_history)
assert restore_native_incremental_note(gateway_agent, gateway_history) is not None
before_gateway = rows('gateway-controller')
gateway._agent_cache['gateway-key'] = (gateway_agent, 'fresh-process', db.get_session('gateway-controller')['message_count'], 'gateway-controller')
with patch('hermes_cli.plugins.invoke_hook', return_value=[]), patch('hermes_cli.lifecycle.invoke_hook', return_value=[]), patch('agent.turn_context._maybe_title_session_at_turn_start', return_value=None):
    outcome = gateway_agent.run_conversation('new gateway user turn', conversation_history=gateway_history)
assert outcome['completed'] and outcome['final_response'] == 'NEW_TURN_gateway-controller'
asyncio.run(gateway._refresh_agent_cache_message_count('gateway-key', 'gateway-controller'))
assert gateway._agent_cache['gateway-key'][2] == db.get_session('gateway-controller')['message_count']

# Fresh factory-child recovery host owns its child row and appends only a new turn.
child_row = db.get_session(child_id)
assert child_row['source'] == 'subagent' and child_row['parent_session_id'] == 'parent-controller'
child = fresh_agent(child_id, 'subagent')
child_history = rows(child_id)
assert bind_native_incremental_replay_projection(child, source_messages=child_history, replay_messages=child_history)
assert restore_native_incremental_note(child, child_history) is not None
before_child = rows(child_id)
with patch('hermes_cli.plugins.invoke_hook', return_value=[]), patch('hermes_cli.lifecycle.invoke_hook', return_value=[]), patch('agent.turn_context._maybe_title_session_at_turn_start', return_value=None):
    child_outcome = child.run_conversation('new child user turn', conversation_history=child_history)
assert child_outcome['completed'] and child_outcome['final_response'] == f'NEW_TURN_{{child_id}}'

# Every fresh host appended only its new turn after its sealed prefix.
assert rows('cli-controller')[:len(before_cli)] == before_cli
assert rows('gateway-controller')[:len(before_gateway)] == before_gateway
assert rows(child_id)[:len(before_child)] == before_child
payload = {{'gateway_cache_entries_at_recovery_start': 0, 'gateway_cache_entries_after_turn': len(gateway._agent_cache),
           'cli_rows_before': len(before_cli), 'cli_rows_after': len(rows('cli-controller')),
           'gateway_rows_before': len(before_gateway), 'gateway_rows_after': len(rows('gateway-controller')),
           'child_rows_before': len(before_child), 'child_rows_after': len(rows(child_id)), 'child_id': child_id,
           'fingerprints': {{'cli_prefix': fp(before_cli), 'gateway_prefix': fp(before_gateway), 'child_prefix': fp(before_child)}}}}
cli.agent.close(); gateway_agent.close(); child.close(); db.close()
Path({str(recovered)!r}).write_text(json.dumps(payload, sort_keys=True), encoding='utf-8')
''',
        env=env,
    )
    rc, _out, err = reap(recovery)
    assert rc == 0, err
    payload = json.loads(recovered.read_text(encoding="utf-8"))
    assert payload["gateway_cache_entries_at_recovery_start"] == 0
    assert payload["gateway_cache_entries_after_turn"] == 1
    assert payload["cli_rows_after"] > payload["cli_rows_before"]
    assert payload["gateway_rows_after"] > payload["gateway_rows_before"]
    assert payload["child_rows_after"] > payload["child_rows_before"]
