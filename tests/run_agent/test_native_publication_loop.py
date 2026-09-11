"""Full automatic loop tests: real SQLite and terminal, fixture provider replies."""
from copy import deepcopy
import json
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
from agent.native_compaction import validate_persisted_native_compaction_history
from agent.native_incremental_handoff import _projection_source_for_messages, restore_native_incremental_note
from tests.run_agent.test_native_preflight_projection_rebind import _agent_with_authenticated_history, _response, _message, NOTE


@pytest.mark.parametrize("in_place", [False, True])
@pytest.mark.parametrize("platform", ["telegram", "subagent"])
def test_two_automatic_cycles_tools_refresh_and_reload(tmp_path, monkeypatch, in_place, platform, caplog):
    agent, db, source = _agent_with_authenticated_history(tmp_path, monkeypatch, toolsets=["terminal","file","continuity"], platform=platform, model="gpt-5.6-terra" if platform == "subagent" else "gpt-6-astra")
    agent.compression_in_place = in_place
    agent.context_compressor.threshold_tokens = 100000
    agent.context_compressor.should_compress_preflight = lambda _: False
    agent.context_compressor.should_compress = lambda tokens: tokens >= 100000
    agent._todo_store.format_for_injection = lambda: "[Current task state]\nTASK_STATE_AMBER; deployment paused"
    assert source[-1].get("role") == "tool"
    assert (source[-1].get("name") or source[-1].get("tool_name")) == "continuity_note"
    calls = []; checkpoints = []; tool_calls = []; refreshes = []; seeded = set(); post_tools = []; written_paths = []

    def provider(request):
        calls.append(deepcopy(request))
        if request["model"] == "gpt-5.6-luna":
            checkpoints.append(len(checkpoints)+1)
            assert len(checkpoints) <= 2
            return _response(NS(type="compaction",id=f"cp-{len(checkpoints)}",encrypted_content=f"fixture-cipher-{len(checkpoints)}"),_message("Continue the authorized task."),model="gpt-5.6-luna")
        if request.get("tool_choice") == {"type":"function","name":"continuity_note"}:
            refreshes.append(len(refreshes)+1)
            assert len(refreshes) <= 8
            return _response(NS(type="function_call",id=f"fc-note-{len(refreshes)}",call_id=f"refresh-{len(refreshes)}",name="continuity_note",arguments=json.dumps(NOTE)))
        if len(checkpoints)<2 or not post_tools:
            i=len(tool_calls); tool_calls.append(i)
            assert i < 8
            name = "write_file" if len(checkpoints)<2 and len(checkpoints) not in seeded else "terminal"
            if name == "write_file":
                seeded.add(len(checkpoints))
            if len(checkpoints)>=2:
                post_tools.append(i)
            write_path = tmp_path / f"fixture-{i}.txt"
            arguments = ({"command":"printf MARBLE_HERON_OK", "workdir":str(tmp_path)}
                         if name == "terminal" else {"path":str(write_path), "content":"verified evidence " * 15000})
            if name == "write_file":
                written_paths.append(write_path)
            response=_response(NS(type="function_call",id=f"fc-tool-{i}",call_id=f"tool-{i}",name=name,arguments=json.dumps(arguments)))
            if len(checkpoints)<2 and name == "terminal":
                response.usage=NS(input_tokens=210000,output_tokens=4,total_tokens=210004)
            return response
        return _response(_message("MARBLE HERON COMPLETE"))
    agent._interruptible_api_call=provider
    try:
        with patch("hermes_cli.plugins.invoke_hook",return_value=[]), patch("agent.turn_context._maybe_title_session_at_turn_start",return_value=None):
            result=agent.run_conversation("STEERING: Keep MARBLE HERON. Exercise tools; do not deploy.",conversation_history=source)
        if not result["completed"]:
            from agent.native_incremental_handoff import _note_fence
            state=agent._native_incremental_replay_projection
            actual=result["messages"]
            diff=[(i, {k:(str(row.get(k))[:100],str(actual[i].get(k))[:100]) for k in set(row)|set(actual[i]) if row.get(k)!=actual[i].get(k)}) for i,row in enumerate(state["replay"]) if i<len(actual) and _note_fence([row])!=_note_fence([actual[i]])]
            pytest.fail(str((result.get("final_response"),[c["model"] for c in calls],diff,[r.getMessage() for r in caplog.records if "maintenance" in r.getMessage()])))
        assert result["final_response"] == "MARBLE HERON COMPLETE"
        assert len(checkpoints)==2 and len(tool_calls)>=3 and len(refreshes)>=2 and post_tools
        assert any("TASK_STATE_AMBER" in str(request) for request in calls)
        assert written_paths and all(
            path.read_text() == "verified evidence " * 15000 for path in written_paths
        )
        assert _projection_source_for_messages(agent,result["messages"])[0] is not None
        persisted=db.get_messages_as_conversation(agent.session_id,repair_alternation=False)
        assert validate_persisted_native_compaction_history(persisted)
        fresh=NS(session_id=agent.session_id)
        assert restore_native_incremental_note(fresh,persisted) is not None
        outputs=[json.loads(m["content"]) for m in persisted if m.get("role")=="tool" and (m.get("name") or m.get("tool_name"))=="terminal"]
        assert any(o.get("exit_code")==0 and "MARBLE_HERON_OK" in o.get("output","") for o in outputs)
        assert any(
            m.get("content")=="STEERING: Keep MARBLE HERON. Exercise tools; do not deploy."
            for m in persisted
        )
        assert "MARBLE HERON" in str(persisted)
    finally:
        agent.close(); db.close()
