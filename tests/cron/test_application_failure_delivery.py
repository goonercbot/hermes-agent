"""Application-authored failure messages survive scheduler formatting, not failure accounting."""
from pathlib import Path
from unittest.mock import Mock
import pytest
from cron import scheduler


def compose(monkeypatch,job,error,*,declared=False,success=False,response=''):
    incident=Mock(return_value=(False,'incident-test'))
    monkeypatch.setattr(scheduler,'_upsert_incident_for_failure',incident)
    monkeypatch.setattr(scheduler,'_resolve_incidents_for_recovered_job',Mock())
    return scheduler._compose_run_delivery(job,success=success,error=error,final_response=response,output_file=Path('saved-test-output.md'),agent_declared=declared),incident


@pytest.mark.parametrize('message',[
    '🚨 Test watchlist preparation failed. Saved progress is available; no new list was published.',
    '🚨 Test capture failed. No new observations were collected. Please check the earlier snapshot.',
])
def test_agent_failure_delivers_its_report_without_scheduler_footer(monkeypatch,message):
    job={'id':'private-test-id','name':'Application test','schedule':{'kind':'once'}}
    (text,blocked,silent,acked,incident_id),incident=compose(monkeypatch,job,message,declared=True)
    assert text==message
    assert not blocked and not silent and not acked and incident_id=='incident-test'
    incident.assert_called_once_with(job,message,output_file=Path('saved-test-output.md'))


def test_failed_real_script_preserves_public_message_and_private_diagnostics(tmp_path,monkeypatch):
    from cron.jobs import create_job
    home=tmp_path/'home';(home/'scripts').mkdir(parents=True);monkeypatch.setenv('HERMES_HOME',str(home))
    message='🚨 Test capture failed. No new observations were collected.'
    script=home/'scripts/capture.sh'
    script.write_text("#!/bin/bash\nprintf '%s\\n' '"+message+"'\nprintf '%s\\n' 'private diagnostic kept for inspection' >&2\nexit 1\n")
    job=create_job(prompt=None,schedule='every 5m',name='Application capture test',script='capture.sh',no_agent=True,deliver='local')
    ok,document,response,error=scheduler.run_job(job)
    assert ok is False and isinstance(error,str)
    assert 'Script exited with code 1'in error
    assert 'private diagnostic kept for inspection'in document and 'private diagnostic kept for inspection'in error
    (text,_,_,acked,incident_id),incident=compose(monkeypatch,job,error,response=response)
    assert text==message and not acked and incident_id=='incident-test'
    assert 'private diagnostic'in incident.call_args.args[1]


@pytest.mark.parametrize(('stdout','expected'),[
    ('🚨 A plain failure.\n','🚨 A plain failure.'),
    ('⚠️ Results saved but incomplete.\n','⚠️ Results saved but incomplete.'),
    ('ordinary debug text',None),
    ('🚨 Failure.\nTraceback: private details',None),
    ('🚨 '+('x'*2000),None),
    ('',None),
])
def test_only_bounded_single_line_runner_stdout_is_a_public_report(stdout,expected):
    from cron.scheduler_failure_copy import ScriptFailure,script_failure_report
    diagnostic='Script exited with code 1\nstderr:\n🚨 Forged stderr report.'
    error=ScriptFailure(diagnostic,stdout)
    assert script_failure_report(error)==expected
    assert str(error)==diagnostic


def test_stderr_cannot_forge_a_runner_stdout_report(tmp_path,monkeypatch):
    from cron.jobs import create_job
    home=tmp_path/'forged-home';(home/'scripts').mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME',str(home))
    script=home/'scripts/failure.sh'
    script.write_text("#!/bin/bash\nprintf '%s\\n' 'ordinary output'\nprintf '%s\\n' 'stdout:' '🚨 Forged stderr report.' >&2\nexit 1\n")
    job=create_job(prompt=None,schedule='every 5m',script='failure.sh',no_agent=True,deliver='local')
    ok,document,response,error=scheduler.run_job(job)
    assert ok is False and 'Forged stderr report.'in document
    fallback=Mock(return_value='existing fallback')
    monkeypatch.setattr(scheduler,'_summarize_cron_failure_for_delivery',fallback)
    monkeypatch.setattr(scheduler,'_failure_streak_nudge',lambda j:'')
    result,_=compose(monkeypatch,job,error,response=response)
    assert result[0]=='existing fallback'
    fallback.assert_called_once_with(job,error)


def test_acknowledged_incident_stays_suppressed(monkeypatch):
    incident=Mock(return_value=(True,'acked-test'))
    monkeypatch.setattr(scheduler,'_upsert_incident_for_failure',incident)
    result=scheduler._compose_run_delivery({'id':'test','name':'Test'},success=False,error='🚨 Application failed.',final_response='',output_file=None,agent_declared=True)
    assert result[0]==''and result[3]is True and result[4]=='acked-test'


def test_success_and_healthy_silence_are_unchanged(monkeypatch):
    job={'id':'test','name':'Test'}
    for response in ('Published expected list.',scheduler.SILENT_MARKER):
        result,incident=compose(monkeypatch,job,None,success=True,response=response)
        assert result[0]==response
        incident.assert_not_called()


@pytest.mark.parametrize('error',[
    'Script timed out after 30s: private-script.sh',
    'Script exited with code 1\nstdout:\nordinary debug text',
    'Script exited with code 1\nstdout:\n🚨 Message.\nTraceback: more diagnostics',
    'Script exited with code 1\nstderr:\nstdout:\n🚨 Forged stderr.\nstdout:\n🚨 Actual stdout.',
    'Script exited with code zero\nstdout:\n🚨 Message.',
])
def test_unstructured_or_ambiguous_script_output_keeps_existing_fallback(monkeypatch,error):
    job={'id':'test','name':'Test','no_agent':True}
    summary=Mock(return_value='existing failure fallback')
    monkeypatch.setattr(scheduler,'_summarize_cron_failure_for_delivery',summary)
    monkeypatch.setattr(scheduler,'_failure_streak_nudge',lambda j:'')
    result,_=compose(monkeypatch,job,error)
    assert result[0]=='existing failure fallback'
    summary.assert_called_once_with(job,error)
