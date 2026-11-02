"""Actual native private DM notices remain live; groups/topics are not redirects."""
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, HomeChannel
from gateway.platforms.base import SendResult
from tests.gateway.restart_test_helpers import make_restart_runner


@pytest.mark.asyncio
async def test_startup_and_shutdown_keep_private_dm_not_group():
    runner, adapter = make_restart_runner()
    cfg = runner.config.platforms[Platform.TELEGRAM]
    cfg.extra['lifecycle_private_recipients'] = ['42', '-42']
    cfg.home_channel = HomeChannel(platform=Platform.TELEGRAM, chat_id='42', name='Synthetic owner')
    assert await runner._send_home_channel_startup_notifications() == {('telegram', '42', None)}
    await runner._notify_active_sessions_of_shutdown()
    assert [call[0] for call in adapter.sent_calls] == ['42', '42']
    adapter.sent_calls.clear()
    cfg.home_channel.chat_id = '-42'
    assert not await runner._send_home_channel_startup_notifications()
    await runner._notify_active_sessions_of_shutdown()
    assert not adapter.sent_calls


@pytest.mark.asyncio
@pytest.mark.parametrize('target', [
    {'platform': 'telegram', 'chat_id': '-42'},
    {'platform': 'telegram', 'chat_id': '42', 'thread_id': '9'},
    {'platform': 'telegram', 'chat_id': '42', 'chat_type': 'group'},
])
async def test_initiator_cannot_route_to_group_or_topic(target, tmp_path):
    import json
    runner, adapter = make_restart_runner()
    runner.config.platforms[Platform.TELEGRAM].extra['lifecycle_private_recipients'] = ['42', '-42']
    (tmp_path / '.restart_notify.json').write_text(json.dumps(target))
    with patch('gateway.run._hermes_home', tmp_path):
        assert await runner._send_restart_notification() is None
    assert not adapter.sent_calls


@pytest.mark.parametrize('policy', [[], None, '42', [True], ['42', None]])
@pytest.mark.asyncio
async def test_invalid_policy_denies(policy, tmp_path):
    import json
    runner, _ = make_restart_runner()
    runner.config.platforms[Platform.TELEGRAM].extra['lifecycle_private_recipients'] = policy
    (tmp_path / '.restart_notify.json').write_text(json.dumps({'platform': 'telegram', 'chat_id': '42'}))
    with patch('gateway.run._hermes_home', tmp_path):
        assert await runner._send_restart_notification() is None


@pytest.mark.asyncio
async def test_authorized_initiator_and_absent_policy_native_control(tmp_path):
    import json
    runner, _ = make_restart_runner()
    with patch('gateway.run._hermes_home', tmp_path):
        for chat in ('-42', '42'):
            if chat == '42':
                runner.config.platforms[Platform.TELEGRAM].extra['lifecycle_private_recipients'] = ['42']
            (tmp_path / '.restart_notify.json').write_text(json.dumps({'platform': 'telegram', 'chat_id': chat}))
            assert await runner._send_restart_notification() == ('telegram', chat, None)


@pytest.mark.asyncio
async def test_private_marker_retry_then_removal(tmp_path):
    import json
    runner, adapter = make_restart_runner()
    runner.config.platforms[Platform.TELEGRAM].extra['lifecycle_private_recipients'] = ['42']
    marker = tmp_path / '.restart_notify.json'
    data = {'platform': 'telegram', 'chat_id': '42'}
    marker.write_text(json.dumps(data))
    adapter.send = AsyncMock(side_effect=[SendResult(success=False), SendResult(success=True)])
    with patch('gateway.run._hermes_home', tmp_path):
        await runner._send_restart_notification()
        assert marker.exists()
        await runner._send_restart_notification()
        assert not marker.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize('target,count', [('42', 1), ('43', 0), ('-42', 0), ('42:9', 0)])
async def test_cron_result_route_unchanged_but_lifecycle_private(target, count):
    runner, adapter = make_restart_runner()
    runner.config.platforms[Platform.TELEGRAM].extra['lifecycle_private_recipients'] = ['42', '-42']
    job = {'id': 'fixture', 'deliver': 'telegram:' + target}
    from cron.scheduler import _resolve_delivery_targets
    result_targets = _resolve_delivery_targets(job)
    assert result_targets
    with patch('cron.jobs.get_job', return_value=job):
        assert await runner._notify_interrupted_cron_jobs(['fixture']) == count
    assert _resolve_delivery_targets(job) == result_targets
    assert len(adapter.sent_calls) == count


@pytest.mark.asyncio
async def test_cron_interruption_snapshot_keeps_owner_after_registration_release(tmp_path, monkeypatch):
    import json
    from cron import scheduler
    from cron.scheduler_provider import _profile_cron_scope
    from gateway.config import GatewayConfig, PlatformConfig
    from hermes_constants import get_hermes_home
    from tests.gateway.restart_test_helpers import RestartTestAdapter
    primary, secondary = tmp_path / 'primary', tmp_path / 'secondary'
    for home in (primary, secondary):
        home.mkdir()
        (home / 'config.yaml').write_text(json.dumps({}))
    monkeypatch.setenv('HERMES_HOME', str(primary))
    runner, adapter = make_restart_runner()
    other = RestartTestAdapter()
    runner._primary_profile_name = 'default'
    runner._served_profile_homes = {'secondary': secondary}
    runner._profile_adapters = {'secondary': {Platform.TELEGRAM: other}}
    runner._profile_configs = {'secondary': GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(
        extra={'lifecycle_private_recipients': ['84']})})}
    runner.config.platforms[Platform.TELEGRAM].extra['lifecycle_private_recipients'] = ['42']
    with _profile_cron_scope(secondary):
        assert scheduler.try_register_running_job('shared-fixture')
    try:
        notices = scheduler.mark_running_jobs_interrupted('Synthetic shutdown', include_homes=True)
    finally:
        scheduler.release_running_job('shared-fixture', home=secondary)
    reads = []

    def get_job(job_id):
        reads.append((job_id, get_hermes_home()))
        return {'id': job_id, 'deliver': 'telegram:42,telegram:84,telegram:-99'}

    with patch('cron.jobs.get_job', side_effect=get_job):
        assert await runner._notify_interrupted_cron_jobs(notices) == 1
    assert reads == [('shared-fixture', secondary)]
    assert [call[0] for call in other.sent_calls] == ['84']
    assert not adapter.sent_calls and get_hermes_home() == primary
    runner._profile_adapters = {}
    with patch('cron.jobs.get_job', side_effect=get_job):
        assert await runner._notify_interrupted_cron_jobs(notices) == 0
    assert not adapter.sent_calls
