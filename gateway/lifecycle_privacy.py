"""Optional exact private lifecycle recipients, independent of result routing.

Retained v0.21.6 private notice/R2 protection; no guessed destination or redirect.
Absent policy retains native behavior for profiles outside the approved rollout.
"""
import re

from gateway.config import Platform


def lifecycle_target_allowed(config, platform, chat_id, thread_id=None, *, chat_type=None):
    extra = getattr(config, 'extra', None) or {}
    if 'lifecycle_private_recipients' not in extra:
        return True
    recipients = extra['lifecycle_private_recipients']
    if not isinstance(recipients, list) or not recipients or thread_id not in (None, ''):
        return False
    if any(isinstance(value, bool) or not isinstance(value, (str, int)) for value in recipients):
        return False
    target = str(chat_id)
    if target not in {str(value) for value in recipients}:
        return False
    if platform == Platform.TELEGRAM:
        return bool(re.fullmatch(r'[1-9][0-9]*', target)) and chat_type not in {'group', 'supergroup', 'channel'}
    return chat_type in {'dm', 'private'}


def _cron_notice_owner(runner, notice):
    from pathlib import Path
    from hermes_constants import get_routing_process_hermes_home, hermes_home_key
    home = get_routing_process_hermes_home()
    if isinstance(notice, str):
        return notice, home, None, runner.config
    if not isinstance(notice, dict) or not notice.get('job_id') or not notice.get('profile_home'):
        return None
    target = Path(notice['profile_home'])
    if hermes_home_key(target) == hermes_home_key(home):
        return notice['job_id'], target, None, runner.config
    for profile, served in (getattr(runner, '_served_profile_homes', None) or {}).items():
        if hermes_home_key(target) == hermes_home_key(served):
            config = (getattr(runner, '_profile_configs', None) or {}).get(profile)
            return (notice['job_id'], target, profile, config) if config is not None else None
    return None


async def notify_interrupted_cron_jobs(runner, notices):
    """Use interruption-time owning store/config/bot even after worker release."""
    import logging
    from cron.jobs import get_job, use_cron_store
    from cron.scheduler import _resolve_delivery_targets
    from agent.i18n import t
    from gateway.run import _async_profile_runtime_scope
    from gateway.run_shutdown import _notice_target_key
    from gateway.warning_notifications import present_notification
    from hermes_constants import hermes_home_key
    logger = logging.getLogger(__name__)
    notified = set()
    for notice in notices or ():
        try:
            owner = _cron_notice_owner(runner, notice)
            if owner is None:
                continue
            job_id, home, profile, config = owner
            async with _async_profile_runtime_scope(home):
                with use_cron_store(home):
                    job = get_job(job_id)
                    if not job:
                        continue
                    targets = _resolve_delivery_targets(job, for_failure=True)
                action = t('gateway.shutdown.action_restarting' if runner._restart_requested else 'gateway.shutdown.action_shutting_down')
                message = t('gateway.shutdown.cron_interrupted', job=job.get('name') or job_id, action=action)
                for target in targets or ():
                    try:
                        platform = Platform(str(target.get('platform', '')).lower())
                    except (ValueError, AttributeError):
                        continue
                    cfg = config.platforms.get(platform)
                    chat, thread = str(target.get('chat_id')), target.get('thread_id')
                    if not lifecycle_target_allowed(cfg, platform, chat, thread, chat_type=target.get('chat_type')):
                        continue
                    adapter = runner._authorization_adapter(platform, profile)
                    if adapter is None or not runner._notice_allowed(platform, 'cron interrupt', cfg):
                        continue
                    key = (hermes_home_key(home), job_id, *_notice_target_key(platform.value, chat, thread))
                    if key in notified:
                        continue
                    try:
                        metadata = runner._thread_metadata_for_target(platform, chat, thread, adapter=adapter)
                    except Exception:
                        continue

                    async def send_notice():
                        if await runner._send_notice_logged(adapter, chat, message, platform.value,
                            'Cron interrupt notice to %s:%s failed: %s', metadata=metadata):
                            notified.add(key)
                    await present_notification(send_notice, platform=platform)
        except Exception:
            logger.debug('Cron interruption notice unavailable', exc_info=True)
    return len(notified)
