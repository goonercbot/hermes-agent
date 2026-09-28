Bukayo standard-worker preservation map

Source examined

- Release base: f97608f178d1ffeca59860195ab7da295f7c8e5f.
- Archive port source: 917e94c4e354db0c88f7ab213bfdc07723a001fa.
- Original local delta: 2237be355906fbe6065ce1815711eee52b2d646e..ecc7fcaafbf2f9925ccc8ab3cbe1735e52e784e7.
- Archive preservation map covered 45 production Python paths. Every original group is accounted for below.

Retained: immutable usage ledger and SQLite deadlines

- hermes_state_common.py: credential-free billing origin sanitizer; immutable usage_events and usage_event_meta tables, immutability triggers, and indexes.
- hermes_state_schema.py: one-time baseline from existing session_model_usage rows is acquired with BEGIN IMMEDIATE before marker inspection; baseline reads/inserts and both metadata markers commit together or roll back together. A partial marker pair fails closed rather than mixing snapshots. Existing first_seen/last_seen are retained only as an interval when both existed; no historical point timestamp is fabricated.
- hermes_state_usage.py: unique admitted direct and queued delta rows; every nonzero signed admitted part survives coalescing even if the aggregate is zero; signed absolute reconciliation, true no-op suppression, auxiliary task/role/interval rows, and sanitized billing evidence are retained.
- hermes_state.py: queue coalescing preserves each event identity and routing fields. Its retained BEGIN IMMEDIATE deadline patch restores the original busy_timeout on success, callback failure, failed acquisition, reopen, and later maintenance.
- hermes_state_messages.py: archived sealed-resume addition is excluded. Official message persistence remains unchanged.
- Official FTS3/FTS migration remains the release implementation; the ledger initialization occurs after existing data migration and FTS setup ordering.

Retained: callback compatibility

- hermes_cli/plugins.py: manager-owned legacy Telegram callback prefix registration, authorization, unload/reload compatibility, consumer compatibility, and None thread IDs.
- agent/tool_executor.py and tools/continuity_note_tool.py: excluded because their archive deltas solely support the retired custom continuity tool.

Retained: gateway delivery and lifecycle

- gateway/config.py and gateway/config_loader.py: separate home-startup notification opt-out. The no-home setup prompt remains DM/private-only; group and forum sources never receive it.
- gateway/run_notifications.py and gateway/run_shutdown.py: pending restart notices respect that home opt-out while direct restart delivery remains independent.
- gateway/platforms/base.py, gateway/run_turn.py, gateway/run_turn_runner.py, gateway/stream_consumer.py, gateway/stream_consumer_fallback.py, and gateway/turn_context.py: cleanup only after the same generation has evidence that the exact complete final body reached the user. A direct final records that evidence at the retry/fallback owner: short plain fallbacks qualify, but deliberate 3,500-character truncation and partial/split retry paths do not. Streamed cleanup requires either an exact consumer match or a successful full reconciliation edit; legacy already-sent/streamed markers, missing edit acknowledgements, failed edits, and reused event evidence cannot authorize cleanup. Failed final sends, exceptions, partial/split streams, and interim-as-final bubbles remain; direct-send settlement precedes cleanup and unrelated post-delivery callbacks retain their ordering.
- gateway/run.py and gateway/session_transcript.py: archive native continuity/hygiene deltas excluded; official release routing and transcript behavior remain unchanged.

Retained: delegation ownership

- tools/delegate_tool_dispatch.py: child work does not inherit or take over a gateway route. Gateway ownership checks retain release generation/concurrency behavior and reject foreign or child completion during /new and /stop races while allowing legitimate same-owner and verified continuation routes.

Archived custom compression/continuity groups deliberately not ported

- Native authenticated continuity and maintenance: agent/native_incremental_handoff.py, agent/native_note_refresh.py, agent/native_compaction_progress.py, and related tests.
- Native request, publication, and continuation: archive deltas in agent/agent_init.py, agent/agent_runtime_helpers.py, agent/chat_completion_helpers.py, agent/chat_completion_nonstream.py, agent/codex_runtime.py, agent/conversation_compression.py, agent/conversation_loop.py, agent/manual_compression_feedback.py, agent/turn_api_call.py, agent/turn_api_request.py, agent/turn_context.py, agent/turn_context_compaction.py, agent/turn_iteration_prep.py, agent/turn_preflight.py, and agent/turn_request_assembly.py.
- Responses transport and pairing: archive deltas in agent/codex_responses_adapter.py, agent/transports/codex.py, and agent/message_sanitization.py.
- Configuration/discoverability: archive deltas in hermes_cli/config_defaults.py, hermes_cli/tools_config.py, and toolsets.py.
- Fleet router fixtures/plugin and all continuity_note tool/toolset, host integration, capability, maintenance, projection, and publication wiring.

Official compression proof retained

- Existing upstream native compaction, summary/nontext retention, message conversion, disabled-feature, persistence, and ordinary context-compressor tests ran unchanged. No custom compression module, tool, toolset, prompt, flag, route-plugin integration, or replacement runner was added.

Regression evidence

- tests/hermes_state/test_usage_event_ledger.py: direct/queued/absolute reconciliation, immutable rows, one-time baseline with preserved sessions/messages, first/second-marker rollback and retry, concurrent initializers, signed net-zero coalescing, true no-op suppression, interval auxiliary evidence, sanitized URL, and reload marker retention.
- tests/hermes_state/test_write_lock_patience.py: timeout restoration on normal writes, callback failure, failed BEGIN deadline, reopened writer, and direct VACUUM.
- tests/test_bukayo_review_reproductions.py, tests/test_bukayo_complete_delivery_reproductions.py, and retained owner tests: failed-result and raised-final preservation; full direct, short fallback, exact-stream, and successful transformed-edit cleanup; truncated fallback, failed transformed edit plus failed normal final, partial fallback failure, unknown legacy marker, interim matching bubble retention, stale generation isolation, and chained callback continuity; DM/private positive controls and group/forum no-home notice rejection; plus Telegram callbacks, lifecycle/restart notices, gateway ownership, and delegate child ownership.

Scope exclusions

No profile, config, service, live database, credential, provider, installation, activation, commit, push, PR, or merge action was taken.
