# Native incremental continuity

## Purpose and scope

This opt-in Codex route creates a provider-native checkpoint using one Luna
Responses operation and a continuity note recorded during ordinary work. It
does not run a separate full-history main-model summarization pass. The main
model remains independently selected. The existing compression attempt budget,
provider watchdogs, durable commit, session lease, and replay validation remain
in force.

The feature is disabled by default. This change does not migrate profiles,
change credentials, restart services, or enable a fleet rollout.

## Configuration

```yaml
compression:
  native_incremental_handoff: true
  native_incremental_model: gpt-5.6-luna
  native_incremental_compact_threshold: 128000
```

The supported route is `openai-codex` with `api_mode: codex_responses`. The
compression model must be available to the account. Enable the `continuity`
toolset explicitly on each intended surface, for example:

```sh
hermes tools enable continuity --platform telegram
```

The tool is unavailable when the feature flag is off. Tool search can expose
it through its deferred catalogue. The host's existing compression threshold
still decides when to attempt compression; `native_incremental_compact_threshold`
is the provider's inline checkpoint setting, not a second host retry trigger.

The default is 128,000 tokens. The former 32,000 setting can trigger repeated
compaction passes within one request when the resulting context still exceeds
that target. Existing explicit settings are respected: updating code alone
does not migrate a profile pinned to 32,000. Any profile-setting change needs
its own activation approval. The host trigger and idle/total limits are unchanged.

## Ordinary-work note

`continuity_note` accepts `objective`, `current_plan`, `next_action`, and optional
`blockers`. Record changes to the objective, accepted plan, verified position,
or next action during normal work. The executor binds the agent-authored fields
to the exact source prefix and persists the paired tool result. There is no
note-generation inference at compression time.

Later messages preserve that prefix and remain protected. Newer user
instructions override historical note claims. Fresh agents restore a recorded
note against canonical history; host-cleaned gateway replay is separately
bound to that source. When a compression lease adopts a newer canonical read,
the note and publication watermark are rebound to that same read.

Missing or stale notes leave the source unchanged. A missing note is not an
instruction to discard history or silently invoke the slow main-model handoff.
When evidence added after the authenticated cursor exceeds 128,000 serialized
characters, the next ordinary Responses request is restricted to the existing
`continuity_note` tool. This is a local maintenance guard, not another provider
token setting or a separate full-history summary-model pass. No unrelated tool
may execute in that response. Normal tools resume on the next request.

The host binds that maintenance request to one opaque, request-local capability.
Every `pre_tool_call` plugin still runs. The host passes the capability through
`maintenance_context`, separate from tool arguments; the exported pure validator
lets a routing policy exempt only its own route-first rule for this exact note.
Independent security, approval and argument-rewrite policies remain effective. The
assistant call and authenticated tool result are written as one database batch,
then read back and revalidated before the cursor is published. Malformed,
duplicated, blocked, interrupted, non-durable, or unchanged maintenance fails
locally, leaves no partial synthetic pair, and does not enter provider retries
or generic-chat fallback. Maintenance prose is never delivered as a user answer.

An old valid note with that much uncovered evidence defers native compaction
before spending a Luna request. The authenticated note tool clears only the
readiness-related structural retry pause; failure cooldowns and ineffective
compression guards remain intact. Failed maintenance cannot repeat against the
same note within one user turn. Transport retries retain their existing budget.

After refresh, keep all post-note rows, the latest user instruction, and the
latest complete/pending tool group verbatim. Do not retain every already-covered
intermediate row merely because the latest user message preceded the note.
Large required user input stays protected and does not itself demand endlessly
repeated notes. Past rows remain in the canonical archive after compression.

The gateway `/compress` helper restores the authenticated note before invoking
the native route, like automatic hygiene. It does not mint or renew a note to
bypass a stale source fence. A blocked no-op reports its reason rather than only
"No changes". Post-note rows remain protected, so an old note can still prevent
useful reduction; record current progress during ordinary work instead of
weakening history validation.

## Request and replay safety

Inline `context_management` is still an ordinary Responses inference turn.
Keep normal assistant instructions when requesting a checkpoint: maintenance
instructions such as "do not answer" or "reply with an acknowledgement only"
can survive inside opaque state and contaminate later replies. A request-local
resume instruction ends legacy maintenance scope only for validated native
incremental checkpoints; it does not rewrite the cached prompt or ciphertext.

Persist the latest valid checkpoint, supported provider suffix, recorded note,
and complete protected live tool groups. Exact duplicate stream deliveries are
deduplicated. Conflicting IDs, invalid ciphertext, unsupported suffix shapes,
invalid routes, and provider errors fail closed without replacing source history.
No-checkpoint responses retain the input and suppress identical immediate retries.
Transport retries remain separately bounded by the existing provider policy.

Native Responses replay preserves validated checkpoint bytes. Generic-chat
fallback instead receives a disposable sanitized projection with checkpoint
carriers and native reasoning sidecars removed; the handoff and ordinary suffix
remain. Durable checkpoint validation is never applied to that repaired chat
projection as though it were the immutable original.

These provenance checks defend against transcript lookalikes and accidental
mutation, not an actor authorized to rewrite the entire local database.

## Blocked-context warning timing

During a public conversation turn, transient `cooldown` and `structural_backoff`
warnings below the model's hard context limit are held until the turn exits.
The current usage and compression block are rechecked before delivery. Successful
recovery or an expired block cancels the stale warning; an unresolved block is
still delivered and deduplicated. Exception, timeout, and interruption exits
also perform the check.

Permanent blockers and pressure at or above the hard model limit warn
immediately. Calls outside the conversation scope retain immediate warning
behavior. This is not a blanket gateway noise filter and does not change
compression admission or retry policy.

## Verification and evidence limits

Regression coverage includes deferred note dispatch, canonical/replay binding,
rotation and in-place persistence, fresh-process restoration, malformed-tail
rejection, real chat-fallback transport switching, and the complete
missing-note → note tool → native checkpoint → normal reply path. Provider
responses in these regression tests are fixtures; no production session is used.

One isolated real-provider copied-history test completed compression in 9.684
seconds and compression plus a correct first reply in 17.618 seconds. Another
turn and a fresh-process reload retained the required facts. A subsequent
natural Riccardo-only compression committed in 52.133 seconds and ordinary
Astra replies resumed. These are individual observations, not a benchmark or
proof of long-term reliability. Those observations predate the progress,
manual-command, and compression-target corrections described below.

Subsequent isolated timing investigation distinguished provider recursion from
the host progress bug: one historical 180-row source at 32,000 emitted repeated
checkpoints and reached the 600-second total ceiling. The same source and prompt
at 128,000 emitted one checkpoint and completed in 102.775 seconds, but no
smaller candidate was committed; its old note left most history protected.
That diagnostic is not a successful compression canary.

A separate real 500-row source with a current authenticated note, at 128,000,
compressed in 55.115 seconds; its subsequent Astra call took 3.231 seconds and
correctly recalled the requested periods and layout restriction. Source rows
were read back unchanged and private databases removed. These observations
support the target correction, not an unconditional latency guarantee. The
progress-fence correction separately keeps genuine native stream activity from
being cancelled by the generic idle timer; real stalls and total ceilings remain.

Run the relevant tests from an isolated development environment:

```sh
scripts/run_tests.sh \
  tests/run_agent/test_native_*.py \
  tests/run_agent/test_single_trigger_native_compaction.py \
  tests/run_agent/test_context_overflow_warning_recovery.py \
  tests/agent/test_turn_context_overflow_warning.py \
  tests/gateway/test_session_hygiene.py \
  tests/gateway/test_telegram_noise_filter.py \
  tests/tui_gateway/test_compression_config_hot_reload.py -q
```

The note-refresh integration tests default to the repository-relative test-only
snapshot in `tests/fixtures/native_maintenance_router/`. This is the complete,
byte-exact actual router candidate, not a rewritten policy or runtime installation.
Its `provenance.json` records the source repository, base, staged tree, relative
path and SHA256. Missing source/provenance or a digest mismatch fails closed.

Run the portable bundled regression with:
`bash scripts/run_tests.sh tests/run_agent/test_native_note_refresh.py tests/run_agent/test_native_note_refresh_capability.py`.
For explicit external-candidate integration, append
`-- --native-router-source=/path/to/router-checkout/plugins/fleet-task-router/__init__.py`.
The `--` passthrough is required. A missing explicit source fails; it never falls
back to the bundle. The old `HERMES_TEST_ROUTER_SOURCE` environment override is
not supported: the canonical runner strips it, so setting it cannot prove which
source ran. Refresh the snapshot and provenance together whenever the reviewed
router pin changes; verify byte equality and SHA256 against that exact staged
source, then run both bundled and explicit integration before acceptance.

Provider replies are fixtures; routed ordinary terminal execution is real.
Live automatic compression and Telegram continuation require separate acceptance
after exact approval.

PR delivery, merge, runtime activation, and long-term observation are separate
states. Any future activation must use the exact reviewed candidate and preserve
profile settings and session history.
