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
  native_incremental_compact_threshold: 32000
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
Ordinary note maintenance is a prerequisite, not a guaranteed autonomous task.

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
proof of long-term reliability. The warning-timing addition is locally tested,
not yet activated or naturally observed.

Run the relevant tests from an isolated development environment:

```sh
python -m pytest -q \
  tests/run_agent/test_native_*.py \
  tests/run_agent/test_single_trigger_native_compaction.py \
  tests/run_agent/test_context_overflow_warning_recovery.py \
  tests/agent/test_turn_context_overflow_warning.py \
  tests/gateway/test_session_hygiene.py \
  tests/gateway/test_telegram_noise_filter.py \
  tests/tui_gateway/test_compression_config_hot_reload.py
```

PR delivery, merge, runtime activation, and long-term observation are separate
states. Any future activation must use the exact reviewed candidate and preserve
profile settings and session history.
