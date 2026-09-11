# v0.21.1 local-addition migration

## Scope

Target: Hermes v2026.9.7 / v0.21.1 (`2237be355906fbe6065ce1815711eee52b2d646e`). Native-continuity source: `317d35c8defbfa693397adf8de492413a48d29b2`, including publication lifecycle repairs, not the earlier note-readback snapshot.

This branch is a preservation port onto the new upstream architecture. It is not a fleet activation. Only Riccardo is authorized for a canary after integrated acceptance; promotion to the other profiles requires a separate decision. Merging and activation are separate from PR delivery.

## Additions carried forward

- **Timestamped immutable usage ledger.** Ported into the upstream state mixins. Preserves historical sessions/messages, existing ledger rows and metadata, one-time baseline seeding, signed corrections, queued/direct updates and absolute reconciliation. Historical timestamp-free usage remains unknown for weekly reporting; the migration does not fabricate timestamps or paid costs.
- **Telegram callback compatibility.** Manager-owned prefix handlers retain authorization, targeted unload/reload and existing consumer contracts. Existing Gooner and William plugin files are not rewritten.
- **Commentary and delivery safeguards.** Interim commentary is removed only after successful final delivery; failed-final and interim-as-final cases are preserved. Post-delivery callback ordering and DM-only lifecycle notification safeguards remain explicit.
- **Authenticated incremental native continuity.** Maintenance notes are bound to durable source history, with a request-scoped capability, middleware enforcement, bounded failure and restoration of ordinary tools. Native checkpoint, historical handoff and protected tail remain distinct from mutable provider-wire projections.
- **Replay, publication and steering.** Source projection survives aggregate tool budgeting, source growth, session rotation/in-place publication and SQLite readback. Native steering is a durable appended user event rather than a mutation of an already-authenticated tool result. Gateway history and hygiene preserve native ownership.
- **Native progress and cancellation.** Request-owned progress, cancellation properties and deadline handling are adapted to the new compression fence and transport architecture. Ordinary non-native behavior remains covered separately.

## Settings and operational boundaries

- Native incremental behavior is opt-in; no other profile is enabled by this port.
- Preserve existing model/provider/fallback configuration, credentials, plugin selections, journal modes and schedules.
- Preserve the safe SQLite interpreter requirement; a passing source suite does not prove packaged module origins or database compatibility.
- The default gateway remains retired. The shared CLI may still serve scheduled scripts and must not be replaced implicitly.
- Declan's router-disabled state is intentional. The fleet-local router is not an upstream/default policy change.

## Acceptance required before activation

1. Final integrated tests, including ordinary disabled behavior, immutable ledger, actual plugin middleware, native refresh denial/timeout, protected replay and fresh SQLite readback.
2. Isolated real-provider pressure through actual file tools and the child-agent constructor, followed by authenticated checkpoint, measured reduced provider context, fresh note, ordinary terminal execution and fresh-process continuation. Synthetic transport fixtures alone do not satisfy this requirement.
3. Exact-candidate independent review and verified PR delivery.
4. Packaged module-origin/dependency checks and a supported Riccardo-only lifecycle handoff with durable receipt. Read back the actual loaded runtime and preserved session before claiming activation.
5. Real use on Riccardo; fleet promotion remains gated on AJ's confirmation.

This document describes scope and acceptance, not a claim that the candidate is deployed or that every acceptance step has passed.
