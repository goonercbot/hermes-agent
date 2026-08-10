---
title: "Hermes Agent Skill Authoring — Use when creating, editing, or retiring Hermes skills"
sidebar_label: "Hermes Agent Skill Authoring"
description: "Use when creating, editing, or retiring Hermes skills"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Hermes Agent Skill Authoring

Use when creating, editing, or retiring Hermes skills.

## Skill metadata

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/software-development/hermes-agent-skill-authoring` |
| Version | `2.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `skills`, `authoring`, `evaluation`, `ownership`, `skill-md` |
| Related skills | [`plan`](/docs/user-guide/skills/bundled/software-development/software-development-plan), [`requesting-code-review`](/docs/user-guide/skills/bundled/software-development/software-development-requesting-code-review) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Hermes Skill Authoring

## Overview

Create, evaluate, improve, and retire Hermes skills. This skill answers one question:

> Should this skill exist, what exactly should it own, how should it be packaged for Hermes, and does evidence show that it improves agent behavior?

It applies to profile skills, repository-owned skills, and bundled Hermes skills. It is the single authoring process for all three; do not create a second metaskill for a specific agent or repository.

**Owns:** skill justification, one-job scope, triggers, boundaries, behavioral evaluation, Hermes packaging, maintenance, and retirement.

**Does NOT Own:** executing the skill's domain task, general repository work, project authority, or runtime policy. Follow the skill or project that owns those concerns.

## When to Use

Use when:

- creating a skill;
- making a meaningful behavior change to an existing skill;
- checking whether a skill is useful, correctly scoped, or colliding with another;
- deciding whether to consolidate or retire a skill.

Do not use merely to execute an existing skill, edit unrelated code or configuration, or force every repeated task into a skill. For a typo-only edit, run structural checks but skip a new behavioral baseline.

## The Six-Step Process

### 1. Identify

Before writing files:

1. Define the responsibility in one sentence. If it contains two independent jobs, split it or choose the real owner.
2. Choose one category from the current skill tree. Add a category only when no existing category fits.
3. Decide whether the skill should exist. Good reasons include a reusable non-obvious workflow, a recurring failure, a safety-critical procedure, or stable domain knowledge that changes behavior. Repetition is evidence, not a fixed gate; one serious incident may justify a skill.
4. Inspect nearby skills and real consumers. Prefer improving the existing owner over creating an overlapping sibling.
5. Write the positive trigger and a clear counter-trigger.
6. Define `Owns` and `Does NOT Own`, then name the closest skill and explain the boundary in one sentence.
7. Draft 3–6 binary acceptance checks and 3–5 realistic cases, including an edge case.

Do not create a skill when the base agent already handles the work reliably and there is no special knowledge, safety constraint, or reusable procedure to add.

**Done when:** one distinct owner, its trigger boundary, and observable proof of value are clear.

### 2. Build Minimally

Choose exactly one canonical owner:

| Skill class | Canonical location | Authoring path |
|---|---|---|
| Profile capability | Intended profile's `$HERMES_HOME/skills/` tree | `skill_manage` |
| Repository procedure | `SKILL.md` in the owning repository | repository file tools + Git |
| Bundled Hermes capability | `skills/<category>/<name>/` in Hermes Agent | repository file tools + Git |
| Hub or official package | Upstream package | supported upstream update or contribution |

Profile selection does not transfer repository ownership. Do not keep editable profile and repository copies of the same procedure.

Start with the smallest instructions that change behavior. Keep always-needed procedure in `SKILL.md`. Use only the support directories Hermes recognizes:

- `references/` for detailed or branch-specific context;
- `scripts/` for deterministic reusable operations;
- `templates/` for reusable text or configuration forms;
- `assets/` for static resources.

Do not put mutable runtime logs, credentials, user data, or arbitrary state such as `config.json` inside a committed skill package. Use normal Hermes configuration, logs, memory, databases, or repository-owned state instead.

**Done when:** the smallest coherent package exists under one owner and contains no runtime sediment.

### 3. Baseline

Before a meaningful new skill or behavior change:

1. Run all representative cases without the proposed guidance, or against the current version.
2. Judge each result only against the binary acceptance checks.
3. Record the failed checks and concrete behavior, not a vague quality score.
4. Keep structural validation separate from behavioral value; a loadable skill can still be useless.

Use `references/behavioral-evaluation.md` for the compact proposal and evaluation format.

**Done when:** the original behavior and exact failures are known before improvement.

### 4. Improve

Use a controlled loop:

1. Choose one failed behavior.
2. Make one meaningful change aimed at that failure.
3. Rerun every representative case, not only the failing one.
4. Keep the change only if the full case set improves without a material regression; otherwise revert it.
5. Record meaningful successful or failed experiments when the evidence will help future maintenance. Do not record trivial wording noise.

Stop when the acceptance checks pass, further changes add no demonstrated value, or the proposed skill does not outperform the base behavior. Do not impose universal run counts, score thresholds, or iteration quotas.

**Done when:** the final wording is supported by before-and-after behavior, not preference alone.

### 5. Review and Activate

Review both kinds of proof:

**Behavioral**

- one responsibility and one owner;
- positive and negative triggers;
- explicit `Owns` / `Does NOT Own` boundary;
- no unresolved nearby-skill collision;
- full representative case set passes the agreed binary checks;
- regressions and rejected experiments are accounted for.

**Structural**

- correct canonical location and support directories;
- valid frontmatter and size limits;
- exact changed tree inspected;
- relevant validator and repository tests pass;
- fresh discovery or loading works after deployment.

Follow the active project's authority and Git rules. Once the exact authoring work is approved, do not ask for approval after every wording iteration. New effects such as publication, merge, deployment, deletion, or production changes still follow their own authority rules.

**Done when:** evidence supports activation and the normal delivery path is complete.

### 6. Observe, Maintain, or Retire

Use real executions and user corrections to watch for:

- missed or false triggers;
- repeated overrides;
- stale commands, paths, APIs, or assumptions;
- failures not represented in the case set;
- overlap created by another skill;
- base-agent behavior catching up.

Turn meaningful failures into cases, then return to the improvement loop. Consolidate or retire a skill when another owner absorbs it, consumers disappear, its guidance becomes wrong, or it no longer improves behavior. Preserve history or rollback material according to the owning repository or profile policy; do not keep a competing active copy.

**Done when:** each active skill still earns its context and has one current owner.

## Hermes Technical Contract

Source of truth: the current Hermes loader and `tools/skill_manager_tool.py`.

### Frontmatter

```yaml
---
name: my-skill-name
description: "Use when <trigger>."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [short, useful, tags]
    related_skills: [existing-skill]
---
```

Current rules:

- `SKILL.md` starts with `---` and has a closing frontmatter fence before a non-empty body.
- Frontmatter parses as a YAML mapping with `name` and `description`.
- Names are at most 64 characters, start with a letter or number, and use lowercase letters, numbers, dots, underscores, or hyphens.
- Descriptions are at most 1,024 characters. New descriptions also fit the 60-character prompt budget. Put the complete trigger first; longer installed descriptions may display only the first 57 characters plus `...` in the skill index.
- Full `SKILL.md` content is at most 100,000 characters. Supporting files are at most 1 MiB when written through `skill_manage`.
- `version`, `author`, `license`, `platforms`, and `metadata` are conventional rather than all validator-required. Match the owning tree's peers instead of inventing metadata.

### Trigger and Body Shape

The description is routing text, not a summary. Start it with `Use when ...` and make the positive trigger understandable inside the prompt budget. Put counter-triggers in `When to Use` when they do not fit cleanly in the description.

A useful body normally includes:

1. overview and responsibility;
2. positive and negative triggers;
3. `Owns` / `Does NOT Own` boundaries;
4. actionable procedure with completion criteria;
5. pitfalls or known failure modes;
6. verification checks.

This is a quality pattern, not a reason to add empty sections.

### Authoring Tools

- Create profile skills with `skill_manage(action='create')` and an explicit category when useful.
- Patch existing skills with `skill_manage(action='patch')`; use `edit` only for a real full rewrite.
- Add support files with `skill_manage(action='write_file')` or repository file tools.
- Create repository and bundled skills with repository file tools, then use the repository's required Git and test workflow.
- Inspect neighboring skills with `skills_list`, `skill_view`, and bounded repository searches.
- A running session may cache discovery. Verify a new or renamed skill in a fresh session or through the supported reload/update path.
- `hermes update` syncs changed bundled skills across opted-in profiles while preserving user-modified copies. Do not overwrite a customized profile copy silently.

## Common Pitfalls

1. Creating a new metaskill instead of improving this owner.
2. Treating “used three times” as a universal creation requirement.
3. Passing YAML validation and calling the skill effective.
4. Testing only the case that motivated the last edit.
5. Broadening the trigger until neighboring skills collide.
6. Keeping both old and replacement skills active after consolidation.
7. Putting mutable logs, credentials, or user state in the skill directory.
8. Growing `SKILL.md` instead of moving optional detail behind a support-file pointer.
9. Recording every editorial tweak rather than meaningful evidence.
10. Applying fixed scoring or lifecycle schedules to every kind of skill.

## Verification Checklist

- [ ] One-sentence responsibility, category, and should-exist judgment are explicit.
- [ ] Positive trigger, counter-trigger, `Owns`, `Does NOT Own`, and nearest collision are clear.
- [ ] There are 3–6 binary checks and 3–5 representative cases including an edge case.
- [ ] A pre-change baseline exists for meaningful behavior changes.
- [ ] Each meaningful iteration changed one thing and reran all cases.
- [ ] Final behavior improved without a material regression.
- [ ] One canonical owner, correct Hermes placement, valid frontmatter, and progressive disclosure are verified.
- [ ] Relevant tests, exact-tree review, and fresh loading or discovery passed.
- [ ] No obsolete runtime machinery, arbitrary thresholds, or duplicate authoring owner was introduced.
