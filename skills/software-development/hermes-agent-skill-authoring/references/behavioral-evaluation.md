# Behavioral Evaluation

Use this only for meaningful skill creation, behavior changes, consolidation, or retirement. A typo-only edit does not need a new evaluation record.

## Proposal

```markdown
Skill: <name>
Responsibility: <one sentence>
Category: <one existing category>
Should exist because: <reusable value, failure, safety, or stable knowledge>
Positive trigger: <when it must load>
Counter-trigger: <when it must not load>
Owns: <one domain>
Does NOT Own: <adjacent domain>
Closest skill: <name>
Boundary: <one sentence>
```

## Binary Acceptance Checks

Write 3–6 checks. Each check must be observable, independent, and answerable with pass or fail.

Good:

```markdown
- [ ] The agent checks the live database before recommending a repair.
- [ ] The agent reports the exact affected row count before mutation.
```

Bad:

```markdown
- [ ] The answer is high quality.
- [ ] The agent is careful.
```

## Representative Cases

Write 3–5 realistic inputs. Include one edge case and, when relevant, one counter-trigger that should not load the skill.

```markdown
| Case | Why it matters | Expected behavior |
|---|---|---|
| Normal | Main workflow | <observable result> |
| Failure | Common failure path | <observable result> |
| Edge | Boundary or safety case | <observable result> |
| Counter-trigger | Nearby but out of scope | <other owner or base behavior> |
```

## Baseline and Result

Run the complete case set before the meaningful change and after every meaningful iteration.

```markdown
| Case | Check 1 | Check 2 | Check 3 | Concrete failure or evidence |
|---|---|---|---|---|
| Normal | pass | fail | pass | <what happened> |
```

Do not reduce the result to one percentage when the failed behavior matters more than the total. Keep the case-level evidence.

## Meaningful Experiment Record

Record an experiment only when it teaches something reusable or explains a retained/reverted behavior.

```markdown
Date: <date>
Hypothesis: <one expected behavior change>
Atomic change: <one meaningful edit>
Cases rerun: <all case names>
Result: <improved, unchanged, or regressed>
Decision: <keep or revert>
Evidence: <specific checks and cases>
```

Do not create permanent records for typo fixes, formatting, or wording changes with no behavioral claim.

## Review Gate

Activate, keep, consolidate, or retire based on:

- one clear owner and no unresolved collision;
- the full case set, including the edge case;
- observable before-and-after behavior;
- current Hermes structural validation;
- the owning profile or repository's authority and delivery rules.

There is no universal pass percentage, run count, recurrence requirement, or review schedule. The evidence needed should match the skill's risk and reuse value.
