---
name: orchestrate
description: Orchestrate a non-trivial task end-to-end. Use when the user explicitly call this skill.
---

# How to orchestrate

You drive the task; you don't write the feature code. Understand it, split it into slices,
and run coding + review **subagents** (`Agent` tool) until it's done and verified. Keep your
own context lean — subagents do the heavy reading and editing.

**Hard rule: every coding subagent is followed by a fresh, independent review subagent.**
No code reaches the user unreviewed.

## 1. Understand

- Read the relevant code and `docs/architecture.md`; dispatch an `Explore` subagent for a
  large surface instead of reading it all yourself.
- Write down what "done" means as concrete, testable acceptance criteria.
- **Ask when it's unclear and not easy.** Big and non-obvious → use the **grill-with-docs**
  skill; small-to-medium → just grill the user directly. Trivial → skip and act.

Don't proceed until you can write a self-contained brief for someone who hasn't seen this chat.

## 2. Decompose

Split into the smallest independently shippable vertical slices; note dependencies.

- Independent slices → run in **parallel** (one message, multiple `Agent` calls).
- Dependent slices → run **sequentially**, each after the prior one's review.

## 3. Code (delegate)

One coding subagent per slice (`subagent_type: general-purpose`). Brief is **behavioral**,
not line-by-line — the subagent explores fresh:

```
Implement one slice. Follow the `write-code` skill.
Summary / Context / Desired behavior (with edge cases)
Acceptance criteria: testable checklist
Out of scope: what NOT to touch
How to run/test: venv activation + test command (it doesn't share your session)
Return: changes, files touched, test results, anything the reviewer should scrutinize.
```

The subagent's final message is a tool result, not shown to the user — relay what matters.

## 4. Review (always, independent)

After every coder, spawn a **different** `general-purpose` subagent — even if the coder
reported success. It must do **both**:

1. Run `/code-review` on the slice's changes.
2. Verify acceptance criteria are met, tests pass, and `write-code` conventions hold.

Return: findings as **blocker** vs **nit**, plus a verdict (pass / needs-fixes).

## 5. Auto-fix loop

Blockers → fix subagent (briefed with the findings) → fresh review again. Cap at **3 rounds**.
If still not converged, stop and escalate to the user with what's stuck. Nits don't force a round.

## 6. Report

Concise summary: what was built, how it was sliced, test status, leftover nits, and any
decisions you made for them. Be honest about anything skipped or uncertain.
