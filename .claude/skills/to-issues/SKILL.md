---
name: to-issues
description: Break a plan, spec, or requirements document into independently-grabbable issues on the project issue tracker using tracer-bullet vertical slices. Use when user wants to convert a plan into issues.
---

# To Issues

Break a plan into independently-grabbable issues using vertical slices (tracer bullets).

The issue tracker config is in `docs/agents/issue-tracker.md` and the triage label vocabulary is in `docs/agents/triage-labels.md`.

## Process

### 1. Gather context

Work from whatever is already in the conversation context. If the user passes an issue reference, fetch it from the issue tracker and read its full body and comments.

### 2. Explore the codebase (optional)

If you have not already explored the codebase, do so. Issue titles and descriptions should use the project's domain glossary vocabulary, and respect architecture decisions in the area you're touching.

### 3. Draft vertical slices

Break the plan into **tracer bullet** issues — thin vertical slices that cut through ALL integration layers end-to-end, NOT horizontal slices of one layer.

Slices may be 'manual' or 'autonomous'. Manual slices require human interaction (architectural decision, design review). Autonomous slices can be implemented and merged by an agent without human interaction. Prefer autonomous over manual.

<vertical-slice-rules>
- Each slice delivers a narrow but COMPLETE path through every layer (schema, API, UI, tests)
- A completed slice is demoable or verifiable on its own
- Prefer many thin slices over few thick ones
</vertical-slice-rules>

### 4. Quiz the user

Present the proposed breakdown as a numbered list. For each slice show: **Title**, **Type** (manual/autonomous), **Blocked by**, **User stories covered**.

Ask the user:

- Does the granularity feel right? (too coarse / too fine)
- Are the dependency relationships correct?
- Should any slices be merged or split further?
- Are the correct slices marked as manual and autonomous?

Iterate until approved.

### 5. Publish the issues to the issue tracker

Publish in dependency order (blockers first) so you can reference real issue identifiers. These issues are considered ready for autonomous agents, so apply the correct triage label unless instructed otherwise.

<issue-template>
## Parent

A reference to the parent issue (if the source was an existing issue, otherwise omit).

## What to build

Concise description of this vertical slice — end-to-end behavior, not layer-by-layer. Avoid file paths or code snippets unless a prototype snippet encodes a decision more precisely than prose can.

## Acceptance criteria

- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

## Blocked by

- A reference to the blocking ticket (if any)

Or "None - can start immediately" if no blockers.

</issue-template>

Do NOT close or modify any parent issue.
