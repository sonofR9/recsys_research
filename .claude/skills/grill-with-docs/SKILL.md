---
name: grill-with-docs
description: Grilling session. Use when user wants to plan his project.
---

<what-to-do>

Interview me relentlessly about every aspect of this plan until we reach a shared understanding. Walk down each branch of the design tree, resolving dependencies between decisions one-by-one. For each question, provide your recommended answer.

Ask the questions one at a time, waiting for feedback on each question before continuing.

If a question can be answered by exploring the codebase, explore the codebase instead.

</what-to-do>

<supporting-info>

## Domain awareness

During codebase exploration, look for existing documentation — see `docs/agents/domain.md` for where `CONTEXT.md` and `docs/architecture/` live.

Create files lazily — only when you have something to write. If no `CONTEXT.md` exists, create one when the first term is resolved. If no `docs/architecture/` exists, create it when the first architecture decision is needed.

## During the session

### Challenge against the glossary

When the user uses a term that conflicts with `CONTEXT.md`, call it out immediately.

### Sharpen fuzzy language

When the user uses vague or overloaded terms, propose a precise canonical term.

### Discuss concrete scenarios

Stress-test domain relationships with specific scenarios that probe edge cases and force precision about boundaries between concepts.

### Cross-reference with code

When the user states how something works, check whether the code agrees. Surface contradictions.

### Update CONTEXT.md inline

When a term is resolved, update `CONTEXT.md` immediately — don't batch. Use the format in [CONTEXT-FORMAT.md](./CONTEXT-FORMAT.md).

`CONTEXT.md` should be totally devoid of implementation details. Do not treat `CONTEXT.md` as a spec, a scratch pad, or a repository for implementation decisions. It  a glossary and nothing else — no implementation details, no specs, no scratch notes.

### Offer architectures sparingly

Only create an architecture decision when all three are true:

1. **Hard to reverse** — changing your mind later is costly
2. **Surprising without context** — a future reader will wonder why
3. **Real trade-off** — genuine alternatives existed

Use the format in [ARCHITECTURE-FORMAT.md](./ARCHITECTURE-FORMAT.md).

</supporting-info>
