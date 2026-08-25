# Architecture Decision Format

Decisions live in `docs/architecture/` with sequential numbering: `0001-slug.md`, `0002-slug.md`, etc. Create the directory lazily — only when the first decision is needed.

## Template

```md
# {Short title of the decision}

{1-3 sentences: what's the context, what did we decide, and why.}
```

An architecture decision can be a single paragraph. The value is in recording *that* a decision was made and *why*.

## Optional sections

Only include when they add genuine value. Most decisions won't need them.

- **Status** frontmatter (`proposed | accepted | deprecated | superseded by NNNN`)
- **Considered Options** — only when rejected alternatives are worth remembering
- **Consequences** — only when non-obvious downstream effects need calling out

## Numbering

Scan `docs/architecture/` for the highest existing number and increment by one.

## When to offer an decision

1. **Hard to reverse** — the cost of changing your mind later is meaningful
2. **Surprising without context**

### What qualifies

- **Architectural shape.** "We're using a monorepo." "The write model is event-sourced, the read model is projected into Postgres."
- **Boundary and scope decisions.** "Customer data is owned by the Customer context; other contexts reference it by ID only." The explicit no-s are as valuable as the yes-s.
- **non-obvious decisions.**
- **Constraints not visible in the code.** "We can't use AWS because of compliance requirements."
- **Rejected alternatives when the rejection is non-obvious.** If you considered GraphQL and picked REST for subtle reasons, record it — otherwise someone will suggest GraphQL again in six months.
