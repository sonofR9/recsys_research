# Out-of-Scope Knowledge Base

The `.out-of-scope/` directory stores persistent records of rejected feature requests for two purposes:

1. **Institutional memory** — why a feature was rejected, so reasoning isn't lost when the issue is closed
2. **Deduplication** — surface prior rejections when similar new issues come in

## Directory structure

```
.out-of-scope/
├── dark-mode.md
├── plugin-system.md
└── graphql-api.md
```

One file per **concept**, not per issue. Multiple issues requesting the same thing are grouped under one file.

## File format

Written in a relaxed, readable style — more like a short design document than a database entry.

```markdown
# Dark Mode

This project does not support dark mode or user-facing theming.

## Why this is out of scope

The rendering pipeline assumes a single color palette defined in
`ThemeConfig`. Supporting multiple themes would require a theme context
provider, per-component theme-aware style resolution, and a persistence
layer for user preferences. This doesn't align with the project's focus
on content authoring. Theming is a concern for downstream
consumers who embed or redistribute the output.

## Prior requests

- #42 — "Add dark mode support"
- #87 — "Night theme for accessibility"
```

### Naming

Use short, descriptive kebab-case: `dark-mode.md`, `plugin-system.md`. Recognizable without opening the file.

### Writing the reason

Substantive, not "we don't want this". Reference project scope/philosophy, technical constraints, or strategic decisions. Avoid temporary circumstances ("we're too busy") — those are deferrals, not rejections.

## When to check `.out-of-scope/`

During triage (Step 1: Gather context), read all files in `.out-of-scope/`. If a new issue matches a prior rejection by concept similarity (not keyword), surface it to the maintainer. They may:

- **Confirm** — add to "Prior requests", close
- **Reconsider** — delete/update the file, proceed with normal triage
- **Disagree** — issues are distinct, proceed normally

## When to write to `.out-of-scope/`

Only when a **feature** (not a bug) is rejected as `out-of-scope`:

1. Check if a matching file already exists
2. If yes: append to "Prior requests"
3. If no: create new file with concept name, decision, reason, and first prior request
4. Post a comment on the issue mentioning the `.out-of-scope/` file
5. Close with `out-of-scope`

## Updating or removing

If the maintainer changes their mind, delete the `.out-of-scope/` file. Old issues stay closed as historical records.
