# Writing Agent Briefs

An agent brief is a structured comment posted on an issue when it moves to `ready-for-agent`. It is the authoritative specification that an autonomous agent will work from.

## Principles

### Durable and behavioral

The issue may sit in `ready-for-agent` for weeks. The codebase will change in the meantime. Describe **what** the system should do, not **how** to implement it. Do not name file paths, line numbers. Functions sometimes are ok to mention. Describe intent, not code. The agent will explore the codebase fresh.

- **Good:** (in refactoring issue) "The `SkillConfig` type should accept an optional `schedule` field of type `CronExpression`"
- **Bad:** "Open src/types/skill.ts and add a schedule field on line 42"
- **Good:** "When a user runs `/triage` with no arguments, they should see a summary of issues needing attention"

### Complete acceptance criteria

Every brief must have concrete, testable acceptance criteria. Each criterion should be independently verifiable.

### Explicit scope boundaries

State what is out of scope to prevent gold-plating.

## Template

```markdown
## Agent Brief

**Category:** bug / feature
**Summary:** one-line description of what needs to happen

**Current behavior:**
What happens now. For bugs, the broken behavior. For features, the status quo.

**Desired behavior:**
What should happen after the agent's work. Be specific about edge cases.

**Acceptance criteria:**
- [ ] Specific, testable criterion 1
- [ ] Specific, testable criterion 2

**Out of scope:**
- Thing that should NOT be changed
- Adjacent feature that is separate
```

## Examples

### Good agent brief (bug)

```markdown
## Agent Brief

**Category:** bug
**Summary:** Skill description truncation drops mid-word, producing broken output

**Current behavior:**
When a skill description exceeds 1024 characters, it is truncated at exactly
1024 characters regardless of word boundaries, producing mid-word cuts.

**Desired behavior:**
Truncation should break at the last word boundary before 1024 characters
and append "...". Total length including "..." must not exceed 1024 chars.

**Acceptance criteria:**
- [ ] Descriptions under 1024 chars are unchanged
- [ ] Descriptions over 1024 chars are truncated at the last word boundary
- [ ] Truncated descriptions end with "..."
- [ ] Total length including "..." does not exceed 1024 chars

**Out of scope:**
- Changing the 1024 char limit itself
- Multi-line description support

### Good agent brief (feature)

```markdown
## Agent Brief

**Category:** feature
**Summary:** Add `.out-of-scope/` directory support for tracking rejected feature requests

**Current behavior:**
When a feature request is rejected, the issue is closed with a `out-of-scope` label
and a comment. There is no persistent record of the decision or reasoning.
Future similar requests require the maintainer to recall or search for the
prior discussion.

**Desired behavior:**
Rejected feature requests should be documented in `.out-of-scope/<concept>.md`
files that capture the decision, reasoning, and links to all issues that
requested the feature. When triaging new issues, these files should be
checked for matches.

**Key interfaces:**
- Markdown file format in `.out-of-scope/` — each file should have a
  `# Concept Name` heading, a `**Decision:**` line, a `**Reason:**` line,
  and a `**Prior requests:**` list with issue links
- The triage workflow should read all `.out-of-scope/*.md` files early
  and match incoming issues against them by concept similarity

**Acceptance criteria:**
- [ ] Closing a feature as out-of-scope creates/updates a file in `.out-of-scope/`
- [ ] The file includes the decision, reasoning, and link to the closed issue
- [ ] If a matching `.out-of-scope/` file already exists, the new issue is
      appended to its "Prior requests" list rather than creating a duplicate
- [ ] During triage, existing `.out-of-scope/` files are checked and surfaced
      when a new issue matches a prior rejection

**Out of scope:**
- Automated matching (human confirms the match)
```
