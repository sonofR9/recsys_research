---
name: write-code
description: Skill for writing the code. Use when user wants to build features or fix bugs.
---

# How to write code

## Workflow

Follow these steps on every non-trivial task. Skip only for genuinely trivial edits (typo, one-line fix).

1. Read and understand the relevant code first.
2. Form a plan. If unsure, ask the user to confirm before implementing.
3. Follow incremental TDD in a loop:
   1. Write test cases and interfaces for this iteration functionality.
   2. Review your test cases.
   3. Implement main logic. Fix all dependencies.
   4. Run tests.
   5. commit
4. Review your own changes, refactor, and re-test.

### TDD guidelines

**Core principle**: Tests verify behavior through public interfaces, not implementation details. Code can change entirely; tests shouldn't.

**Good tests** exercise real code paths through public APIs. They describe _what_ the system does, not _how_. They survive refactors because they don't care about internal structure and fail when important logic is broken. Even if important logic does not have public interface on its own.

**Bad tests** are coupled to implementation. They mock internal collaborators, test private methods, or verify through external mean (like querying a database directly instead of using the interface). Warning sign: test breaks on refactor but behavior hasn't changed.  If you rename an internal function and tests fail, those tests were testing implementation, not behavior.

Write incrementally: one test → one implementation → repeat. For example, implementing an OrderProcessor with stock and payment validation. Let's say it has only 1 public method - process. First focus on stock validation. Come up with test cases for stock validation. Write tests. Implement the matching logic and ensure test passes. Then do the same with payment validation, then test coordination as a whole.

### Checklist per cycle

[ ] Tests describe behavior, not implementation
[ ] Tests use public interface only
[ ] Tests would survive internal refactor
[ ] Tests written before implementation
[ ] Implementation passes tests
[ ] Code reviewed and refactored
[ ] All edge cases covered
[ ] No uncommited code exists

### A few words about refactoring

After all tests pass, review and refactor your code:

Duplication → Extract function/class
Long methods → Break into private helpers (keep tests on public interface)
Primitive obsession → Introduce value objects

All tests must pass after refactoring!

### Checklist per workflow cycle

[ ] you've followed tdd
[ ] your've reviewed your code and you are happy with code quality
[ ] your implementation passes tests
[ ] your tests cover all edge cases
[ ] No uncommited code exists

## Code style — clean code is mandatory

- **No unnecessary comments.** Only add when the WHY is non-obvious. Never restate what the code does.
- **No docstrings on the obvious.** Docstrings are for non-trivial public API only.
- **Names carry the meaning.** Spend effort on naming before reaching for explanation.
- **No abbreviations in names.** `index` not `idx`, `config` not `cfg`. Universal initialisms (`id`, `url`, `cpu`) are fine.
- **Never remove a `FIXME` / `TODO` you have not actually fixed.** If out of scope, leave untouched. If stale, ask before removing.
- **Small, focused functions and classes.** Single responsibility.
- **No dead code, no commented-out code.** Git remembers.
- **DRY, but not at the cost of clarity.** Duplicating two short blocks is fine; coupling unrelated callers through a shared helper is not.
- **Type hints everywhere.**
- **Public code is minimal.** Only what truly needs to be public.
- **Consolidate common functionality into dedicated utilities.**
