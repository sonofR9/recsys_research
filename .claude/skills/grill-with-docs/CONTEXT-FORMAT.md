# CONTEXT.md Format

## Structure

```md
# {Context Name}

{One or two sentence description of what this context is and why it exists.}

## Language

**Order**:
{A one or two sentence description of the term}
_Avoid_: Purchase, transaction

**Invoice**:
A request for payment sent to a customer after delivery.
_Avoid_: Bill, payment request

**Customer**:
A person or organization that places orders.
_Avoid_: Client, buyer, account
```

## Rules

- **Be opinionated.** Pick the best term; list others as aliases to avoid.
- **Flag conflicts explicitly.** If a term is used ambiguously, call it out with a clear resolution.
- **Keep definitions tight.** One or two sentences max. Define what it IS, not what it does.
- **Show relationships.** Use bold term names and express cardinality where obvious.
- **Only project-specific terms.** General programming concepts don't belong. Before adding a term, ask: is this a concept unique to this context, or a general programming concept? Only the former belongs.
- **Group terms under subheadings** when natural clusters emerge.
- **Write an example dialogue** between a dev and a domain expert that demonstrates how the terms interact.

## Single context repos=

**Single context (most repos):** One `CONTEXT.md` at the repo root.
