---
name: to-requirements-doc
description: Turn the current conversation context into a requirements document and publish it to the project issue tracker. Use when user wants to create a requirements document from the current context.
---

This skill takes the current conversation context and codebase understanding and produces a requirements document. Do NOT interview the user — just synthesize what you already know.

The issue tracker config is in `docs/agents/issue-tracker.md` and the triage label vocabulary is in `docs/agents/triage-labels.md`.

## Process

1. Explore the repo to understand the current state of the codebase, if you haven't already. Use the project's domain glossary vocabulary throughout, and respect any architecture decisions in the area you're touching.

2. Sketch out the major modules you will need to build or modify. Check with the user that these modules match their expectations.

3. Write the requirements document using the template below, then publish it to the project issue tracker. Apply the `ready-for-agent` triage label.

<requirements-template>

## Problem Statement

The problem from the user's perspective.

## Solution

The solution from the user's perspective.

## User Stories

A LONG, numbered list of user stories. Each user story should be in the format of:

1. As an <actor>, I want a <feature>, so that <benefit>

<user-story-example>
1. As a mobile bank customer, I want to see balance on my accounts, so that I can make better informed decisions about my spending
</user-story-example>

This list of user stories should be extremely extensive and cover all aspects of the feature.

## Implementation Decisions

A list of implementation decisions: modules to build/modify, architectural decisions, schema changes, API contracts. No file paths or code snippets unless a prototype snippet encodes a decision more precisely than prose can.


## Out of Scope

What is out of scope for this requirements document.

## Further Notes

Any further notes about the feature.

</requirements-template>
