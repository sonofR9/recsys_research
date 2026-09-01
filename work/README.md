# Work tracker

Task location is its canonical status:

- `not_started/`
- `wip/`
- `blocked/`
- `human_review/`
- `done/`

Copy `task.template.yaml` to the appropriate directory and give it the same
meaningful kebab-case name as its `id`. Only the lead agent may create, edit,
move, or delete task YAML files. Other agents treat all of `work/` as read-only.
The lead moves a human-review task to `done` only after user acceptance.

To synchronize an RQ in `experiments/ideas.md`, add a unique tag to the task's
`ideas_tags`, then put `<!-- work:the-tag -->` on the RQ's status line. The task
folder maps to `not_started`, `wip`, `wip`, `review`, or `complete`, respectively.

Validate or synchronize manually:

```bash
python -m work.workflow validate
python -m work.workflow sync
```
