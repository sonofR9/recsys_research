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

The monitor synchronizes tagged RQs, checks immediately, sends only the generic
reminder when `not_started` or `wip` tasks exist, then repeats every 30 minutes.
Blocked and human-review tasks do not trigger reminders:

```bash
./work/monitor_service.sh start
./work/monitor_service.sh status
./work/monitor_service.sh stop
```

Codex has no supported CLI for injecting a prompt into an existing live thread.
The service therefore captures the explicitly active tmux pane, inserts the
reminder, and verifies that Codex consumed it after submission. Typed or
unrecognized composer content defers delivery until the next cycle. It submits
with Enter when Codex is idle, queues with Tab while Codex is working, retries
an ignored queue key, and falls back to Enter if queuing is unavailable.

Tmux cannot make the screen-content check and later send conditional atomic, so
a human keystroke in that narrow interval remains a residual race. Use the
non-input fallback when that is unacceptable:

```bash
WORK_MONITOR_DELIVERY=log ./work/monitor_service.sh start
tail -f work/.runtime/monitor.log
```

The service refuses non-Codex panes for input delivery and must start inside
tmux. Runtime logs stay under ignored `work/.runtime/`. The monitor itself runs
in a dedicated, detached tmux service session so it survives the shell that
starts it. A failed cycle is logged and retried after the configured interval.
