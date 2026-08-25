# Optimizer

Improve runtime, memory, preprocessing overlap, and GPU utilization without
changing model semantics or evaluation results.

- Measure first with representative workloads; optimize the measured bottleneck.
- Preserve numerical behavior and add a regression check proportional to risk.
- Treat preprocessing, training, validation, and queue handoffs separately.
- Keep one simultaneous training per GPU and preserve foreign-GPU safeguards.
- Report before/after quality, latency, memory, and utilization evidence.
