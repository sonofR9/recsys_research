---
name: running-tests-on-a100-box
description: How to actually run pytest on the a100 box — the venv in CLAUDE.md is missing and three imports have to be worked around
metadata:
  type: project
---

On the a100 box (`/home/sashanovak/tmp_a100/ysda_recsys`), the run command in
CLAUDE.md does not work as written: `/home/sonofr/python_venvs/.venv` does not
exist. Use `/home/sashanovak/envs/base/bin/python` and set
`CUDA_VISIBLE_DEVICES=""`, otherwise `dcn/nn/transformer.py` imports
`flash_attn`, whose `.so` is built against a different torch ABI and fails with
an undefined-symbol error.

`duckdb` and `kagglehub` are not installed anywhere on this box, and
`dcn/nn/__init__.py` pulls them in transitively, so even pure-nn tests fail to
collect. Stub them on `PYTHONPATH` to get the suite running (`duckdb` needs a
`DuckDBPyConnection` attribute, `kagglehub` a `KaggleDatasetAdapter`).

Two failures are pre-existing and unrelated to any change:
`dcn/tests/test_main_e2e.py::test_run_experiment_trains_end_to_end` (needs real
duckdb) and
`data/tests/test_ema_counter.py::TestEmaCounter::test_ema_decay_multiple_days`
on the `coutners_to_one_column` branch.
