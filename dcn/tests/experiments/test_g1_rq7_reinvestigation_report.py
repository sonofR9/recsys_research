from __future__ import annotations

from pathlib import Path
import json

from experiments.g1_sasrec_item_ids_likes.analysis.rq7_reinvestigation_candidates import (
    candidate_by_run,
    diagnostic_candidates,
    initial_candidates,
    legacy_concat_diagnostic_candidates,
    make_boundary_candidate,
    make_confirmation_candidate,
    make_rope_base_extension_candidates,
    rope_base_candidates,
)
from experiments.g1_sasrec_item_ids_likes.analysis.rq7_reinvestigation_report import (
    Run,
    build_report_bundle,
    sync_readme,
    write_report_bundle,
)


METRICS = {
    "recall@100": 0.13,
    "ndcg@100": 0.05,
    "recall@10": 0.027,
    "ndcg@10": 0.021,
    "coverage@100": 0.60,
}


def _run(candidate, validation: float, final: float | None = None) -> Run:
    metrics = dict(METRICS)
    metrics["recall@100"] = validation if final is None else final
    return Run(candidate, 2, 20, validation, validation / 2, metrics)


def _initial_runs() -> list[Run]:
    scores = {0.006: 0.12, 0.012: 0.14, 0.024: 0.13}
    return [
        _run(candidate, scores[candidate.deep_lr]) for candidate in initial_candidates()
    ]


def test_incomplete_program_omits_pending_rows_from_reader_tables() -> None:
    bundle = build_report_bundle([])

    assert bundle.reader_markdown.count("| learned-position treatment |") == 1
    assert bundle.reader_markdown.count("| RoPE / ALiBi treatment |") == 1
    assert "pending native-500M LR surface" not in bundle.reader_markdown
    assert "50m" not in bundle.tuning_markdown.lower()
    assert bundle.evidence["claims_status"] == "pending"


def test_selection_uses_validation_winner_but_renders_full_user_metrics() -> None:
    runs = _initial_runs()
    treatment = "learned_forward_add"
    for index, run in enumerate(runs):
        if run.candidate.treatment == treatment and run.candidate.deep_lr == 0.006:
            runs[index] = _run(run.candidate, 0.12, 0.90)
        if run.candidate.treatment == treatment and run.candidate.deep_lr == 0.012:
            runs[index] = _run(run.candidate, 0.14, 0.15)

    bundle = build_report_bundle(runs)
    row = next(
        line
        for line in bundle.reader_markdown.splitlines()
        if "forward additive" in line and line.startswith("|")
    )

    assert "0.150" in row
    assert "0.900" not in row
    assert "| **0.064** | **0.012** |" in bundle.tuning_markdown


def test_boundary_winner_is_pending_until_the_next_geometric_probe() -> None:
    runs = _initial_runs()
    alibi = [run for run in runs if run.candidate.treatment == "alibi"]
    for run in alibi:
        runs[runs.index(run)] = _run(run.candidate, run.candidate.deep_lr)

    pending = build_report_bundle(runs)
    assert pending.evidence["treatments"]["alibi"]["status"] == "boundary pending"

    surface = next(run.candidate for run in runs if run.candidate.treatment == "alibi")
    continuation = make_boundary_candidate(surface, "high", 1)
    runs.append(_run(continuation, 0.01))
    resolved = build_report_bundle(runs)

    assert resolved.evidence["treatments"]["alibi"]["selected_deep_lr"] == 0.024
    assert "| 0.064 | 0.048 |" in resolved.tuning_markdown


def test_complete_confirmations_replace_seed42_with_three_seed_mean() -> None:
    runs = _initial_runs()
    winner = next(
        run.candidate
        for run in runs
        if run.candidate.treatment == "alibi" and run.candidate.deep_lr == 0.012
    )
    for index, run in enumerate(runs):
        if run.candidate == winner:
            runs[index] = _run(winner, 0.14, 0.12)
        if (
            run.candidate.treatment == "rope_forward_base10000"
            and run.candidate.deep_lr == 0.012
        ):
            runs[index] = _run(run.candidate, 0.14, 0.12)
    runs.extend(
        [
            _run(make_confirmation_candidate(winner, 43), 0.14, 0.15),
            _run(make_confirmation_candidate(winner, 44), 0.14, 0.18),
        ]
    )

    bundle = build_report_bundle(runs)
    row = next(
        line for line in bundle.reader_markdown.splitlines() if "**ALiBi**" in line
    )

    assert "0.150" in row
    assert bundle.evidence["treatments"]["alibi"]["reader_evidence"] == "3-seed mean"


def test_comparability_claims_wait_for_every_declared_confirmation() -> None:
    runs = _initial_runs()
    winners = [
        run.candidate
        for run in runs
        if run.candidate.deep_lr == 0.012
        and run.candidate.treatment
        in {
            "alibi",
            "rope_forward_base10000",
            "rope_forward_base10000_alibi",
        }
    ]
    runs.append(_run(make_confirmation_candidate(winners[0], 43), 0.14))

    pending = build_report_bundle(runs)
    assert pending.evidence["claims_status"] == "pending"
    assert "confirmations pending" not in pending.reader_markdown

    runs.extend(
        _run(make_confirmation_candidate(winner, seed), 0.14)
        for winner in winners
        for seed in (43, 44)
        if make_confirmation_candidate(winner, seed).run_name
        not in {run.candidate.run_name for run in runs}
    )
    complete = build_report_bundle(runs)

    assert complete.evidence["rope_claims_status"] == "ready"
    assert complete.reader_markdown.count("3-seed mean") == 0


def test_plain_rope_selection_resolves_lower_and_outer_base_surfaces() -> None:
    runs = _initial_runs()
    for index, run in enumerate(runs):
        if run.candidate.treatment == "rope_forward_base10000":
            runs[index] = _run(
                run.candidate,
                {0.006: 0.09, 0.012: 0.10, 0.024: 0.095}[run.candidate.deep_lr],
            )
    runs.extend(
        _run(
            candidate,
            {
                "rope_forward_base100": {0.006: 0.14, 0.012: 0.16, 0.024: 0.15},
                "rope_forward_base1000": {0.006: 0.13, 0.012: 0.15, 0.024: 0.14},
            }[candidate.treatment][candidate.deep_lr],
        )
        for candidate in rope_base_candidates()
    )

    pending = build_report_bundle(runs)
    assert pending.evidence["treatments"]["plain_rope"]["status"] == (
        "native-500M RoPE-base surface pending"
    )

    runs.extend(
        _run(candidate, {0.006: 0.13, 0.012: 0.14, 0.024: 0.135}[candidate.deep_lr])
        for candidate in make_rope_base_extension_candidates("low")
    )
    resolved = build_report_bundle(runs)

    assert "plain forward RoPE (base 100)" in resolved.reader_markdown


def test_tuning_ledger_excludes_50m_and_legacy_concat_diagnostics() -> None:
    runs = _initial_runs()
    runs.extend(_run(candidate, 0.99) for candidate in diagnostic_candidates())
    runs.extend(
        _run(candidate, 0.99)
        for revision in (1, 2)
        for candidate in legacy_concat_diagnostic_candidates(revision)
    )

    bundle = build_report_bundle(runs)

    assert bundle.evidence["eligible_native_runs"] == 36
    assert "diagnostic" not in bundle.tuning_markdown
    assert "_r1_50m" not in bundle.tuning_markdown
    assert "_r2_50m" not in bundle.tuning_markdown
    assert "| seed |" not in bundle.tuning_markdown
    assert "| stage |" not in bundle.tuning_markdown


def test_report_excludes_historical_r6_combined_native_artifacts() -> None:
    current = next(
        candidate
        for candidate in initial_candidates()
        if candidate.treatment == "learned_forward_reverse_add"
        and candidate.deep_lr == 0.012
    )
    historical = candidate_by_run(current.run_name.replace("_r7_", "_r6_"))

    bundle = build_report_bundle([_run(historical, 0.99)])

    assert bundle.evidence["eligible_native_runs"] == 0
    assert bundle.evidence["learned_claims_status"] == "pending"
    assert "forward + reverse additive" not in bundle.reader_markdown


def test_bundle_writer_and_readme_sync_replace_only_generated_rq7_tables(
    tmp_path: Path,
) -> None:
    bundle = build_report_bundle([])
    paths = write_report_bundle(bundle, tmp_path / "scratchpad", tmp_path / "evidence")
    readme = tmp_path / "README.md"
    readme.write_text(
        "before\n| historical comparison |\n"
        "<!-- rq7-reinvestigation-generated:start -->\nold\n"
        "<!-- rq7-reinvestigation-generated:end -->\nafter\n"
    )

    sync_readme(readme, bundle.reader_markdown)

    assert set(paths) == {"tuning", "reader", "evidence"}
    assert "old" not in readme.read_text()
    assert "| historical comparison |" in readme.read_text()
    assert "learned-position treatment" in readme.read_text()
    assert readme.read_text().startswith("before\n")
    assert readme.read_text().endswith("after\n")


def test_checked_in_report_retains_historical_tables_and_reports_r7_ready() -> None:
    root = Path(__file__).parents[3]
    report = root / "experiments/g1_sasrec_item_ids_likes"
    readme = (report / "README.md").read_text()
    reader_50m = (report / "scratchpad/research_questions_50m.md").read_text()
    tuning_50m = (report / "scratchpad/hyperparameter_tuning_50m.md").read_text()
    reader_500m = (report / "scratchpad/research_questions_500m.md").read_text()
    generated_reader = (report / "scratchpad/rq7_reinvestigation_reader_500m.md").read_text()
    tuning_500m = (report / "scratchpad/rq7_reinvestigation_tuning_500m.md").read_text()
    evidence = json.loads(
        (report / "evidence/rq7_reinvestigation_results.json").read_text()
    )

    assert readme.count("<!-- rq7-reinvestigation-generated:start -->") == 1
    assert readme.count("<!-- rq7-reinvestigation-generated:end -->") == 1
    assert "max_scale * tanh(gate) * reverse" in readme
    assert "DenseNet([item; forward; reverse])" in readme
    assert "relative phases" in readme
    assert "### Earlier broad position-encoding comparison" in readme
    assert "position rope reverse learned reverse" in readme
    assert "No reader table is generated from Yambda-50M" in reader_50m
    assert "No RQ7 tuning table is generated from Yambda-50M" in tuning_50m
    assert reader_500m.count("<!-- rq7-reinvestigation-generated:start -->") == 1
    assert reader_500m.count("<!-- rq7-reinvestigation-generated:end -->") == 1
    assert generated_reader.strip() in reader_500m
    assert "### Earlier broad position-encoding comparison" in reader_500m
    assert "position rope reverse learned reverse" in reader_500m
    assert "pending native-500M LR surface" not in reader_500m
    assert "| evidence |" not in generated_reader
    assert "selected seed" not in generated_reader
    assert "3-seed mean" not in generated_reader
    assert " r3" not in generated_reader
    assert " r7" not in generated_reader
    assert "diagnostic" not in tuning_500m
    assert "_50m" not in tuning_500m
    assert "| stage |" not in tuning_500m
    assert "| seed |" not in tuning_500m
    assert evidence["claims_status"] == "ready"
    assert evidence["learned_claims_status"] == "ready"
    assert evidence["rope_claims_status"] == "ready"
    assert evidence["treatments"]["plain_rope"]["reader_evidence"] == "3-seed mean"
    assert "every forward+reverse row is pending" not in readme
    assert "No forward+reverse learned treatment is accepted" not in readme
    assert "All requested forward+reverse variants satisfy" in readme
