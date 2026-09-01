from experiments.g3_pretrained_item_embeddings.analysis.reports import (
    ReaderRow,
    ReaderSection,
    TuningRow,
    build_compact_report,
    build_reader_scaffold,
    build_tuning_report,
)


def _tuning_row(
    trial_id: str,
    recall: float,
    *,
    status: str = "usable",
    capacity: int | None = None,
) -> TuningRow:
    return TuningRow(
        research_question="RQ2",
        family="content concatenation",
        trial_id=trial_id,
        status=status,
        embedding_learning_rate=0.1,
        deep_learning_rate=0.02,
        declared_horizon_epochs=25,
        completed_horizon_epochs=25 if status == "usable" else 12,
        restored_best_epoch=18,
        capacity=capacity,
        validation_recall_at_100=recall,
        validation_ndcg_at_100=0.04,
        training_seconds=100.0,
    )


def test_tuning_report_keeps_every_usable_row_and_bolds_each_family_winner() -> None:
    report = build_tuning_report(
        (
            _tuning_row("trial-1", 0.10, capacity=64),
            _tuning_row("trial-2", 0.11, capacity=128),
            _tuning_row("failed-trial", 0.99, status="failed", capacity=256),
        )
    )

    assert "## RQ2" in report
    assert "### content concatenation" in report
    assert "trial-1" in report
    assert "**trial-2**" in report
    assert "failed-trial" not in report
    assert "embedding lr" in report
    assert "declared horizon" in report
    assert "restored epoch" in report
    assert "capacity" in report


def test_compact_and_reader_scaffolds_have_reader_metrics_not_tuning_fields() -> None:
    rq1 = ReaderSection(
        question="Does pretrained content improve history input?",
        reference_variant="learned item ID",
        rows=(
            ReaderRow(
                "learned item ID",
                (("recall@100", 0.10), ("ndcg@100", 0.05)),
            ),
            ReaderRow(
                "content only",
                (("recall@100", 0.105), ("ndcg@100", 0.052)),
            ),
            ReaderRow(
                "content plus ID",
                (("recall@100", 0.12), ("ndcg@100", 0.065)),
            ),
        ),
    )
    aggregate = ReaderSection(
        question="Aggregated improvement",
        reference_variant="original tied baseline",
        rows=(
            ReaderRow(
                "original tied baseline",
                (("recall@100", 0.10), ("ndcg@100", 0.05)),
            ),
            ReaderRow(
                "selected compatible G3 combination",
                (("recall@100", 0.12), ("ndcg@100", 0.065)),
            ),
        ),
    )
    dispersions = {"recall@100": 0.10, "ndcg@100": 0.10}

    compact = build_compact_report((rq1,), relative_dispersions=dispersions)
    scaffold = build_reader_scaffold(
        title="G3 pretrained item embeddings",
        description="Approved native-50M experiment with a native-size companion.",
        sections=(rq1,),
        aggregate=aggregate,
        relative_dispersions=dispersions,
    )

    assert compact.startswith("## Does pretrained content improve history input?")
    assert "+5.0% (0.105)" in compact
    assert '<span style="color: green">+20.0% (0.120)</span>' in compact
    assert "embedding learning rate" not in compact
    assert "| variant | recall@100 | ndcg@100 |" in compact
    assert "runs" not in compact
    assert scaffold.startswith("# G3 pretrained item embeddings\n\n")
    assert scaffold.index("## Aggregated improvement") > scaffold.index(
        "## Does pretrained content improve history input?"
    )
    assert scaffold.count("## Aggregated improvement") == 1
