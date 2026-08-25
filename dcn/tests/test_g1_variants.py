import json
import runpy
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.g1_sasrec_item_ids_likes.launchers import verify_artifact


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "experiments/g1_sasrec_item_ids_likes/configs/variant.py"
)
EXPERIMENT = SCRIPT.parents[1]
COLLECT = EXPERIMENT / "analysis/collect.py"
METRIC_REGRESSION = EXPERIMENT / "checks/metric_regression_50m.py"
FINAL_SWEEP = EXPERIMENT / "launchers/core/final_sweep.sh"
FOLLOWUP_SWEEP = EXPERIMENT / "launchers/core/followup_sweep.sh"
SEED_SWEEP = EXPERIMENT / "launchers/core/seeds.sh"
SWEEP = EXPERIMENT / "launchers/core/sweep.sh"


def _namespace(monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setenv("G1_VARIANT", "baseline")
    monkeypatch.delenv("G1_DATASET_SIZE", raising=False)
    monkeypatch.delenv("G1_MAX_USERS", raising=False)
    return runpy.run_path(str(SCRIPT))


def test_every_architecture_variant_has_a_cosine_warmup_cross(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _namespace(monkeypatch)
    variants = namespace["VARIANTS"]
    collect = runpy.run_path(str(COLLECT))
    expected = {name for _, names in collect["ARCHITECTURE_AXES"] for name in names}
    expected.update(dict(collect["BASE_AXES"])[collect["SHARED_WINDOW_TITLE"]])

    assert set(namespace["ARCHITECTURE_VARIANTS"]) == expected

    for name in namespace["ARCHITECTURE_VARIANTS"]:
        crossed = variants[f"cosine_{name}"]
        assert crossed == replace(
            variants[name],
            run_name=f"g1_calibrated_cosine_{name}_ts2_500m",
            lr_schedule=namespace["COSINE_WARMUP"],
        )


def test_g1_variants_do_not_compile_ragged_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variants = _namespace(monkeypatch)["VARIANTS"]

    assert all(not variant.runtime.compile for variant in variants.values())


@pytest.mark.parametrize(
    ("name", "rope"),
    [
        ("pos_rope_learned_reverse", "forward"),
        ("pos_rope_reverse_learned_reverse", "reverse"),
    ],
)
def test_missing_reverse_position_crosses_are_runnable(
    monkeypatch: pytest.MonkeyPatch, name: str, rope: str
) -> None:
    variant = _namespace(monkeypatch)["VARIANTS"][name]

    assert variant.transformer.learned_positions == "reverse"
    assert variant.transformer.rope == rope
    assert not variant.transformer.alibi


@pytest.mark.parametrize(
    ("name", "positions", "alibi"),
    [
        ("pos_learned_forward_concat", "forward", False),
        ("pos_learned_forward_reverse_concat", ("forward", "reverse"), False),
        ("pos_learned_forward_concat_alibi", "forward", True),
        (
            "pos_learned_forward_reverse_concat_alibi",
            ("forward", "reverse"),
            True,
        ),
    ],
)
def test_position_concatenation_variants_are_exact_treatments(
    monkeypatch: pytest.MonkeyPatch, name: str, positions, alibi: bool
) -> None:
    variant = _namespace(monkeypatch)["VARIANTS"][name]

    assert variant.transformer.learned_positions == positions
    assert variant.transformer.learned_position_fusion == "concat"
    assert variant.transformer.alibi is alibi
    assert variant.transformer.rope is None


@pytest.mark.parametrize(
    ("name", "base"),
    [("pos_rope_base100", 100.0), ("pos_rope_base1000", 1000.0)],
)
def test_rope_base_diagnostic_variants_are_exact_treatments(
    monkeypatch: pytest.MonkeyPatch, name: str, base: float
) -> None:
    variant = _namespace(monkeypatch)["VARIANTS"][name]

    assert variant.transformer.rope == "forward"
    assert variant.transformer.rope_base == base
    assert variant.transformer.learned_positions is None
    assert not variant.transformer.alibi


def test_default_position_fields_remain_compatible_with_historical_recipes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = _namespace(monkeypatch)["VARIANTS"]["baseline"]

    _, invariants = verify_artifact._expected_metadata(experiment)

    transformer = invariants["transformer"]
    assert "rope_base" not in transformer
    assert "learned_position_fusion" not in transformer


def test_cosine_screen_is_reported_against_its_own_baseline() -> None:
    namespace = runpy.run_path(str(COLLECT))
    variants = {
        "baseline": [{"recall@100": 0.10}],
        "lr_cosine_warmup": [{"recall@100": 0.12}],
        "cosine_dim_32": [{"recall@100": 0.11}],
    }

    rows, reference_name, reference = namespace["_by_score"](
        "Dimension under cosine warmup", ["cosine_dim_32"], variants
    )

    assert reference_name == "lr_cosine_warmup"
    assert reference == variants[reference_name]
    assert rows == [
        ("lr_cosine_warmup", variants[reference_name]),
        ("cosine_dim_32", variants["cosine_dim_32"]),
    ]


def test_report_splits_learning_rate_axes() -> None:
    namespace = runpy.run_path(str(COLLECT))
    axes = dict(namespace["BASE_AXES"])

    assert "Embedding and deep learning rates" not in axes
    assert axes[namespace["EMBEDDING_LR_TITLE"]] == [
        "embedding_lr_1e4",
        "embedding_lr_2e4",
        "embedding_lr_5e4",
        "embedding_lr_2e3",
        "embedding_lr_3e3",
        "embedding_lr_5e3",
    ]
    assert axes[namespace["DEEP_LR_TITLE"]] == [
        "deep_lr_5e4",
        "deep_lr_2e3",
        "deep_lr_3e3",
        "deep_lr_5e3",
    ]


def test_every_report_table_states_the_reference_configuration_in_its_row() -> None:
    namespace = runpy.run_path(str(COLLECT))
    baseline = [{"recall@100": 0.1}]

    rendered = namespace["_table"](
        [("baseline", baseline)],
        "baseline",
        baseline,
        title=namespace["DEEP_LR_TITLE"],
    )

    assert "| reference configuration |" in rendered.splitlines()[0]
    assert "| baseline | deep LR=0.001; embedding LR=0.001 | 1 |" in rendered
    assert all(
        title in namespace["_REFERENCE_DETAILS"] for title, _ in namespace["AXES"]
    )
    assert all(
        namespace["_resolve_table"](table)[0] in namespace["_REFERENCE_DETAILS"]
        for question in namespace["QUESTIONS"]
        for table in question.tables
    )


def test_report_titles_state_architecture_references() -> None:
    namespace = runpy.run_path(str(COLLECT))
    base_titles = {title for title, _ in namespace["ARCHITECTURE_AXES"]}
    cosine_titles = {title for title, _ in namespace["COSINE_AXES"]}

    assert "Depth (baseline: depth=2)" in base_titles
    assert "Depth under cosine warmup (baseline: depth=2)" in cosine_titles
    assert (
        "Number of attention heads under cosine warmup "
        "(baseline: heads=2, kv_heads=2)" in cosine_titles
    )


def test_best_metrics_question_contains_only_final_candidates() -> None:
    namespace = runpy.run_path(str(COLLECT))
    question = next(
        question
        for question in namespace["QUESTIONS"]
        if question.title.startswith("rq2 ")
    )

    assert question.tables == [
        (
                "Final metric candidates",
                [
                    "selected_quality",
                    "selected_balanced",
                ],
        )
    ]


def test_question_axes_have_single_owners() -> None:
    namespace = runpy.run_path(str(COLLECT))
    questions = {question.title[:3]: question for question in namespace["QUESTIONS"]}

    assert namespace["FEEDFORWARD_COSINE_TITLE"] in questions["rq4"].tables
    assert namespace["FEEDFORWARD_COSINE_TITLE"] not in questions["rq8"].tables
    assert namespace["POSITION_COSINE_TITLE"] in questions["rq7"].tables
    assert namespace["POSITION_COSINE_TITLE"] not in questions["rq8"].tables
    assert namespace["POSITION_BASE_TITLE"] not in questions["rq7"].tables
    assert not any(
        table in questions["rq8"].tables
        for table in namespace["ARCHITECTURE_BASE_TITLES"].values()
    )


@pytest.mark.parametrize(
    ("name", "negative_sampling"),
    [
        ("neg_online_logq", "online_logq"),
        ("neg_random", "random"),
        ("neg_in_batch_no_logq", "in_batch_no_logq"),
        ("neg_mixed_online_logq", "mixed_online_logq"),
        ("neg_mixed_offline_logq", "mixed_offline_logq"),
    ],
)
def test_negative_sampling_variants_change_only_the_sampling_method(
    monkeypatch: pytest.MonkeyPatch, name: str, negative_sampling: str
) -> None:
    namespace = _namespace(monkeypatch)
    variant = namespace["VARIANTS"][name]
    baseline = namespace["VARIANTS"]["baseline"]

    assert variant == replace(
        baseline,
        run_name=f"g1_calibrated_{name}_ts2_500m",
        negative_sampling=negative_sampling,
    )


def test_fixed_logq_objectives_have_descriptive_distinct_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variants = _namespace(monkeypatch)["VARIANTS"]
    yi2019 = variants["neg_fixed_inbatch_global_q_yi2019"]
    leave_one_out = variants["neg_fixed_inbatch_leave_one_out"]

    assert yi2019.negative_sampling == leave_one_out.negative_sampling == "offline_logq"
    assert yi2019.logq_correction == "yi2019"
    assert yi2019.correct_positive_logq
    assert leave_one_out.logq_correction == "baseline"
    assert not yi2019.mask_false_negatives
    assert not yi2019.exclude_own_group_negatives
    assert not leave_one_out.mask_false_negatives
    assert not leave_one_out.exclude_own_group_negatives


@pytest.mark.parametrize(
    "name",
    [
        "neg_fixed_inbatch_global_q_yi2019",
        "neg_streaming_inbatch_global_q_yi2019",
        "neg_popularity_random_global_q_yi2019",
    ],
)
def test_global_q_objectives_do_not_apply_query_dependent_sampling_changes(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    variant = _namespace(monkeypatch)["VARIANTS"][name]

    assert variant.correct_positive_logq
    assert not variant.mask_false_negatives
    assert not variant.exclude_own_group_negatives


@pytest.mark.parametrize(
    "name",
    [
        "neg_mixed_streaming_logq_negative_only",
        "neg_mixed_fixed_logq_negative_only",
    ],
)
def test_mixed_objectives_correct_only_the_logq_negative_component(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    variant = _namespace(monkeypatch)["VARIANTS"][name]

    assert not variant.correct_positive_logq
    assert not variant.mask_false_negatives
    assert not variant.exclude_own_group_negatives


def test_rq11_aggregate_streaming_variants_are_globally_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variants = _namespace(monkeypatch)["VARIANTS"]
    primary = variants["neg_aggregate_uniform_streaming_global_q_yi2019"]
    diagnostic = variants["neg_aggregate_uniform_streaming_global_q_negative_only"]

    assert primary.negative_sampling == "mixed_online_global_q"
    assert primary.correct_positive_logq
    assert diagnostic.negative_sampling == "mixed_online_global_q_negative_only"
    assert not diagnostic.correct_positive_logq
    for variant in (primary, diagnostic):
        assert not variant.mask_false_negatives
        assert not variant.exclude_own_group_negatives


def test_negative_sampling_question_has_one_controlled_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    variants = _namespace(monkeypatch)["VARIANTS"]
    question = next(
        question
        for question in namespace["QUESTIONS"]
        if question.title.startswith("rq11 ")
    )

    assert question.tables == [namespace["NEGATIVE_SAMPLING_TITLE"]]
    assert "neg_offline_logq" not in variants


def test_random_logq_uses_random_negatives_with_offline_correction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variant = _namespace(monkeypatch)["VARIANTS"]["neg_random_offline_logq"]

    assert variant.negative_sampling == "random_offline_logq"


def test_mu_transfer_table_has_same_width_standard_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    names = dict(namespace["AXES"])[namespace["MUTRANSFER_TITLE"]]

    assert "cosine_dim_32" in names
    assert "cosine_dim_128" in names
    assert set(names) == {
        "cosine_dim_32",
        "lr_cosine_warmup",
        "cosine_dim_128",
        "mup_dim32_lr5e2",
        "mup_dim32_lr1e1",
        "mup_dim128_lr5e2",
        "mup_dim128_lr1e1",
    }


def test_inverse_sqrt_control_scales_its_timescale_with_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variant = _namespace(monkeypatch)["VARIANTS"]["lr_inverse_sqrt"]

    assert variant.lr_schedule.timescale_steps is None
    assert variant.lr_schedule.timescale_fraction == 0.05


@pytest.mark.parametrize("name", ["baseline", "homework_reproduction"])
def test_g1_baseline_matches_the_homework_reference_contract(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    variant = _namespace(monkeypatch)["VARIANTS"][name]

    assert variant.user_sample is None
    assert variant.window == "next_item"
    assert variant.event_type_filter == "like"
    assert variant.min_item_interactions_per_item == 5
    assert variant.validation_interval_seconds == 7 * 24 * 60 * 60
    assert variant.day_range.end_day == 300
    assert variant.evaluation_catalog == "all"
    assert not variant.exclude_seen_from_evaluation
    assert variant.num_epochs == 20
    assert variant.eval_every_n_epochs == 1
    assert variant.early_stopping_patience == 3
    assert variant.early_stopping_min_delta == 0
    assert variant.restore_best_weights
    assert variant.negative_sampling == "offline_logq"
    assert not variant.correct_positive_logq
    assert not variant.mask_false_negatives
    assert not variant.exclude_own_group_negatives
    assert variant.embedding_learning_rate == variant.deep_learning_rate == 1e-3
    assert variant.weight_decay == 0
    assert variant.transformer.dim == 64
    assert variant.transformer.nhead == variant.transformer.num_kv_heads == 2
    assert variant.transformer.ffn == "gelu"
    assert variant.transformer.ffn_intermediate_dim == 256
    assert variant.transformer.ffn_dropout == 0.1
    assert variant.transformer.norm == "layer"
    assert variant.transformer.learned_positions == "forward"
    assert not variant.transformer.alibi
    assert variant.initializer_std == 0.02
    assert variant.dataloader.batch_size == 128
    assert variant.dataloader.val_batch_size == 8192


def test_architecture_axes_are_one_factor_changes_from_the_corrected_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variants = _namespace(monkeypatch)["VARIANTS"]
    baseline = variants["baseline"].transformer

    assert [
        variants[name].transformer.dim
        for name in ("dim_16", "dim_32", "dim_128", "dim_256")
    ] == [16, 32, 128, 256]
    assert [
        variants[name].transformer.ffn_intermediate_dim
        for name in ("dim_16", "dim_32", "dim_128", "dim_256")
    ] == [64, 128, 512, 1024]
    assert [
        variants[name].transformer.nhead for name in ("heads_1", "heads_4", "heads_8")
    ] == [1, 4, 8]
    assert all(
        variants[name].transformer.nhead == variants[name].transformer.num_kv_heads
        for name in ("heads_1", "heads_4", "heads_8")
    )
    assert variants["heads_gqa"].transformer.nhead == baseline.nhead == 2
    assert variants["heads_gqa"].transformer.num_kv_heads == 1
    assert variants["seq_12"].max_seq_len == 12
    assert variants["seq_25"].max_seq_len == 25
    assert variants["seq_128"].max_seq_len == 128
    assert variants["cls"].effective_cls_token_mode == "end_only"
    assert variants["cls_interleaved"].effective_cls_token_mode == "interleaved"
    assert variants["dropout_0"].transformer.dropout == 0
    assert variants["dropout_0"].transformer.input_dropout == 0
    assert variants["dropout_0"].transformer.ffn_dropout == 0
    for name, probability in (("dropout_30", 0.3), ("dropout_50", 0.5)):
        transformer = variants[name].transformer
        assert transformer.dropout == probability
        assert transformer.input_dropout == probability
        assert transformer.ffn_dropout == probability


def test_position_and_feedforward_axes_have_no_duplicate_baseline_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variants = _namespace(monkeypatch)["VARIANTS"]
    baseline = variants["baseline"].transformer

    assert variants["ffn_swiglu"].transformer.ffn == "swiglu"
    assert variants["ffn_swiglu"].transformer.ffn_intermediate_dim == 256
    assert variants["ffn_swiglu_matched"].transformer.ffn_intermediate_dim == 171
    assert variants["pos_alibi"].transformer.alibi
    assert variants["pos_alibi"].transformer.learned_positions is None
    assert variants["pos_all"].transformer.alibi
    assert variants["pos_all"].transformer.rope == "forward"
    assert variants["pos_all"].transformer.learned_positions == "forward"
    learned_both = variants["pos_learned_forward_reverse"].transformer
    assert learned_both.learned_positions == ("forward", "reverse")
    assert learned_both.rope is None
    assert not learned_both.alibi
    for name in (
        "ffn_swiglu",
        "ffn_swiglu_matched",
        "pos_none",
        "pos_alibi",
        "pos_learned_reverse",
        "pos_learned_forward_reverse",
        "pos_rope",
        "pos_rope_reverse",
        "pos_rope_alibi",
        "pos_rope_learned",
        "pos_rope_learned_reverse",
        "pos_rope_reverse_learned_reverse",
        "pos_rope_reverse_learned",
        "pos_rope_reverse_alibi",
        "pos_learned_alibi",
        "pos_learned_reverse_alibi",
        "pos_rope_learned_reverse_alibi",
        "pos_rope_reverse_learned_alibi",
        "pos_all",
        "pos_reverse_all",
        "norm_rms",
        "norm_batch",
        "norm_all_rms",
        "norm_input_layer",
        "norm_input_rms",
        "norm_no_final",
    ):
        assert variants[name].transformer != baseline


def test_architecture_variants_do_not_schedule_duplicate_post_layer_norm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _namespace(monkeypatch)

    assert "norm_post_layer" not in namespace["VARIANTS"]
    assert "norm_post_layer" not in namespace["ARCHITECTURE_VARIANTS"]


def test_homework_reproduction_is_only_a_compatibility_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variants = _namespace(monkeypatch)["VARIANTS"]

    assert variants["baseline"] == replace(
        variants["homework_reproduction"],
        run_name="g1_calibrated_baseline_ts2_500m",
    )


def test_validation_batch_tuning_uses_separate_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("G1_VARIANT", "baseline")
    monkeypatch.setenv("G1_DATASET_SIZE", "50m")
    monkeypatch.setenv("G1_VAL_BATCH_SIZE", "2048")

    experiment = runpy.run_path(str(SCRIPT))["experiment"]

    assert experiment.dataloader.val_batch_size == 2048
    assert experiment.run_name == "g1_calibrated_baseline_ts2_50m_val2048"


def test_validation_batch_tuning_rejects_non_positive_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("G1_VARIANT", "baseline")
    monkeypatch.setenv("G1_VAL_BATCH_SIZE", "0")

    with pytest.raises(ValueError, match="G1_VAL_BATCH_SIZE"):
        runpy.run_path(str(SCRIPT))


@pytest.mark.parametrize(
    ("name", "rate"),
    [
        ("embedding_lr_1e4", 1e-4),
        ("embedding_lr_2e4", 2e-4),
        ("deep_lr_3e3", 3e-3),
        ("deep_lr_5e3", 5e-3),
    ],
)
def test_followup_learning_rate_variants_extend_both_sides_of_the_grid(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    rate: float,
) -> None:
    variant = _namespace(monkeypatch)["VARIANTS"][name]

    selected = (
        variant.embedding_learning_rate
        if name.startswith("embedding")
        else variant.deep_learning_rate
    )
    assert selected == rate
    assert variant.lr_schedule == _namespace(monkeypatch)["COSINE_WARMUP"]


@pytest.mark.parametrize(
    ("name", "cycles"), [("lr_cosine_cycles2", 2), ("lr_cosine_cycles4", 4)]
)
def test_cosine_restart_variants_change_only_the_cycle_count(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    cycles: int,
) -> None:
    variant = _namespace(monkeypatch)["VARIANTS"][name]

    assert variant.lr_schedule.shape == "cosine"
    assert variant.lr_schedule.warmup_fraction == 0.05
    assert variant.lr_schedule.cycles == cycles


@pytest.mark.parametrize(
    ("name", "num_bins", "rope"),
    [
        ("time_bins_8", 8, None),
        ("time_bins_16", 16, None),
        ("time_bins_64", 64, None),
        ("time_bins_reverse_rope", 32, "timestamp_reverse"),
    ],
)
def test_timestamp_followups_cover_bin_resolution_and_reverse_rope(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    num_bins: int,
    rope: str | None,
) -> None:
    variant = _namespace(monkeypatch)["VARIANTS"][name]

    assert variant.timestamp_delta == "bins"
    assert variant.timestamp_num_bins == num_bins
    assert variant.transformer.rope == rope


def test_followup_sweep_is_targeted_and_seeded() -> None:
    source = FOLLOWUP_SWEEP.read_text()

    for variant in (
        "lr_cosine_warmup",
        "embedding_lr_1e4",
        "deep_lr_5e3",
        "cosine_ffn_swiglu",
        "lr_cosine_cycles2",
        "cosine_pos_learned_reverse",
        "cosine_seq_128",
        "cosine_dropout_0",
        "cosine_heads_gqa",
        "cosine_norm_post",
        "window_50",
        "time_bins_reverse_rope",
        "neg_random_offline_logq",
        "mup_dim128_lr5e2",
        "selected_quality",
        "selected_balanced",
    ):
        assert variant in source
    assert 'G1_SEEDS=${G1_SEEDS:-"0 1 2 3"}' in source


@pytest.mark.parametrize(
    ("variable", "value"),
    [("G1_MAX_USERS", "0001"), ("G1_VAL_BATCH_SIZE", "02048")],
)
def test_tuning_overrides_reject_noncanonical_numbers(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    value: str,
) -> None:
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValueError, match=variable):
        runpy.run_path(str(SCRIPT))


def test_metric_regression_clears_validation_tuning_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("G1_VAL_BATCH_SIZE", "2048")
    monkeypatch.delitem(
        sys.modules,
        "experiments.g1_sasrec_item_ids_likes.configs.variant",
        raising=False,
    )

    experiment = runpy.run_path(str(METRIC_REGRESSION))["experiment"]

    assert experiment.dataloader.val_batch_size == 8192
    assert experiment.run_name == "g1_metric_regression_50m_b128_s0"


def test_every_g1_variant_uses_the_corrected_data_and_evaluation_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variants = _namespace(monkeypatch)["VARIANTS"].values()

    for variant in variants:
        assert variant.size == "500m"
        assert variant.run_name.endswith("_500m")
        assert variant.user_sample is None
        assert variant.event_type_filter == "like"
        assert variant.min_item_interactions_per_item == 5
        assert variant.drop_unmapped_items
        assert variant.window == "next_item"
        assert variant.validation_interval_seconds == 7 * 24 * 60 * 60
        assert variant.day_range.end_day == 300
        assert variant.evaluation_catalog == "all"
        assert not variant.exclude_seen_from_evaluation


@pytest.mark.parametrize(
    ("launcher", "queue_label"),
    [
        (
            SEED_SWEEP,
            "g1_calibrated_${variant}${cap}_ts2_${dataset_size}${run_suffix}_s${seed}",
        ),
    ],
)
def test_seeded_sweep_queue_labels_match_calibrated_artifact_directories(
    launcher: Path,
    queue_label: str,
) -> None:
    source = launcher.read_text()

    assert f'enqueue "{queue_label}"' in source


def test_final_sweep_runs_once_unless_a_seed_is_explicit() -> None:
    source = FINAL_SWEEP.read_text()

    assert 'enqueue "g1_calibrated_${variant}${cap}_ts2_500m"' in source
    assert 'enqueue "g1_calibrated_${variant}${cap}_ts2_500m_s${G1_SEED}"' in source
    assert "G1_SEEDS" not in source


@pytest.mark.parametrize("launcher", [SWEEP, SEED_SWEEP])
def test_tuning_sweep_queue_labels_include_artifact_suffixes(launcher: Path) -> None:
    source = launcher.read_text()

    assert "${G1_MAX_USERS}users_seed42" in source
    assert "_val${G1_VAL_BATCH_SIZE}" in source
    assert "${run_suffix}" in source


def test_final_sweep_clears_validation_batch_tuning_override() -> None:
    assert "unset G1_VAL_BATCH_SIZE" in FINAL_SWEEP.read_text()


def test_g1_tuning_scale_is_explicit_and_keeps_separate_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("G1_VARIANT", "baseline")
    monkeypatch.setenv("G1_DATASET_SIZE", "50m")
    monkeypatch.setenv("G1_MAX_USERS", "1000")

    variant = runpy.run_path(str(SCRIPT))["experiment"]

    assert variant.size == "50m"
    assert variant.user_sample.max_users == 1_000
    assert variant.run_name == "g1_calibrated_baseline_ts2_50m_1000users_seed42"


def test_renamed_report_title_finds_legacy_table() -> None:
    namespace = runpy.run_path(str(COLLECT))
    legacy = "### Depth under cosine warmup\n\nlegacy"
    previous = {"Depth under cosine warmup": legacy}
    title = namespace["ARCHITECTURE_COSINE_TITLES"]["Depth"]

    assert namespace["_previous_table"](previous, title) == legacy


@pytest.mark.parametrize("renderer", ["render", "render_questions"])
def test_legacy_raw_result_renderers_are_disabled(renderer: str) -> None:
    namespace = runpy.run_path(str(COLLECT))

    with pytest.raises(RuntimeError, match="legacy raw-result renderer is disabled"):
        namespace[renderer]("archived pre-calibration tables")


def test_collector_ignores_homework_compatibility_alias(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    logs = tmp_path / "logs"
    for run_name in [
        *(f"g1_calibrated_baseline_500m_s{seed}" for seed in range(4)),
        "g1_calibrated_homework_reproduction_500m_s0",
    ]:
        directory = logs / run_name
        directory.mkdir(parents=True)
        (directory / "final_metrics.json").write_text("{}")
    monkeypatch.setitem(namespace["_collect"].__globals__, "GENERATED", tmp_path)

    assert set(namespace["_collect"]()) == {"baseline"}


def test_calibrated_question_notes_are_hypotheses_not_legacy_results() -> None:
    questions = runpy.run_path(str(COLLECT))["QUESTIONS"]

    assert all(question.note.startswith("Hypothesis:") for question in questions)


def test_split_lr_question_tables_preserve_only_their_legacy_rows() -> None:
    namespace = runpy.run_path(str(COLLECT))
    table = namespace["_table"]
    baseline = [{"recall@100": 0.1}]
    previous = "\n\n".join(
        [
            "## rq3",
            "**Embedding and deep learning rates**",
            table(
                [
                    ("lr_cosine_warmup", baseline),
                    ("embedding_lr_2e3", [{"recall@100": 0.12}]),
                    ("deep_lr_5e4", [{"recall@100": 0.11}]),
                ],
                "lr_cosine_warmup",
                baseline,
                costs=False,
            ),
        ]
    )
    current = "\n\n".join(
        [
            "## rq3",
            f"**{namespace['EMBEDDING_LR_TITLE']}**",
            table([], "lr_cosine_warmup", [], costs=False),
            f"**{namespace['DEEP_LR_TITLE']}**",
            table([], "lr_cosine_warmup", [], costs=False),
        ]
    )

    merged = namespace["_merge_question_tables"](
        previous,
        current,
        {
            namespace["EMBEDDING_LR_TITLE"]: "lr_cosine_warmup",
            namespace["DEEP_LR_TITLE"]: "lr_cosine_warmup",
        },
        allowed_rows={
            namespace["EMBEDDING_LR_TITLE"]: {
                "lr_cosine_warmup",
                "embedding_lr_2e3",
            },
            namespace["DEEP_LR_TITLE"]: {
                "lr_cosine_warmup",
                "deep_lr_5e4",
            },
        },
    )

    embedding, deep = merged.split(f"**{namespace['DEEP_LR_TITLE']}**")
    assert "embedding_lr_2e3" in embedding
    assert "deep_lr_5e4" not in embedding
    assert "deep_lr_5e4" in deep
    assert "embedding_lr_2e3" not in deep


def test_unrun_cosine_axis_does_not_render_a_baseline_only_table() -> None:
    namespace = runpy.run_path(str(COLLECT))

    rows, _, _ = namespace["_by_score"](
        "Dimension under cosine warmup",
        ["cosine_dim_32"],
        {"lr_cosine_warmup": [{"recall@100": 0.12}]},
    )

    assert rows == []


def test_stale_baseline_does_not_replace_scheduled_table_reference() -> None:
    namespace = runpy.run_path(str(COLLECT))
    table = namespace["_table"]
    scheduled = [{"recall@100": 0.12}]
    current = "### Crosses\n\n" + table(
        [("lr_cosine_warmup", scheduled), ("combo", [{"recall@100": 0.124}])],
        "lr_cosine_warmup",
        scheduled,
        costs=False,
    )
    previous = "### Crosses\n\n" + table(
        [("baseline", [{"recall@100": 0.09}])],
        "baseline",
        [{"recall@100": 0.09}],
        costs=False,
    )

    merged = namespace["_merge_table_rows"](previous, current, "lr_cosine_warmup")

    assert "| baseline |" not in merged
    assert "| lr_cosine_warmup | lr_cosine_warmup | 1 | 0.120 |" in merged
    assert "| combo | — | 1 | +3% (0.124) |" in merged


def test_preserved_baseline_remains_the_reference_when_its_log_is_gone() -> None:
    namespace = runpy.run_path(str(COLLECT))
    table = namespace["_table"]
    previous = "### Position\n\n" + table(
        [("baseline", [{"recall@100": 0.1}])],
        "baseline",
        [{"recall@100": 0.1}],
        costs=False,
    )
    current = "### Position\n\n" + table(
        [("rope", [{"recall@100": 0.09}])], "baseline", [], costs=False
    )

    merged = namespace["_merge_table_rows"](previous, current, "baseline")

    assert "| baseline | baseline | 1 | 0.100 |" in merged
    assert "| rope | — | 1 | -10% (0.090) |" in merged


def test_each_table_preserves_its_own_reference_description(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    baseline = [{"recall@100": 0.1}]
    depth_title = namespace["ARCHITECTURE_BASE_TITLES"]["Depth"]
    heads_title = namespace["ARCHITECTURE_BASE_TITLES"][
        "Number of attention heads"
    ]
    existing = "\n\n".join(
        [
            namespace["PROVENANCE_MARKER"],
            f"### {depth_title}\n\n"
            + namespace["_table"](
                [("baseline", baseline)],
                "baseline",
                baseline,
                title=depth_title,
            ),
            f"### {heads_title}\n\n"
            + namespace["_table"](
                [("baseline", baseline)],
                "baseline",
                baseline,
                title=heads_title,
            ),
        ]
    )
    monkeypatch.setitem(
        namespace["render"].__globals__,
        "_collect",
        lambda: {
            "depth_1": [{"recall@100": 0.09}],
            "heads_1": [{"recall@100": 0.08}],
        },
    )

    with pytest.raises(RuntimeError, match="legacy raw-result renderer is disabled"):
        namespace["render"](existing)


def test_reference_from_another_table_repairs_pruned_logs() -> None:
    namespace = runpy.run_path(str(COLLECT))
    table = namespace["_table"]
    current = "### Schedule\n\n" + table(
        [("cosine", [{"recall@100": 0.12}])], "baseline", [], costs=False
    )
    fallback = (
        "| baseline | constant LR=0.001 | 4 | 0.1000 | 0.0200 | 0.0100 | "
        "0.0050 | 0.2000 |"
    )

    merged = namespace["_merge_table_rows"]("", current, "baseline", fallback)

    assert "| baseline | constant LR=0.001 | 4 | 0.1000 |" in merged
    assert "| cosine | — | 1 | +20% (0.120) |" in merged


def test_preserved_rows_match_the_target_table_width() -> None:
    namespace = runpy.run_path(str(COLLECT))
    with_costs = (
        "| model | 4 | 0.1200 | 0.0300 | 0.0200 | 0.0100 | 0.2000 | "
        "3.0 | 4.0 | 1.000M | 2.0M | 10 |"
    )
    without_costs = "| model | 4 | 0.1200 | 0.0300 | 0.0200 | 0.0100 | 0.2000 |"

    rows = namespace["_table_rows"](f"{with_costs}\n{without_costs}", 12)

    assert rows == {"model": with_costs}


def test_merge_replaces_a_wrong_width_local_reference() -> None:
    namespace = runpy.run_path(str(COLLECT))
    table = namespace["_table"]
    current = "### Cost\n\n" + table(
        [("model", [{"recall@100": 0.12}])], "baseline", [], costs=True
    )
    wrong_local = (
        "### Cost\n\n| variant | runs | recall@100 | ndcg@100 | recall@10 | "
        "ndcg@10 | coverage@100 |\n| --- | --- | --- | --- | --- | --- | --- |\n"
        "| baseline | 4 | 0.1000 | 0.0200 | 0.0100 | 0.0050 | 0.2000 |"
    )
    fallback = (
        "| baseline | corrected baseline | 4 | 0.1000 | 0.0200 | 0.0100 | "
        "0.0050 | 0.2000 | 3.0 | 4.0 | 1.000M | 2.0M | 10 |"
    )

    merged = namespace["_merge_table_rows"](wrong_local, current, "baseline", fallback)

    assert fallback in merged
    assert "| model | — | 1 | +20% (0.120) |" in merged


def test_merge_rejects_a_same_width_reordered_schema() -> None:
    namespace = runpy.run_path(str(COLLECT))
    table = namespace["_table"]
    current = "### Quality\n\n" + table(
        [("model", [{"recall@100": 0.12}])], "baseline", [], costs=False
    )
    previous = current.replace(
        "recall@100 | ndcg@100", "ndcg@100 | recall@100"
    ).replace("| model |", "| baseline |")
    fallback = (
        "| baseline | corrected baseline | 4 | 0.1000 | 0.0200 | 0.0100 | "
        "0.0050 | 0.2000 |"
    )

    merged = namespace["_merge_table_rows"](previous, current, "baseline", fallback)

    assert fallback in merged
    assert "| model | — | 1 | +20% (0.120) |" in merged


def test_removed_question_tables_are_not_preserved() -> None:
    namespace = runpy.run_path(str(COLLECT))
    table = namespace["_table"]
    baseline = [{"recall@100": 0.1}]
    previous = "## rq\n\n**Old axis**\n\n" + table(
        [("baseline", baseline)], "baseline", baseline, costs=False
    )
    current = "## rq\n\n**Current axis**\n\n" + table(
        [("baseline", baseline)], "baseline", baseline, costs=False
    )

    merged = namespace["_merge_question_tables"](
        previous, current, {"Current axis": "baseline"}
    )

    assert "Current axis" in merged
    assert "Old axis" not in merged


def test_collect_prefers_legacy_unsuffixed_seed_over_explicit_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    monkeypatch.setitem(namespace["_collect"].__globals__, "GENERATED", tmp_path)
    for name, score in [
        ("g1_calibrated_example", 0.2),
        ("g1_calibrated_example_s0", 0.1),
        ("g1_calibrated_example_s1", 0.3),
        ("g1_calibrated_example_s2", 0.4),
        ("g1_calibrated_example_s3", 0.5),
        ("g1_calibrated_example_s4", 0.9),
        ("g1_calibrated_perf_probe", 0.9),
        ("g1_homework_reproduction_500m", 0.9),
    ]:
        run = tmp_path / "logs" / name
        run.mkdir(parents=True)
        (run / "final_metrics.json").write_text(json.dumps({"recall@100": score}))

    collected = namespace["_collect"]("50m")

    assert set(collected) == {"example"}
    assert [run["recall@100"] for run in collected["example"]] == [
        0.3,
        0.4,
        0.5,
        0.2,
    ]


def test_legacy_collector_ignores_ambiguous_unsuffixed_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    monkeypatch.setitem(namespace["_collect"].__globals__, "GENERATED", tmp_path)
    run = tmp_path / "logs" / "g1_calibrated_example"
    run.mkdir(parents=True)
    (run / "final_metrics.json").write_text(json.dumps({"recall@100": 0.2}))

    assert namespace["_collect"]("50m") == {}


def test_collect_uses_explicit_final_seeds_and_ignores_calibration_alias(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    monkeypatch.setitem(namespace["_collect"].__globals__, "GENERATED", tmp_path)
    for name, score in [
        ("g1_calibrated_example_500m", 0.9),
        ("g1_calibrated_example_500m_s0", 0.2),
        ("g1_calibrated_example_500m_s1", 0.3),
        ("g1_calibrated_example_500m_s2", 0.4),
        ("g1_calibrated_example_500m_s3", 0.5),
        ("g1_example", 0.1),
    ]:
        run = tmp_path / "logs" / name
        run.mkdir(parents=True)
        (run / "final_metrics.json").write_text(json.dumps({"recall@100": score}))

    collected = namespace["_collect"]()

    assert set(collected) == {"example"}
    assert [run["recall@100"] for run in collected["example"]] == [
        0.2,
        0.3,
        0.4,
        0.5,
    ]


def test_collect_ignores_mu_transfer_runs_from_before_the_initialization_fix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    monkeypatch.setitem(namespace["_collect"].__globals__, "GENERATED", tmp_path)
    for seed in range(4):
        for variant in ("mup_dim32_lr2e3", "mup_dim32_lr5e2"):
            run = tmp_path / "logs" / f"g1_calibrated_{variant}_500m_s{seed}"
            run.mkdir(parents=True)
            (run / "final_metrics.json").write_text(
                json.dumps({"recall@100": 0.1})
            )

    collected = namespace["_collect"]()

    assert set(collected) == {"mup_dim32_lr5e2"}


def test_collect_ignores_complete_legacy_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    namespace = runpy.run_path(str(COLLECT))
    monkeypatch.setitem(namespace["_collect"].__globals__, "GENERATED", tmp_path)
    for seed in range(4):
        for prefix, score in (("g1", 0.1), ("g1_calibrated", 0.2)):
            run = tmp_path / "logs" / f"{prefix}_example_500m_s{seed}"
            run.mkdir(parents=True)
            (run / "final_metrics.json").write_text(json.dumps({"recall@100": score}))

    collected = namespace["_collect"]()

    assert set(collected) == {"example"}
    assert {run["recall@100"] for run in collected["example"]} == {0.2}


@pytest.mark.parametrize(
    ("name", "rope"),
    [
        ("time_rope", "timestamp"),
        ("time_rope_reverse", "timestamp_reverse"),
        ("time_log_rope", "timestamp_log"),
        ("time_log_rope_reverse", "timestamp_log_reverse"),
    ],
)
def test_timestamp_rope_variants_use_the_scheduled_baseline(
    monkeypatch: pytest.MonkeyPatch, name: str, rope: str
) -> None:
    variant = _namespace(monkeypatch)["VARIANTS"][name]

    assert variant.transformer.rope == rope
    assert variant.lr_schedule == _namespace(monkeypatch)["COSINE_WARMUP"]


@pytest.mark.parametrize(
    ("name", "kind", "combination"),
    [
        ("time_plain_add", "plain", "add"),
        ("time_log_add", "log", "add"),
        ("time_bins_add", "bins", "add"),
        ("time_log_concat", "log", "concat"),
        ("time_bins_concat", "bins", "concat"),
    ],
)
def test_timestamp_embedding_variants_are_runnable(
    monkeypatch: pytest.MonkeyPatch, name: str, kind: str, combination: str
) -> None:
    variant = _namespace(monkeypatch)["VARIANTS"][name]

    assert variant.timestamp_delta == kind
    assert variant.timestamp_combination == combination
    assert variant.lr_schedule == _namespace(monkeypatch)["COSINE_WARMUP"]


def test_remaining_scaling_variants_are_scheduled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _namespace(monkeypatch)
    variants = namespace["VARIANTS"]

    assert variants["window_25"].transformer.attention_window == 25
    assert variants["window_50"].transformer.attention_window == 50
    assert variants["per_layer_embeddings"].per_layer_item_embeddings
    assert all(
        variants[name].lr_schedule == namespace["COSINE_WARMUP"]
        for name in ["window_25", "window_50", "per_layer_embeddings"]
    )


def test_additional_scheduler_variants_are_runnable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variants = _namespace(monkeypatch)["VARIANTS"]

    assert variants["lr_step"].lr_schedule.shape == "step"
    assert variants["lr_exponential"].lr_schedule.shape == "exponential"
    assert variants["lr_polynomial"].lr_schedule.shape == "polynomial"
    assert variants["lr_wsd"].lr_schedule == replace(
        variants["lr_wsd"].lr_schedule,
        shape="warmup_stable_decay",
        warmup_fraction=0.05,
    )


@pytest.mark.parametrize(
    ("name", "embedding_rate", "deep_rate"),
    [
        ("embedding_lr_5e4", 5e-4, 1e-3),
        ("embedding_lr_2e3", 2e-3, 1e-3),
        ("embedding_lr_3e3", 3e-3, 1e-3),
        ("embedding_lr_5e3", 5e-3, 1e-3),
        ("deep_lr_5e4", 1e-3, 5e-4),
        ("deep_lr_2e3", 1e-3, 2e-3),
    ],
)
def test_learning_rate_axes_change_only_the_named_rate(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    embedding_rate: float,
    deep_rate: float,
) -> None:
    variant = _namespace(monkeypatch)["VARIANTS"][name]

    assert variant.embedding_learning_rate == embedding_rate
    assert variant.deep_learning_rate == deep_rate
    assert variant.lr_schedule == _namespace(monkeypatch)["COSINE_WARMUP"]


def test_final_rate_combinations_cross_the_screen_winners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variants = _namespace(monkeypatch)["VARIANTS"]

    rates = variants["combo_lr_rates"]
    timestamp = variants["combo_embedding_time"]
    assert (rates.embedding_learning_rate, rates.deep_learning_rate) == (2e-3, 5e-4)
    assert timestamp.embedding_learning_rate == 2e-3
    assert timestamp.timestamp_delta == "bins"
    assert timestamp.transformer.rope == "timestamp_log"

    position = variants["combo_embedding_position"]
    assert position.embedding_learning_rate == 2e-3
    assert position.transformer.rope == "reverse"
    assert position.transformer.learned_positions == "reverse"
    assert not position.transformer.alibi


def test_normalization_kind_and_placement_variants_are_runnable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variants = _namespace(monkeypatch)["VARIANTS"]

    assert variants["norm_batch"].transformer.norm == "batch"
    assert variants["norm_rms"].transformer.norm == "rms"
    assert variants["norm_all_rms"].transformer.input_norm == "rms"
    assert variants["norm_all_rms"].transformer.final_norm == "rms"
    assert variants["norm_input_layer"].transformer.input_norm == "layer"
    assert variants["norm_input_rms"].transformer.input_norm == "rms"
    assert variants["norm_no_final"].transformer.final_norm is None


def test_selected_future_baselines_state_every_chosen_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variants = _namespace(monkeypatch)["VARIANTS"]
    quality = variants["selected_quality"]
    balanced = variants["selected_balanced"]

    for variant in (quality, balanced):
        assert variant.max_seq_len == 128
        assert variant.lr_schedule.shape == "linear"
        assert variant.lr_schedule_horizon_epochs == 20
        assert variant.lr_schedule.warmup_fraction == 0.0
        assert variant.timestamp_delta == "bins"
        assert variant.timestamp_num_bins == 16
        assert variant.embedding_learning_rate == 1e-3
        assert variant.deep_learning_rate == 3e-3
        assert variant.transformer.ffn == "swiglu"
        assert variant.transformer.ffn_intermediate_dim == 171
        assert variant.transformer.nhead == 2
        assert variant.transformer.num_kv_heads == 1
        assert variant.transformer.attention_window == 50
    assert quality.negative_sampling == "random"
    assert quality.dense_random_negative_scores
    assert quality.dataloader.batch_size == 128
    assert balanced.negative_sampling == "offline_logq"
    assert balanced.dataloader.batch_size == 128
    throughput = variants["selected_quality_b1024"]
    assert throughput.dataloader.batch_size == 1024
    assert throughput.run_name.endswith("selected_quality_b1024_ts2_500m")
    future = variants["future_baseline"]
    assert future.dataloader.batch_size == 512
    assert future.embedding_learning_rate == 32e-3
    assert future.deep_learning_rate == 12e-3
    assert future.transformer.input_norm == "rms"
    assert future.run_name.endswith("future_baseline_ts2_500m")
    tuning = variants["selected_quality_b1024_e2e3_d6e3"]
    assert tuning.dataloader.batch_size == 1024
    assert tuning.embedding_learning_rate == 2e-3
    assert tuning.deep_learning_rate == 6e-3
    scaled_tuning = variants["selected_quality_b1024_e12e3_d36e3"]
    assert scaled_tuning.embedding_learning_rate == 12e-3
    assert scaled_tuning.deep_learning_rate == 36e-3


@pytest.mark.parametrize(
    ("name", "dim", "deep_rate"),
    [
        ("mup_dim32_lr5e4", 32, 5e-4),
        ("mup_dim32_lr1e3", 32, 1e-3),
        ("mup_dim32_lr2e3", 32, 2e-3),
        ("mup_dim32_lr3e3", 32, 3e-3),
        ("mup_dim32_lr5e3", 32, 5e-3),
        ("mup_dim32_lr1e2", 32, 1e-2),
        ("mup_dim32_lr2e2", 32, 2e-2),
        ("mup_dim32_lr3e2", 32, 3e-2),
        ("mup_dim32_lr5e2", 32, 5e-2),
        ("mup_dim32_lr1e1", 32, 1e-1),
        ("mup_dim32_lr2e1", 32, 2e-1),
        ("mup_dim128_lr5e4", 128, 5e-4),
        ("mup_dim128_lr1e3", 128, 1e-3),
        ("mup_dim128_lr2e3", 128, 2e-3),
        ("mup_dim128_lr3e3", 128, 3e-3),
        ("mup_dim128_lr5e3", 128, 5e-3),
        ("mup_dim128_lr1e2", 128, 1e-2),
        ("mup_dim128_lr2e2", 128, 2e-2),
        ("mup_dim128_lr3e2", 128, 3e-2),
        ("mup_dim128_lr5e2", 128, 5e-2),
        ("mup_dim128_lr1e1", 128, 1e-1),
        ("mup_dim128_lr2e1", 128, 2e-1),
    ],
)
def test_mu_transfer_grid_uses_one_rate_grid_at_two_widths(
    monkeypatch: pytest.MonkeyPatch, name: str, dim: int, deep_rate: float
) -> None:
    variant = _namespace(monkeypatch)["VARIANTS"][name]

    assert type(variant).__name__ == "MuTransferGenerationExperiment"
    assert variant.item_embedding_dim == 64
    assert variant.transformer.dim == dim
    assert variant.transformer.ffn_intermediate_dim == 4 * dim
    assert variant.deep_learning_rate == deep_rate
    assert variant.lr_schedule_horizon_epochs == 20
