from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import statistics
import tempfile
from typing import Any

from dcn.training_metadata import GENERATION_TRAINING_SEMANTICS_REVISION
from experiments.g1_sasrec_item_ids_likes.analysis.rq10_reinvestigation_candidates import (
    Family,
    Rq10Candidate,
    candidate_by_run,
    initial_candidates,
)


_FAMILIES: tuple[Family, ...] = (
    "input_output_only",
    "direct_add",
    "concat_residual",
    "gemma_ple",
)
_INITIAL_WIDTHS = {
    "concat_residual": (16, 32, 64),
    "gemma_ple": (8, 16, 32),
}
_INITIAL_LRS = (0.006, 0.012, 0.024)
_METRICS = ("recall@100", "ndcg@100", "recall@10", "ndcg@10", "coverage@100")
_METRIC_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_RECALL_BAND = 0.003
_NDCG_BAND = 0.001


class Rq10ReportError(RuntimeError):
    pass


@dataclass(frozen=True)
class Run:
    candidate: Rq10Candidate
    best_epoch: int
    stopped_epoch: int
    validation_recall: float
    validation_ndcg: float
    params_total: int
    median_train_epoch_seconds: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class FamilyResolution:
    selected: Run | None
    required_followups: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return self.selected is not None and not self.required_followups


@dataclass(frozen=True)
class Rq10ReportBundle:
    reader_markdown: str
    tuning_markdown: str
    evidence: dict[str, object]


def collect_report_bundle(logs: Path) -> Rq10ReportBundle:
    runs = []
    for directory in sorted(logs.glob("g1_rq10_*_500m")):
        if not directory.is_dir():
            continue
        try:
            candidate = candidate_by_run(directory.name)
        except ValueError:
            continue
        files = [
            directory / name
            for name in (
                "training_metadata.json",
                "final_metrics.json",
                "sweep.log",
            )
        ]
        if not all(path.is_file() for path in files):
            continue
        runs.append(_load_run(directory, candidate))
    return build_report_bundle(runs)


def build_report_bundle(runs: list[Run]) -> Rq10ReportBundle:
    by_name = {run.candidate.run_name: run for run in runs}
    if len(by_name) != len(runs):
        raise Rq10ReportError("duplicate RQ10 artifact identity")
    resolutions = {family: _resolve_family(family, runs) for family in _FAMILIES}
    required_followups = sorted(
        {
            name
            for resolution in resolutions.values()
            for name in resolution.required_followups
        }
    )
    ready = not required_followups and all(
        resolution.ready for resolution in resolutions.values()
    )
    selected = {
        family: resolution.selected
        for family, resolution in resolutions.items()
        if resolution.selected is not None
    }
    comparisons = _comparisons(selected, resolutions)
    accepted_treatments = [
        selected[family]
        for family, outcome in comparisons.items()
        if outcome == "non_inferior"
    ]
    selected_treatment = _best(accepted_treatments)
    evidence: dict[str, object] = {
        "dataset_size": "500m",
        "claims_status": "ready" if ready else "pending",
        "acceptance_status": (
            "pending"
            if not ready
            else "accepted" if selected_treatment is not None else "not_met"
        ),
        "selection_rule": "validation recall@100, then same-epoch NDCG@100",
        "non_inferiority_bands": {
            "recall@100": _RECALL_BAND,
            "ndcg@100": _NDCG_BAND,
        },
        "required_followups": required_followups,
        "family_status": {
            family: "ready" if resolution.ready else "pending"
            for family, resolution in resolutions.items()
        },
        "selected": {family: _selected_record(run) for family, run in selected.items()},
        "selected_added_feature": (
            None
            if selected_treatment is None
            else {
                "family": selected_treatment.candidate.family,
                **_selected_record(selected_treatment),
            }
        ),
        "control_comparisons": comparisons,
        "architectural_mechanism_claims": "not_made",
        "unexpected_degradation_diagnostics": {
            "exact_initial_control_equality": "focused_test",
            "gradient_flow": "focused_test",
            "gate_trajectories": "not_recorded",
            "branch_hidden_rms": "not_recorded",
            "item_frequency_strata": "not_recorded",
        },
        "validated_native_artifacts": sorted(by_name),
        "historical_two_layer_comparison": {
            "selection_eligible": False,
            "input_output_only": {
                "runs": 4,
                "recall@100": 0.13950,
                "ndcg@100": 0.05334,
            },
            "direct_add": {"runs": 4, "recall@100": 0.13920, "ndcg@100": 0.05305},
        },
    }
    return Rq10ReportBundle(
        reader_markdown=_render_reader(selected, resolutions),
        tuning_markdown=_render_tuning(runs, resolutions),
        evidence=evidence,
    )


def write_report_bundle(
    bundle: Rq10ReportBundle, scratchpad: Path, evidence: Path
) -> dict[str, Path]:
    paths = {
        "tuning": scratchpad / "rq10_reinvestigation_tuning_500m.md",
        "reader": scratchpad / "rq10_reinvestigation_reader_500m.md",
        "evidence": evidence / "rq10_reinvestigation_results.json",
    }
    _write(paths["tuning"], bundle.tuning_markdown)
    _write(paths["reader"], bundle.reader_markdown)
    _write(
        paths["evidence"],
        json.dumps(bundle.evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return paths


def _load_run(directory: Path, candidate: Rq10Candidate) -> Run:
    try:
        metadata = json.loads((directory / "training_metadata.json").read_text())
        metrics = json.loads((directory / "final_metrics.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise Rq10ReportError(f"{candidate.run_name}: unreadable artifact") from error
    _validate_metadata(metadata, candidate)
    validation_recall, validation_ndcg = _selection_metrics(directory, metadata)
    params_total, median_train_epoch_seconds = _resource_metrics(directory)
    final_metrics = {
        metric: _bounded_metric(metrics, metric, candidate.run_name)
        for metric in _METRICS
    }
    _require_equal(metrics.get("num_users"), 37018, candidate.run_name, "num_users")
    return Run(
        candidate=candidate,
        best_epoch=_required_int(metadata, "best_epoch", candidate.run_name),
        stopped_epoch=_required_int(metadata, "stopped_epoch", candidate.run_name),
        validation_recall=validation_recall,
        validation_ndcg=validation_ndcg,
        params_total=params_total,
        median_train_epoch_seconds=median_train_epoch_seconds,
        metrics=final_metrics,
    )


def _validate_metadata(metadata: dict[str, Any], candidate: Rq10Candidate) -> None:
    context = candidate.run_name
    expected = {
        "training_semantics_revision": GENERATION_TRAINING_SEMANTICS_REVISION,
        "dataset_size": "500m",
        "seed": 42,
        "num_epochs": 20,
        "max_epochs": 20,
        "epochs_trained": 20,
        "stopped_epoch": 20,
        "early_stopped": False,
        "lr_horizon_complete": True,
        "selection_resolved": True,
        "batch_size": 1280,
        "effective_batch_size": 1280,
        "model_dim": 64,
        "item_embedding_dim": 64,
        "initializer_std": 0.02,
        "embedding_learning_rate": 0.064,
        "deep_learning_rate": candidate.deep_lr,
        "weight_decay": 0.0,
        "gradient_clip_norm": None,
        "runtime_dtype": "torch.bfloat16",
        "negative_sampling": "random",
        "lr_schedule_horizon_epochs": 20,
    }
    for key, value in expected.items():
        _require_equal(metadata.get(key), value, context, key)
    if not metadata.get("lr_horizon_complete") or not metadata.get(
        "selection_resolved"
    ):
        raise Rq10ReportError(f"{context}: run must be horizon-complete")
    best_epoch = _required_int(metadata, "best_epoch", context)
    if not 1 <= best_epoch <= 20:
        raise Rq10ReportError(f"{context}: best_epoch must be in [1, 20]")
    invariants = metadata.get("transfer_invariants")
    if not isinstance(invariants, dict):
        raise Rq10ReportError(f"{context}: transfer_invariants is absent")
    family = "none" if candidate.family == "input_output_only" else candidate.family
    invariant_expected = {
        "experiment_class": "MuTransferGenerationExperiment",
        "dataset_size": "500m",
        "batch_size": 1280,
        "physical_batch_size": 1280,
        "effective_batch_size": 1280,
        "gradient_accumulation_steps": 1,
        "model_dim": 64,
        "item_embedding_dim": 64,
        "per_layer_item_features": family,
        "per_layer_item_feature_dim": (
            candidate.feature_width
            if candidate.family in {"concat_residual", "gemma_ple"}
            else None
        ),
        "negative_sampling": "random",
        "num_in_batch_negatives": 512,
        "dense_random_negative_scores": True,
        "mask_false_negatives": False,
        "exclude_own_group_negatives": False,
        "correct_positive_logq": False,
        "random_negative_fraction": 0.5,
        "logq_alpha": 0.01,
        "logq_correction": "yi2019",
        "max_seq_len": 128,
        "bos": False,
        "cls_token": False,
        "cls_token_mode": "none",
        "timestamp_delta": "bins",
        "timestamp_combination": "add",
        "timestamp_num_bins": 16,
        "timestamp_bin_semantics_revision": 2,
        "event_type_filter": "like",
        "window": "next_item",
        "drop_unmapped_items": True,
        "min_item_interactions_per_item": 5,
        "day_range": {"start_day": 0, "end_day": 300},
        "validation_interval_seconds": 604800,
        "user_sample": None,
        "evaluation_catalog": "all",
        "exclude_seen_from_evaluation": False,
        "eval_ks": [10, 50, 100],
        "eval_max_users": 20000,
        "eval_every_n_epochs": 1,
        "restore_best_weights": True,
        "selection_k": 100,
        "early_stopping_metric": "recall@100",
        "early_stopping_metric_prefix": "epoch/val_true",
        "mup_base_dim": 16,
        "mup_delta_dim": 32,
        "mup_base_ffn_dim": None,
        "mup_delta_ffn_dim": None,
    }
    for key, value in invariant_expected.items():
        _require_equal(
            invariants.get(key), value, context, f"transfer_invariants.{key}"
        )
    transformer_expected = {
        "alibi": False,
        "attention_window": 50,
        "dim": 64,
        "dropout": 0.1,
        "ffn": "swiglu",
        "ffn_dropout": 0.1,
        "ffn_intermediate_dim": 171,
        "final_norm": "layer",
        "gated_ffn_dropout": False,
        "input_dropout": 0.1,
        "input_norm": None,
        "learned_positions": "forward",
        "nhead": 2,
        "norm": "layer",
        "norm_place": "pre",
        "num_kv_heads": 1,
        "num_layers": 4,
        "rope": None,
    }
    _require_equal(
        invariants.get("transformer"),
        transformer_expected,
        context,
        "transfer_invariants.transformer",
    )
    schedule_expected = {
        "cycles": 1,
        "min_lr_fraction": 0.0,
        "optimizer_group_scope": "both",
        "power_exponent": -0.51,
        "power_transition_tokens": None,
        "shape": "linear",
        "timescale_fraction": None,
        "timescale_steps": None,
        "warmup_fraction": 0.0,
    }
    _require_equal(
        invariants.get("lr_schedule"),
        schedule_expected,
        context,
        "transfer_invariants.lr_schedule",
    )
    _require_equal(
        invariants.get("lr_schedule_horizon_epochs"),
        20,
        context,
        "transfer_invariants.lr_schedule_horizon_epochs",
    )


def _selection_metrics(
    directory: Path, metadata: dict[str, Any]
) -> tuple[float, float]:
    epoch = _required_int(metadata, "best_epoch", directory.name) - 1
    pairs = set()
    for line in (directory / "sweep.log").read_text().splitlines():
        if re.search(rf"\bepoch {epoch} finished\b", line) is None:
            continue
        recall = re.search(rf"\bepoch/val_true\.recall@100=({_METRIC_NUMBER})\b", line)
        ndcg = re.search(rf"\bepoch/val_true\.ndcg@100=({_METRIC_NUMBER})\b", line)
        if recall is not None and ndcg is not None:
            pair = (float(recall.group(1)), float(ndcg.group(1)))
            if all(math.isfinite(value) and 0 <= value <= 1 for value in pair):
                pairs.add(pair)
    if len(pairs) != 1:
        raise Rq10ReportError(
            f"{directory.name}: best epoch has missing or conflicting validation metrics"
        )
    return pairs.pop()


def _resource_metrics(directory: Path) -> tuple[int, float]:
    parameter_counts: set[int] = set()
    epoch_times: list[float] = []
    for line in (directory / "sweep.log").read_text().splitlines():
        if re.search(r"\bepoch \d+ finished\b", line) is None:
            continue
        parameter_count = re.search(r"\bresources\.params_total=(\d+(?:\.0+)?)\b", line)
        epoch_time = re.search(
            rf"\btiming\.train_epoch_time=({_METRIC_NUMBER})\b", line
        )
        if parameter_count is not None:
            parameter_counts.add(int(float(parameter_count.group(1))))
        if epoch_time is not None:
            value = float(epoch_time.group(1))
            if math.isfinite(value) and value > 0:
                epoch_times.append(value)
    if len(parameter_counts) != 1:
        raise Rq10ReportError(f"{directory.name}: inconsistent parameter count")
    if len(epoch_times) != 20:
        raise Rq10ReportError(
            f"{directory.name}: expected 20 positive training epoch times"
        )
    return parameter_counts.pop(), statistics.median(epoch_times)


def _resolve_family(family: Family, runs: list[Run]) -> FamilyResolution:
    family_runs = [run for run in runs if run.candidate.family == family]
    missing_initial = [
        candidate.run_name
        for candidate in initial_candidates()
        if candidate.family == family
        and all(run.candidate != candidate for run in family_runs)
    ]
    if missing_initial:
        return FamilyResolution(_best(family_runs), tuple(missing_initial))
    if family in _INITIAL_WIDTHS:
        central = [run for run in family_runs if run.candidate.deep_lr == 0.012]
        width_winner = _best(central)
        assert (
            width_winner is not None
            and width_winner.candidate.feature_width is not None
        )
        width_followup = _boundary_followup(width_winner, central, "width")
        if width_followup is not None:
            return FamilyResolution(width_winner, (width_followup,))
        selected_width = width_winner.candidate.feature_width
        lr_runs = [
            run for run in family_runs if run.candidate.feature_width == selected_width
        ]
        missing_lrs = [
            Rq10Candidate(family, selected_width, rate, "selected_width_lr").run_name
            for rate in (0.006, 0.024)
            if all(run.candidate.deep_lr != rate for run in lr_runs)
        ]
        if missing_lrs:
            return FamilyResolution(_best(lr_runs), tuple(missing_lrs))
    else:
        lr_runs = family_runs
    winner = _best(lr_runs)
    assert winner is not None
    lr_followup = _boundary_followup(winner, lr_runs, "lr")
    return FamilyResolution(
        winner,
        () if lr_followup is None else (lr_followup,),
    )


def _boundary_followup(winner: Run, runs: list[Run], axis: str) -> str | None:
    if axis == "width":
        values = sorted({run.candidate.feature_width for run in runs})
        value = winner.candidate.feature_width
        assert value is not None and all(item is not None for item in values)
        if value == values[0] and value > 1:
            next_value = max(1, value // 2)
        elif value == values[-1]:
            next_value = value * 2
        else:
            return None
        return Rq10Candidate(
            winner.candidate.family,
            next_value,
            0.012,
            "width_boundary",
        ).run_name
    values = sorted({run.candidate.deep_lr for run in runs})
    value = winner.candidate.deep_lr
    if value == values[0]:
        next_value = value / 2
    elif value == values[-1]:
        next_value = value * 2
    else:
        return None
    return Rq10Candidate(
        winner.candidate.family,
        winner.candidate.feature_width,
        next_value,
        "lr_boundary",
    ).run_name


def _best(runs: list[Run]) -> Run | None:
    if not runs:
        return None
    return max(
        sorted(runs, key=lambda run: run.candidate.run_name),
        key=lambda run: (run.validation_recall, run.validation_ndcg),
    )


def _comparisons(
    selected: dict[Family, Run],
    resolutions: dict[Family, FamilyResolution],
) -> dict[str, str]:
    control = selected.get("input_output_only")
    if control is None or not resolutions["input_output_only"].ready:
        return {}
    comparisons = {}
    for family in _FAMILIES[1:]:
        run = selected.get(family)
        if run is None or not resolutions[family].ready:
            continue
        comparisons[family] = (
            "non_inferior"
            if run.metrics["recall@100"] >= control.metrics["recall@100"] - _RECALL_BAND
            and run.metrics["ndcg@100"] >= control.metrics["ndcg@100"] - _NDCG_BAND
            else "inferior"
        )
    return comparisons


def _selected_record(run: Run) -> dict[str, object]:
    return {
        "run_name": run.candidate.run_name,
        "feature_width": run.candidate.feature_width,
        "embedding_lr": run.candidate.embedding_lr,
        "deep_lr": run.candidate.deep_lr,
        "best_epoch": run.best_epoch,
        "stopped_epoch": run.stopped_epoch,
        "validation": {
            "recall@100": run.validation_recall,
            "ndcg@100": run.validation_ndcg,
        },
        "params_total": run.params_total,
        "median_train_epoch_seconds": run.median_train_epoch_seconds,
        "metrics": run.metrics,
    }


def _family_label(family: str) -> str:
    try:
        return {
            "input_output_only": "Input/output item embedding only",
            "direct_add": "Direct full-width addition",
            "concat_residual": "Zero-start concatenated DenseNet residual",
            "gemma_ple": "Zero-start Gemma-style PLE",
        }[family]
    except KeyError as error:
        raise Rq10ReportError(f"unknown RQ10 family {family!r}") from error


def _render_reader(
    selected: dict[Family, Run],
    resolutions: dict[Family, FamilyResolution],
) -> str:
    historical = "\n".join(
        (
            "### Earlier valid two-layer comparison",
            "",
            "| item-feature path | recall@100 | ndcg@100 |",
            "| --- | ---: | ---: |",
            "| input/output item embedding only | 0.140 | 0.053 |",
            "| direct full-width addition before every layer | 0.139 | 0.053 |",
        )
    )
    control = selected.get("input_output_only")
    rows = []
    if control is not None and resolutions["input_output_only"].ready:
        ready_runs = [
            selected[family]
            for family in _FAMILIES
            if family in selected and resolutions[family].ready
        ]
        best_recall = max(run.metrics["recall@100"] for run in ready_runs)
        for run in ready_runs:
            family = run.candidate.family
            label = _family_label(family)
            cells = [
                label,
                (
                    "—"
                    if run.candidate.feature_width is None
                    else str(run.candidate.feature_width)
                ),
                *[
                    (
                        _absolute(run.metrics[metric])
                        if family == "input_output_only"
                        else _relative(
                            run.metrics[metric], control.metrics[metric], metric
                        )
                    )
                    for metric in _METRICS
                ],
                f"{run.params_total / 1_000_000:.1f}",
                f"{run.median_train_epoch_seconds:.2f}",
            ]
            if run.metrics["recall@100"] == best_recall:
                cells = [f"**{cell}**" for cell in cells]
            rows.append("| " + " | ".join(cells) + " |")
    current = "\n".join(
        (
            "### Four-layer reinvestigation",
            "",
            "| item-feature path | feature width | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 | parameters (M) | median epoch (s) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *rows,
        )
    )
    return historical + "\n\n" + current + "\n"


def _method_details() -> str:
    return "\n".join(
        (
            "Method details:",
            "",
            "- Input/output-only uses the tied item table only at token input and sampled-softmax output.",
            "- Direct addition injects an independent full-width item lookup before every transformer block.",
            "- Concatenated residual fuses normalized hidden and projected item features through a zero-start DenseNet residual before each block.",
            "- Gemma-style PLE follows [Gemma 3n per-layer embeddings](https://ai.google.dev/gemma/docs/gemma-3n): a compact lookup combines with the original token, is gated by the post-block hidden state, projected to model width, normalized, and added through a zero-start residual.",
        )
    )


def render_readme_section(bundle: Rq10ReportBundle) -> str:
    heading = "## RQ10 — Do separate item embeddings at every transformer layer help?"
    opening = (
        "The earlier matched native-500M sanity comparison is retained as "
        "two-layer context; it cannot select the four-layer treatment. The "
        "reinvestigation independently tunes the four-layer input/output-only "
        "control, direct addition, a zero-start concatenated DenseNet residual, "
        "and zero-start Gemma-style PLE."
    )
    if bundle.evidence.get("claims_status") != "ready":
        conclusion = (
            "Selection remains pending. The table contains only families whose "
            "learning-rate and width boundaries are already closed."
        )
    else:
        selected = bundle.evidence.get("selected")
        treatment = bundle.evidence.get("selected_added_feature")
        if not isinstance(selected, dict):
            raise Rq10ReportError("ready RQ10 evidence has no selected families")
        control = selected["input_output_only"]
        comparisons = bundle.evidence.get("control_comparisons")
        if not isinstance(comparisons, dict):
            raise Rq10ReportError("ready RQ10 evidence has no control comparisons")
        not_selected = [
            _family_label(family)
            for family, outcome in comparisons.items()
            if outcome != "non_inferior"
        ]
        if isinstance(treatment, dict):
            family = treatment["family"]
            control_metrics = control["metrics"]
            treatment_metrics = treatment["metrics"]
            recall_loss = (
                control_metrics["recall@100"] - treatment_metrics["recall@100"]
            )
            ndcg_loss = control_metrics["ndcg@100"] - treatment_metrics["ndcg@100"]
            analysis = (
                "Selection uses validation recall@100 and then same-epoch NDCG@100. "
                f"{_family_label(family)} selects width {treatment['feature_width']} "
                f"and deep LR {treatment['deep_lr']:g}. Against the tuned control, "
                f"its exact final recall/NDCG losses are {recall_loss:.9f} and "
                f"{ndcg_loss:.9f}; both are inside the approved 0.003/0.001 "
                "non-inferiority bands."
            )
            if not_selected:
                analysis += (
                    " Not selected by the decision rule: "
                    + ", ".join(not_selected)
                    + ". Their internal degradation mechanisms remain unresolved, "
                    "so no architectural-harm claim is made."
                )
            conclusion = (
                f"Conclusion: Select {_family_label(family)} with width "
                f"{treatment['feature_width']} and deep LR {treatment['deep_lr']:g} "
                "for the added-feature treatment. It satisfies the not-worse-than-"
                "baseline acceptance gate. The run does not establish a metric "
                "improvement over the control. The other tested fusions are not "
                "selected on this surface, without attributing their losses to the "
                "architectures themselves."
            )
        else:
            analysis = (
                "Selection uses validation recall@100 and then same-epoch NDCG@100. "
                "No added-feature family finishes inside both approved 0.003/0.001 "
                "non-inferiority bands."
            )
            conclusion = (
                "Conclusion: Keep the input/output-only control; no tested "
                "per-layer item-feature treatment satisfies the acceptance gate. "
                "This result does not identify the mechanism behind the observed "
                "losses. Internal optimization diagnostics are required before "
                "attributing them to the architectures themselves."
            )
        return "\n\n".join(
            (
                heading,
                opening,
                bundle.reader_markdown.strip(),
                _method_details(),
                analysis,
                conclusion,
            )
        )
    return "\n\n".join(
        (
            heading,
            opening,
            bundle.reader_markdown.strip(),
            _method_details(),
            conclusion,
        )
    )


def _render_tuning(runs: list[Run], resolutions: dict[Family, FamilyResolution]) -> str:
    selected_names = {
        resolution.selected.candidate.run_name
        for resolution in resolutions.values()
        if resolution.ready and resolution.selected is not None
    }
    lines = [
        "# RQ10 reinvestigation tuning — Yambda-500M",
        "",
        "Median epoch time is the median of all 20 recorded training epochs.",
        "",
        "| family | feature width | stage | embedding LR | deep LR | best/stopped epoch | validation recall@100 | validation ndcg@100 | recall@100 | ndcg@100 | recall@10 | ndcg@10 | coverage@100 | parameters | median epoch (s) |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in sorted(
        runs,
        key=lambda item: (
            _FAMILIES.index(item.candidate.family),
            item.candidate.feature_width or 0,
            item.candidate.deep_lr,
        ),
    ):
        cells = [
            run.candidate.family,
            (
                "—"
                if run.candidate.feature_width is None
                else str(run.candidate.feature_width)
            ),
            run.candidate.stage,
            f"{run.candidate.embedding_lr:g}",
            f"{run.candidate.deep_lr:g}",
            f"{run.best_epoch}/{run.stopped_epoch}",
            f"{run.validation_recall:.6f}",
            f"{run.validation_ndcg:.6f}",
            *[f"{run.metrics[metric]:.6f}" for metric in _METRICS],
            str(run.params_total),
            f"{run.median_train_epoch_seconds:.4f}",
        ]
        if run.candidate.run_name in selected_names:
            cells = [f"**{cell}**" for cell in cells]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _absolute(value: float) -> str:
    return f"{value:.3f}"


def _relative(value: float, control: float, metric: str) -> str:
    percent = 100 * (value / control - 1)
    cell = f"{percent:+.0f}% ({value:.3f})"
    band = (
        0.1
        if metric == "coverage@100"
        else _RECALL_BAND if "recall" in metric else _NDCG_BAND
    )
    difference = value - control
    if difference > band:
        return f'<span style="color: green">{cell}</span>'
    if difference < -band:
        return f'<span style="color: red">{cell}</span>'
    return cell


def _bounded_metric(values: dict[str, Any], key: str, context: str) -> float:
    try:
        value = float(values[key])
    except (KeyError, TypeError, ValueError) as error:
        raise Rq10ReportError(f"{context}: missing metric {key}") from error
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise Rq10ReportError(f"{context}: invalid metric {key}")
    return value


def _required_int(values: dict[str, Any], key: str, context: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise Rq10ReportError(f"{context}: {key} must be an integer")
    return value


def _require_equal(actual: object, expected: object, context: str, key: str) -> None:
    if actual != expected:
        if key in {"lr_horizon_complete", "selection_resolved"}:
            raise Rq10ReportError(f"{context}: run must be horizon-complete")
        raise Rq10ReportError(f"{context}: {key} expected {expected!r}, got {actual!r}")


def _write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(contents)
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--scratchpad", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    arguments = parser.parse_args()
    bundle = collect_report_bundle(arguments.logs)
    for path in write_report_bundle(
        bundle, arguments.scratchpad, arguments.evidence
    ).values():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
