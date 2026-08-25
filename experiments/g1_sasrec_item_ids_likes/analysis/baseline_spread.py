"""Summarize the shared empirical metric bands from ten 500M controls."""

import argparse
import json
import statistics
from pathlib import Path

from dcn.training_metadata import GENERATION_TRAINING_SEMANTICS_REVISION


EXPERIMENT = Path(__file__).resolve().parent.parent
GENERATED = EXPERIMENT.parents[1] / "generated"
DEFAULT_PREFIX = (
    "g1_calibrated_homework_baseline_native500_r3_cap40_ts2_500m_s"
)
DEFAULT_OUTPUT = EXPERIMENT / "scratchpad/baseline_spread_500m.json"
EXPECTED_SEEDS = tuple(range(10))
EXPECTED_HOMEWORK_FIELDS = {
    ("training_semantics_revision",): GENERATION_TRAINING_SEMANTICS_REVISION,
    ("batch_size",): 1280,
    ("physical_batch_size",): 1280,
    ("gradient_accumulation_steps",): 1,
    ("effective_batch_size",): 1280,
    ("val_batch_size",): 8192,
    ("num_workers",): 4,
    ("prefetch_factor",): 4,
    ("model_dim",): 64,
    ("item_embedding_dim",): 64,
    ("embedding_learning_rate",): 0.001,
    ("deep_learning_rate",): 0.002,
    ("weight_decay",): 0.0,
    ("initializer_std",): 0.02,
    ("runtime_dtype",): "torch.bfloat16",
    ("runtime_compile",): False,
    ("gradient_clip_norm",): None,
    ("negative_sampling",): "offline_logq",
    ("transfer_invariants", "experiment_class"): "GenerationExperiment",
    ("transfer_invariants", "mup_base_dim"): None,
    ("transfer_invariants", "mup_delta_dim"): None,
    ("transfer_invariants", "user_sample"): None,
    ("transfer_invariants", "event_type_filter"): "like",
    ("transfer_invariants", "min_item_interactions_per_item"): 5,
    ("transfer_invariants", "drop_unmapped_items"): True,
    ("transfer_invariants", "validation_interval_seconds"): 7 * 24 * 60 * 60,
    ("transfer_invariants", "day_range"): {"start_day": 0, "end_day": 300},
    ("transfer_invariants", "batch_size"): 1280,
    ("transfer_invariants", "physical_batch_size"): 1280,
    ("transfer_invariants", "gradient_accumulation_steps"): 1,
    ("transfer_invariants", "effective_batch_size"): 1280,
    ("transfer_invariants", "max_seq_len"): 100,
    ("transfer_invariants", "window"): "next_item",
    ("transfer_invariants", "bos"): False,
    ("transfer_invariants", "cls_token"): False,
    ("transfer_invariants", "timestamp_delta"): None,
    ("transfer_invariants", "timestamp_combination"): "add",
    ("transfer_invariants", "timestamp_num_bins"): 32,
    ("transfer_invariants", "per_layer_item_embeddings"): False,
    ("transfer_invariants", "negative_sampling"): "offline_logq",
    ("transfer_invariants", "num_in_batch_negatives"): 512,
    ("transfer_invariants", "logq_correction"): "baseline",
    ("transfer_invariants", "random_negative_fraction"): 0.5,
    ("transfer_invariants", "logq_alpha"): 0.01,
    ("transfer_invariants", "correct_positive_logq"): False,
    ("transfer_invariants", "mask_false_negatives"): False,
    ("transfer_invariants", "exclude_own_group_negatives"): False,
    ("transfer_invariants", "dense_random_negative_scores"): False,
    ("transfer_invariants", "selection_k"): 100,
    ("transfer_invariants", "evaluation_catalog"): "all",
    ("transfer_invariants", "exclude_seen_from_evaluation"): False,
    ("transfer_invariants", "eval_ks"): [10, 50, 100],
    ("transfer_invariants", "eval_max_users"): 20_000,
    ("transfer_invariants", "eval_every_n_epochs"): 1,
    ("transfer_invariants", "early_stopping_patience"): 3,
    ("transfer_invariants", "early_stopping_min_delta"): 0.0,
    ("transfer_invariants", "early_stopping_metric"): "recall@100",
    ("transfer_invariants", "early_stopping_metric_prefix"): "epoch/val_true",
    ("transfer_invariants", "restore_best_weights"): True,
    ("transfer_invariants", "transformer"): {
        "alibi": False,
        "attention_window": None,
        "dim": 64,
        "dropout": 0.1,
        "ffn": "gelu",
        "ffn_dropout": 0.1,
        "ffn_intermediate_dim": 256,
        "final_norm": "layer",
        "input_dropout": 0.1,
        "input_norm": None,
        "learned_positions": "forward",
        "nhead": 2,
        "norm": "layer",
        "norm_place": "pre",
        "num_kv_heads": 2,
        "num_layers": 2,
        "rope": None,
    },
    ("transfer_invariants", "lr_schedule"): {
        "cycles": 1,
        "min_lr_fraction": 0.0,
        "power_exponent": -0.51,
        "power_transition_tokens": None,
        "shape": "constant",
        "timescale_fraction": None,
        "timescale_steps": None,
        "warmup_fraction": 0.0,
    },
}


def _nested(mapping: dict, path: tuple[str, ...]) -> object:
    value = mapping
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def homework_metadata_errors(
    metadata: dict,
    *,
    batch_size: int = 1280,
    seed: int | None = None,
) -> list[str]:
    expected_fields = dict(EXPECTED_HOMEWORK_FIELDS)
    expected_fields[("batch_size",)] = batch_size
    expected_fields[("physical_batch_size",)] = batch_size
    expected_fields[("effective_batch_size",)] = batch_size
    expected_fields[("transfer_invariants", "batch_size")] = batch_size
    expected_fields[("transfer_invariants", "physical_batch_size")] = batch_size
    expected_fields[("transfer_invariants", "effective_batch_size")] = batch_size
    errors = []
    if metadata.get("dataset_size") != "500m":
        errors.append(f"dataset_size={metadata.get('dataset_size')!r}")
    if seed is not None and metadata.get("seed") != seed:
        errors.append(f"seed={metadata.get('seed')!r}")
    for path, expected in expected_fields.items():
        actual = _nested(metadata, path)
        if actual != expected:
            errors.append(f"{'.'.join(path)}={actual!r}, expected {expected!r}")
    epochs_trained = metadata.get("epochs_trained")
    best_epoch = metadata.get("best_epoch")
    stopped_epoch = metadata.get("stopped_epoch")
    max_epochs = metadata.get("max_epochs")
    if (
        not isinstance(max_epochs, int)
        or isinstance(max_epochs, bool)
        or max_epochs < 20
        or metadata.get("num_epochs") != max_epochs
    ):
        errors.append(f"max_epochs={max_epochs!r}")
    if (
        not isinstance(epochs_trained, int)
        or isinstance(epochs_trained, bool)
        or not isinstance(max_epochs, int)
        or not 1 <= epochs_trained <= max_epochs
    ):
        errors.append(f"epochs_trained={epochs_trained!r}")
    if stopped_epoch != epochs_trained:
        errors.append(
            f"stopped_epoch={stopped_epoch!r}, expected {epochs_trained!r}"
        )
    if (
        isinstance(stopped_epoch, int)
        and not isinstance(stopped_epoch, bool)
        and isinstance(max_epochs, int)
        and stopped_epoch >= max_epochs
    ):
        errors.append(
            f"stopped_epoch={stopped_epoch!r}, expected less than {max_epochs!r}"
        )
    if (
        not isinstance(best_epoch, int)
        or not isinstance(stopped_epoch, int)
        or not 1 <= best_epoch <= stopped_epoch
    ):
        errors.append(f"best_epoch={best_epoch!r}")
    if metadata.get("best_epoch_at_cap") is not False:
        errors.append(f"best_epoch_at_cap={metadata.get('best_epoch_at_cap')!r}")
    if metadata.get("early_stopped") is not True:
        errors.append(f"early_stopped={metadata.get('early_stopped')!r}")
    if metadata.get("selection_resolved") is not True:
        errors.append(f"selection_resolved={metadata.get('selection_resolved')!r}")
    targets_per_epoch = metadata.get("targets_per_epoch")
    tokens_per_epoch = metadata.get("tokens_per_epoch")
    if not isinstance(targets_per_epoch, int) or isinstance(
        targets_per_epoch, bool
    ) or targets_per_epoch < 1:
        errors.append(f"targets_per_epoch={targets_per_epoch!r}")
    elif isinstance(epochs_trained, int) and (
        metadata.get("training_horizon") != targets_per_epoch * epochs_trained
    ):
        errors.append("training_horizon does not match epochs_trained")
    if not isinstance(tokens_per_epoch, int) or isinstance(
        tokens_per_epoch, bool
    ) or tokens_per_epoch < 1:
        errors.append(f"tokens_per_epoch={tokens_per_epoch!r}")
    elif isinstance(epochs_trained, int):
        if (
            metadata.get("token_horizon") != tokens_per_epoch * epochs_trained
            or metadata.get("tokens_seen") != tokens_per_epoch * epochs_trained
        ):
            errors.append("token horizon does not match epochs_trained")
    optimizer_steps = metadata.get("optimizer_steps")
    if (
        not isinstance(optimizer_steps, int)
        or isinstance(optimizer_steps, bool)
        or optimizer_steps < 1
    ):
        errors.append(f"optimizer_steps={optimizer_steps!r}")
    return errors


def load_repeats(prefix: str) -> dict[int, dict[str, float]]:
    repeats = {}
    reference_metadata = None
    for seed in EXPECTED_SEEDS:
        run_dir = GENERATED / "logs" / f"{prefix}{seed}"
        metrics_path = run_dir / "final_metrics.json"
        metadata_path = run_dir / "training_metadata.json"
        if not metrics_path.exists() or not metadata_path.exists():
            missing = [
                path.name
                for path in (metrics_path, metadata_path)
                if not path.exists()
            ]
            raise FileNotFoundError(f"{run_dir.name}: missing {', '.join(missing)}")
        if metrics_path.stat().st_mtime_ns < metadata_path.stat().st_mtime_ns:
            raise ValueError(f"{run_dir.name}: metrics predate training metadata")
        metadata = json.loads(metadata_path.read_text())
        errors = homework_metadata_errors(metadata, seed=seed)
        if errors:
            raise ValueError(f"{run_dir.name}: " + "; ".join(errors))
        comparable_metadata = {
            key: value
            for key, value in metadata.items()
            if key
            not in {
                "seed",
                "validation_loss",
                "epochs_trained",
                "best_epoch",
                "stopped_epoch",
                "early_stopped",
                "training_horizon",
                "token_horizon",
                "tokens_seen",
                "optimizer_steps",
            }
        }
        if reference_metadata is None:
            reference_metadata = comparable_metadata
        elif comparable_metadata != reference_metadata:
            raise ValueError(
                f"{run_dir.name}: configuration differs from seed 0; "
                "the empirical band requires ten unchanged controls"
            )
        raw_metrics = json.loads(metrics_path.read_text())
        repeats[seed] = {
            key: float(value)
            for key, value in raw_metrics.items()
            if key != "num_users" and isinstance(value, (int, float))
        }
        repeats[seed]["__num_users"] = float(raw_metrics["num_users"])
    user_counts = {metrics.pop("__num_users") for metrics in repeats.values()}
    if len(user_counts) != 1:
        raise ValueError("baseline repeats do not score the same user population")
    metric_sets = {frozenset(metrics) for metrics in repeats.values()}
    if len(metric_sets) != 1:
        raise ValueError("baseline repeats do not report the same metrics")
    return repeats


def summarize(prefix: str = DEFAULT_PREFIX) -> dict:
    repeats = load_repeats(prefix)
    metrics = {}
    for metric in sorted(next(iter(repeats.values()))):
        values = [repeats[seed][metric] for seed in EXPECTED_SEEDS]
        mean = statistics.fmean(values)
        sample_stddev = statistics.stdev(values)
        metrics[metric] = {
            "mean": mean,
            "sample_stddev": sample_stddev,
            "absolute_band": sample_stddev,
            "stddev_percent_of_mean": 100 * sample_stddev / mean,
        }
    return {
        "description": (
            "Shared empirical resolution bands from ten unchanged independent "
            "Yambda-500M accepted-control repeats; not confidence intervals or "
            "treatment-specific significance tests."
        ),
        "run_prefix": prefix,
        "seeds": list(EXPECTED_SEEDS),
        "n": len(EXPECTED_SEEDS),
        "metrics": metrics,
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# G1 shared empirical metric bands",
        "",
        summary["description"],
        "",
        "| metric | mean | sample stddev / absolute band | stddev as % of mean |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric, values in summary["metrics"].items():
        lines.append(
            f"| {metric} | {values['mean']:.8f} | "
            f"{values['sample_stddev']:.8f} | "
            f"{values['stddev_percent_of_mean']:.3f}% |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    summary = summarize(args.prefix)
    markdown = render_markdown(summary)
    if not args.write:
        print(markdown, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.output.with_suffix(".md").write_text(markdown)
    print(f"wrote {args.output}")
    print(f"wrote {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
