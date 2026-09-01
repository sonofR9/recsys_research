#!/usr/bin/env bash
set -u

mode=run
if [[ "${1:-}" == "--list" ]]; then
    mode=list
    shift
elif [[ "${1:-}" == "--preflight" ]]; then
    mode=preflight
    shift
fi
if [[ "$#" -ne 0 ]]; then
    echo "usage: $0 [--list|--preflight]" >&2
    exit 2
fi

launcher_dir=$(cd "$(dirname "$0")" && pwd) || exit 1
repo_root=$(cd "$launcher_dir/../../../.." && pwd) || exit 1
cd "$repo_root" || exit 1

mapfile -t candidate_rows < <(
    python - <<'PY'
from experiments.g1_sasrec_item_ids_likes.analysis.rq14_pretrained_lesion_candidates import diagnostic_candidates
for candidate in diagnostic_candidates():
    print(f"{candidate.run_name}\t{candidate.treatment}")
PY
) || exit 2
if [[ "${#candidate_rows[@]}" -ne 4 ]]; then
    echo "RQ14 lesion launcher must contain exactly four selected cells" >&2
    exit 2
fi
if [[ "$mode" == list ]]; then
    printf '%s\n' "${candidate_rows[@]}"
    exit 0
fi

logs=${G1_RQ14_PRETRAINED_LOGS:-$repo_root/generated/logs}
experiment_dir=experiments/g1_sasrec_item_ids_likes
config=$experiment_dir/configs/rq14_pretrained_lesion_variant.py
source "$launcher_dir/../artifacts.sh" || exit 1

read -r source_run checkpoint < <(
    python - "$logs" <<'PY'
from pathlib import Path
import sys
from experiments.g1_sasrec_item_ids_likes.analysis.rq15_training_candidates import selected_source_candidate
candidate = selected_source_candidate(Path(sys.argv[1]))
print(candidate.run_name, candidate.checkpoint_path(Path(sys.argv[1])))
PY
) || exit 2
if [[ ! -f "$checkpoint" ]]; then
    echo "Missing selected RQ15 source checkpoint: $checkpoint" >&2
    exit 2
fi

export G1_DATASET_SIZE=500m
export WANDB_MODE=${WANDB_MODE:-offline}
export TRAINING_QUEUE_MONITOR_LIGHT_GPUS=${TRAINING_QUEUE_MONITOR_LIGHT_GPUS:-1}
unset G1_MAX_USERS G1_MAX_EPOCHS G1_SEED G1_TRAIN_BATCH_SIZE G1_VAL_BATCH_SIZE
unset G1_VARIANT G1_RQ14_PRETRAINED_RUN G1_RQ14_LESION_RUN

if [[ "$mode" == preflight ]]; then
    python - "$config" "$source_run" "$checkpoint" <<'PY'
import sys
from pathlib import Path
from experiments.g1_sasrec_item_ids_likes.analysis.rq14_pretrained_lesion_candidates import diagnostic_candidates
from experiments.g1_sasrec_item_ids_likes.launchers.verify_artifact import _config_assignments, _config_experiment
config, source_run, checkpoint = sys.argv[1:]
for candidate in diagnostic_candidates():
    assignments = _config_assignments([
        f"G1_RQ14_LESION_RUN={candidate.run_name}",
        f"G1_RQ15_SOURCE_RUN={source_run}",
        f"G1_RQ15_FIRST_STAGE_CHECKPOINT={checkpoint}",
    ])
    experiment = _config_experiment(Path(config), assignments)
    assert experiment.run_name == candidate.run_name
    assert experiment.diagnostic_lesions == candidate.lesions
print("RQ14 lesion preflight passed for four selected cells")
PY
    exit $?
fi

TRAINING_QUEUE_SCRIPT=$config
source "${G1_TRAINING_QUEUE_LIBRARY:-utils/training_queue/queue.sh}" || exit 1
enqueued_count=0
skipped_count=0
for row in "${candidate_rows[@]}"; do
    IFS=$'\t' read -r run treatment extra <<< "$row"
    if [[ -z "$run" || -z "$treatment" || -n "$extra" ]]; then
        echo "Malformed RQ14 lesion candidate row" >&2
        exit 2
    fi
    verifier_args=(
        "G1_RQ14_LESION_RUN=$run"
        "G1_RQ15_SOURCE_RUN=$source_run"
        "G1_RQ15_FIRST_STAGE_CHECKPOINT=$checkpoint"
    )
    artifact_status=0
    g1_require_config_compatible_or_absent "$logs/$run" "$config" \
        "${verifier_args[@]}" || artifact_status=$?
    if [[ "$artifact_status" -eq 0 ]]; then
        if [[ -f "$logs/$run/rq14_lesion_diagnostics.json" ]]; then
            skipped_count=$((skipped_count + 1))
            continue
        fi
        g1_archive_artifact "$logs/$run" incomplete || exit 2
        artifact_status=1
    fi
    [[ "$artifact_status" -eq 1 ]] || exit "$artifact_status"
    TRAINING_QUEUE_DATA_GROUP="g1-rq14-pretrained-lesion-${treatment}-500m-seq128"
    enqueue "$run" "${verifier_args[@]}" || exit 1
    enqueued_count=$((enqueued_count + 1))
done
echo "=== RQ14 lesion diagnostics: enqueued=${enqueued_count}, skipped=${skipped_count} ==="
g1_stop_artifact_verifier
drain || exit 1

python -m experiments.g1_sasrec_item_ids_likes.analysis.rq14_pretrained_lesion_evidence \
    --logs "$logs" \
    --rq14-results "$experiment_dir/evidence/rq14_pretrained_results.json" \
    --output "$experiment_dir/evidence/rq14_pretrained_lesion_results.json" \
    --explanation "$experiment_dir/evidence/rq14_pretrained_lesion_explanation.json"
