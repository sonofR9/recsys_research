#!/usr/bin/env bash
set -u

launcher_dir=$(cd "$(dirname "$0")" && pwd) || exit 1
source "$launcher_dir/../artifacts.sh" || exit 1
repo_root=$(cd "$launcher_dir/../../../.." && pwd) || exit 1
cd "$repo_root" || exit 1

if [[ -z "${G1_RQ10_FOLLOWUP_RUNS:-}" ]]; then
    echo "G1_RQ10_FOLLOWUP_RUNS must name at least one candidate" >&2
    exit 2
fi
logs=${G1_RQ10_LOGS:-$repo_root/generated/logs}
export WANDB_MODE=${WANDB_MODE:-offline}
export G1_DATASET_SIZE=500m
export TRAINING_QUEUE_MONITOR_LIGHT_GPUS=${TRAINING_QUEUE_MONITOR_LIGHT_GPUS:-1}
unset G1_MAX_USERS G1_MAX_EPOCHS G1_SEED G1_TRAIN_BATCH_SIZE G1_VAL_BATCH_SIZE
unset G1_VARIANT G1_RQ10_RUN

config=experiments/g1_sasrec_item_ids_likes/configs/rq10_reinvestigation_variant.py
TRAINING_QUEUE_SCRIPT=$config
source "${G1_TRAINING_QUEUE_LIBRARY:-utils/training_queue/queue.sh}" || exit 1

mapfile -t candidate_rows < <(
    python - <<'PY'
import os

from experiments.g1_sasrec_item_ids_likes.analysis.rq10_reinvestigation_candidates import candidate_by_run

for run_name in os.environ["G1_RQ10_FOLLOWUP_RUNS"].split():
    candidate = candidate_by_run(run_name)
    if candidate.stage == "initial":
        raise SystemExit("initial candidates belong to the initial launcher")
    print(candidate.run_name)
PY
)
if [[ "${#candidate_rows[@]}" -eq 0 ]]; then
    echo "RQ10 follow-up surface is empty" >&2
    exit 2
fi

declare -A seen
enqueued_count=0
skipped_count=0
for run in "${candidate_rows[@]}"; do
    if [[ -z "$run" || -n "${seen[$run]+x}" ]]; then
        echo "Invalid or duplicate RQ10 follow-up: $run" >&2
        exit 2
    fi
    seen[$run]=1
    directory="$logs/$run"
    verifier_args=("G1_RQ10_RUN=$run")
    artifact_status=0
    g1_require_config_recipe_compatible_or_absent "$directory" "$config" \
        "${verifier_args[@]}" || artifact_status=$?
    if [[ "$artifact_status" -eq 0 ]]; then
        echo "=== skipped compatible $run ==="
        skipped_count=$((skipped_count + 1))
        continue
    fi
    [[ "$artifact_status" -eq 1 ]] || exit "$artifact_status"
    TRAINING_QUEUE_DATA_GROUP="g1-rq10-500m-seq128" \
        enqueue "$run" "${verifier_args[@]}" || exit 1
    enqueued_count=$((enqueued_count + 1))
done

echo "=== rq10 follow-up: enqueued=${enqueued_count}, skipped=${skipped_count} ==="
g1_stop_artifact_verifier
drain || exit 1
