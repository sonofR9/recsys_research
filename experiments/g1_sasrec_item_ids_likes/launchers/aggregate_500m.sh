#!/usr/bin/env bash
set -u

launcher_dir=$(cd "$(dirname "$0")" && pwd) || exit 1
source "$launcher_dir/artifacts.sh" || exit 1
repo_root=$(cd "$launcher_dir/../../.." && pwd) || exit 1
cd "$repo_root" || exit 1

stage=${G1_AGGREGATE_STAGE:-initial}
logs=${G1_AGGREGATE_LOGS:-$repo_root/generated/logs}
export WANDB_MODE=${WANDB_MODE:-offline}
export G1_DATASET_SIZE=500m
export TRAINING_QUEUE_MONITOR_LIGHT_GPUS=${TRAINING_QUEUE_MONITOR_LIGHT_GPUS:-1}
unset G1_MAX_USERS G1_MAX_EPOCHS G1_SEED G1_TRAIN_BATCH_SIZE G1_VAL_BATCH_SIZE
unset G1_VARIANT G1_AGGREGATE_RUN

config=experiments/g1_sasrec_item_ids_likes/configs/aggregate_variant.py
TRAINING_QUEUE_SCRIPT=$config
source "${G1_TRAINING_QUEUE_LIBRARY:-utils/training_queue/queue.sh}" || exit 1

case "$stage" in
    initial)
        mapfile -t candidate_rows < <(
            python - <<'PY'
from experiments.g1_sasrec_item_ids_likes.analysis.aggregate_candidates import initial_candidates

for candidate in initial_candidates():
    print(candidate.run_name)
PY
        )
        if [[ "${#candidate_rows[@]}" -ne 12 ]]; then
            echo "Aggregate initial surface expected 12 runs" >&2
            exit 2
        fi
        ;;
    recovery)
        mapfile -t candidate_rows < <(
            python - <<'PY'
from experiments.g1_sasrec_item_ids_likes.analysis.aggregate_candidates import recovery_candidates

for candidate in recovery_candidates():
    print(candidate.run_name)
PY
        )
        if [[ "${#candidate_rows[@]}" -ne 8 ]]; then
            echo "Aggregate recovery surface expected 8 runs" >&2
            exit 2
        fi
        ;;
    followups|bridges)
        required_output=$(
            python -m experiments.g1_sasrec_item_ids_likes.analysis.aggregate_report \
                --logs "$logs" --required all
        ) || exit 2
        required_rows=()
        if [[ -n "$required_output" ]]; then
            mapfile -t required_rows <<< "$required_output"
        fi
        candidate_output=$(
            G1_AGGREGATE_REQUIRED="${required_rows[*]}" \
            G1_AGGREGATE_REQUESTED_STAGE="$stage" python - <<'PY'
import os

from experiments.g1_sasrec_item_ids_likes.analysis.aggregate_candidates import candidate_by_run

requested = os.environ["G1_AGGREGATE_REQUESTED_STAGE"]
candidates = [
    candidate_by_run(name)
    for name in os.environ.get("G1_AGGREGATE_REQUIRED", "").split()
]
if requested == "followups":
    if any(candidate.stage == "initial" for candidate in candidates):
        raise SystemExit("the exact initial surface is incomplete; use stage=initial")
    selected = [candidate for candidate in candidates if candidate.family != "bridge"]
else:
    if any(candidate.family != "bridge" for candidate in candidates):
        raise SystemExit("aggregate selection is unresolved; run required follow-ups first")
    selected = candidates
    if len(selected) > 11:
        raise SystemExit("the bridge stage exceeds the exact eleven selected bridges")
for candidate in selected:
    print(candidate.run_name)
PY
        ) || exit 2
        candidate_rows=()
        if [[ -n "$candidate_output" ]]; then
            mapfile -t candidate_rows <<< "$candidate_output"
        fi
        if [[ "${#candidate_rows[@]}" -eq 0 ]]; then
            echo "Aggregate $stage stage has no required candidates" >&2
            exit 2
        fi
        ;;
    *)
        echo "G1_AGGREGATE_STAGE must be initial, recovery, followups, or bridges" >&2
        exit 2
        ;;
esac

declare -A seen
enqueued_count=0
skipped_count=0
for run in "${candidate_rows[@]}"; do
    if [[ -z "$run" || -n "${seen[$run]+x}" ]]; then
        echo "Invalid or duplicate aggregate candidate: $run" >&2
        exit 2
    fi
    seen[$run]=1
    directory="$logs/$run"
    verifier_args=("G1_AGGREGATE_RUN=$run")
    artifact_status=0
    g1_require_config_recipe_compatible_or_absent "$directory" "$config" \
        "${verifier_args[@]}" || artifact_status=$?
    if [[ "$artifact_status" -eq 0 ]]; then
        echo "=== skipped compatible $run ==="
        skipped_count=$((skipped_count + 1))
        continue
    fi
    [[ "$artifact_status" -eq 1 ]] || exit "$artifact_status"
    TRAINING_QUEUE_DATA_GROUP="g1-aggregate-500m-seq100" \
        enqueue "$run" "${verifier_args[@]}" || exit 1
    enqueued_count=$((enqueued_count + 1))
done

echo "=== aggregate $stage: enqueued=${enqueued_count}, skipped=${skipped_count} ==="
g1_stop_artifact_verifier
drain || exit 1
