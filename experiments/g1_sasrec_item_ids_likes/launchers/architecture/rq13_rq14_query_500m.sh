#!/usr/bin/env bash
set -u

launcher_dir=$(cd "$(dirname "$0")" && pwd) || exit 1
repo_root=$(cd "$launcher_dir/../../../.." && pwd) || exit 1
cd "$repo_root" || exit 1

if [[ "${1:-}" == "--list-initial" ]]; then
    python - <<'PY'
from experiments.g1_sasrec_item_ids_likes.analysis.rq13_rq14_query_candidates import rq13_initial_candidates

for candidate in rq13_initial_candidates():
    print(candidate.run_name)
PY
    exit $?
fi

source "$launcher_dir/../artifacts.sh" || exit 1
logs=${G1_QUERY_LOGS:-$repo_root/generated/logs}
export WANDB_MODE=${WANDB_MODE:-offline}
export G1_DATASET_SIZE=500m
export TRAINING_QUEUE_MONITOR_LIGHT_GPUS=${TRAINING_QUEUE_MONITOR_LIGHT_GPUS:-1}
unset G1_MAX_USERS G1_MAX_EPOCHS G1_SEED G1_TRAIN_BATCH_SIZE
unset G1_VAL_BATCH_SIZE G1_VARIANT G1_QUERY_RUN

config=experiments/g1_sasrec_item_ids_likes/configs/rq13_rq14_query_variant.py
TRAINING_QUEUE_SCRIPT=$config
source "${G1_TRAINING_QUEUE_LIBRARY:-utils/training_queue/queue.sh}" || exit 1

mapfile -t candidate_rows < <(
    python - <<'PY'
from experiments.g1_sasrec_item_ids_likes.analysis.rq13_rq14_query_candidates import rq13_initial_candidates

for candidate in rq13_initial_candidates():
    print(f"{candidate.run_name}\t{candidate.study}\t{candidate.treatment}")
PY
)
if [[ "${#candidate_rows[@]}" -ne 15 ]]; then
    echo "RQ13 initial stage must contain exactly 15 runs" >&2
    exit 2
fi

declare -A seen
enqueued_count=0
skipped_count=0
for row in "${candidate_rows[@]}"; do
    IFS=$'\t' read -r run study treatment <<< "$row"
    if [[ -z "$run" || -n "${seen[$run]+x}" ]]; then
        echo "Invalid or duplicate RQ13 row: $row" >&2
        exit 2
    fi
    seen[$run]=1
    TRAINING_QUEUE_DATA_GROUP="g1-${study}-${treatment}-500m-seq128"
    directory="$logs/$run"
    verifier_args=("G1_QUERY_RUN=$run")
    artifact_status=0
    g1_require_config_compatible_or_absent "$directory" "$config" \
        "${verifier_args[@]}" || artifact_status=$?
    if [[ "$artifact_status" -eq 0 ]]; then
        echo "=== skipped compatible $run ==="
        skipped_count=$((skipped_count + 1))
        continue
    fi
    [[ "$artifact_status" -eq 1 ]] || exit "$artifact_status"
    enqueue "$run" "${verifier_args[@]}" || exit 1
    enqueued_count=$((enqueued_count + 1))
done

echo "=== rq13 native-500M: enqueued=${enqueued_count}, skipped=${skipped_count} ==="
g1_stop_artifact_verifier
drain || exit 1
