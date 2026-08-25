#!/usr/bin/env bash
set -u

launcher_dir=$(cd "$(dirname "$0")" && pwd)
source "$launcher_dir/../artifacts.sh"

repo_root=$(cd "$launcher_dir/../../../.." && pwd)
cd "$repo_root"
logs=${G1_RQ8_LOGS:-$repo_root/generated/logs}
export WANDB_MODE=${WANDB_MODE:-offline}
export G1_DATASET_SIZE=500m
export TRAINING_QUEUE_MONITOR_LIGHT_GPUS=${TRAINING_QUEUE_MONITOR_LIGHT_GPUS:-1}
unset G1_MAX_USERS G1_MAX_EPOCHS G1_SEED G1_TRAIN_BATCH_SIZE G1_VAL_BATCH_SIZE
unset G1_VARIANT G1_RQ8_RUN

config=experiments/g1_sasrec_item_ids_likes/configs/rq8_reinvestigation_variant.py
TRAINING_QUEUE_SCRIPT=$config
source "${G1_TRAINING_QUEUE_LIBRARY:-utils/training_queue/queue.sh}" || exit 1

mapfile -t candidate_rows < <(
    python - <<'PY'
from experiments.g1_sasrec_item_ids_likes.analysis.rq8_reinvestigation_candidates import sequence_initial_candidates

for candidate in sequence_initial_candidates():
    print(f"{candidate.run_name}\t{candidate.max_seq_len}")
PY
)
if [[ "${#candidate_rows[@]}" -ne 48 ]]; then
    echo "RQ8 corrected sequence stage must contain exactly 48 runs" >&2
    exit 2
fi

declare -A seen
enqueued_count=0
skipped_count=0
enqueue_candidate() {
    local run=$1
    local max_seq_len=$2
    local TRAINING_QUEUE_DATA_GROUP="g1-rq8-fullcausal-500m-seq${max_seq_len}"
    local directory="$logs/$run"
    local verifier_args=("G1_RQ8_RUN=$run")
    local artifact_status=0
    g1_require_config_recipe_compatible_or_absent "$directory" "$config" \
        "${verifier_args[@]}" || artifact_status=$?
    if [[ "$artifact_status" -eq 0 ]]; then
        echo "=== skipped compatible $run ==="
        skipped_count=$((skipped_count + 1))
        return 0
    fi
    [[ "$artifact_status" -eq 1 ]] || return "$artifact_status"
    enqueue "$run" "${verifier_args[@]}" || return 1
    enqueued_count=$((enqueued_count + 1))
}

for row in "${candidate_rows[@]}"; do
    IFS=$'\t' read -r run max_seq_len <<< "$row"
    if [[ -z "$run" || -z "$max_seq_len" || -n "${seen[$run]+x}" ]]; then
        echo "Invalid or duplicate RQ8 candidate row: $row" >&2
        exit 2
    fi
    seen[$run]=1
    enqueue_candidate "$run" "$max_seq_len" || exit $?
done

echo "=== rq8 full-causal sequence 500m: enqueued=${enqueued_count}, skipped=${skipped_count} ==="
g1_stop_artifact_verifier
drain || exit 1
