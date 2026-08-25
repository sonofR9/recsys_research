#!/usr/bin/env bash
set -u

launcher_dir=$(cd "$(dirname "$0")" && pwd) || exit 1
repo_root=$(cd "$launcher_dir/../../../.." && pwd) || exit 1
cd "$repo_root" || exit 1

if [[ "${1:-}" == "--list-initial" ]]; then
    python - <<'PY'
from experiments.g1_sasrec_item_ids_likes.analysis.rq11_mixed_streaming_candidates import initial_candidates

for candidate in initial_candidates():
    print(candidate.run_name)
PY
    exit $?
fi

stage=${G1_RQ11_STAGE:-initial}
if [[ "$stage" != "initial" && "$stage" != "followups" ]]; then
    echo "G1_RQ11_STAGE must be initial or followups" >&2
    exit 2
fi

logs=${G1_RQ11_LOGS:-$repo_root/generated/logs}
if [[ "$stage" == "initial" ]]; then
    mapfile -t runs < <(
        python - <<'PY'
from experiments.g1_sasrec_item_ids_likes.analysis.rq11_mixed_streaming_candidates import initial_candidates

for candidate in initial_candidates():
    print(candidate.run_name)
PY
    )
    if [[ "${#runs[@]}" -ne 24 ]]; then
        echo "RQ11 initial stage must contain exactly 24 runs" >&2
        exit 2
    fi
else
    output=$(python -m \
        experiments.g1_sasrec_item_ids_likes.analysis.rq11_mixed_streaming_selection \
        --logs "$logs") || exit $?
    runs=()
    if [[ -n "$output" ]]; then
        mapfile -t runs <<<"$output"
    fi
fi

source "$launcher_dir/../artifacts.sh" || exit 1
export WANDB_MODE=${WANDB_MODE:-offline}
export G1_DATASET_SIZE=500m
export TRAINING_QUEUE_MONITOR_LIGHT_GPUS=${TRAINING_QUEUE_MONITOR_LIGHT_GPUS:-1}
export TRAINING_QUEUE_DATA_GROUP=g1-rq11-mixed-streaming-500m-seq128
unset G1_MAX_USERS G1_SEED G1_TRAIN_BATCH_SIZE G1_VAL_BATCH_SIZE G1_VARIANT G1_RQ11_RUN

config=experiments/g1_sasrec_item_ids_likes/configs/rq11_mixed_streaming_variant.py
TRAINING_QUEUE_SCRIPT=$config
source "${G1_TRAINING_QUEUE_LIBRARY:-utils/training_queue/queue.sh}" || exit 1

declare -A seen
enqueued_count=0
skipped_count=0
for run in "${runs[@]}"; do
    if [[ -n "${seen[$run]+x}" ]]; then
        echo "Duplicate RQ11 run identity: $run" >&2
        exit 2
    fi
    seen[$run]=1
    directory="$logs/$run"
    verifier_args=("G1_RQ11_RUN=$run")
    artifact_status=0
    g1_require_config_recipe_compatible_or_absent "$directory" "$config" \
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

echo "=== rq11 native-500M ${stage}: enqueued=${enqueued_count}, skipped=${skipped_count} ==="
g1_stop_artifact_verifier
drain || exit 1
