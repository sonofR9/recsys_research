#!/usr/bin/env bash
set -u

launcher_dir=$(cd "$(dirname "$0")" && pwd)
source "$launcher_dir/../artifacts.sh"

stage=${G1_RQ5_STAGE:-initial}
if [[ "$stage" != "initial" && "$stage" != "probes" && "$stage" != "corrections" ]]; then
    echo "G1_RQ5_STAGE must be initial, probes, or corrections" >&2
    exit 2
fi

repo_root=$(cd "$launcher_dir/../../../.." && pwd)
cd "$repo_root"
manifest=${G1_RQ5_MANIFEST:-$repo_root/experiments/g1_sasrec_item_ids_likes/scratchpad/rq5_scheduler_candidate_manifest.json}
export WANDB_MODE=${WANDB_MODE:-offline}
export G1_DATASET_SIZE=500m
export TRAINING_QUEUE_MONITOR_LIGHT_GPUS=${TRAINING_QUEUE_MONITOR_LIGHT_GPUS:-1}
export TRAINING_QUEUE_DATA_GROUP=g1-rq5-schedulers-500m-seq128

config=experiments/g1_sasrec_item_ids_likes/configs/rq5_scheduler_variant.py
TRAINING_QUEUE_SCRIPT=$config
source "${G1_TRAINING_QUEUE_LIBRARY:-utils/training_queue/queue.sh}" || exit 1

if [[ "$stage" == "initial" ]]; then
    mapfile -t runs < <(
        python - <<'PY'
from experiments.g1_sasrec_item_ids_likes.analysis.rq5_scheduler_candidates import initial_candidates

for candidate in initial_candidates():
    print(candidate.run_name)
PY
    )
    if [[ "${#runs[@]}" -ne 67 ]]; then
        echo "RQ5 initial stage must contain exactly 67 unique runs" >&2
        exit 2
    fi
elif [[ "$stage" == "probes" ]]; then
    correction_output=$(python -m \
        experiments.g1_sasrec_item_ids_likes.analysis.rq5_scheduler_selection \
        --logs "$repo_root/generated/logs" \
        --manifest "$manifest") || exit $?
    runs=()
    if [[ -n "$correction_output" ]]; then
        mapfile -t runs <<<"$correction_output"
    fi
else
    planner_args=(--logs "$repo_root/generated/logs")
    if [[ -e "$manifest" || -L "$manifest" ]]; then
        planner_args+=(--manifest "$manifest")
    fi
    correction_output=$(python -m \
        experiments.g1_sasrec_item_ids_likes.analysis.rq5_scheduler_corrections \
        "${planner_args[@]}") || exit $?
    runs=()
    if [[ -n "$correction_output" ]]; then
        mapfile -t runs <<<"$correction_output"
    fi
fi

declare -A seen
enqueued_count=0
skipped_count=0
for run in "${runs[@]}"; do
    if [[ -n "${seen[$run]+x}" ]]; then
        echo "Duplicate RQ5 run identity: $run" >&2
        exit 2
    fi
    seen[$run]=1
    directory="$repo_root/generated/logs/$run"
    verifier_args=("G1_RQ5_RUN=$run")
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

echo "=== rq5 schedulers 500m ${stage}: enqueued=${enqueued_count}, skipped=${skipped_count} ==="
g1_stop_artifact_verifier
drain || exit 1
