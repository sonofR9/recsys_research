#!/usr/bin/env bash

g1_rq7_launch_stage() {
    local candidate_factory=$1
    local expected_count=$2
    local dataset_size=$3
    local stage_label=$4
    local launcher_dir
    launcher_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) || return 1
    source "$launcher_dir/../artifacts.sh" || return 1

    local repo_root
    repo_root=$(cd "$launcher_dir/../../../.." && pwd) || return 1
    cd "$repo_root" || return 1
    local logs=${G1_RQ7_LOGS:-$repo_root/generated/logs}
    export WANDB_MODE=${WANDB_MODE:-offline}
    export G1_DATASET_SIZE=$dataset_size
    export TRAINING_QUEUE_MONITOR_LIGHT_GPUS=${TRAINING_QUEUE_MONITOR_LIGHT_GPUS:-1}
    unset G1_MAX_USERS G1_MAX_EPOCHS G1_SEED G1_TRAIN_BATCH_SIZE G1_VAL_BATCH_SIZE
    unset G1_VARIANT G1_RQ7_RUN

    local config=experiments/g1_sasrec_item_ids_likes/configs/rq7_reinvestigation_variant.py
    TRAINING_QUEUE_SCRIPT=$config
    source "${G1_TRAINING_QUEUE_LIBRARY:-utils/training_queue/queue.sh}" || return 1

    local candidate_output
    candidate_output=$(python - "$candidate_factory" <<'PY'
import sys

from experiments.g1_sasrec_item_ids_likes.analysis import rq7_reinvestigation_candidates

factory = getattr(rq7_reinvestigation_candidates, sys.argv[1])
for candidate in factory():
    print(candidate.run_name, candidate.dataset_size, sep="\t")
PY
    ) || return $?
    local -a candidate_rows
    mapfile -t candidate_rows <<< "$candidate_output"
    if [[ "${#candidate_rows[@]}" -ne "$expected_count" ]]; then
        echo "RQ7 $stage_label stage must contain exactly $expected_count runs" >&2
        return 2
    fi

    local -A seen=()
    local enqueued_count=0
    local skipped_count=0
    local row run candidate_dataset extra directory artifact_status
    local -a verifier_args
    for row in "${candidate_rows[@]}"; do
        IFS=$'\t' read -r run candidate_dataset extra <<< "$row"
        if [[ -z "$run" || "$candidate_dataset" != "$dataset_size" || \
              -n "$extra" || -n "${seen[$run]+x}" ]]; then
            echo "Invalid or duplicate RQ7 candidate row: $row" >&2
            return 2
        fi
        seen[$run]=1
        directory="$logs/$run"
        verifier_args=("G1_RQ7_RUN=$run")
        artifact_status=0
        g1_require_config_recipe_compatible_or_absent "$directory" "$config" \
            "${verifier_args[@]}" || artifact_status=$?
        if [[ "$artifact_status" -eq 0 ]]; then
            echo "=== skipped compatible $run ==="
            skipped_count=$((skipped_count + 1))
            continue
        fi
        [[ "$artifact_status" -eq 1 ]] || return "$artifact_status"
        TRAINING_QUEUE_DATA_GROUP="g1-rq7-${dataset_size}-seq128" \
            enqueue "$run" "${verifier_args[@]}" || return 1
        enqueued_count=$((enqueued_count + 1))
    done

    echo "=== rq7 $stage_label: enqueued=${enqueued_count}, skipped=${skipped_count} ==="
    g1_stop_artifact_verifier
    drain || return 1
}
