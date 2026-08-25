#!/usr/bin/env bash
set -u

launcher_dir=$(cd "$(dirname "$0")" && pwd)
source "$launcher_dir/../artifacts.sh"

repo_root=$(cd "$launcher_dir/../../../.." && pwd)
cd "$repo_root"
logs=${G1_RQ8_LOGS:-$repo_root/generated/logs}
selector=${G1_RQ8_FOLLOWUP_SELECTOR:-$repo_root/experiments/g1_sasrec_item_ids_likes/analysis/rq8_reinvestigation_selection.py}
export WANDB_MODE=${WANDB_MODE:-offline}
export G1_DATASET_SIZE=500m
export TRAINING_QUEUE_MONITOR_LIGHT_GPUS=${TRAINING_QUEUE_MONITOR_LIGHT_GPUS:-1}
unset G1_MAX_USERS G1_MAX_EPOCHS G1_SEED G1_TRAIN_BATCH_SIZE G1_VAL_BATCH_SIZE
unset G1_VARIANT G1_RQ8_RUN

config=experiments/g1_sasrec_item_ids_likes/configs/rq8_reinvestigation_variant.py
TRAINING_QUEUE_SCRIPT=$config
queue_library=${G1_TRAINING_QUEUE_LIBRARY:-utils/training_queue/queue.sh}

wave=0
while true; do
    candidate_output=$(python "$selector" --logs "$logs") || exit $?
    [[ -n "$candidate_output" ]] || break
    mapfile -t candidate_rows <<< "$candidate_output"
    source "$queue_library" || exit 1
    declare -A seen=()
    enqueued_count=0
    skipped_count=0
    for row in "${candidate_rows[@]}"; do
        IFS=$'\t' read -r run max_seq_len extra <<< "$row"
        if [[ -z "$run" || -z "$max_seq_len" || -n "$extra" || \
              ! "$max_seq_len" =~ ^[0-9]+$ || -n "${seen[$run]+x}" ]]; then
            echo "Invalid or duplicate RQ8 follow-up row: $row" >&2
            exit 2
        fi
        seen[$run]=1
        directory="$logs/$run"
        verifier_args=("G1_RQ8_RUN=$run")
        artifact_status=0
        g1_require_config_recipe_compatible_or_absent "$directory" "$config" \
            "${verifier_args[@]}" || artifact_status=$?
        if [[ "$artifact_status" -eq 0 ]]; then
            echo "=== skipped compatible $run ==="
            skipped_count=$((skipped_count + 1))
            continue
        fi
        [[ "$artifact_status" -eq 1 ]] || exit "$artifact_status"
        if [[ "$run" == *"_sequence_fullcausal_"* ]]; then
            data_group="g1-rq8-fullcausal-500m-seq${max_seq_len}"
        else
            data_group="g1-rq8-500m-seq${max_seq_len}"
        fi
        TRAINING_QUEUE_DATA_GROUP="$data_group" \
            enqueue "$run" "${verifier_args[@]}" || exit 1
        enqueued_count=$((enqueued_count + 1))
    done
    wave=$((wave + 1))
    echo "=== rq8 follow-up wave ${wave}: enqueued=${enqueued_count}, skipped=${skipped_count} ==="
    g1_stop_artifact_verifier
    drain || exit 1
done

g1_stop_artifact_verifier
echo "=== rq8 follow-ups resolved after ${wave} queue waves ==="
