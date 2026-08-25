#!/usr/bin/env bash
set -u

launcher_dir=$(cd "$(dirname "$0")" && pwd)
source "$launcher_dir/../artifacts.sh"

seeds=${G1_BASELINE_SPREAD_SEEDS:-0 1 2 3 4 5 6 7 8 9}
if [[ -z "$seeds" ]]; then
    echo "G1_BASELINE_SPREAD_SEEDS must contain at least one seed" >&2
    exit 2
fi
seen_seeds=" "
for seed in $seeds; do
    if [[ ! "$seed" =~ ^(0|[1-9][0-9]*)$ ]]; then
        echo "G1_BASELINE_SPREAD_SEEDS must contain canonical nonnegative integers" >&2
        exit 2
    fi
    if [[ "$seen_seeds" == *" $seed "* ]]; then
        echo "G1_BASELINE_SPREAD_SEEDS contains duplicate seed: $seed" >&2
        exit 2
    fi
    seen_seeds+="$seed "
done

cd "$(dirname "$0")/../../../.."
export WANDB_MODE=${WANDB_MODE:-offline}
export G1_DATASET_SIZE=500m
export TRAINING_QUEUE_MONITOR_LIGHT_GPUS=${TRAINING_QUEUE_MONITOR_LIGHT_GPUS:-1}
export TRAINING_QUEUE_DATA_GROUP=g1-baseline-spread-500m-seq100
unset G1_MAX_USERS
unset G1_VAL_BATCH_SIZE
unset G1_TRAIN_BATCH_SIZE
unset G1_HOMEWORK_BATCH_SIZE
unset G1_HOMEWORK_RUN_TAG
unset G1_SEED
unset G1_VARIANT
max_epochs=${G1_MAX_EPOCHS:-40}
if [[ ! "$max_epochs" =~ ^[1-9][0-9]*$ || "$max_epochs" -lt 20 ]]; then
    echo "G1_MAX_EPOCHS must be a canonical integer of at least 20" >&2
    exit 2
fi
variant=homework_baseline_native500_r3
TRAINING_QUEUE_SCRIPT=experiments/g1_sasrec_item_ids_likes/configs/variant.py
source "${G1_TRAINING_QUEUE_LIBRARY:-utils/training_queue/queue.sh}" || exit 1

for seed in $seeds; do
    cap=
    [[ "$max_epochs" -eq 20 ]] || cap="_cap${max_epochs}"
    run="g1_calibrated_${variant}${cap}_ts2_500m_s${seed}"
    directory="$PWD/generated/logs/$run"
    config="$PWD/experiments/g1_sasrec_item_ids_likes/configs/variant.py"
    assignments=(
        "G1_VARIANT=${variant}"
        "G1_SEED=${seed}"
        "G1_MAX_EPOCHS=${max_epochs}"
    )
    if g1_artifact_exists "$directory"; then
        verify_status=0
        g1_verify_config_artifact "$directory" "$config" \
            "${assignments[@]}" || verify_status=$?
        if [[ "$verify_status" -eq 0 ]]; then
            echo "=== skipped compatible $run ==="
            continue
        fi
        [[ "$verify_status" -eq 1 ]] || exit "$verify_status"
        recipe_status=0
        g1_verify_config_recipe_artifact "$directory" "$config" \
            "${assignments[@]}" || recipe_status=$?
        if [[ "$recipe_status" -eq 0 ]]; then
            echo "$run reached its validation cap; increase G1_MAX_EPOCHS" >&2
            exit 2
        fi
        [[ "$recipe_status" -eq 1 ]] || exit "$recipe_status"
    fi
    artifact_status=0
    g1_require_config_compatible_or_absent "$directory" "$config" \
        "${assignments[@]}" || artifact_status=$?
    if [[ "$artifact_status" -eq 0 ]]; then
        echo "=== skipped compatible $run ==="
        continue
    fi
    [[ "$artifact_status" -eq 1 ]] || exit "$artifact_status"
    enqueue "$run" "${assignments[@]}"
done

g1_stop_artifact_verifier
drain || exit 1
