#!/usr/bin/env bash
set -u

launcher_dir=$(cd "$(dirname "$0")" && pwd)
source "$launcher_dir/../artifacts.sh"

batch_size=${G1_HOMEWORK_BATCH_SIZE:?Set G1_HOMEWORK_BATCH_SIZE to the selected batch size}
if [[ ! "$batch_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "G1_HOMEWORK_BATCH_SIZE must be a canonical positive integer" >&2
    exit 2
fi

cd "$(dirname "$0")/../../../.."
export WANDB_MODE=${WANDB_MODE:-offline}
export G1_DATASET_SIZE=500m
export TRAINING_QUEUE_MONITOR_LIGHT_GPUS=${TRAINING_QUEUE_MONITOR_LIGHT_GPUS:-1}
export TRAINING_QUEUE_DATA_GROUP=g1-homework-calibration-500m-seq100
unset G1_MAX_USERS
unset G1_VAL_BATCH_SIZE
unset G1_TRAIN_BATCH_SIZE
unset G1_SEED
max_epochs=${G1_MAX_EPOCHS:-20}
if [[ ! "$max_epochs" =~ ^[1-9][0-9]*$ || "$max_epochs" -lt 20 ]]; then
    echo "G1_MAX_EPOCHS must be a canonical integer of at least 20" >&2
    exit 2
fi
export G1_HOMEWORK_RUN_TAG=r2

TRAINING_QUEUE_SCRIPT=experiments/g1_sasrec_item_ids_likes/configs/homework_500m.py
source utils/training_queue/queue.sh || exit 1

cap=
[[ "$max_epochs" -eq 20 ]] || cap="_cap${max_epochs}"
run="g1_calibrated_homework_baseline${cap}_ts2_b${batch_size}_r2_500m"
directory="$PWD/generated/logs/$run"
config="$PWD/experiments/g1_sasrec_item_ids_likes/configs/homework_500m.py"
assignments=("G1_HOMEWORK_BATCH_SIZE=${batch_size}" "G1_HOMEWORK_RUN_TAG=r2" "G1_MAX_EPOCHS=${max_epochs}")
artifact_status=0
g1_require_config_compatible_or_absent "$directory" "$config" \
    "${assignments[@]}" || artifact_status=$?
if [[ "$artifact_status" -eq 0 ]]; then
    echo "=== skipped compatible $run ==="
elif [[ "$artifact_status" -eq 1 ]]; then
    enqueue "$run" "${assignments[@]}"
else
    exit "$artifact_status"
fi

g1_stop_artifact_verifier
drain || exit 1
