#!/usr/bin/env bash
set -u

transfer_epochs=${G1_TRANSFER_EPOCHS:-20}

source "$(dirname "$0")/common.sh"

if [[ ! "$transfer_epochs" =~ ^[1-9][0-9]*$ || "$transfer_epochs" -lt 20 ]]; then
    echo "G1_TRANSFER_EPOCHS must be a canonical integer of at least 20" >&2
    exit 2
fi
cap=
[[ "$transfer_epochs" -eq 20 ]] || cap="_cap${transfer_epochs}"

cd "$(dirname "$0")/../../../.."
export WANDB_MODE=${WANDB_MODE:-offline}
export G1_DATASET_SIZE=50m
export TRAINING_QUEUE_MONITOR_LIGHT_GPUS=${TRAINING_QUEUE_MONITOR_LIGHT_GPUS:-1}
export TRAINING_QUEUE_DATA_GROUP=g1-batch-scaling-50m-seq100
unset G1_MAX_USERS G1_VAL_BATCH_SIZE G1_TRAIN_BATCH_SIZE G1_SEED

batches=${G1_BATCH_SCALING_BATCHES:-1280}
embedding_rates_b128=${G1_BATCH_SCALING_EMBEDDING_RATES_B128:-0.0005 0.001 0.002}
deep_rates_b128=${G1_BATCH_SCALING_DEEP_RATES_B128:-0.0005 0.001 0.002}
embedding_rates_b1280=${G1_BATCH_SCALING_EMBEDDING_RATES_B1280:-0.008 0.016 0.032 0.064}
deep_rates_b1280=${G1_BATCH_SCALING_DEEP_RATES_B1280:-0.002 0.004 0.008 0.016 0.032}
sparse_points=${G1_BATCH_SCALING_POINTS:-}

slug() {
    local value=${1//./p}
    echo "${value//-/m}"
}

declare -a planned_batches planned_embedding_lrs planned_deep_lrs planned_runs
declare -A planned_run_names

add_point() {
    local batch_size=$1
    local embedding_lr=$2
    local deep_lr=$3
    if [[ "$batch_size" != 128 && "$batch_size" != 1280 ]] ||
       ! g1_require_canonical_positive_decimal "$embedding_lr" ||
       ! g1_require_canonical_positive_decimal "$deep_lr"; then
        echo "Invalid batch-scaling point: $batch_size:$embedding_lr:$deep_lr" >&2
        exit 2
    fi
    local run="batchscale_b${batch_size}_e$(slug "$embedding_lr")"
    run+="_d$(slug "$deep_lr")${cap}_ts2_r2"
    if [[ -n "${planned_run_names[$run]+x}" ]]; then
        echo "Duplicate batch-scaling run: $run" >&2
        exit 2
    fi
    planned_run_names[$run]=1
    planned_batches+=("$batch_size")
    planned_embedding_lrs+=("$embedding_lr")
    planned_deep_lrs+=("$deep_lr")
    planned_runs+=("$run")
}

if [[ -n "$sparse_points" ]]; then
    for point in $sparse_points; do
        [[ "$point" =~ ^(128|1280):([^:]+):([^:]+)$ ]] || {
            echo "Invalid G1_BATCH_SCALING_POINTS entry: $point" >&2
            exit 2
        }
        IFS=: read -r -a fields <<< "$point"
        add_point "${fields[0]}" "${fields[1]}" "${fields[2]}"
    done
else
    for batch_size in $batches; do
        if [[ "$batch_size" == 128 ]]; then
            embedding_rates=$embedding_rates_b128
            deep_rates=$deep_rates_b128
        else
            embedding_rates=$embedding_rates_b1280
            deep_rates=$deep_rates_b1280
        fi
        for embedding_lr in $embedding_rates; do
            for deep_lr in $deep_rates; do
                add_point "$batch_size" "$embedding_lr" "$deep_lr"
            done
        done
    done
fi

TRAINING_QUEUE_SCRIPT=experiments/g1_sasrec_item_ids_likes/configs/transfer_variant.py
source utils/training_queue/queue.sh || exit 1

for index in "${!planned_runs[@]}"; do
    g1_enqueue_transfer_recipe "${planned_runs[$index]}" "$transfer_epochs" \
        "${planned_embedding_lrs[$index]}" "${planned_deep_lrs[$index]}" \
        conventional "${planned_batches[$index]}" \
        homework_fixed_leave_one_out || exit 2
done

g1_stop_artifact_verifier
drain || exit 1
