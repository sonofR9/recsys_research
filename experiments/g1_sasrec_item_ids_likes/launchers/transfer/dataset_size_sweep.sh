#!/usr/bin/env bash
set -u

source "$(dirname "$0")/common.sh"

cd "$(dirname "$0")/../../../.."
export WANDB_MODE=${WANDB_MODE:-offline}
export G1_DATASET_SIZE=${G1_SIZE_SWEEP_DATASET_SIZE:-50m}
export TRAINING_QUEUE_MONITOR_LIGHT_GPUS=${TRAINING_QUEUE_MONITOR_LIGHT_GPUS:-1}
export TRAINING_QUEUE_DATA_GROUP=g1-transfer-sizesweep-${G1_DATASET_SIZE}
unset G1_MAX_USERS
unset G1_VAL_BATCH_SIZE
unset G1_TRAIN_BATCH_SIZE
unset G1_SEED

if [[ "$G1_DATASET_SIZE" != 50m && "$G1_DATASET_SIZE" != 500m ]]; then
    echo "G1_SIZE_SWEEP_DATASET_SIZE must be 50m or 500m" >&2
    exit 2
fi

deep_rates=${G1_SIZE_SWEEP_DEEP_LRS:-"0.003 0.006 0.012 0.024 0.048 0.096"}
embedding_rates=${G1_SIZE_SWEEP_EMBEDDING_LRS:-0.032}
# The transition is a token count, not a fraction of the run: the schedule is
# budget-agnostic only if both dataset sizes share the same absolute value.
power_tokens=${G1_SIZE_SWEEP_POWER_TOKENS:-1500000}
power_epochs=${G1_SIZE_SWEEP_POWER_EPOCHS:-40}

for embedding_rate in $embedding_rates; do
    g1_require_canonical_positive_decimal "$embedding_rate" || exit 2
done
for deep_rate in $deep_rates; do
    g1_require_canonical_positive_decimal "$deep_rate" || exit 2
done
for value in "$power_tokens" "$power_epochs"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "Power token count and epoch cap must be canonical positive integers" >&2
        exit 2
    fi
done

TRAINING_QUEUE_SCRIPT=experiments/g1_sasrec_item_ids_likes/configs/transfer_variant.py
source utils/training_queue/queue.sh || exit 1

schedules=${G1_SIZE_SWEEP_SCHEDULES:-"linear power"}
for schedule in $schedules; do
    if [[ "$schedule" != linear && "$schedule" != power ]]; then
        echo "G1_SIZE_SWEEP_SCHEDULES may contain only linear and power" >&2
        exit 2
    fi
done

for embedding_rate in $embedding_rates; do
    for deep_rate in $deep_rates; do
        rates="e${embedding_rate//./p}_d${deep_rate//./p}"
        for schedule in $schedules; do
            if [[ "$schedule" == linear ]]; then
                g1_enqueue_transfer_recipe "sizesweep_linear_${rates}_ts2_r2" 20 \
                    "$embedding_rate" "$deep_rate" mup 1280 \
                    selected_quality_b1280 || exit 2
            else
                g1_enqueue_transfer_recipe \
                    "sizesweep_power_${rates}_cap${power_epochs}_ts2_r2" \
                    "$power_epochs" "$embedding_rate" "$deep_rate" mup 1280 \
                    selected_quality_b1280 \
                    "G1_TRANSFER_POWER_TOKENS=${power_tokens}" || exit 2
            fi
        done
    done
done

g1_stop_artifact_verifier
drain || exit 1
