#!/usr/bin/env bash
set -u

source "$(dirname "$0")/common.sh"

cd "$(dirname "$0")/../../../.."
export WANDB_MODE=${WANDB_MODE:-offline}
export G1_DATASET_SIZE=50m
export TRAINING_QUEUE_MONITOR_LIGHT_GPUS=${TRAINING_QUEUE_MONITOR_LIGHT_GPUS:-1}
export TRAINING_QUEUE_DATA_GROUP=g1-transfer-horizon-50m-seq128
unset G1_MAX_USERS
unset G1_VAL_BATCH_SIZE
unset G1_TRAIN_BATCH_SIZE
unset G1_SEED

TRAINING_QUEUE_SCRIPT=experiments/g1_sasrec_item_ids_likes/configs/transfer_variant.py
source utils/training_queue/queue.sh || exit 1

for rates in e8e3_d12e3:0.008:0.012 e16e3_d6e3:0.016:0.006 \
    e16e3_d12e3:0.016:0.012; do
    rate_name=${rates%%:*}
    rest=${rates#*:}
    embedding_rate=${rest%%:*}
    deep_rate=${rest#*:}
    for transition in t500k:500000 t1500k:1500000 t4000k:4000000; do
        transition_name=${transition%%:*}
        transition_tokens=${transition#*:}
        name="power_${rate_name}_${transition_name}_ts2_r2"
        g1_enqueue_transfer "$name" 20 "$embedding_rate" "$deep_rate" \
            "G1_TRANSFER_POWER_TOKENS=${transition_tokens}" || exit 2
    done
done

g1_stop_artifact_verifier
drain || exit 1
