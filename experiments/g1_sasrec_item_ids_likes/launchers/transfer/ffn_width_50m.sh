#!/usr/bin/env bash
set -u

source "$(dirname "$0")/common.sh"

cd "$(dirname "$0")/../../../.."
export WANDB_MODE=${WANDB_MODE:-offline}
export G1_DATASET_SIZE=50m
export TRAINING_QUEUE_MONITOR_LIGHT_GPUS=${TRAINING_QUEUE_MONITOR_LIGHT_GPUS:-1}
export TRAINING_QUEUE_DATA_GROUP=g1-transfer-ffnwidth-50m-seq128
unset G1_MAX_USERS
unset G1_VAL_BATCH_SIZE
unset G1_TRAIN_BATCH_SIZE
unset G1_SEED

ffn_widths=${G1_FFN_WIDTH_TRANSFER_WIDTHS:-"32 64 128 224"}
deep_rates=${G1_FFN_WIDTH_TRANSFER_DEEP_LRS:-"0.003 0.006 0.012 0.024 0.048 0.096"}
embedding_rate=${G1_FFN_WIDTH_TRANSFER_EMBEDDING_LR:-0.032}
mup_ffn_base=${G1_FFN_WIDTH_TRANSFER_MUP_FFN_BASE:-32}

for width in $ffn_widths; do
    if [[ ! "$width" =~ ^[1-9][0-9]*$ ]]; then
        echo "G1_FFN_WIDTH_TRANSFER_WIDTHS must be canonical positive integers" >&2
        exit 2
    fi
done
g1_require_canonical_positive_decimal "$embedding_rate" || exit 2
for deep_rate in $deep_rates; do
    g1_require_canonical_positive_decimal "$deep_rate" || exit 2
done
if [[ ! "$mup_ffn_base" =~ ^[1-9][0-9]*$ ]]; then
    echo "G1_FFN_WIDTH_TRANSFER_MUP_FFN_BASE must be a canonical positive integer" >&2
    exit 2
fi

TRAINING_QUEUE_SCRIPT=experiments/g1_sasrec_item_ids_likes/configs/transfer_variant.py
source utils/training_queue/queue.sh || exit 1

rate_slug() {
    local value=$1
    echo "${value//./p}"
}

for width in $ffn_widths; do
    for deep_rate in $deep_rates; do
        suffix="ffn${width}_e$(rate_slug "$embedding_rate")_d$(rate_slug "$deep_rate")_ts2_r2"
        g1_enqueue_transfer_recipe "ffnratio_${suffix}" 20 \
            "$embedding_rate" "$deep_rate" mup 1280 selected_quality_b1280 \
            "G1_TRANSFER_FFN_DIM=${width}" || exit 2
        g1_enqueue_transfer_recipe "ffnbase_${suffix}" 20 \
            "$embedding_rate" "$deep_rate" mup 1280 selected_quality_b1280 \
            "G1_TRANSFER_FFN_DIM=${width}" \
            "G1_TRANSFER_MUP_FFN_BASE=${mup_ffn_base}" || exit 2
    done
done

g1_stop_artifact_verifier
drain || exit 1
