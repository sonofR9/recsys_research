#!/usr/bin/env bash
set -u

launcher_dir=$(cd "$(dirname "$0")" && pwd)
repo_root=$(cd "$launcher_dir/../../../.." && pwd)
source "$launcher_dir/common.sh"

if [[ -n "${G1_FINAL_EPOCHS+x}${G1_FINAL_RUN_REVISION+x}" ]] && \
   [[ -z "${G1_FINAL_EPOCHS+x}" || -z "${G1_FINAL_RUN_REVISION+x}" ]]; then
    echo "G1_FINAL_EPOCHS and G1_FINAL_RUN_REVISION must be set together" >&2
    exit 2
fi
final_epochs=${G1_FINAL_EPOCHS:-40}
final_run_revision=${G1_FINAL_RUN_REVISION:-2}
if [[ ! "$final_epochs" =~ ^[1-9][0-9]*$ || "$final_epochs" -lt 20 ]]; then
    echo "G1_FINAL_EPOCHS must be a canonical integer of at least 20" >&2
    exit 2
fi
if [[ ! "$final_run_revision" =~ ^[1-9][0-9]*$ ]]; then
    echo "G1_FINAL_RUN_REVISION must be a canonical positive integer" >&2
    exit 2
fi

selector=${G1_NATIVE_SELECTOR:-$repo_root/experiments/g1_sasrec_item_ids_likes/analysis/select_native_500m.py}
source_generated=${G1_NATIVE_SOURCE_GENERATED:-$repo_root/generated}
selection=$(python "$selector" --generated "$source_generated" --format provenance-tsv) || exit 2
IFS=$'\t' read -r source_digest source_id embedding_lr deep_lr source_artifacts winner_run <<< "$selection"
if [[ ! "$source_digest" =~ ^[0-9a-f]{64}$ || \
      "$source_id" != "${source_digest:0:12}" || \
      "$embedding_lr" != 0.001 || \
      "$deep_lr" != 0.002 || "$source_artifacts" != 42 ]]; then
    echo "Invalid native-50M selection: $selection" >&2
    exit 2
fi

cd "$repo_root"
export WANDB_MODE=${WANDB_MODE:-offline}
export G1_DATASET_SIZE=500m
export TRAINING_QUEUE_MONITOR_LIGHT_GPUS=${TRAINING_QUEUE_MONITOR_LIGHT_GPUS:-1}
export TRAINING_QUEUE_DATA_GROUP=g1-selected-native-500m-seq100
unset G1_MAX_USERS G1_VAL_BATCH_SIZE G1_TRAIN_BATCH_SIZE G1_SEED

run="selected_native50_${source_id}_e0p001_d0p002"
run+="_cap${final_epochs}_ts2_r${final_run_revision}"
TRAINING_QUEUE_SCRIPT=experiments/g1_sasrec_item_ids_likes/configs/transfer_variant.py
source "${G1_TRAINING_QUEUE_LIBRARY:-utils/training_queue/queue.sh}" || exit 1

export G1_TRANSFER_RUN_REVISION=$final_run_revision
g1_enqueue_transfer_recipe "$run" "$final_epochs" "$embedding_lr" "$deep_lr" \
    conventional 1280 homework_fixed_leave_one_out || exit 2
g1_stop_artifact_verifier
drain || exit 1
provenance_tool=${G1_NATIVE_PROVENANCE_TOOL:-$repo_root/experiments/g1_sasrec_item_ids_likes/analysis/native_500m_provenance.py}
python "$provenance_tool" \
    --target "$repo_root/generated/logs/g1_transfer_${run}_500m" \
    --source-digest "$source_digest" \
    --winner-run "$winner_run" \
    --embedding-lr "$embedding_lr" \
    --deep-lr "$deep_lr" \
    --source-artifacts "$source_artifacts" \
    --write || exit 2
