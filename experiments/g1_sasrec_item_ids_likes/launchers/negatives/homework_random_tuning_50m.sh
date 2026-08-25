#!/usr/bin/env bash
set -u

launcher_dir=$(cd "$(dirname "$0")" && pwd)
source "$launcher_dir/../artifacts.sh"

if [[ -n "${G1_HOMEWORK_RANDOM_EPOCHS+x}${G1_HOMEWORK_RANDOM_RUN_REVISION+x}" ]]; then
    if [[ -z "${G1_HOMEWORK_RANDOM_EPOCHS+x}" || \
          -z "${G1_HOMEWORK_RANDOM_RUN_REVISION+x}" ]]; then
        echo "G1_HOMEWORK_RANDOM_EPOCHS and G1_HOMEWORK_RANDOM_RUN_REVISION must be set together" >&2
        exit 2
    fi
fi
epochs=${G1_HOMEWORK_RANDOM_EPOCHS:-20}
run_revision=${G1_HOMEWORK_RANDOM_RUN_REVISION:-1}
if [[ ! "$epochs" =~ ^[1-9][0-9]*$ || "$epochs" -lt 20 ]]; then
    echo "G1_HOMEWORK_RANDOM_EPOCHS must be a canonical integer of at least 20" >&2
    exit 2
fi
if [[ ! "$run_revision" =~ ^[1-9][0-9]*$ ]]; then
    echo "G1_HOMEWORK_RANDOM_RUN_REVISION must be a canonical positive integer" >&2
    exit 2
fi

default_embedding_lrs="0.0005 0.001 0.002"
default_deep_lrs="0.001 0.002 0.004"
if [[ -n "${G1_HOMEWORK_RANDOM_EMBEDDING_LRS+x}" && \
      -z "$G1_HOMEWORK_RANDOM_EMBEDDING_LRS" ]] || \
   [[ -n "${G1_HOMEWORK_RANDOM_DEEP_LRS+x}" && \
      -z "$G1_HOMEWORK_RANDOM_DEEP_LRS" ]]; then
    echo "G1_HOMEWORK_RANDOM_EMBEDDING_LRS and G1_HOMEWORK_RANDOM_DEEP_LRS cannot be empty" >&2
    exit 2
fi
embedding_lrs=${G1_HOMEWORK_RANDOM_EMBEDDING_LRS:-$default_embedding_lrs}
deep_lrs=${G1_HOMEWORK_RANDOM_DEEP_LRS:-$default_deep_lrs}
read -r -a raw_embedding_lr_values <<< "$embedding_lrs"
read -r -a raw_deep_lr_values <<< "$deep_lrs"
if [[ -n "${G1_HOMEWORK_RANDOM_EMBEDDING_LRS+x}${G1_HOMEWORK_RANDOM_DEEP_LRS+x}" && \
      -z "${G1_HOMEWORK_RANDOM_RUN_TAG:-}" ]]; then
    echo "Set a unique G1_HOMEWORK_RANDOM_RUN_TAG for a boundary extension" >&2
    exit 2
fi
default_run_tag=initial
[[ "$epochs" -eq 20 ]] || default_run_tag=capcontinue
run_tag=${G1_HOMEWORK_RANDOM_RUN_TAG-$default_run_tag}
if [[ ! "$run_tag" =~ ^[a-z0-9_]+$ ]]; then
    echo "G1_HOMEWORK_RANDOM_RUN_TAG must contain lowercase letters, digits, and _" >&2
    exit 2
fi
if [[ "$run_tag" =~ _r[1-9][0-9]*$ || "$run_tag" =~ (^|_)cap[1-9][0-9]*$ ]]; then
    echo "G1_HOMEWORK_RANDOM_RUN_TAG cannot reuse reserved provenance suffixes" >&2
    exit 2
fi
if [[ "$run_tag" == initial ]] && \
   [[ "$epochs" -ne 20 || "$run_revision" -ne 1 || \
      -n "${G1_HOMEWORK_RANDOM_EMBEDDING_LRS+x}${G1_HOMEWORK_RANDOM_DEEP_LRS+x}" ]]; then
    echo "G1_HOMEWORK_RANDOM_RUN_TAG=initial is reserved for the initial grid" >&2
    exit 2
fi
if [[ "$run_tag" == capcontinue ]] && \
   [[ "$epochs" -eq 20 || "$run_revision" -eq 1 ]]; then
    echo "G1_HOMEWORK_RANDOM_RUN_TAG=capcontinue is reserved for cap continuations" >&2
    exit 2
fi

repo_root=$(cd "$launcher_dir/../../../.." && pwd)
selector=$repo_root/experiments/g1_sasrec_item_ids_likes/analysis/select_homework_negative_control.py
if ! canonical_embedding_lrs=$(
    python "$selector" --canonicalize-lrs "${raw_embedding_lr_values[@]}"
); then
    exit 2
fi
if ! canonical_deep_lrs=$(
    python "$selector" --canonicalize-lrs "${raw_deep_lr_values[@]}"
); then
    exit 2
fi
mapfile -t embedding_lr_values <<< "$canonical_embedding_lrs"
mapfile -t deep_lr_values <<< "$canonical_deep_lrs"
cd "$repo_root"
export WANDB_MODE=${WANDB_MODE:-offline}
export G1_DATASET_SIZE=50m
export TRAINING_QUEUE_MONITOR_LIGHT_GPUS=${TRAINING_QUEUE_MONITOR_LIGHT_GPUS:-1}
export TRAINING_QUEUE_DATA_GROUP=g1-homework-random-50m-seq100
unset G1_MAX_USERS G1_VAL_BATCH_SIZE G1_TRAIN_BATCH_SIZE G1_SEED G1_MAX_EPOCHS
unset G1_HOMEWORK_RANDOM_RUN G1_HOMEWORK_RANDOM_EMBEDDING_LR
unset G1_HOMEWORK_RANDOM_DEEP_LR

config=experiments/g1_sasrec_item_ids_likes/configs/homework_random_control.py
TRAINING_QUEUE_SCRIPT=$config
source "${G1_TRAINING_QUEUE_LIBRARY:-utils/training_queue/queue.sh}" || exit 1

slug() {
    local value=${1//./p}
    value=${value//-/m}
    echo "${value//+/p}"
}

cap=
[[ "$epochs" -eq 20 ]] || cap="_cap${epochs}"
enqueued_count=0
skipped_count=0
for embedding_lr in "${embedding_lr_values[@]}"; do
    for deep_lr in "${deep_lr_values[@]}"; do
        run="${run_tag}_e$(slug "$embedding_lr")_d$(slug "$deep_lr")${cap}_ts2_r${run_revision}"
        name="g1_homework_random_${run}_50m"
        directory="$repo_root/generated/logs/$name"
        assignments=(
            "G1_HOMEWORK_RANDOM_RUN=${run}"
            "G1_HOMEWORK_RANDOM_EPOCHS=${epochs}"
            "G1_HOMEWORK_RANDOM_RUN_REVISION=${run_revision}"
            "G1_HOMEWORK_RANDOM_EMBEDDING_LR=${embedding_lr}"
            "G1_HOMEWORK_RANDOM_DEEP_LR=${deep_lr}"
        )
        artifact_status=0
        g1_require_config_compatible_or_absent \
            "$directory" "$repo_root/$config" "${assignments[@]}" || artifact_status=$?
        if [[ "$artifact_status" -eq 0 ]]; then
            echo "=== skipped compatible $name ==="
            skipped_count=$((skipped_count + 1))
        elif [[ "$artifact_status" -eq 1 ]]; then
            enqueue "$name" "${assignments[@]}" || exit 2
            enqueued_count=$((enqueued_count + 1))
        else
            exit "$artifact_status"
        fi
    done
done

echo "=== homework random 50m: enqueued=${enqueued_count}, skipped=${skipped_count} ==="
g1_stop_artifact_verifier
drain || exit 1
