#!/usr/bin/env bash
set -u

launcher_dir=$(cd "$(dirname "$0")" && pwd)
source "$launcher_dir/../artifacts.sh"
source "$launcher_dir/manifest.sh"
source "$launcher_dir/../global_batch.sh"

valid_families=$G1_NEGATIVE_VALID_FAMILIES
selected_specs=${G1_SELECTED_NEGATIVE_WINNERS:-}
if [[ -z "$selected_specs" ]]; then
    echo "G1_SELECTED_NEGATIVE_WINNERS is required" >&2
    exit 2
fi
global_batch_size=${G1_GLOBAL_BATCH_SIZE:-}
if [[ ! "$global_batch_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "G1_GLOBAL_BATCH_SIZE must be a canonical positive integer" >&2
    exit 2
fi
if [[ -n "${G1_FINAL_EPOCHS+x}${G1_FINAL_RUN_REVISION+x}" ]] && \
   [[ -z "${G1_FINAL_EPOCHS+x}" || -z "${G1_FINAL_RUN_REVISION+x}" ]]; then
    echo "G1_FINAL_EPOCHS and G1_FINAL_RUN_REVISION must be set together" >&2
    exit 2
fi
final_epochs=${G1_FINAL_EPOCHS:-20}
final_run_revision=${G1_FINAL_RUN_REVISION:-2}
if [[ ! "$final_epochs" =~ ^[1-9][0-9]*$ || "$final_epochs" -lt 20 ]]; then
    echo "G1_FINAL_EPOCHS must be a canonical integer of at least 20" >&2
    exit 2
fi
if [[ ! "$final_run_revision" =~ ^[1-9][0-9]*$ ]]; then
    echo "G1_FINAL_RUN_REVISION must be a canonical positive integer" >&2
    exit 2
fi
final_provenance_suffix=
[[ "$final_epochs" -ne 20 ]] && final_provenance_suffix="_cap${final_epochs}"
final_provenance_suffix+="_ts2_r${final_run_revision}"

families=("${G1_NEGATIVE_FAMILY_SPECS[@]}")

require_positive_integer() {
    local name=$1
    local value=$2
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$name must be a canonical positive integer, got $value" >&2
        exit 2
    fi
}

require_positive_number() {
    local name=$1
    local value=$2
    if [[ ! "$value" =~ ^(0\.[0-9]*[1-9][0-9]*|[1-9][0-9]*(\.[0-9]+)?)([eE][+-]?[0-9]+)?$ ]]; then
        echo "$name must be a positive number, got $value" >&2
        exit 2
    fi
}

require_fraction() {
    local name=$1
    local value=$2
    if [[ ! "$value" =~ ^0\.[0-9]*[1-9][0-9]*$ ]]; then
        echo "$name must be a decimal in (0, 1), got $value" >&2
        exit 2
    fi
}

slug() {
    local value=${1,,}
    value=${value//./p}
    value=${value//-/m}
    echo "${value//+/}"
}

same_number() {
    awk -v left="$1" -v right="$2" 'BEGIN { exit !(left == right) }'
}

winner_for() {
    local requested=$1
    local spec
    local family
    local matches=0
    for spec in $selected_specs; do
        IFS=: read -r family embedding_lr deep_lr negative_count alpha random_fraction extra <<< "$spec"
        if [[ -n "${extra:-}" || -z "$family" || -z "$embedding_lr" || \
              -z "$deep_lr" || -z "$negative_count" || \
              -z "$alpha" || -z "$random_fraction" ]]; then
            echo "Invalid G1_SELECTED_NEGATIVE_WINNERS entry: $spec" >&2
            exit 2
        fi
        if [[ " $valid_families " != *" $family "* ]]; then
            echo "Unknown family in G1_SELECTED_NEGATIVE_WINNERS: $family" >&2
            exit 2
        fi
        if [[ "$family" == "$requested" ]]; then
            require_positive_number embedding_lr "$embedding_lr"
            require_positive_number deep_lr "$deep_lr"
            require_positive_integer negative_count "$negative_count"
            require_positive_number alpha "$alpha"
            require_fraction random_fraction "$random_fraction"
            selected_embedding_lr=$embedding_lr
            selected_deep_lr=$deep_lr
            selected_negative_count=$negative_count
            selected_alpha=$alpha
            selected_random_fraction=$random_fraction
            matches=$((matches + 1))
        fi
    done
    if [[ "$matches" -ne 1 ]]; then
        echo "Expected exactly one 500M winner for $requested, found $matches" >&2
        exit 2
    fi
}

for entry in "${families[@]}"; do
    winner_for "${entry%%:*}"
done

repo_root=$(cd "$launcher_dir/../../../.." && pwd)
g1_require_global_batch_selection "$repo_root" "$global_batch_size" || exit 2
cd "$repo_root"
export WANDB_MODE=${WANDB_MODE:-offline}
export G1_DATASET_SIZE=500m
export TRAINING_QUEUE_MONITOR_LIGHT_GPUS=${TRAINING_QUEUE_MONITOR_LIGHT_GPUS:-1}
export TRAINING_QUEUE_DATA_GROUP=g1-rq-negative-500m-seq128
unset G1_MAX_USERS
unset G1_VAL_BATCH_SIZE
unset G1_TRAIN_BATCH_SIZE
unset G1_SEED
unset G1_TUNE_BATCH_SIZE
unset G1_TUNE_NUM_NEGATIVES
unset G1_TUNE_NUM_WORKERS
unset G1_TUNE_LOGQ_ALPHA
unset G1_TUNE_LOGQ_CORRECTION
unset G1_TUNE_CORRECT_POSITIVE_LOGQ
unset G1_TUNE_MASK_FALSE_NEGATIVES
unset G1_TUNE_EXCLUDE_OWN_GROUP_NEGATIVES
unset G1_TUNE_RANDOM_FRACTION
unset G1_TUNE_FFN_DIM
unset G1_TUNE_TRANSFORMER_FIELDS
unset G1_TUNE_EXPERIMENT_FIELDS
unset G1_TUNE_EMBEDDING_LR
unset G1_TUNE_DEEP_LR
unset G1_TUNE_SOURCE_VARIANT
unset G1_TUNE_RUN

TRAINING_QUEUE_SCRIPT=experiments/g1_sasrec_item_ids_likes/configs/rq_tuning_variant.py
source "${G1_TRAINING_QUEUE_LIBRARY:-utils/training_queue/queue.sh}" || exit 1

enqueued_count=0
skipped_count=0
reused_count=0

enqueue_selected() {
    local family=$1
    local source=$2
    local embedding_lr=$3
    local deep_lr=$4
    local batch_size=$5
    local negative_count=$6
    local alpha=$7
    local random_fraction=$8
    local slug_embedding
    local slug_deep
    local slug_alpha
    local slug_random
    slug_embedding=$(slug "$embedding_lr")
    slug_deep=$(slug "$deep_lr")
    slug_alpha=$(slug "$alpha")
    slug_random=$(slug "$random_fraction")
    local exact="e${slug_embedding}_d${slug_deep}_b${batch_size}_n${negative_count}_a${slug_alpha}_r${slug_random}"
    local overrides=(
        "G1_TUNE_BATCH_SIZE=${batch_size}"
        "G1_TUNE_NUM_NEGATIVES=${negative_count}"
        "G1_TUNE_LOGQ_ALPHA=${alpha}"
    )
    if [[ "$family" == uniform_random_plus_streaming_logq_negative_only || \
          "$family" == uniform_random_plus_fixed_logq_negative_only ]]; then
        overrides+=("G1_TUNE_RANDOM_FRACTION=${random_fraction}")
    fi
    local run="rqfinal_neg_${family}_${exact}${final_provenance_suffix}"
    local directory="$repo_root/generated/logs/g1_rqtune_${run}_500m"
    local -a verifier_args=(
        "G1_TUNE_RUN=${run}"
        "G1_TUNE_RUN_REVISION=${final_run_revision}"
        "G1_TUNE_EPOCHS=${final_epochs}"
        "G1_TUNE_SOURCE_VARIANT=${source}"
        "G1_TUNE_TRANSFORMER_FIELDS="
        "G1_TUNE_EXPERIMENT_FIELDS=negative_sampling,logq_correction,correct_positive_logq,mask_false_negatives,exclude_own_group_negatives"
        "G1_TUNE_EMBEDDING_LR=${embedding_lr}"
        "G1_TUNE_DEEP_LR=${deep_lr}"
        "${overrides[@]}"
    )
    local artifact_status=0
    g1_require_compatible_or_absent "$directory" 500m \
        "${verifier_args[@]}" || artifact_status=$?
    if [[ "$artifact_status" -eq 0 ]]; then
        echo "=== skipped compatible $(basename "$directory") ==="
        skipped_count=$((skipped_count + 1))
        return 0
    fi
    [[ "$artifact_status" -eq 1 ]] || return "$artifact_status"

    if [[ "$family" == uniform_random && "$negative_count" == 512 ]] && \
          same_number "$alpha" 0.01; then
        artifact_status=0
        g1_reuse_first_compatible_artifact "$directory" 500m \
            "$repo_root/generated/logs/g1_rqtune_rqfinal_architecture_control_*_500m" \
            "${verifier_args[@]}" || artifact_status=$?
        if [[ "$artifact_status" -eq 0 ]]; then
            reused_count=$((reused_count + 1))
            return 0
        fi
        [[ "$artifact_status" -eq 1 ]] || return "$artifact_status"
    fi

    enqueue "g1_rqtune_${run}_500m" \
        "G1_TUNE_RUN=${run}" \
        "G1_TUNE_RUN_REVISION=${final_run_revision}" \
        "G1_TUNE_EPOCHS=${final_epochs}" \
        "G1_TUNE_SOURCE_VARIANT=${source}" \
        "G1_TUNE_EXPERIMENT_FIELDS=negative_sampling,logq_correction,correct_positive_logq,mask_false_negatives,exclude_own_group_negatives" \
        "G1_TUNE_EMBEDDING_LR=${embedding_lr}" \
        "G1_TUNE_DEEP_LR=${deep_lr}" "${overrides[@]}"
    enqueued_count=$((enqueued_count + 1))
}

for entry in "${families[@]}"; do
    family=${entry%%:*}
    source=${entry#*:}
    winner_for "$family"
    enqueue_selected "$family" "$source" \
        "$selected_embedding_lr" "$selected_deep_lr" "$global_batch_size" \
        "$selected_negative_count" "$selected_alpha" "$selected_random_fraction" || exit 2
done

echo "=== final negatives: enqueued=${enqueued_count}, skipped=${skipped_count}, reused=${reused_count} ==="
g1_stop_artifact_verifier
drain || exit 1
