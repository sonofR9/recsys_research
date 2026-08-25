#!/usr/bin/env bash
set -u

launcher_dir=$(cd "$(dirname "$0")" && pwd)
source "$launcher_dir/../architecture/manifest.sh"
source "$launcher_dir/../artifacts.sh"

# RQ4 was decided on 50M-selected FFN widths, and 50M prefers small capacity on
# every axis 500M reverses. This tunes the deep rate at native size for the two
# parameter-matched arms instead: RegularMLP has two dim x width matrices and
# SwiGLU has three, so GELU 171 and SwiGLU 114 carry the same parameter count.
# The 20-epoch anneal comes from the base variant; G1_TUNE_EPOCHS only raises the
# early-stopping cap, so these runs stay comparable with the settled 500M arms.
requested_arms=${G1_FFN_TUNING_ARMS:-"gelu171 swiglu114"}
requested_deep_lrs=${G1_FFN_TUNING_DEEP_LRS:-"0.006 0.012 0.024"}
embedding_lr=${G1_FFN_TUNING_EMBEDDING_LR:-0.064}
final_batch_size=1280
final_epochs=40
final_run_revision=3
provenance_suffix="_cap${final_epochs}_ts2_r${final_run_revision}"

declare -A seen_arms
tuning_arms=()
for arm in $requested_arms; do
    if [[ -n "${seen_arms[$arm]+x}" ]]; then
        echo "Duplicate G1_FFN_TUNING_ARMS value: $arm" >&2
        exit 2
    fi
    seen_arms[$arm]=1
    tuning_arms+=("$arm")
done
if [[ "${#tuning_arms[@]}" -eq 0 ]]; then
    echo "G1_FFN_TUNING_ARMS must select at least one arm" >&2
    exit 2
fi

declare -A seen_rates
deep_lrs=()
for rate in $requested_deep_lrs; do
    if [[ ! "$rate" =~ ^0\.[0-9]+$ ]]; then
        echo "G1_FFN_TUNING_DEEP_LRS must be canonical decimals: $rate" >&2
        exit 2
    fi
    if [[ -n "${seen_rates[$rate]+x}" ]]; then
        echo "Duplicate G1_FFN_TUNING_DEEP_LRS value: $rate" >&2
        exit 2
    fi
    seen_rates[$rate]=1
    deep_lrs+=("$rate")
done
if [[ "${#deep_lrs[@]}" -eq 0 ]]; then
    echo "G1_FFN_TUNING_DEEP_LRS must select at least one rate" >&2
    exit 2
fi

declare -A manifest_sources
declare -A manifest_transformer_fields
declare -A manifest_experiment_fields
declare -A manifest_extras
while IFS='|' read -r axis treatment source transformer_fields experiment_fields extras alias; do
    [[ "$axis" == ffn ]] || continue
    manifest_sources[$treatment]=$source
    manifest_transformer_fields[$treatment]=$transformer_fields
    manifest_experiment_fields[$treatment]=$experiment_fields
    manifest_extras[$treatment]=$extras
done < <(g1_manifest_rows)
for arm in "${tuning_arms[@]}"; do
    if [[ -z "${manifest_sources[$arm]+x}" || \
          "${manifest_transformer_fields[$arm]}" != ffn || \
          -n "${manifest_experiment_fields[$arm]}" || \
          -z "${manifest_extras[$arm]}" ]]; then
        echo "Missing compatible ffn/$arm manifest treatment" >&2
        exit 2
    fi
done

slug() { echo "${1//./p}"; }

repo_root=$(cd "$launcher_dir/../../../.." && pwd)
cd "$repo_root"
export WANDB_MODE=${WANDB_MODE:-offline}
export G1_DATASET_SIZE=500m
export TRAINING_QUEUE_MONITOR_LIGHT_GPUS=${TRAINING_QUEUE_MONITOR_LIGHT_GPUS:-1}
export TRAINING_QUEUE_DATA_GROUP=g1-rq-architecture-500m-seq128
unset G1_MAX_USERS G1_VAL_BATCH_SIZE G1_TRAIN_BATCH_SIZE G1_SEED
unset G1_TUNE_BATCH_SIZE G1_TUNE_NUM_NEGATIVES G1_TUNE_NUM_WORKERS
unset G1_TUNE_GRADIENT_ACCUMULATION_STEPS G1_TUNE_LOGQ_ALPHA
unset G1_TUNE_LOGQ_CORRECTION G1_TUNE_CORRECT_POSITIVE_LOGQ
unset G1_TUNE_MASK_FALSE_NEGATIVES G1_TUNE_EXCLUDE_OWN_GROUP_NEGATIVES
unset G1_TUNE_RANDOM_FRACTION G1_TUNE_FFN_DIM
unset G1_TUNE_TRANSFORMER_FIELDS G1_TUNE_EXPERIMENT_FIELDS
unset G1_TUNE_EMBEDDING_LR G1_TUNE_DEEP_LR G1_TUNE_SOURCE_VARIANT G1_TUNE_RUN

TRAINING_QUEUE_SCRIPT=experiments/g1_sasrec_item_ids_likes/configs/rq_tuning_variant.py
source "${G1_TRAINING_QUEUE_LIBRARY:-utils/training_queue/queue.sh}" || exit 1

enqueued_count=0
skipped_count=0
for arm in "${tuning_arms[@]}"; do
    for deep_lr in "${deep_lrs[@]}"; do
        run="rqfinal_ffn_${arm}_e$(slug "$embedding_lr")_d$(slug "$deep_lr")"
        run+="_b${final_batch_size}${provenance_suffix}"
        directory="$repo_root/generated/logs/g1_rqtune_${run}_500m"
        read -ra extra_args <<< "${manifest_extras[$arm]}"
        verifier_args=(
            "G1_TUNE_RUN=${run}"
            "G1_TUNE_RUN_REVISION=${final_run_revision}"
            "G1_TUNE_EPOCHS=${final_epochs}"
            "G1_TUNE_SOURCE_VARIANT=${manifest_sources[$arm]}"
            "G1_TUNE_TRANSFORMER_FIELDS=${manifest_transformer_fields[$arm]}"
            "G1_TUNE_EXPERIMENT_FIELDS=${manifest_experiment_fields[$arm]}"
            "G1_TUNE_EMBEDDING_LR=${embedding_lr}"
            "G1_TUNE_DEEP_LR=${deep_lr}"
            "G1_TUNE_BATCH_SIZE=${final_batch_size}"
            "G1_TUNE_GRADIENT_ACCUMULATION_STEPS=1"
            "${extra_args[@]}"
        )
        artifact_status=0
        g1_require_compatible_or_absent "$directory" 500m \
            "${verifier_args[@]}" || artifact_status=$?
        if [[ "$artifact_status" -eq 0 ]]; then
            echo "=== skipped compatible $(basename "$directory") ==="
            skipped_count=$((skipped_count + 1))
            continue
        fi
        [[ "$artifact_status" -eq 1 ]] || exit "$artifact_status"
        if ! enqueue "g1_rqtune_${run}_500m" "${verifier_args[@]}"; then
            echo "Failed to enqueue FFN 500M tuning run $run" >&2
            exit 1
        fi
        enqueued_count=$((enqueued_count + 1))
    done
done

echo "=== ffn 500m tuning: enqueued=${enqueued_count}, skipped=${skipped_count} ==="
g1_stop_artifact_verifier
drain || exit 1
