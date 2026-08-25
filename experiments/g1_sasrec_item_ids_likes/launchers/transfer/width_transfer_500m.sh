#!/usr/bin/env bash
set -u

launcher_dir=$(cd "$(dirname "$0")" && pwd)
source "$launcher_dir/../architecture/manifest.sh"
source "$launcher_dir/../artifacts.sh"
source "$launcher_dir/../global_batch.sh"

if [[ -n "${G1_WIDTH_TRANSFER_EPOCHS+x}${G1_WIDTH_TRANSFER_RUN_REVISION+x}" ]] && \
   [[ -z "${G1_WIDTH_TRANSFER_EPOCHS+x}" || \
      -z "${G1_WIDTH_TRANSFER_RUN_REVISION+x}" ]]; then
    echo "G1_WIDTH_TRANSFER_EPOCHS and G1_WIDTH_TRANSFER_RUN_REVISION must be set together" >&2
    exit 2
fi
final_epochs=${G1_WIDTH_TRANSFER_EPOCHS:-20}
final_run_revision=${G1_WIDTH_TRANSFER_RUN_REVISION:-2}
if [[ ! "$final_epochs" =~ ^[1-9][0-9]*$ || "$final_epochs" -lt 20 ]]; then
    echo "G1_WIDTH_TRANSFER_EPOCHS must be a canonical integer of at least 20" >&2
    exit 2
fi
if [[ ! "$final_run_revision" =~ ^[1-9][0-9]*$ ]]; then
    echo "G1_WIDTH_TRANSFER_RUN_REVISION must be a canonical positive integer" >&2
    exit 2
fi
provenance_suffix=
[[ "$final_epochs" -ne 20 ]] && provenance_suffix="_cap${final_epochs}"
provenance_suffix+="_ts2_r${final_run_revision}"

requested_widths=${G1_WIDTH_TRANSFER_WIDTHS:-"16 256"}
declare -A requested_width_names
transfer_widths=()
for width in $requested_widths; do
    if [[ ! "$width" =~ ^(16|256)$ ]]; then
        echo "G1_WIDTH_TRANSFER_WIDTHS may contain only 16 and 256" >&2
        exit 2
    fi
    if [[ -n "${requested_width_names[$width]+x}" ]]; then
        echo "Duplicate G1_WIDTH_TRANSFER_WIDTHS value: $width" >&2
        exit 2
    fi
    requested_width_names[$width]=1
    transfer_widths+=("$width")
done
if [[ "${#transfer_widths[@]}" -eq 0 ]]; then
    echo "G1_WIDTH_TRANSFER_WIDTHS must select at least one width" >&2
    exit 2
fi

repo_root=$(cd "$launcher_dir/../../../.." && pwd)
final_batch_size=1280
embedding_lr=0.032
deep_lr=0.012
g1_require_global_batch_selection "$repo_root" "$final_batch_size" || exit 2
if ! g1_same_number "$G1_VERIFIED_CONTROL_EMBEDDING_LR" "$embedding_lr" || \
   ! g1_same_number "$G1_VERIFIED_CONTROL_DEEP_LR" "$deep_lr"; then
    echo "The global batch control must select the approved width-transfer rates 0.032/0.012" >&2
    exit 2
fi

width_selector=${G1_WIDTH_TRANSFER_SELECTOR:-$repo_root/experiments/g1_sasrec_item_ids_likes/analysis/select_width_transfer_500m.py}
if [[ ! -f "$width_selector" ]]; then
    echo "Missing RQ1 width-transfer selector: $width_selector" >&2
    exit 2
fi
if ! selector_output=$(python "$width_selector" --generated "$repo_root/generated"); then
    echo "RQ1 width-transfer selection preflight failed" >&2
    exit 2
fi

declare -A selected_proxy_runs
while IFS=$'\t' read -r width proxy_run selected_embedding_lr selected_deep_lr selected_batch extra; do
    [[ -n "$width" ]] || continue
    if [[ -n "${extra:-}" || ! "$width" =~ ^(16|256)$ || -z "$proxy_run" || \
          -z "$selected_embedding_lr" || -z "$selected_deep_lr" || \
          "$selected_batch" != "$final_batch_size" ]]; then
        echo "Invalid RQ1 width-transfer selector row for width ${width:-missing}" >&2
        exit 2
    fi
    if ! g1_same_number "$selected_embedding_lr" "$embedding_lr" || \
       ! g1_same_number "$selected_deep_lr" "$deep_lr"; then
        echo "RQ1 width $width does not use approved rates 0.032/0.012" >&2
        exit 2
    fi
    if [[ -n "${selected_proxy_runs[$width]+x}" ]]; then
        echo "Duplicate RQ1 width-transfer selector row: $width" >&2
        exit 2
    fi
    selected_proxy_runs[$width]=$proxy_run
done <<< "$selector_output"
for width in "${transfer_widths[@]}"; do
    if [[ -z "${selected_proxy_runs[$width]+x}" ]]; then
        echo "RQ1 width-transfer selector is missing width $width" >&2
        exit 2
    fi
done
if [[ "${#selected_proxy_runs[@]}" -ne 2 ]]; then
    echo "RQ1 width-transfer selector must return exactly widths 16 and 256" >&2
    exit 2
fi

declare -A manifest_sources
declare -A manifest_transformer_fields
declare -A manifest_experiment_fields
declare -A manifest_extras
while IFS='|' read -r axis treatment source transformer_fields experiment_fields extras alias; do
    if [[ "$axis" == dimension && "$treatment" =~ ^(16|256)$ ]]; then
        manifest_sources[$treatment]=$source
        manifest_transformer_fields[$treatment]=$transformer_fields
        manifest_experiment_fields[$treatment]=$experiment_fields
        manifest_extras[$treatment]=$extras
    fi
done < <(g1_manifest_rows)
for width in "${transfer_widths[@]}"; do
    if [[ -z "${manifest_sources[$width]+x}" || \
          "${manifest_transformer_fields[$width]}" != dim || \
          -n "${manifest_experiment_fields[$width]}" || \
          -z "${manifest_extras[$width]}" ]]; then
        echo "Missing compatible dimension/$width manifest treatment" >&2
        exit 2
    fi
done

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
for width in "${transfer_widths[@]}"; do
    run="rqfinal_dimension_${width}_e0p032_d0p012_b1280${provenance_suffix}"
    directory="$repo_root/generated/logs/g1_rqtune_${run}_500m"
    read -ra extra_args <<< "${manifest_extras[$width]}"
    verifier_args=(
        "G1_TUNE_RUN=${run}"
        "G1_TUNE_RUN_REVISION=${final_run_revision}"
        "G1_TUNE_EPOCHS=${final_epochs}"
        "G1_TUNE_SOURCE_VARIANT=${manifest_sources[$width]}"
        "G1_TUNE_TRANSFORMER_FIELDS=${manifest_transformer_fields[$width]}"
        "G1_TUNE_EXPERIMENT_FIELDS=${manifest_experiment_fields[$width]}"
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
    if ! enqueue "g1_rqtune_${run}_500m" \
        "${verifier_args[@]}"; then
        echo "Failed to enqueue RQ1 width-transfer confirmation for width $width" >&2
        exit 1
    fi
    enqueued_count=$((enqueued_count + 1))
done

echo "=== width-transfer confirmations: enqueued=${enqueued_count}, skipped=${skipped_count} ==="
g1_stop_artifact_verifier
drain || exit 1
