#!/usr/bin/env bash
set -u

launcher_dir=$(cd "$(dirname "$0")" && pwd)
source "$launcher_dir/manifest.sh"
source "$launcher_dir/../artifacts.sh"
source "$launcher_dir/../global_batch.sh"

selected_axes=${G1_FINAL_AXES:-$G1_DEFAULT_AXES}
selections=${G1_FINAL_SELECTIONS:-}
if [[ -z "$selections" ]]; then
    echo "G1_FINAL_SELECTIONS is required: treatment:embedding_lr:deep_lr" >&2
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
if [[ -n "${G1_FINAL_CONTROL_EPOCHS+x}${G1_FINAL_CONTROL_RUN_REVISION+x}" ]] && \
   [[ -z "${G1_FINAL_CONTROL_EPOCHS+x}" || \
      -z "${G1_FINAL_CONTROL_RUN_REVISION+x}" ]]; then
    echo "G1_FINAL_CONTROL_EPOCHS and G1_FINAL_CONTROL_RUN_REVISION must be set together" >&2
    exit 2
fi
control_epochs=${G1_FINAL_CONTROL_EPOCHS:-20}
control_run_revision=${G1_FINAL_CONTROL_RUN_REVISION:-2}
if [[ ! "$control_epochs" =~ ^[1-9][0-9]*$ || "$control_epochs" -lt 20 ]]; then
    echo "G1_FINAL_CONTROL_EPOCHS must be a canonical integer of at least 20" >&2
    exit 2
fi
if [[ ! "$control_run_revision" =~ ^[1-9][0-9]*$ ]]; then
    echo "G1_FINAL_CONTROL_RUN_REVISION must be a canonical positive integer" >&2
    exit 2
fi
final_batch_size=${G1_GLOBAL_BATCH_SIZE:-}
if [[ ! "$final_batch_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "G1_GLOBAL_BATCH_SIZE must be a canonical positive integer" >&2
    exit 2
fi
final_run_tag=${G1_FINAL_RUN_TAG:-}
if [[ -n "$final_run_tag" && ! "$final_run_tag" =~ ^[a-z0-9][a-z0-9_-]*$ ]]; then
    echo "G1_FINAL_RUN_TAG must contain lowercase letters, digits, underscores, or hyphens" >&2
    exit 2
fi
declare -A selected_axis_names
for axis in $selected_axes; do
    if [[ " $G1_MANIFEST_AXES " != *" $axis "* ]]; then
        echo "Unknown G1_FINAL_AXES value: $axis" >&2
        exit 2
    fi
    if [[ -n "${selected_axis_names[$axis]+x}" ]]; then
        echo "Duplicate G1_FINAL_AXES value: $axis" >&2
        exit 2
    fi
    selected_axis_names[$axis]=1
done
if [[ -n "${selected_axis_names[sequence]+x}" && ${#selected_axis_names[@]} -ne 1 ]]; then
    echo "Launch sequence as its own queue stage so other axes reuse preprocessing" >&2
    exit 2
fi
if [[ -n "${selected_axis_names[time]+x}" && ${#selected_axis_names[@]} -ne 1 ]]; then
    echo "Launch time as its own stage so it reuses its preprocessing group" >&2
    exit 2
fi

axis_selected() {
    [[ -n "${selected_axis_names[$1]+x}" ]]
}

require_positive_float() {
    python - "$1" "$2" <<'PY'
import math
import sys

name, raw = sys.argv[1:]
try:
    value = float(raw)
except ValueError:
    raise SystemExit(f"{name} must contain positive finite numbers: {raw}")
if not math.isfinite(value) or value <= 0:
    raise SystemExit(f"{name} must contain positive finite numbers: {raw}")
PY
}

slug() {
    local value=${1,,}
    value=${value//./p}
    echo "${value//-/m}"
}

declare -A manifest_sources
declare -A manifest_transformer_fields
declare -A manifest_experiment_fields
declare -A manifest_extras
declare -A manifest_aliases
manifest_order=()
selected_has_alias=0
while IFS='|' read -r axis treatment source transformer_fields experiment_fields extras alias; do
    key="$axis/$treatment"
    if [[ -n "${manifest_sources[$key]+x}" ]]; then
        echo "Duplicate manifest treatment: $key" >&2
        exit 2
    fi
    manifest_sources[$key]=$source
    manifest_transformer_fields[$key]=$transformer_fields
    manifest_experiment_fields[$key]=$experiment_fields
    manifest_extras[$key]=$extras
    manifest_aliases[$key]=$alias
    manifest_order+=("$key")
done < <(g1_manifest_rows)

selected_treatments=${G1_FINAL_TREATMENTS:-}
declare -A selected_treatment_keys
for key in $selected_treatments; do
    if [[ -z "${manifest_sources[$key]+x}" ]]; then
        echo "Unknown G1_FINAL_TREATMENTS value: $key" >&2
        exit 2
    fi
    if ! axis_selected "${key%%/*}"; then
        echo "Treatment $key is outside G1_FINAL_AXES" >&2
        exit 2
    fi
    if [[ -n "${selected_treatment_keys[$key]+x}" ]]; then
        echo "Duplicate G1_FINAL_TREATMENTS value: $key" >&2
        exit 2
    fi
    selected_treatment_keys[$key]=1
done

treatment_selected() {
    [[ -z "$selected_treatments" || -n "${selected_treatment_keys[$1]+x}" ]]
}

exploratory_treatments=${G1_FINAL_EXPLORATORY_TREATMENTS:-}
declare -A exploratory_treatment_keys
for key in $exploratory_treatments; do
    if [[ -z "${manifest_sources[$key]+x}" ]]; then
        echo "Unknown G1_FINAL_EXPLORATORY_TREATMENTS value: $key" >&2
        exit 2
    fi
    if ! treatment_selected "$key" || ! axis_selected "${key%%/*}"; then
        echo "Exploratory treatment $key is outside the selected treatments" >&2
        exit 2
    fi
    exploratory_treatment_keys[$key]=1
done
if [[ -n "$exploratory_treatments" && -z "${G1_FINAL_RUN_TAG:-}" ]]; then
    echo "Set G1_FINAL_RUN_TAG so an exploratory confirmation is named apart" >&2
    exit 2
fi

if axis_selected sequence && treatment_selected sequence/512; then
    seq512_accumulation_steps=${G1_FINAL_SEQ512_GRADIENT_ACCUMULATION_STEPS:-2}
    if [[ ! "$seq512_accumulation_steps" =~ ^[1-9][0-9]*$ ]]; then
        echo "G1_FINAL_SEQ512_GRADIENT_ACCUMULATION_STEPS must be a canonical positive integer" >&2
        exit 2
    fi
    seq512_physical_batch_size=${G1_FINAL_SEQ512_PHYSICAL_BATCH_SIZE:-}
    if [[ -z "$seq512_physical_batch_size" ]]; then
        if (( final_batch_size % seq512_accumulation_steps != 0 )); then
            echo "G1_GLOBAL_BATCH_SIZE must be divisible by sequence-512 accumulation steps" >&2
            exit 2
        fi
        seq512_physical_batch_size=$((final_batch_size / seq512_accumulation_steps))
    fi
    if [[ ! "$seq512_physical_batch_size" =~ ^[1-9][0-9]*$ ]] ||
       (( seq512_physical_batch_size * seq512_accumulation_steps != final_batch_size )); then
        echo "sequence-512 physical batch times accumulation must equal G1_GLOBAL_BATCH_SIZE" >&2
        exit 2
    fi
fi

for key in "${manifest_order[@]}"; do
    axis=${key%%/*}
    if axis_selected "$axis" && treatment_selected "$key" && \
       [[ -n "${manifest_aliases[$key]}" ]]; then
        selected_has_alias=1
    fi
done

declare -A selected_embedding_lrs
declare -A selected_deep_lrs
for spec in $selections; do
    IFS=: read -r key embedding_lr deep_lr batch_size extra <<< "$spec"
    if [[ -n "${extra:-}" || -z "$key" || -z "$embedding_lr" || -z "$deep_lr" || \
          -n "${batch_size:-}" ]]; then
        echo "Invalid G1_FINAL_SELECTIONS entry: $spec" >&2
        exit 2
    fi
    if [[ -z "${manifest_sources[$key]+x}" ]]; then
        echo "Unknown treatment in G1_FINAL_SELECTIONS: $key" >&2
        exit 2
    fi
    if { ! axis_selected "${key%%/*}" || ! treatment_selected "$key"; } && \
       [[ "$key" != control/control ]]; then
        echo "Selection $key is outside G1_FINAL_AXES" >&2
        exit 2
    fi
    if [[ -n "${manifest_aliases[$key]}" ]]; then
        echo "Alias treatment $key reuses ${manifest_aliases[$key]} and cannot have a selection" >&2
        exit 2
    fi
    if [[ -n "${selected_embedding_lrs[$key]+x}" ]]; then
        echo "Duplicate G1_FINAL_SELECTIONS treatment: $key" >&2
        exit 2
    fi
    require_positive_float G1_FINAL_SELECTIONS "$embedding_lr" || exit 2
    require_positive_float G1_FINAL_SELECTIONS "$deep_lr" || exit 2
    selected_embedding_lrs[$key]=$embedding_lr
    selected_deep_lrs[$key]=$deep_lr
done

for key in "${manifest_order[@]}"; do
    axis=${key%%/*}
    axis_selected "$axis" || continue
    treatment_selected "$key" || continue
    [[ -n "${manifest_aliases[$key]}" ]] && continue
    if [[ -z "${selected_embedding_lrs[$key]+x}" ]]; then
        echo "Missing G1_FINAL_SELECTIONS treatment: $key" >&2
        exit 2
    fi
done

repo_root=$(cd "$launcher_dir/../../../.." && pwd)
batch_control_key=control/control
if [[ -z "${selected_embedding_lrs[$batch_control_key]+x}" ]]; then
    echo "G1_FINAL_SELECTIONS must include batch-control provenance: $batch_control_key" >&2
    exit 2
fi
g1_require_global_batch_selection "$repo_root" "$final_batch_size" || exit 2
if ! g1_same_number "${selected_embedding_lrs[$batch_control_key]}" \
        "$G1_VERIFIED_CONTROL_EMBEDDING_LR" || \
   ! g1_same_number "${selected_deep_lrs[$batch_control_key]}" \
        "$G1_VERIFIED_CONTROL_DEEP_LR"; then
    echo "control/control rates must match G1_GLOBAL_BATCH_SELECTION" >&2
    exit 2
fi
read -ra batch_control_extra_args <<< "${manifest_extras[$batch_control_key]}"
batch_control_stem=$(g1_run_stem "$batch_control_key")
compatible_proxy_control=0
for directory in \
    "$repo_root"/generated/logs/g1_rqtune_${batch_control_stem}_*_50m; do
    [[ -d "$directory" ]] || continue
    batch_control_run=$(basename "$directory")
    batch_control_run=${batch_control_run#g1_rqtune_}
    batch_control_run=${batch_control_run%_50m}
    if g1_verify_tuning_artifact "$directory" 50m \
        "G1_TUNE_RUN=${batch_control_run}" \
        "G1_TUNE_RUN_REVISION=${control_run_revision}" \
        "G1_TUNE_EPOCHS=${control_epochs}" \
        "G1_TUNE_SOURCE_VARIANT=${manifest_sources[$batch_control_key]}" \
        "G1_TUNE_TRANSFORMER_FIELDS=${manifest_transformer_fields[$batch_control_key]}" \
        "G1_TUNE_EXPERIMENT_FIELDS=${manifest_experiment_fields[$batch_control_key]}" \
        "G1_TUNE_EMBEDDING_LR=${selected_embedding_lrs[$batch_control_key]}" \
        "G1_TUNE_DEEP_LR=${selected_deep_lrs[$batch_control_key]}" \
        "G1_TUNE_BATCH_SIZE=${final_batch_size}" \
        "${batch_control_extra_args[@]}"; then
        compatible_proxy_control=1
        break
    fi
done
if [[ "$compatible_proxy_control" -ne 1 ]]; then
    echo "Missing compatible selected 50M batch-control artifact" >&2
    exit 2
fi

architecture_selector=${G1_ARCHITECTURE_SELECTOR:-$repo_root/experiments/g1_sasrec_item_ids_likes/analysis/select_architecture_500m.py}
if [[ ! -f "$architecture_selector" ]]; then
    echo "Missing native-50M architecture selector: $architecture_selector" >&2
    exit 2
fi
declare -A preflight_specs
declare -A preflight_exploratory
preflight_specs[sequence_128]="sequence_128:${selected_embedding_lrs[$batch_control_key]}:${selected_deep_lrs[$batch_control_key]}"
for key in "${manifest_order[@]}"; do
    axis=${key%%/*}
    axis_selected "$axis" || continue
    treatment_selected "$key" || continue
    base=$(g1_run_stem "$key")
    selection_key=$key
    if [[ -n "${manifest_aliases[$key]}" ]]; then
        selection_key=${manifest_aliases[$key]}
    fi
    spec="$base:${selected_embedding_lrs[$selection_key]}:${selected_deep_lrs[$selection_key]}"
    if [[ -n "${preflight_specs[$base]+x}" && "${preflight_specs[$base]}" != "$spec" ]]; then
        echo "Conflicting native-50M preflight selection for $base" >&2
        exit 2
    fi
    preflight_specs[$base]=$spec
    if [[ -n "${exploratory_treatment_keys[$key]+x}" ]]; then
        preflight_exploratory[$base]=1
    fi
done
selector_args=(--generated "$repo_root/generated")
for base in "${!preflight_specs[@]}"; do
    if [[ -n "${preflight_exploratory[$base]+x}" ]]; then
        selector_args+=(--exploratory-selection "${preflight_specs[$base]}")
    else
        selector_args+=(--selection "${preflight_specs[$base]}")
    fi
done
if ! preflight_output=$(python "$architecture_selector" "${selector_args[@]}"); then
    echo "Native-50M final-selection preflight failed" >&2
    exit 2
fi
while IFS= read -r verified_selection; do
    [[ -n "$verified_selection" ]] || continue
    echo "=== verified native-50M winner: $verified_selection ==="
done <<< "$preflight_output"

if [[ "$selected_has_alias" -eq 1 ]] && ! axis_selected control; then
    if [[ -z "${selected_embedding_lrs[control/control]+x}" ]]; then
        echo "G1_FINAL_SELECTIONS must include control/control provenance" >&2
        exit 2
    fi
    control_embedding_lr=${selected_embedding_lrs[control/control]}
    control_deep_lr=${selected_deep_lrs[control/control]}
    compatible_control=0
    for directory in "$repo_root"/generated/logs/g1_rqtune_rqfinal_architecture_control_*_500m; do
        [[ -d "$directory" ]] || continue
        if g1_verify_control_artifact "$directory" 500m "$final_batch_size" \
            "$control_embedding_lr" "$control_deep_lr" \
            "$final_epochs" "$final_run_revision"; then
            compatible_control=1
            break
        fi
    done
    if [[ "$compatible_control" -ne 1 ]]; then
        echo "Missing compatible final control artifact for aliased treatment" >&2
        exit 2
    fi
fi

cd "$repo_root"
export WANDB_MODE=${WANDB_MODE:-offline}
export G1_DATASET_SIZE=500m
export TRAINING_QUEUE_MONITOR_LIGHT_GPUS=${TRAINING_QUEUE_MONITOR_LIGHT_GPUS:-1}
if axis_selected sequence; then
    unset TRAINING_QUEUE_DATA_GROUP
else
    export TRAINING_QUEUE_DATA_GROUP=g1-rq-architecture-500m-seq128
fi
unset G1_MAX_USERS G1_VAL_BATCH_SIZE G1_TRAIN_BATCH_SIZE G1_SEED
unset G1_TUNE_BATCH_SIZE G1_TUNE_NUM_NEGATIVES G1_TUNE_NUM_WORKERS
unset G1_TUNE_GRADIENT_ACCUMULATION_STEPS
unset G1_TUNE_LOGQ_ALPHA
unset G1_TUNE_LOGQ_CORRECTION G1_TUNE_CORRECT_POSITIVE_LOGQ
unset G1_TUNE_MASK_FALSE_NEGATIVES G1_TUNE_EXCLUDE_OWN_GROUP_NEGATIVES
unset G1_TUNE_RANDOM_FRACTION G1_TUNE_FFN_DIM
unset G1_TUNE_TRANSFORMER_FIELDS G1_TUNE_EXPERIMENT_FIELDS
unset G1_TUNE_EMBEDDING_LR G1_TUNE_DEEP_LR G1_TUNE_SOURCE_VARIANT G1_TUNE_RUN

TRAINING_QUEUE_SCRIPT=experiments/g1_sasrec_item_ids_likes/configs/rq_tuning_variant.py
source "${G1_TRAINING_QUEUE_LIBRARY:-utils/training_queue/queue.sh}" || exit 1

enqueued_count=0
skipped_count=0
while IFS='|' read -r axis treatment source transformer_fields experiment_fields extras alias; do
    axis_selected "$axis" || continue
    key="$axis/$treatment"
    treatment_selected "$key" || continue
    [[ -n "$alias" ]] && continue
    rate_suffix="e$(slug "${selected_embedding_lrs[$key]}")"
    rate_suffix+="_d$(slug "${selected_deep_lrs[$key]}")"
    physical_batch_size=$final_batch_size
    gradient_accumulation_steps=1
    accumulation_suffix=
    if [[ "$key" == sequence/512 ]]; then
        physical_batch_size=$seq512_physical_batch_size
        gradient_accumulation_steps=$seq512_accumulation_steps
        accumulation_suffix="_pb${physical_batch_size}_ga${gradient_accumulation_steps}"
    fi
    run="rqfinal_$(g1_run_stem "$key")_${rate_suffix}_b${final_batch_size}"
    run+="${accumulation_suffix}${final_run_tag:+_${final_run_tag}}"
    run+="${final_provenance_suffix}"
    read -ra extra_args <<< "$extras"
    directory="$repo_root/generated/logs/g1_rqtune_${run}_500m"
    verifier_args=(
        "G1_TUNE_RUN=${run}"
        "G1_TUNE_RUN_REVISION=${final_run_revision}"
        "G1_TUNE_EPOCHS=${final_epochs}"
        "G1_TUNE_SOURCE_VARIANT=${source}"
        "G1_TUNE_TRANSFORMER_FIELDS=${transformer_fields}"
        "G1_TUNE_EXPERIMENT_FIELDS=${experiment_fields}"
        "G1_TUNE_EMBEDDING_LR=${selected_embedding_lrs[$key]}"
        "G1_TUNE_DEEP_LR=${selected_deep_lrs[$key]}"
        "G1_TUNE_BATCH_SIZE=${physical_batch_size}"
        "G1_TUNE_GRADIENT_ACCUMULATION_STEPS=${gradient_accumulation_steps}"
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
    enqueue "g1_rqtune_${run}_500m" \
        "G1_TUNE_RUN=${run}" \
        "G1_TUNE_RUN_REVISION=${final_run_revision}" \
        "G1_TUNE_EPOCHS=${final_epochs}" \
        "G1_TUNE_SOURCE_VARIANT=${source}" \
        "G1_TUNE_TRANSFORMER_FIELDS=${transformer_fields}" \
        "G1_TUNE_EXPERIMENT_FIELDS=${experiment_fields}" \
        "G1_TUNE_EMBEDDING_LR=${selected_embedding_lrs[$key]}" \
        "G1_TUNE_DEEP_LR=${selected_deep_lrs[$key]}" \
        "G1_TUNE_BATCH_SIZE=${physical_batch_size}" \
        "G1_TUNE_GRADIENT_ACCUMULATION_STEPS=${gradient_accumulation_steps}" \
        "${extra_args[@]}"
    enqueued_count=$((enqueued_count + 1))
done < <(g1_manifest_rows)

echo "=== final manifest: enqueued=${enqueued_count}, skipped=${skipped_count} ==="
g1_stop_artifact_verifier
drain || exit 1
