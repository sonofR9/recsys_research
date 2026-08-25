#!/usr/bin/env bash
set -u

launcher_dir=$(cd "$(dirname "$0")" && pwd)
source "$launcher_dir/manifest.sh"
source "$launcher_dir/../artifacts.sh"
source "$launcher_dir/../global_batch.sh"

selected_axes=${G1_TUNE_AXES:-$G1_DEFAULT_AXES}
stage=${G1_TUNE_STAGE:-lr}
if [[ "$stage" != lr && "$stage" != batch && "$stage" != batch_lr ]]; then
    echo "G1_TUNE_STAGE must be lr, batch, or batch_lr" >&2
    exit 2
fi
explicit_provenance=0
if [[ -n "${G1_TUNE_EPOCHS+x}${G1_TUNE_RUN_REVISION+x}" ]]; then
    if [[ -z "${G1_TUNE_EPOCHS+x}" || -z "${G1_TUNE_RUN_REVISION+x}" ]]; then
        echo "G1_TUNE_EPOCHS and G1_TUNE_RUN_REVISION must be set together" >&2
        exit 2
    fi
    explicit_provenance=1
fi
epochs=${G1_TUNE_EPOCHS:-20}
run_revision=${G1_TUNE_RUN_REVISION:-2}
if [[ ! "$epochs" =~ ^[1-9][0-9]*$ || ! "$run_revision" =~ ^[1-9][0-9]*$ ]]; then
    echo "G1_TUNE_EPOCHS and G1_TUNE_RUN_REVISION must be canonical positive integers" >&2
    exit 2
fi
if [[ "$epochs" -lt 20 ]]; then
    echo "G1_TUNE_EPOCHS must be at least the 20-epoch safety cap" >&2
    exit 2
fi
if [[ -n "${G1_TUNE_CONTROL_EPOCHS+x}${G1_TUNE_CONTROL_RUN_REVISION+x}" ]] && \
   [[ -z "${G1_TUNE_CONTROL_EPOCHS+x}" || \
      -z "${G1_TUNE_CONTROL_RUN_REVISION+x}" ]]; then
    echo "G1_TUNE_CONTROL_EPOCHS and G1_TUNE_CONTROL_RUN_REVISION must be set together" >&2
    exit 2
fi
control_epochs=${G1_TUNE_CONTROL_EPOCHS:-20}
control_run_revision=${G1_TUNE_CONTROL_RUN_REVISION:-2}
if [[ ! "$control_epochs" =~ ^[1-9][0-9]*$ || "$control_epochs" -lt 20 ]]; then
    echo "G1_TUNE_CONTROL_EPOCHS must be a canonical integer of at least 20" >&2
    exit 2
fi
if [[ ! "$control_run_revision" =~ ^[1-9][0-9]*$ ]]; then
    echo "G1_TUNE_CONTROL_RUN_REVISION must be a canonical positive integer" >&2
    exit 2
fi
training_semantics_revision=2
provenance_suffix="_ts${training_semantics_revision}_r2"
if [[ "$explicit_provenance" -eq 1 ]]; then
    provenance_suffix="_ts${training_semantics_revision}_r${run_revision}"
    [[ "$epochs" -ne 20 ]] && provenance_suffix="_cap${epochs}${provenance_suffix}"
fi
fixed_batch_size=${G1_GLOBAL_BATCH_SIZE:-1280}
if [[ ! "$fixed_batch_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "G1_GLOBAL_BATCH_SIZE must be a canonical positive integer" >&2
    exit 2
fi
if [[ "$stage" != lr && -n "${G1_GLOBAL_BATCH_SIZE+x}" ]]; then
    echo "G1_GLOBAL_BATCH_SIZE is set only after control batch selection" >&2
    exit 2
fi

require_unique_words() {
    local name=$1
    local values=$2
    local value
    local seen=" "
    for value in $values; do
        if [[ "$seen" == *" $value "* ]]; then
            echo "$name contains duplicate value: $value" >&2
            exit 2
        fi
        seen+="$value "
    done
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
    local value=${1//./p}
    echo "${value//-/m}"
}

lr_slug() {
    case $1 in
        0.003) echo 3 ;;
        0.006) echo 6 ;;
        0.008) echo 8 ;;
        0.012) echo 12 ;;
        0.016) echo 16 ;;
        0.024) echo 24 ;;
        0.032) echo 32 ;;
        0.064) echo 64 ;;
        *) slug "$1" ;;
    esac
}

same_rates() {
    python - "$1" "$2" "$3" "$4" <<'PY'
import sys

raise SystemExit(
    0
    if float(sys.argv[1]) == float(sys.argv[3])
    and float(sys.argv[2]) == float(sys.argv[4])
    else 1
)
PY
}

require_unique_words G1_TUNE_AXES "$selected_axes"
for axis in $selected_axes; do
    if [[ " $G1_MANIFEST_AXES " != *" $axis "* ]]; then
        echo "Unknown G1_TUNE_AXES value: $axis" >&2
        exit 2
    fi
done
if [[ " $selected_axes " == *" sequence " && "$selected_axes" != sequence ]]; then
    echo "Launch sequence as its own queue stage so other axes reuse preprocessing" >&2
    exit 2
fi
if [[ " $selected_axes " == *" time " && "$selected_axes" != time ]]; then
    echo "Launch time as its own stage so it reuses its preprocessing group" >&2
    exit 2
fi

run_tag=${G1_TUNE_RUN_TAG:-}
if [[ -n "$run_tag" && ! "$run_tag" =~ ^[a-z0-9_]+$ ]]; then
    echo "G1_TUNE_RUN_TAG must contain only lowercase letters, digits, and _" >&2
    exit 2
fi
if [[ "$run_tag" == mup || "$run_tag" == batch || "$run_tag" == batchlr || \
      "$run_tag" =~ _r[1-9][0-9]*$ || \
      "$run_tag" =~ (^|_)cap[1-9][0-9]*$ ]]; then
    echo "G1_TUNE_RUN_TAG cannot reuse reserved tag $run_tag" >&2
    exit 2
fi

lr_points=${G1_TUNE_LR_POINTS:-}
if [[ -n "$lr_points" && "$stage" != lr && "$stage" != batch_lr ]]; then
    echo "G1_TUNE_LR_POINTS is supported only for LR or batch-LR tuning" >&2
    exit 2
fi

if [[ "$stage" == lr || "$stage" == batch_lr ]]; then
    embedding_lrs=${G1_TUNE_EMBEDDING_LRS:-0.008 0.016 0.032}
    deep_lrs=${G1_TUNE_DEEP_LRS:-0.003 0.006 0.012}
    require_unique_words G1_TUNE_EMBEDDING_LRS "$embedding_lrs"
    require_unique_words G1_TUNE_DEEP_LRS "$deep_lrs"
    for embedding_lr in $embedding_lrs; do
        require_positive_float G1_TUNE_EMBEDDING_LRS "$embedding_lr" || exit 2
    done
    for deep_lr in $deep_lrs; do
        require_positive_float G1_TUNE_DEEP_LRS "$deep_lr" || exit 2
    done
    if [[ -n "$lr_points" && \
          -n "${G1_TUNE_EMBEDDING_LRS+x}${G1_TUNE_DEEP_LRS+x}" ]]; then
        echo "G1_TUNE_LR_POINTS cannot be combined with LR-grid overrides" >&2
        exit 2
    fi
    if [[ -n "${G1_TUNE_EMBEDDING_LRS+x}${G1_TUNE_DEEP_LRS+x}$lr_points" && \
          -z "$run_tag" ]]; then
        echo "Set a unique G1_TUNE_RUN_TAG for an LR-grid extension" >&2
        exit 2
    fi
fi

if [[ "$stage" == batch ]]; then
    batch_sizes=${G1_TUNE_BATCH_SIZES:-1024 1536 2048}
    require_unique_words G1_TUNE_BATCH_SIZES "$batch_sizes"
    for batch_size in $batch_sizes; do
        if [[ ! "$batch_size" =~ ^[1-9][0-9]*$ || "$batch_size" == 1280 ]]; then
            echo "Batch screens must contain canonical positive sizes other than the LR-grid batch 1280" >&2
            exit 2
        fi
    done
    if [[ -n "${G1_TUNE_BATCH_SIZES+x}" && -z "$run_tag" ]]; then
        echo "Set a unique G1_TUNE_RUN_TAG for a batch-grid extension" >&2
        exit 2
    fi
fi

axis_selected() {
    [[ " $selected_axes " == *" $1 "* ]]
}

declare -A manifest_sources
declare -A manifest_transformer_fields
declare -A manifest_experiment_fields
declare -A manifest_extras
declare -A manifest_aliases
manifest_order=()
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

selected_treatments=${G1_TUNE_TREATMENTS:-}
require_unique_words G1_TUNE_TREATMENTS "$selected_treatments"
declare -A selected_treatment_keys
for key in $selected_treatments; do
    if [[ -z "${manifest_sources[$key]+x}" ]]; then
        echo "Unknown G1_TUNE_TREATMENTS value: $key" >&2
        exit 2
    fi
    if ! axis_selected "${key%%/*}"; then
        echo "Treatment $key is outside G1_TUNE_AXES" >&2
        exit 2
    fi
    selected_treatment_keys[$key]=1
done
if [[ "$stage" != lr ]]; then
    read -ra control_stage_treatments <<< "$selected_treatments"
    if [[ "${#control_stage_treatments[@]}" -ne 1 ]]; then
        echo "$stage tuning requires exactly one control treatment" >&2
        exit 2
    fi
    control_stage_key=${control_stage_treatments[0]}
    if [[ "$control_stage_key" != control/control ]]; then
        echo "$stage tuning requires the global control/control treatment" >&2
        exit 2
    fi
fi

treatment_selected() {
    [[ -z "$selected_treatments" || -n "${selected_treatment_keys[$1]+x}" ]]
}

declare -a sparse_keys
declare -a sparse_embedding_lrs
declare -a sparse_deep_lrs
declare -a sparse_batch_sizes
declare -A sparse_specs
for spec in $lr_points; do
    IFS=: read -r key embedding_lr deep_lr batch_size extra <<< "$spec"
    if [[ -n "${extra:-}" || -z "$key" || -z "$embedding_lr" || -z "$deep_lr" ||
          ( "$stage" == lr && -n "${batch_size:-}" ) ||
          ( "$stage" == batch_lr && ( -z "${batch_size:-}" || ! "$batch_size" =~ ^[1-9][0-9]*$ ) ) ]]; then
        echo "Invalid G1_TUNE_LR_POINTS entry: $spec" >&2
        exit 2
    fi
    if [[ -z "${manifest_sources[$key]+x}" ]]; then
        echo "Unknown treatment in G1_TUNE_LR_POINTS: $key" >&2
        exit 2
    fi
    if ! axis_selected "${key%%/*}" || ! treatment_selected "$key"; then
        echo "Sparse LR point $key is outside the selected treatments" >&2
        exit 2
    fi
    if [[ -n "${manifest_aliases[$key]}" ]]; then
        echo "Alias treatment $key reuses ${manifest_aliases[$key]} and cannot have sparse LR points" >&2
        exit 2
    fi
    require_positive_float G1_TUNE_LR_POINTS "$embedding_lr" || exit 2
    require_positive_float G1_TUNE_LR_POINTS "$deep_lr" || exit 2
    sparse_key="$key:$embedding_lr:$deep_lr:${batch_size:-}"
    if [[ -n "${sparse_specs[$sparse_key]+x}" ]]; then
        echo "Duplicate G1_TUNE_LR_POINTS entry: $spec" >&2
        exit 2
    fi
    sparse_specs[$sparse_key]=1
    sparse_keys+=("$key")
    sparse_embedding_lrs+=("$embedding_lr")
    sparse_deep_lrs+=("$deep_lr")
    sparse_batch_sizes+=("${batch_size:-}")
done

declare -A selected_embedding_lrs
declare -A selected_deep_lrs
declare -A selected_batch_sizes
if [[ "$stage" != lr && -z "$lr_points" ]]; then
    selections=${G1_TUNE_SELECTIONS:-}
    if [[ -z "$selections" ]]; then
        echo "G1_TUNE_SELECTIONS is required for $stage tuning" >&2
        exit 2
    fi
    for spec in $selections; do
        IFS=: read -r key embedding_lr deep_lr batch_size extra <<< "$spec"
        if [[ -n "${extra:-}" || -z "$key" || -z "$embedding_lr" || -z "$deep_lr" ]]; then
            echo "Invalid G1_TUNE_SELECTIONS entry: $spec" >&2
            exit 2
        fi
        if [[ -z "${manifest_sources[$key]+x}" ]]; then
            echo "Unknown treatment in G1_TUNE_SELECTIONS: $key" >&2
            exit 2
        fi
        if [[ "$key" != "$control_stage_key" ]]; then
            echo "$stage tuning accepts only the selected control $control_stage_key" >&2
            exit 2
        fi
        if [[ -n "${manifest_aliases[$key]}" ]]; then
            echo "Alias treatment $key reuses ${manifest_aliases[$key]} and cannot have a selection" >&2
            exit 2
        fi
        if [[ -n "${selected_embedding_lrs[$key]+x}" ]]; then
            echo "Duplicate G1_TUNE_SELECTIONS treatment: $key" >&2
            exit 2
        fi
        if [[ "$stage" == batch && -n "${batch_size:-}" ]]; then
            echo "Batch-stage selections use treatment:embedding_lr:deep_lr" >&2
            exit 2
        fi
        if [[ "$stage" == batch_lr && ( -z "${batch_size:-}" || ! "$batch_size" =~ ^[1-9][0-9]*$ ) ]]; then
            echo "Batch-LR selections require a canonical positive batch size" >&2
            exit 2
        fi
        require_positive_float G1_TUNE_SELECTIONS "$embedding_lr" || exit 2
        require_positive_float G1_TUNE_SELECTIONS "$deep_lr" || exit 2
        selected_embedding_lrs[$key]=$embedding_lr
        selected_deep_lrs[$key]=$deep_lr
        selected_batch_sizes[$key]=${batch_size:-}
    done

    if [[ -z "${selected_embedding_lrs[$control_stage_key]+x}" ]]; then
        echo "Missing G1_TUNE_SELECTIONS control: $control_stage_key" >&2
        exit 2
    fi
fi

repo_root=$(cd "$launcher_dir/../../../.." && pwd)
if [[ "$stage" == lr ]]; then
    read -ra lr_treatments <<< "$selected_treatments"
    expected_batch_control=control/control
    control_only=0
    if [[ "${#lr_treatments[@]}" -eq 1 && \
          "${lr_treatments[0]}" == "$expected_batch_control" ]]; then
        control_only=1
    fi
    if [[ "$control_only" -eq 1 && "$fixed_batch_size" != 1280 ]]; then
        echo "Retune the selected global control with G1_TUNE_STAGE=batch_lr" >&2
        exit 2
    fi
    if [[ "$control_only" -eq 0 ]]; then
        batch_control=${G1_TUNE_BATCH_CONTROL:-}
        IFS=: read -r batch_control_key control_embedding_lr control_deep_lr \
            control_batch_size extra <<< "$batch_control"
        if [[ -n "${extra:-}" || \
              "$batch_control_key" != "$expected_batch_control" || \
              -z "${control_embedding_lr:-}" || -z "${control_deep_lr:-}" || \
              ! "${control_batch_size:-}" =~ ^[1-9][0-9]*$ ]]; then
            echo "G1_TUNE_BATCH_CONTROL must be control_key:embedding_lr:deep_lr:batch_size" >&2
            exit 2
        fi
        require_positive_float G1_TUNE_BATCH_CONTROL "$control_embedding_lr" || exit 2
        require_positive_float G1_TUNE_BATCH_CONTROL "$control_deep_lr" || exit 2
        if [[ "$control_batch_size" != "$fixed_batch_size" ]]; then
            echo "G1_GLOBAL_BATCH_SIZE must match G1_TUNE_BATCH_CONTROL" >&2
            exit 2
        fi
        compatible_control=0
        read -ra control_extra_args <<< "${manifest_extras[$batch_control_key]}"
        control_stem=$(g1_run_stem "$batch_control_key")
        for directory in \
            "$repo_root"/generated/logs/g1_rqtune_${control_stem}_*_50m; do
            [[ -d "$directory" ]] || continue
            control_run=$(basename "$directory")
            control_run=${control_run#g1_rqtune_}
            control_run=${control_run%_50m}
            if g1_verify_tuning_artifact "$directory" 50m \
                "G1_TUNE_RUN=${control_run}" \
                "G1_TUNE_RUN_REVISION=${control_run_revision}" \
                "G1_TUNE_EPOCHS=${control_epochs}" \
                "G1_TUNE_SOURCE_VARIANT=${manifest_sources[$batch_control_key]}" \
                "G1_TUNE_TRANSFORMER_FIELDS=${manifest_transformer_fields[$batch_control_key]}" \
                "G1_TUNE_EXPERIMENT_FIELDS=${manifest_experiment_fields[$batch_control_key]}" \
                "G1_TUNE_EMBEDDING_LR=${control_embedding_lr}" \
                "G1_TUNE_DEEP_LR=${control_deep_lr}" \
                "G1_TUNE_BATCH_SIZE=${control_batch_size}" \
                "G1_TUNE_GRADIENT_ACCUMULATION_STEPS=1" \
                "${control_extra_args[@]}"; then
                compatible_control=1
                break
            fi
        done
        if [[ "$compatible_control" -ne 1 ]]; then
            echo "Missing compatible selected batch-control artifact" >&2
            exit 2
        fi
        g1_require_global_batch_selection "$repo_root" "$fixed_batch_size" || exit 2
        if ! g1_same_number "$control_embedding_lr" \
                "$G1_VERIFIED_CONTROL_EMBEDDING_LR" || \
           ! g1_same_number "$control_deep_lr" "$G1_VERIFIED_CONTROL_DEEP_LR"; then
            echo "G1_TUNE_BATCH_CONTROL rates must match G1_GLOBAL_BATCH_SELECTION" >&2
            exit 2
        fi
    fi
fi

cd "$repo_root"
export WANDB_MODE=${WANDB_MODE:-offline}
export G1_DATASET_SIZE=50m
export TRAINING_QUEUE_MONITOR_LIGHT_GPUS=${TRAINING_QUEUE_MONITOR_LIGHT_GPUS:-1}
export TRAINING_QUEUE_DATA_GROUP=g1-rq-architecture-50m-seq128
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

set_batch_contract() {
    local key=$1
    local effective_batch_size=$2
    tuning_physical_batch_size=$effective_batch_size
    tuning_accumulation_steps=1
    tuning_batch_suffix=
    if [[ "$effective_batch_size" != 1280 ]]; then
        tuning_batch_suffix="_b${effective_batch_size}"
    fi
    if [[ "$key" == sequence/512 ]]; then
        if [[ "$effective_batch_size" != 1280 ]]; then
            echo "sequence-512 tuning requires effective batch 1280" >&2
            return 2
        fi
        tuning_physical_batch_size=640
        tuning_accumulation_steps=2
        tuning_batch_suffix="_b1280_pb640_ga2"
    fi
}

enqueue_treatment() {
    local key=$1
    local suffix=$2
    local embedding_lr=$3
    local deep_lr=$4
    shift 4
    local data_group=$TRAINING_QUEUE_DATA_GROUP
    if [[ "$key" == sequence/* ]]; then
        data_group="g1-rq-architecture-50m-seq${key#sequence/}"
    fi
    local TRAINING_QUEUE_DATA_GROUP=$data_group
    local run="$(g1_run_stem "$key")_${suffix}"
    local -a extra_args=()
    if [[ -n "${manifest_extras[$key]}" ]]; then
        read -ra extra_args <<< "${manifest_extras[$key]}"
    fi
    local directory="$repo_root/generated/logs/g1_rqtune_${run}_50m"
    local -a verifier_args=(
        "G1_TUNE_RUN=${run}"
        "G1_TUNE_RUN_REVISION=${run_revision}"
        "G1_TUNE_EPOCHS=${epochs}"
        "G1_TUNE_SOURCE_VARIANT=${manifest_sources[$key]}"
        "G1_TUNE_TRANSFORMER_FIELDS=${manifest_transformer_fields[$key]}"
        "G1_TUNE_EXPERIMENT_FIELDS=${manifest_experiment_fields[$key]}"
        "G1_TUNE_EMBEDDING_LR=${embedding_lr}"
        "G1_TUNE_DEEP_LR=${deep_lr}"
        "${extra_args[@]}" "$@"
    )
    local artifact_status=0
    g1_require_compatible_or_absent "$directory" 50m \
        "${verifier_args[@]}" || artifact_status=$?
    if [[ "$artifact_status" -eq 0 ]]; then
        echo "=== skipped compatible $(basename "$directory") ==="
        skipped_count=$((skipped_count + 1))
        return 0
    fi
    [[ "$artifact_status" -eq 1 ]] || return "$artifact_status"
    enqueue "g1_rqtune_${run}_50m" \
        "G1_TUNE_RUN=${run}" \
        "G1_TUNE_RUN_REVISION=${run_revision}" \
        "G1_TUNE_EPOCHS=${epochs}" \
        "G1_TUNE_SOURCE_VARIANT=${manifest_sources[$key]}" \
        "G1_TUNE_TRANSFORMER_FIELDS=${manifest_transformer_fields[$key]}" \
        "G1_TUNE_EXPERIMENT_FIELDS=${manifest_experiment_fields[$key]}" \
        "G1_TUNE_EMBEDDING_LR=${embedding_lr}" \
        "G1_TUNE_DEEP_LR=${deep_lr}" \
        "${extra_args[@]}" "$@"
    enqueued_count=$((enqueued_count + 1))
}

if [[ -n "$lr_points" ]]; then
    for index in "${!sparse_keys[@]}"; do
        key=${sparse_keys[$index]}
        embedding_lr=${sparse_embedding_lrs[$index]}
        deep_lr=${sparse_deep_lrs[$index]}
        batch_size=${sparse_batch_sizes[$index]}
        if [[ "$stage" == lr ]]; then
            set_batch_contract "$key" "$fixed_batch_size" || exit 2
            suffix="e$(lr_slug "$embedding_lr")d$(lr_slug "$deep_lr")"
            suffix+="${tuning_batch_suffix}_${run_tag}${provenance_suffix}"
            enqueue_treatment "$key" "$suffix" "$embedding_lr" "$deep_lr" \
                "G1_TUNE_BATCH_SIZE=${tuning_physical_batch_size}" \
                "G1_TUNE_GRADIENT_ACCUMULATION_STEPS=${tuning_accumulation_steps}" || exit 2
        else
            suffix="e$(slug "$embedding_lr")_d$(slug "$deep_lr")"
            suffix+="_b${batch_size}_${run_tag}${provenance_suffix}"
            enqueue_treatment "$key" "$suffix" "$embedding_lr" "$deep_lr" \
                "G1_TUNE_BATCH_SIZE=${batch_size}" \
                "G1_TUNE_GRADIENT_ACCUMULATION_STEPS=1" || exit 2
        fi
    done
else
    for key in "${manifest_order[@]}"; do
        axis=${key%%/*}
        axis_selected "$axis" || continue
        treatment_selected "$key" || continue
        [[ -n "${manifest_aliases[$key]}" ]] && continue

        if [[ "$stage" == lr ]]; then
            for embedding_lr in $embedding_lrs; do
                for deep_lr in $deep_lrs; do
                    tag=${run_tag:-mup}
                    set_batch_contract "$key" "$fixed_batch_size" || exit 2
                    suffix="e$(lr_slug "$embedding_lr")d$(lr_slug "$deep_lr")"
                    suffix+="${tuning_batch_suffix}_${tag}${provenance_suffix}"
                    enqueue_treatment "$key" "$suffix" "$embedding_lr" "$deep_lr" \
                        "G1_TUNE_BATCH_SIZE=${tuning_physical_batch_size}" \
                        "G1_TUNE_GRADIENT_ACCUMULATION_STEPS=${tuning_accumulation_steps}" || exit 2
                done
            done
        elif [[ "$stage" == batch ]]; then
            embedding_lr=${selected_embedding_lrs[$key]}
            deep_lr=${selected_deep_lrs[$key]}
            tag="${run_tag:-batch}${provenance_suffix}"
            for batch_size in $batch_sizes; do
                enqueue_treatment "$key" \
                    "e$(slug "$embedding_lr")_d$(slug "$deep_lr")_b${batch_size}_${tag}" \
                    "$embedding_lr" "$deep_lr" \
                    "G1_TUNE_BATCH_SIZE=${batch_size}" \
                    "G1_TUNE_GRADIENT_ACCUMULATION_STEPS=1" || exit 2
            done
        else
            batch_size=${selected_batch_sizes[$key]}
            [[ "$batch_size" == 1280 ]] && continue
            tag="${run_tag:-batchlr}${provenance_suffix}"
            for embedding_lr in $embedding_lrs; do
                for deep_lr in $deep_lrs; do
                    if same_rates "$embedding_lr" "$deep_lr" \
                        "${selected_embedding_lrs[$key]}" "${selected_deep_lrs[$key]}"; then
                        continue
                    fi
                    enqueue_treatment "$key" \
                        "e$(slug "$embedding_lr")_d$(slug "$deep_lr")_b${batch_size}_${tag}" \
                        "$embedding_lr" "$deep_lr" \
                        "G1_TUNE_BATCH_SIZE=${batch_size}" \
                        "G1_TUNE_GRADIENT_ACCUMULATION_STEPS=1" || exit 2
                done
            done
        fi
    done
fi

echo "=== manifest stage=${stage}: enqueued=${enqueued_count}, skipped=${skipped_count} ==="
g1_stop_artifact_verifier
drain || exit 1
