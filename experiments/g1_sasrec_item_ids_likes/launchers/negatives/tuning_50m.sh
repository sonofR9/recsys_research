#!/usr/bin/env bash
set -u

launcher_dir=$(cd "$(dirname "$0")" && pwd)
source "$launcher_dir/../artifacts.sh"
source "$launcher_dir/manifest.sh"
source "$launcher_dir/../global_batch.sh"

valid_families=$G1_NEGATIVE_VALID_FAMILIES
for requested_family in ${G1_TUNE_NEGATIVE_FAMILIES:-all}; do
    if [[ "$requested_family" != all && " $valid_families " != *" $requested_family "* ]]; then
        echo "Unknown G1_TUNE_NEGATIVE_FAMILIES value: $requested_family" >&2
        exit 2
    fi
done

stage=${G1_TUNE_NEGATIVE_STAGE:-lr}
if [[ "$stage" != lr && "$stage" != secondary && "$stage" != local_lr ]]; then
    echo "G1_TUNE_NEGATIVE_STAGE must be lr, secondary, or local_lr" >&2
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
training_semantics_revision=2
if [[ ! "$epochs" =~ ^[1-9][0-9]*$ || ! "$run_revision" =~ ^[1-9][0-9]*$ ]]; then
    echo "G1_TUNE_EPOCHS and G1_TUNE_RUN_REVISION must be canonical positive integers" >&2
    exit 2
fi
if [[ "$epochs" -lt 20 ]]; then
    echo "G1_TUNE_EPOCHS must be at least the 20-epoch safety cap" >&2
    exit 2
fi
global_batch_size=${G1_GLOBAL_BATCH_SIZE:-}
if [[ ! "$global_batch_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "G1_GLOBAL_BATCH_SIZE must be a canonical positive integer" >&2
    exit 2
fi
if [[ -n "${G1_TUNE_BATCH_SIZES+x}" ]]; then
    echo "G1_TUNE_BATCH_SIZES is not supported: training batch is global" >&2
    exit 2
fi

default_tag=initial
[[ "$stage" == secondary ]] && default_tag=secondary
[[ "$stage" == local_lr ]] && default_tag=local
requested_run_tag=${G1_TUNE_RUN_TAG:-}
if [[ -n "$requested_run_tag" && ! "$requested_run_tag" =~ ^[a-z0-9_]+$ ]]; then
    echo "G1_TUNE_RUN_TAG must contain only lowercase letters, digits, and _" >&2
    exit 2
fi
if [[ -n "${G1_TUNE_RUN_TAG+x}" && \
      ( "$requested_run_tag" == initial || "$requested_run_tag" == secondary || \
        "$requested_run_tag" == local || \
        "$requested_run_tag" =~ _r[1-9][0-9]*$ || \
        "$requested_run_tag" =~ (^|_)cap[1-9][0-9]*$ ) ]]; then
    echo "G1_TUNE_RUN_TAG cannot reuse reserved tag $requested_run_tag" >&2
    exit 2
fi
run_tag=${requested_run_tag:-$default_tag}
global_batch_suffix=
if [[ "$global_batch_size" != 1280 ]]; then
    global_batch_suffix="_b${global_batch_size}"
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

if [[ "$stage" == lr || "$stage" == local_lr ]] && \
    [[ -n "${G1_TUNE_EMBEDDING_LRS+x}${G1_TUNE_DEEP_LRS+x}" ]] && \
    [[ -z "$requested_run_tag" ]]; then
    echo "Set a unique G1_TUNE_RUN_TAG for an LR-grid extension" >&2
    exit 2
fi

if [[ "$stage" == secondary ]] && \
    [[ -n "${G1_TUNE_NEGATIVE_COUNTS+x}${G1_TUNE_LOGQ_ALPHAS+x}${G1_TUNE_RANDOM_FRACTIONS+x}" ]] && \
    [[ -z "$requested_run_tag" ]]; then
    echo "Set a unique G1_TUNE_RUN_TAG for a secondary-grid extension" >&2
    exit 2
fi

selected_lr_specs=${G1_TUNE_SELECTED_LRS:-}
if [[ "$stage" != lr && -z "$selected_lr_specs" ]]; then
    echo "G1_TUNE_SELECTED_LRS is required for secondary and local LR tuning" >&2
    exit 2
fi

selected_secondary_specs=${G1_TUNE_SELECTED_SECONDARY:-}
if [[ "$stage" == local_lr && -z "$selected_secondary_specs" ]]; then
    echo "G1_TUNE_SELECTED_SECONDARY is required for exact local LR tuning" >&2
    exit 2
fi

if [[ "$stage" == lr || "$stage" == local_lr ]]; then
    embedding_lr_values=${G1_TUNE_EMBEDDING_LRS:-$G1_NEGATIVE_INITIAL_EMBEDDING_LRS}
    deep_lr_values=${G1_TUNE_DEEP_LRS:-$G1_NEGATIVE_INITIAL_DEEP_LRS}
    require_unique_words G1_TUNE_EMBEDDING_LRS "$embedding_lr_values"
    require_unique_words G1_TUNE_DEEP_LRS "$deep_lr_values"
    for value in $embedding_lr_values; do
        require_positive_number G1_TUNE_EMBEDDING_LRS "$value"
    done
    for value in $deep_lr_values; do
        require_positive_number G1_TUNE_DEEP_LRS "$value"
    done
else
    negative_count_values=${G1_TUNE_NEGATIVE_COUNTS:-$G1_NEGATIVE_SECONDARY_COUNTS}
    alpha_values=${G1_TUNE_LOGQ_ALPHAS:-$G1_NEGATIVE_SECONDARY_ALPHAS}
    random_fraction_values=${G1_TUNE_RANDOM_FRACTIONS:-$G1_NEGATIVE_SECONDARY_RANDOM_FRACTIONS}
    require_unique_words G1_TUNE_NEGATIVE_COUNTS "$negative_count_values"
    require_unique_words G1_TUNE_LOGQ_ALPHAS "$alpha_values"
    require_unique_words G1_TUNE_RANDOM_FRACTIONS "$random_fraction_values"
    for value in $negative_count_values; do
        require_positive_integer G1_TUNE_NEGATIVE_COUNTS "$value"
    done
    for value in $alpha_values; do
        require_positive_number G1_TUNE_LOGQ_ALPHAS "$value"
    done
    for value in $random_fraction_values; do
        require_fraction G1_TUNE_RANDOM_FRACTIONS "$value"
    done
fi

families=("${G1_NEGATIVE_FAMILY_SPECS[@]}")

family_selected() {
    local family=$1
    local selected
    for selected in ${G1_TUNE_NEGATIVE_FAMILIES:-all}; do
        if [[ "$selected" == all || "$selected" == "$family" ]]; then
            return 0
        fi
    done
    return 1
}

selected_rates_for() {
    local requested=$1
    local spec
    local family
    local embedding_lr
    local deep_lr
    local matches=0
    for spec in $selected_lr_specs; do
        IFS=: read -r family embedding_lr deep_lr extra <<< "$spec"
        if [[ -n "${extra:-}" || -z "$family" || -z "$embedding_lr" || -z "$deep_lr" ]]; then
            echo "Invalid G1_TUNE_SELECTED_LRS entry: $spec" >&2
            exit 2
        fi
        if [[ " $valid_families " != *" $family "* ]]; then
            echo "Unknown family in G1_TUNE_SELECTED_LRS: $family" >&2
            exit 2
        fi
        if [[ "$family" == "$requested" ]]; then
            require_positive_number embedding_lr "$embedding_lr"
            require_positive_number deep_lr "$deep_lr"
            selected_embedding_lr=$embedding_lr
            selected_deep_lr=$deep_lr
            matches=$((matches + 1))
        fi
    done
    if [[ "$matches" -ne 1 ]]; then
        echo "Expected exactly one selected LR pair for $requested, found $matches" >&2
        exit 2
    fi
}

selected_secondary_for() {
    local requested=$1
    local spec
    local family
    local negative_count
    local alpha
    local random_fraction
    local matches=0
    for spec in $selected_secondary_specs; do
        IFS=: read -r family negative_count alpha random_fraction extra <<< "$spec"
        if [[ -n "${extra:-}" || -z "$family" || -z "$negative_count" || \
              -z "$alpha" || -z "$random_fraction" ]]; then
            echo "Invalid G1_TUNE_SELECTED_SECONDARY entry: $spec" >&2
            exit 2
        fi
        if [[ " $valid_families " != *" $family "* ]]; then
            echo "Unknown family in G1_TUNE_SELECTED_SECONDARY: $family" >&2
            exit 2
        fi
        if [[ "$family" == "$requested" ]]; then
            require_positive_integer negative_count "$negative_count"
            require_positive_number alpha "$alpha"
            require_fraction random_fraction "$random_fraction"
            selected_negative_count=$negative_count
            selected_alpha=$alpha
            selected_random_fraction=$random_fraction
            matches=$((matches + 1))
        fi
    done
    if [[ "$matches" -ne 1 ]]; then
        echo "Expected exactly one selected secondary configuration for $requested, found $matches" >&2
        exit 2
    fi
}

if [[ "$stage" == secondary ]]; then
    for entry in "${families[@]}"; do
        family=${entry%%:*}
        family_selected "$family" || continue
        selected_rates_for "$family"
    done
fi

if [[ "$stage" == local_lr ]]; then
    for entry in "${families[@]}"; do
        family=${entry%%:*}
        family_selected "$family" || continue
        selected_rates_for "$family"
        selected_secondary_for "$family"
    done
fi

repo_root=$(cd "$launcher_dir/../../../.." && pwd)
g1_require_global_batch_selection "$repo_root" "$global_batch_size" || exit 2
cd "$repo_root"
export WANDB_MODE=${WANDB_MODE:-offline}
export G1_DATASET_SIZE=50m
export TRAINING_QUEUE_MONITOR_LIGHT_GPUS=${TRAINING_QUEUE_MONITOR_LIGHT_GPUS:-1}
export TRAINING_QUEUE_DATA_GROUP=g1-rq-negative-50m-seq128
unset G1_MAX_USERS
unset G1_VAL_BATCH_SIZE
unset G1_TRAIN_BATCH_SIZE
unset G1_SEED
unset G1_TUNE_BATCH_SIZE
unset G1_TUNE_GRADIENT_ACCUMULATION_STEPS
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

slug() {
    local value=${1//./p}
    echo "${value//-/m}"
}

can_reuse_architecture_control() {
    local family=$1
    local negative_count=$2
    local alpha=$3
    local random_fraction=$4
    [[ "$family" == uniform_random ]] || return 1
    [[ "$negative_count" == 512 ]] || return 1
    same_number "$alpha" 0.01 || return 1
    same_number "$random_fraction" 0.5 || return 1
}

enqueued_count=0
skipped_count=0
reused_count=0

enqueue_negative() {
    local family=$1
    local source=$2
    local suffix=$3
    local embedding_lr=$4
    local deep_lr=$5
    shift 5
    local name="neg_${family}_${suffix}"
    if [[ "$explicit_provenance" -eq 1 ]]; then
        [[ "$epochs" -ne 20 ]] && name+="_cap${epochs}"
        name+="_ts${training_semantics_revision}_r${run_revision}"
    else
        name+="_ts${training_semantics_revision}_r2"
    fi
    local negative_count=512
    local alpha=0.01
    local random_fraction=0.5
    local assignment
    for assignment in "$@"; do
        case $assignment in
            G1_TUNE_NUM_NEGATIVES=*) negative_count=${assignment#*=} ;;
            G1_TUNE_LOGQ_ALPHA=*) alpha=${assignment#*=} ;;
            G1_TUNE_RANDOM_FRACTION=*) random_fraction=${assignment#*=} ;;
        esac
    done
    local directory="$repo_root/generated/logs/g1_rqtune_${name}_50m"
    local -a provenance_args=(
        "G1_TUNE_RUN_REVISION=${run_revision}"
        "G1_TUNE_EPOCHS=${epochs}"
    )
    local -a verifier_args=(
        "G1_TUNE_RUN=${name}"
        "${provenance_args[@]}"
        "G1_TUNE_SOURCE_VARIANT=${source}"
        "G1_TUNE_TRANSFORMER_FIELDS="
        "G1_TUNE_EXPERIMENT_FIELDS=negative_sampling,logq_correction,correct_positive_logq,mask_false_negatives,exclude_own_group_negatives"
        "G1_TUNE_EMBEDDING_LR=${embedding_lr}"
        "G1_TUNE_DEEP_LR=${deep_lr}"
        "G1_TUNE_BATCH_SIZE=${global_batch_size}"
        "G1_TUNE_GRADIENT_ACCUMULATION_STEPS=1"
        "$@"
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

    if can_reuse_architecture_control "$family" "$negative_count" "$alpha" \
        "$random_fraction"; then
        artifact_status=0
        g1_reuse_first_compatible_artifact "$directory" 50m \
            "$repo_root/generated/logs/g1_rqtune_architecture_control_*_50m" \
            "${verifier_args[@]}" || artifact_status=$?
        if [[ "$artifact_status" -eq 0 ]]; then
            reused_count=$((reused_count + 1))
            return 0
        fi
        [[ "$artifact_status" -eq 1 ]] || return "$artifact_status"
    fi
    enqueue "g1_rqtune_${name}_50m" \
        "G1_TUNE_RUN=${name}" \
        "${provenance_args[@]}" \
        "G1_TUNE_SOURCE_VARIANT=${source}" \
        "G1_TUNE_EXPERIMENT_FIELDS=negative_sampling,logq_correction,correct_positive_logq,mask_false_negatives,exclude_own_group_negatives" \
        "G1_TUNE_EMBEDDING_LR=${embedding_lr}" \
        "G1_TUNE_DEEP_LR=${deep_lr}" \
        "G1_TUNE_BATCH_SIZE=${global_batch_size}" \
        "G1_TUNE_GRADIENT_ACCUMULATION_STEPS=1" "$@"
    enqueued_count=$((enqueued_count + 1))
}

is_mixed_family() {
    [[ "$1" == uniform_random_plus_streaming_logq_negative_only || \
       "$1" == uniform_random_plus_fixed_logq_negative_only ]]
}

is_streaming_family() {
    [[ "$1" == streaming_inbatch_global_q_yi2019 || \
       "$1" == uniform_random_plus_streaming_logq_negative_only ]]
}

same_number() {
    awk -v left="$1" -v right="$2" 'BEGIN { exit !(left == right) }'
}

secondary_change_count() {
    local family=$1
    local changes=0
    [[ "$selected_negative_count" != 512 ]] && changes=$((changes + 1))
    if is_streaming_family "$family" && ! same_number "$selected_alpha" 0.01; then
        changes=$((changes + 1))
    fi
    if is_mixed_family "$family" && \
       ! same_number "$selected_random_fraction" 0.5; then
        changes=$((changes + 1))
    fi
    echo "$changes"
}

exact_suffix() {
    local embedding_lr=$1
    local deep_lr=$2
    local batch_size=$3
    local negative_count=$4
    local alpha=$5
    local random_fraction=$6
    echo "e$(slug "$embedding_lr")_d$(slug "$deep_lr")_b${batch_size}_n${negative_count}_a$(slug "$alpha")_r$(slug "$random_fraction")"
}

enqueue_exact_negative() {
    local family=$1
    local source=$2
    local suffix=$3
    local embedding_lr=$4
    local deep_lr=$5
    local batch_size=$6
    local negative_count=$7
    local alpha=$8
    local random_fraction=$9
    local overrides=(
        "G1_TUNE_NUM_NEGATIVES=${negative_count}"
        "G1_TUNE_LOGQ_ALPHA=${alpha}"
    )
    if is_mixed_family "$family"; then
        overrides+=("G1_TUNE_RANDOM_FRACTION=${random_fraction}")
    fi
    enqueue_negative "$family" "$source" "$suffix" \
        "$embedding_lr" "$deep_lr" "${overrides[@]}"
}

if [[ "$stage" == lr ]]; then
    read -ra embedding_lrs <<< "${G1_TUNE_EMBEDDING_LRS:-$G1_NEGATIVE_INITIAL_EMBEDDING_LRS}"
    read -ra deep_lrs <<< "${G1_TUNE_DEEP_LRS:-$G1_NEGATIVE_INITIAL_DEEP_LRS}"
    for entry in "${families[@]}"; do
        family=${entry%%:*}
        source=${entry#*:}
        family_selected "$family" || continue
        for embedding_lr in "${embedding_lrs[@]}"; do
            for deep_lr in "${deep_lrs[@]}"; do
                suffix="${run_tag}_e$(slug "$embedding_lr")_d$(slug "$deep_lr")"
                suffix+="$global_batch_suffix"
                enqueue_negative "$family" "$source" \
                    "$suffix" \
                    "$embedding_lr" "$deep_lr" || exit 2
            done
        done
    done
    echo "=== negatives stage=${stage}: enqueued=${enqueued_count}, skipped=${skipped_count}, reused=${reused_count} ==="
    g1_stop_artifact_verifier
    drain || exit 1
    exit 0
fi

if [[ "$stage" == local_lr ]]; then
    read -ra embedding_lrs <<< "${G1_TUNE_EMBEDDING_LRS:-$G1_NEGATIVE_INITIAL_EMBEDDING_LRS}"
    read -ra deep_lrs <<< "${G1_TUNE_DEEP_LRS:-$G1_NEGATIVE_INITIAL_DEEP_LRS}"
    for entry in "${families[@]}"; do
        family=${entry%%:*}
        source=${entry#*:}
        family_selected "$family" || continue
        selected_rates_for "$family"
        selected_secondary_for "$family"
        changed_axes=$(secondary_change_count "$family")
        for embedding_lr in "${embedding_lrs[@]}"; do
            for deep_lr in "${deep_lrs[@]}"; do
                if [[ "$changed_axes" -le 1 ]] && \
                   same_number "$embedding_lr" "$selected_embedding_lr" && \
                   same_number "$deep_lr" "$selected_deep_lr"; then
                    continue
                fi
                suffix=$(exact_suffix \
                    "$embedding_lr" "$deep_lr" "$global_batch_size" \
                    "$selected_negative_count" "$selected_alpha" \
                    "$selected_random_fraction")
                enqueue_exact_negative "$family" "$source" \
                    "${run_tag}_${suffix}" "$embedding_lr" "$deep_lr" \
                    "$global_batch_size" "$selected_negative_count" \
                    "$selected_alpha" "$selected_random_fraction" || exit 2
            done
        done
    done
    echo "=== negatives stage=${stage}: enqueued=${enqueued_count}, skipped=${skipped_count}, reused=${reused_count} ==="
    g1_stop_artifact_verifier
    drain || exit 1
    exit 0
fi

read -ra negative_counts <<< "${G1_TUNE_NEGATIVE_COUNTS:-$G1_NEGATIVE_SECONDARY_COUNTS}"
read -ra logq_alphas <<< "${G1_TUNE_LOGQ_ALPHAS:-$G1_NEGATIVE_SECONDARY_ALPHAS}"
read -ra random_fractions <<< "${G1_TUNE_RANDOM_FRACTIONS:-$G1_NEGATIVE_SECONDARY_RANDOM_FRACTIONS}"

for entry in "${families[@]}"; do
    family=${entry%%:*}
    source=${entry#*:}
    family_selected "$family" || continue
    selected_rates_for "$family"
    rate_suffix="e$(slug "$selected_embedding_lr")_d$(slug "$selected_deep_lr")"
    rate_suffix+="$global_batch_suffix"

    for count in "${negative_counts[@]}"; do
        enqueue_negative "$family" "$source" \
            "${run_tag}_${rate_suffix}_n${count}" \
            "$selected_embedding_lr" "$selected_deep_lr" \
            "G1_TUNE_NUM_NEGATIVES=${count}" || exit 2
    done
    if [[ "$family" == streaming_inbatch_global_q_yi2019 || \
          "$family" == uniform_random_plus_streaming_logq_negative_only ]]; then
        for alpha in "${logq_alphas[@]}"; do
            enqueue_negative "$family" "$source" \
                "${run_tag}_${rate_suffix}_alpha$(slug "$alpha")" \
                "$selected_embedding_lr" "$selected_deep_lr" \
                "G1_TUNE_LOGQ_ALPHA=${alpha}" || exit 2
        done
    fi

    if [[ "$family" == uniform_random_plus_streaming_logq_negative_only || \
          "$family" == uniform_random_plus_fixed_logq_negative_only ]]; then
        for fraction in "${random_fractions[@]}"; do
            enqueue_negative "$family" "$source" \
                "${run_tag}_${rate_suffix}_random$(slug "$fraction")" \
                "$selected_embedding_lr" "$selected_deep_lr" \
                "G1_TUNE_RANDOM_FRACTION=${fraction}" || exit 2
        done
    fi
done

echo "=== negatives stage=${stage}: enqueued=${enqueued_count}, skipped=${skipped_count}, reused=${reused_count} ==="
g1_stop_artifact_verifier
drain || exit 1
