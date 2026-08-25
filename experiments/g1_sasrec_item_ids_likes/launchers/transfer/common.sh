#!/usr/bin/env bash

g1_transfer_launcher_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
g1_transfer_repo_root=$(cd "$g1_transfer_launcher_dir/../../../.." && pwd)
source "$g1_transfer_launcher_dir/../artifacts.sh"

unset G1_DATASET_SIZE G1_MAX_USERS G1_SEED G1_TRAIN_BATCH_SIZE G1_VAL_BATCH_SIZE
unset G1_HOMEWORK_BATCH_SIZE G1_HOMEWORK_RUN_TAG
unset G1_TRANSFER_BATCH_SIZE G1_TRANSFER_DEEP_LR G1_TRANSFER_DIM
unset G1_TRANSFER_EMBEDDING_LR G1_TRANSFER_EPOCHS
unset G1_TRANSFER_PARAMETERIZATION G1_TRANSFER_POWER_TOKENS G1_TRANSFER_RUN
unset G1_TRANSFER_RUN_REVISION G1_TRANSFER_SOURCE_VARIANT
unset G1_TUNE_BATCH_SIZE G1_TUNE_CORRECT_POSITIVE_LOGQ G1_TUNE_DEEP_LR
unset G1_TUNE_EMBEDDING_LR G1_TUNE_EXCLUDE_OWN_GROUP_NEGATIVES
unset G1_TUNE_EXPERIMENT_FIELDS G1_TUNE_FFN_DIM G1_TUNE_LOGQ_ALPHA
unset G1_TUNE_LOGQ_CORRECTION G1_TUNE_MASK_FALSE_NEGATIVES
unset G1_TUNE_NUM_NEGATIVES G1_TUNE_RANDOM_FRACTION G1_TUNE_RUN
unset G1_TUNE_SOURCE_VARIANT G1_TUNE_TRANSFORMER_FIELDS

g1_require_canonical_positive_decimal() {
    python - "$1" <<'PY'
from decimal import Decimal, InvalidOperation
import math
import sys

raw = sys.argv[1]
try:
    decimal_value = Decimal(raw)
    float_value = float(raw)
except (InvalidOperation, ValueError, OverflowError):
    raise SystemExit(1)
canonical = format(decimal_value.normalize(), "f")
if (
    not decimal_value.is_finite()
    or decimal_value <= 0
    or not math.isfinite(float_value)
    or float_value <= 0
    or raw != canonical
):
    raise SystemExit(1)
PY
}

g1_require_known_transfer_assignments() {
    local assignment
    for assignment in "$@"; do
        case "$assignment" in
            G1_TRANSFER_POWER_TOKENS=*|G1_TRANSFER_FFN_DIM=*|G1_TRANSFER_MUP_FFN_BASE=*) ;;
            *)
                echo "Unknown transfer assignment: ${assignment%%=*}" >&2
                return 2
                ;;
        esac
    done
}

g1_enqueue_transfer() {
    local run=$1
    local epochs=$2
    local embedding_lr=$3
    local deep_lr=$4
    shift 4
    if [[ ! "$epochs" =~ ^[1-9][0-9]*$ || "$epochs" -lt 20 ]]; then
        echo "Transfer epoch cap must be a canonical integer of at least 20" >&2
        return 2
    fi
    local cap=
    [[ "$epochs" -eq 20 ]] || cap="_cap${epochs}"
    local run_revision=${G1_TRANSFER_RUN_REVISION:-2}
    if [[ ! "$run_revision" =~ ^[1-9][0-9]*$ ]]; then
        echo "Transfer run revision must be a canonical positive integer" >&2
        return 2
    fi
    if [[ "$run" != *"${cap}_ts2_r${run_revision}" && \
          ( "$epochs" -ne 20 || "$run" != *_cap20_ts2_r${run_revision} ) ]]; then
        echo "Current transfer run must end in ${cap}_ts2_r${run_revision}: $run" >&2
        return 2
    fi
    g1_require_known_transfer_assignments "$@" || return 2
    g1_enqueue_transfer_recipe "$run" "$epochs" "$embedding_lr" "$deep_lr" \
        conventional 1280 selected_quality_b1280 "$@"
}

g1_enqueue_transfer_recipe() {
    local run=$1
    local epochs=$2
    local embedding_lr=$3
    local deep_lr=$4
    local parameterization=$5
    local batch_size=$6
    local source_variant=$7
    shift 7
    if [[ ! "$epochs" =~ ^[1-9][0-9]*$ || "$epochs" -lt 20 ]]; then
        echo "Transfer epoch cap must be a canonical integer of at least 20" >&2
        return 2
    fi
    local cap=
    [[ "$epochs" -eq 20 ]] || cap="_cap${epochs}"
    local run_revision=${G1_TRANSFER_RUN_REVISION:-2}
    if [[ ! "$run_revision" =~ ^[1-9][0-9]*$ ]]; then
        echo "Transfer run revision must be a canonical positive integer" >&2
        return 2
    fi
    if [[ "$run" != *"${cap}_ts2_r${run_revision}" && \
          ( "$epochs" -ne 20 || "$run" != *_cap20_ts2_r${run_revision} ) ]]; then
        echo "Current transfer run must end in ${cap}_ts2_r${run_revision}: $run" >&2
        return 2
    fi
    g1_require_known_transfer_assignments "$@" || return 2
    local config="$g1_transfer_repo_root/experiments/g1_sasrec_item_ids_likes/configs/transfer_variant.py"
    local directory="$g1_transfer_repo_root/generated/logs/g1_transfer_${run}_${G1_DATASET_SIZE}"
    local -a assignments=(
        "G1_DATASET_SIZE=${G1_DATASET_SIZE}"
        "G1_TRANSFER_RUN=${run}"
        "G1_TRANSFER_RUN_REVISION=${run_revision}"
        "G1_TRANSFER_EPOCHS=${epochs}"
        "G1_TRANSFER_EMBEDDING_LR=${embedding_lr}"
        "G1_TRANSFER_DEEP_LR=${deep_lr}"
        "G1_TRANSFER_PARAMETERIZATION=${parameterization}"
        "G1_TRANSFER_BATCH_SIZE=${batch_size}"
        "G1_TRANSFER_DIM=64"
        "G1_TRANSFER_SOURCE_VARIANT=${source_variant}"
        "$@"
    )
    local artifact_status=0
    g1_require_config_compatible_or_absent "$directory" "$config" \
        "${assignments[@]}" || artifact_status=$?
    if [[ "$artifact_status" -eq 0 ]]; then
        echo "=== skipped compatible $(basename "$directory") ==="
        return 0
    fi
    [[ "$artifact_status" -eq 1 ]] || return "$artifact_status"
    enqueue "$(basename "$directory")" "${assignments[@]}"
}
