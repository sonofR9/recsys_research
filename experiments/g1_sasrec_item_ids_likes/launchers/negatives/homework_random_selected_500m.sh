#!/usr/bin/env bash
set -u

launcher_dir=$(cd "$(dirname "$0")" && pwd)
source "$launcher_dir/../artifacts.sh"

selection=${G1_HOMEWORK_RANDOM_SELECTION:?Set G1_HOMEWORK_RANDOM_SELECTION=embedding_lr:deep_lr}
IFS=: read -r embedding_lr deep_lr extra <<< "$selection"
if [[ -n "${extra:-}" || -z "$embedding_lr" || -z "$deep_lr" ]]; then
    echo "G1_HOMEWORK_RANDOM_SELECTION must be embedding_lr:deep_lr" >&2
    exit 2
fi

repo_root=$(cd "$launcher_dir/../../../.." && pwd)
selector=${G1_TEST_HOMEWORK_CONTROL_SELECTOR:-$repo_root/experiments/g1_sasrec_item_ids_likes/analysis/select_homework_negative_control.py}
if [[ -n "${G1_TEST_HOMEWORK_CONTROL_SELECTOR+x}" && \
      -z "${G1_TRAINING_QUEUE_LIBRARY:-}" ]]; then
    echo "G1_TEST_HOMEWORK_CONTROL_SELECTOR requires an injected queue library" >&2
    exit 2
fi
if ! canonical_selection=$(
    python "$selector" \
        --family random --embedding-lr "$embedding_lr" --deep-lr "$deep_lr"
); then
    exit 2
fi
IFS=: read -r embedding_lr deep_lr extra <<< "$canonical_selection"
if [[ -n "${extra:-}" || -z "$embedding_lr" || -z "$deep_lr" ]]; then
    echo "homework control selector returned an invalid LR pair" >&2
    exit 2
fi

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
expected_epochs=20
expected_revision=1
while (( expected_epochs < epochs )); do
    expected_epochs=$((expected_epochs * 2))
    expected_revision=$((expected_revision + 1))
done
if (( epochs != expected_epochs || run_revision != expected_revision )); then
    echo "G1_HOMEWORK_RANDOM_EPOCHS/RUN_REVISION must follow 20/r1, 40/r2, 80/r3, ..." >&2
    exit 2
fi

slug() {
    local value=${1//./p}
    value=${value//-/m}
    echo "${value//+/p}"
}

cd "$repo_root"
export WANDB_MODE=${WANDB_MODE:-offline}
export G1_DATASET_SIZE=500m
export TRAINING_QUEUE_MONITOR_LIGHT_GPUS=${TRAINING_QUEUE_MONITOR_LIGHT_GPUS:-1}
export TRAINING_QUEUE_DATA_GROUP=g1-homework-random-500m-seq100
unset G1_MAX_USERS G1_VAL_BATCH_SIZE G1_TRAIN_BATCH_SIZE G1_SEED G1_MAX_EPOCHS
unset G1_HOMEWORK_RANDOM_RUN G1_HOMEWORK_RANDOM_EMBEDDING_LR
unset G1_HOMEWORK_RANDOM_DEEP_LR G1_HOMEWORK_RANDOM_DATASET_SIZE

cap=
[[ "$epochs" -eq 20 ]] || cap="_cap${epochs}"
run="selected_e$(slug "$embedding_lr")_d$(slug "$deep_lr")${cap}_ts2_r${run_revision}"
control_run_name="g1_homework_random_${run}_500m"
config=experiments/g1_sasrec_item_ids_likes/configs/homework_random_control.py
assignments=(
    "G1_HOMEWORK_RANDOM_RUN=${run}"
    "G1_HOMEWORK_RANDOM_EPOCHS=${epochs}"
    "G1_HOMEWORK_RANDOM_RUN_REVISION=${run_revision}"
    "G1_HOMEWORK_RANDOM_EMBEDDING_LR=${embedding_lr}"
    "G1_HOMEWORK_RANDOM_DEEP_LR=${deep_lr}"
    "G1_HOMEWORK_RANDOM_DATASET_SIZE=500m"
)

TRAINING_QUEUE_SCRIPT=$config
source "${G1_TRAINING_QUEUE_LIBRARY:-utils/training_queue/queue.sh}" || exit 1
if (( epochs > 20 )); then
    predecessor_epochs=20
    predecessor_revision=1
    while (( predecessor_epochs < epochs )); do
        predecessor_cap=
        [[ "$predecessor_epochs" -eq 20 ]] || predecessor_cap="_cap${predecessor_epochs}"
        predecessor_run="selected_e$(slug "$embedding_lr")_d$(slug "$deep_lr")"
        predecessor_run+="${predecessor_cap}_ts2_r${predecessor_revision}"
        predecessor_name="g1_homework_random_${predecessor_run}_500m"
        predecessor_directory="$repo_root/generated/logs/$predecessor_name"
        predecessor_assignments=(
            "G1_HOMEWORK_RANDOM_RUN=${predecessor_run}"
            "G1_HOMEWORK_RANDOM_EPOCHS=${predecessor_epochs}"
            "G1_HOMEWORK_RANDOM_RUN_REVISION=${predecessor_revision}"
            "G1_HOMEWORK_RANDOM_EMBEDDING_LR=${embedding_lr}"
            "G1_HOMEWORK_RANDOM_DEEP_LR=${deep_lr}"
            "G1_HOMEWORK_RANDOM_DATASET_SIZE=500m"
        )
        if ! g1_artifact_exists "$predecessor_directory"; then
            echo "Missing selected 500M cap predecessor: $predecessor_name" >&2
            exit 2
        fi
        g1_classify_config_artifact \
            "$predecessor_directory" "$repo_root/$config" \
            "${predecessor_assignments[@]}" || exit 2
        if [[ "$_g1_artifact_state" != resumable ]]; then
            echo "Selected 500M cap predecessor did not hit its cap: $predecessor_name" >&2
            exit 2
        fi
        predecessor_epochs=$((predecessor_epochs * 2))
        predecessor_revision=$((predecessor_revision + 1))
    done
fi
artifact_status=0
g1_require_config_compatible_or_absent \
    "$repo_root/generated/logs/$control_run_name" "$repo_root/$config" \
    "${assignments[@]}" || artifact_status=$?
if [[ "$artifact_status" -eq 0 ]]; then
    echo "=== skipped compatible $control_run_name ==="
elif [[ "$artifact_status" -eq 1 ]]; then
    enqueue "$control_run_name" "${assignments[@]}" || exit 2
else
    exit "$artifact_status"
fi

g1_stop_artifact_verifier
drain || exit 1
