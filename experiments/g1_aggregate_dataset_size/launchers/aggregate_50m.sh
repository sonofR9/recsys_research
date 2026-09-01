#!/usr/bin/env bash
set -u

launcher_dir=$(cd "$(dirname "$0")" && pwd) || exit 1
repo_root=$(cd "$launcher_dir/../../.." && pwd) || exit 1
cd "$repo_root" || exit 1

stage=${G1_AGGREGATE_STAGE:-batch_lr_calibration}
canonical_logs=$repo_root/generated/logs
requested_logs=${G1_AGGREGATE_LOGS:-$canonical_logs}
resolved_logs=$(realpath -m "$requested_logs") || exit 2
if [[ "$resolved_logs" != "$canonical_logs" ]]; then
    echo "G1_AGGREGATE_LOGS must resolve to $canonical_logs" >&2
    exit 2
fi
logs=$canonical_logs
infeasible_ledger=${G1_AGGREGATE_INFEASIBLE_LEDGER:-$logs/.g1-aggregate-50m-infeasible.json}
depth=${G1_AGGREGATE_DEPTH:-}
export WANDB_MODE=${WANDB_MODE:-offline}
export G1_DATASET_SIZE=50m
export TRAINING_QUEUE_MONITOR_LIGHT_GPUS=${TRAINING_QUEUE_MONITOR_LIGHT_GPUS:-1}
unset G1_MAX_USERS G1_MAX_EPOCHS G1_SEED G1_TRAIN_BATCH_SIZE G1_VAL_BATCH_SIZE
unset G1_VARIANT G1_AGGREGATE_RUN

config=experiments/g1_aggregate_dataset_size/configs/aggregate_variant.py
runtime=experiments.g1_aggregate_dataset_size.launchers.runtime
TRAINING_QUEUE_SCRIPT=$config
if [[ " ${TRAINING_QUEUE_FORWARD_ENV:-} " != *" G1_DATASET_SIZE "* ]]; then
    TRAINING_QUEUE_FORWARD_ENV="${TRAINING_QUEUE_FORWARD_ENV:+${TRAINING_QUEUE_FORWARD_ENV} }G1_DATASET_SIZE"
fi
if [[ " ${TRAINING_QUEUE_FORWARD_ENV:-} " != *" WANDB_MODE "* ]]; then
    TRAINING_QUEUE_FORWARD_ENV="${TRAINING_QUEUE_FORWARD_ENV:+${TRAINING_QUEUE_FORWARD_ENV} }WANDB_MODE"
fi
for name in G1_DATASET_SIZE WANDB_MODE; do
    if [[ " ${TRAINING_QUEUE_REQUIRED_FORWARD_ENV:-} " != *" ${name} "* ]]; then
        TRAINING_QUEUE_REQUIRED_FORWARD_ENV="${TRAINING_QUEUE_REQUIRED_FORWARD_ENV:+${TRAINING_QUEUE_REQUIRED_FORWARD_ENV} }${name}"
    fi
done
export TRAINING_QUEUE_FORWARD_ENV TRAINING_QUEUE_REQUIRED_FORWARD_ENV

manifest_args=(
    manifest
    --stage "$stage"
    --logs "$logs"
    --infeasible-ledger "$infeasible_ledger"
)
if [[ -n "$depth" ]]; then
    manifest_args+=(--depth "$depth")
fi
candidate_output=$(python -m "$runtime" "${manifest_args[@]}") || exit 2
candidate_rows=()
if [[ -n "$candidate_output" ]]; then
    mapfile -t candidate_rows <<< "$candidate_output"
fi
if [[ "${#candidate_rows[@]}" -eq 0 ]]; then
    echo "Aggregate $stage has no required candidates"
    exit 0
fi
source "${G1_TRAINING_QUEUE_LIBRARY:-utils/training_queue/queue.sh}" || exit 1

declare -A seen
enqueued_count=0
skipped_count=0
for run in "${candidate_rows[@]}"; do
    if [[ -z "$run" || -n "${seen[$run]+x}" ]]; then
        echo "Invalid or duplicate native-50M aggregate candidate: $run" >&2
        exit 2
    fi
    seen[$run]=1
    directory="$logs/$run"
    if [[ -e "$directory" || -L "$directory" ]]; then
        if python -m "$runtime" verify "$directory" "$run"; then
            echo "=== skipped compatible $run ==="
            skipped_count=$((skipped_count + 1))
            continue
        fi
        python -m "$runtime" archive-infeasible \
            "$directory" "$run" "$infeasible_ledger"
        infeasible_status=$?
        if [[ "$infeasible_status" -eq 0 ]]; then
            echo "=== recorded infeasible batch cell $run ==="
            skipped_count=$((skipped_count + 1))
            continue
        fi
        if [[ "$infeasible_status" -ne 3 ]]; then
            echo "Could not classify or preserve failed artifact: $directory" >&2
            exit 2
        fi
        python -m "$runtime" archive-retry "$directory" || exit 2
        echo "=== preserved failed artifact before retrying $run ==="
    fi
    TRAINING_QUEUE_DATA_GROUP="g1-aggregate-50m-seq100" \
        enqueue "$run" "G1_AGGREGATE_RUN=$run" || exit 1
    enqueued_count=$((enqueued_count + 1))
done

echo "=== aggregate $stage: enqueued=${enqueued_count}, skipped=${skipped_count} ==="
drain || exit 1
