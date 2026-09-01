#!/usr/bin/env bash
set -u

launcher_dir=$(cd "$(dirname "$0")" && pwd) || exit 1
repo_root=$(cd "$launcher_dir/../../.." && pwd) || exit 1
cd "$repo_root" || exit 1

logs=$repo_root/generated/logs
runtime=experiments.g1_aggregate_dataset_size.launchers.fixed26_calibration
if ! python -m "$runtime" verify-source --logs "$logs"; then
    echo "Selected batch-512 MuTransfer source artifacts failed authentication" >&2
    exit 2
fi
python -m "$runtime" verify-jobs || exit 2
mapfile -t runs < <(python -m "$runtime" manifest) || exit 2
if [[ "${#runs[@]}" -ne 10 ]]; then
    echo "Fixed-26 calibration manifest must contain exactly ten jobs" >&2
    exit 2
fi
declare -A seen
for run in "${runs[@]}"; do
    if [[ -z "$run" || -n "${seen[$run]+x}" ]]; then
        echo "Invalid or duplicate fixed-26 calibration run: $run" >&2
        exit 2
    fi
    seen[$run]=1
done

export WANDB_MODE=${WANDB_MODE:-offline}
service=utils/training_queue/service.py
state=${TRAINING_QUEUE_SERVICE_STATE_DIR:-generated/training-queue-service}
python "$service" --state-dir "$state" status --json >/dev/null || exit 1
specification=$(mktemp "$repo_root/generated/.fixed26-queue-spec.XXXXXX") || exit 1
trap 'rm -f "$specification"' EXIT
python -m "$runtime" queue-spec --wandb-mode "$WANDB_MODE" \
    >"$specification" || exit 2
batch_id=$(python "$service" --state-dir "$state" find-batch "$specification")
find_status=$?
if [[ "$find_status" -eq 3 ]]; then
    batch_id=
elif [[ "$find_status" -ne 0 ]]; then
    exit 1
fi
if [[ -z "$batch_id" ]]; then
    for run in "${runs[@]}"; do
        if [[ -e "$logs/$run" || -L "$logs/$run" ]]; then
            echo "Immutable fixed-26 run path already exists: $logs/$run" >&2
            exit 2
        fi
    done
    batch_id=$(
        python "$service" --state-dir "$state" submit-batch "$specification"
    ) || exit 1
fi
if [[ -z "$batch_id" ]]; then
    echo "Atomic fixed-26 submission returned no batch id" >&2
    exit 1
fi
echo "=== fixed-26 calibration batch: $batch_id ==="
python "$service" --state-dir "$state" wait-batch "$batch_id" || exit 1
