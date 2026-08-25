#!/usr/bin/env bash
# 1.1 round 2: re-run the named variants across seeds, so the sweep's ranking
# can be read against its own noise instead of taken at face value.
#
#     ./seeds.sh dropout_30 dim_128 baseline
set -u

launcher_dir=$(cd "$(dirname "$0")" && pwd)
source "$launcher_dir/../artifacts.sh"
cd "$launcher_dir/../../../.."
export WANDB_MODE=${WANDB_MODE:-offline}
export G1_DATASET_SIZE=${G1_DATASET_SIZE:-500m}
TRAINING_QUEUE_SCRIPT=experiments/g1_sasrec_item_ids_likes/configs/variant.py
source utils/training_queue/queue.sh || exit 1

seeds=${G1_SEEDS:-"1 2 3"}
dataset_size=$G1_DATASET_SIZE
max_epochs=${G1_MAX_EPOCHS:-20}
if [[ ! "$max_epochs" =~ ^[1-9][0-9]*$ || "$max_epochs" -lt 20 ]]; then
    echo "G1_MAX_EPOCHS must be a canonical integer of at least 20" >&2
    exit 2
fi
cap=
[[ "$max_epochs" -eq 20 ]] || cap="_cap${max_epochs}"
run_suffix=
if [[ -n ${G1_MAX_USERS:-} ]]; then
    run_suffix="_${G1_MAX_USERS}users_seed42"
fi
if [[ -n ${G1_VAL_BATCH_SIZE:-} ]]; then
    run_suffix="${run_suffix}_val${G1_VAL_BATCH_SIZE}"
fi

for variant in "$@"; do
    for seed in $seeds; do
        enqueue "g1_calibrated_${variant}${cap}_ts2_${dataset_size}${run_suffix}_s${seed}" \
            "G1_VARIANT=${variant}" "G1_SEED=${seed}" \
            "G1_MAX_EPOCHS=${max_epochs}"
    done
done
drain || exit 1

echo "=== seeds done $(date +%H:%M:%S) ==="
