#!/usr/bin/env bash
set -u

launcher_dir=$(cd "$(dirname "$0")" && pwd)
source "$launcher_dir/../artifacts.sh"
cd "$launcher_dir/../../../.."
export WANDB_MODE=${WANDB_MODE:-offline}
export G1_DATASET_SIZE=500m
unset G1_MAX_USERS
unset G1_VAL_BATCH_SIZE
TRAINING_QUEUE_SCRIPT=experiments/g1_sasrec_item_ids_likes/configs/variant.py
source utils/training_queue/queue.sh || exit 1

variants=${*:-$(G1_VARIANT=baseline python - <<'EOF'
from experiments.g1_sasrec_item_ids_likes.configs.variant import VARIANTS

names = list(VARIANTS)
names.remove("homework_reproduction")
print("\n".join(names))
EOF
)}
max_epochs=${G1_MAX_EPOCHS:-20}
if [[ ! "$max_epochs" =~ ^[1-9][0-9]*$ || "$max_epochs" -lt 20 ]]; then
    echo "G1_MAX_EPOCHS must be a canonical integer of at least 20" >&2
    exit 2
fi
cap=
[[ "$max_epochs" -eq 20 ]] || cap="_cap${max_epochs}"
for variant in $variants; do
    if [[ -n ${G1_SEED:-} ]]; then
        enqueue "g1_calibrated_${variant}${cap}_ts2_500m_s${G1_SEED}" \
            "G1_VARIANT=${variant}" "G1_SEED=${G1_SEED}" \
            "G1_MAX_EPOCHS=${max_epochs}"
    else
        enqueue "g1_calibrated_${variant}${cap}_ts2_500m" \
            "G1_VARIANT=${variant}" "G1_MAX_EPOCHS=${max_epochs}"
    fi
done
drain || exit 1

echo "=== final sweep done $(date +%H:%M:%S) ==="
