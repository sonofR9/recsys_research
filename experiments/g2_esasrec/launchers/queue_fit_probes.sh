#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
python "$project_root/utils/training_queue/service.py" status --json >/dev/null
TRAINING_QUEUE_SCRIPT=$project_root/experiments/g2_esasrec/launchers/run_fit_probe.py
TRAINING_QUEUE_DATA_GROUP=g2-native50m-likes-next-item-diagnostic
source "$project_root/utils/training_queue/queue.sh"
for batch_size in 128 256 512 1024 1280; do
    enqueue "g2_fit_probe_ligr_m6_b${batch_size}_native50m_diagnostic" \
        "G2_FIT_BATCH_SIZE=$batch_size" "WANDB_MODE=offline"
done
drain
