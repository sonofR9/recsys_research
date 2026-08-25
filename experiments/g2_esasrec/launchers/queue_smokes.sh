#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
python "$project_root/utils/training_queue/service.py" status --json >/dev/null
TRAINING_QUEUE_SCRIPT="$project_root/experiments/g2_esasrec/launchers/run_smoke.py"
TRAINING_QUEUE_DATA_GROUP=g2-smoke-2000users-likes-next-item
source "$project_root/utils/training_queue/queue.sh"

for method in \
    standard_sampled_softmax \
    standard_gbce \
    ligr_sampled_softmax \
    ligr_gbce; do
    enqueue "g2_smoke_${method}_2000users_seed42" \
        "G2_SMOKE_METHOD=$method" "WANDB_MODE=offline"
done
drain
