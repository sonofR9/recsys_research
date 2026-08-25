#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 SELECTED_RUN_NAME COMPILED_JOB_B64 OUTPUT_PATH" >&2
    exit 2
fi

selected_run_name=$1
compiled_job=$2
output_path=$3
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
queue_run="g2_selected_benchmark_${selected_run_name}_deterministic_reproduction_offline"
assignments=(
    "G2_COMPILED_JOB_B64=$compiled_job"
    "G2_BENCHMARK_OUTPUT=$output_path"
    "WANDB_MODE=offline"
)

if [[ ${G2_QUEUE_DRY_RUN:-0} == 1 ]]; then
    printf '%s\n' "$queue_run" "${assignments[@]}"
    exit 0
fi

python "$project_root/utils/training_queue/service.py" status --json >/dev/null
TRAINING_QUEUE_SCRIPT=$project_root/experiments/g2_esasrec/launchers/run_selected_benchmark.py
TRAINING_QUEUE_DATA_GROUP=g2-native50m-likes-next-item-selected-benchmark
source "$project_root/utils/training_queue/queue.sh"
enqueue "$queue_run" "${assignments[@]}"
drain
