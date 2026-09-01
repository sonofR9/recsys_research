#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 COMPILED_MANIFEST.json" >&2
    exit 2
fi

compiled_manifest=$1
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
compiled_rows=$(
    python -m experiments.g6_rqkmeans_history.launchers.compiled "$compiled_manifest"
)
rows=()
if [ -n "$compiled_rows" ]; then
    mapfile -t rows <<< "$compiled_rows"
fi

if [ "${G6_RQ0_QUEUE_DRY_RUN:-0}" = "1" ]; then
    for row in "${rows[@]}"; do
        IFS=$'\t' read -r run_name _ <<< "$row"
        printf '%s\n' "$run_name"
    done
    exit 0
fi

queue_status=$(
    python "$project_root/utils/training_queue/service.py" status --json
)
if ! QUEUE_STATUS_JSON="$queue_status" python -c '
import json
import os
import sys

status = json.loads(os.environ["QUEUE_STATUS_JSON"])
sys.exit(0 if status.get("running") is True else 1)
'; then
    echo "persistent training queue service is not running" >&2
    exit 1
fi

TRAINING_QUEUE_SCRIPT=$project_root/experiments/g6_rqkmeans_history/launchers/run_compiled.py
TRAINING_QUEUE_DATA_GROUP=g6-rq0-native50m-likes-next-item
source "$project_root/utils/training_queue/queue.sh"
for row in "${rows[@]}"; do
    IFS=$'\t' read -r run_name payload <<< "$row"
    enqueue "$run_name" "G6_RQ0_COMPILED_JOB_B64=$payload" "WANDB_MODE=offline"
done
drain
