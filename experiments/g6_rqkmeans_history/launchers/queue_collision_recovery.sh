#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
rows_file=$(mktemp)
trap 'rm -f "$rows_file"' EXIT
python -m experiments.g6_rqkmeans_history.launchers.collision_recovery_runtime > "$rows_file"
mapfile -t rows < "$rows_file"
if [ "${#rows[@]}" -ne 1 ] || [ -z "${rows[0]}" ] || [[ "${rows[0]}" != *$'\t'* ]]; then
    echo "collision recovery manifest produced an invalid job" >&2
    exit 1
fi

if [ "${G6_RQ23_RECOVERY_QUEUE_DRY_RUN:-0}" = "1" ]; then
    IFS=$'\t' read -r run_name _ <<< "${rows[0]}"
    printf '%s\n' "$run_name"
    exit 0
fi

queue_status=$(python "$project_root/utils/training_queue/service.py" status --json)
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

TRAINING_QUEUE_SCRIPT=$project_root/experiments/g6_rqkmeans_history/launchers/run_collision_recovery.py
TRAINING_QUEUE_DATA_GROUP=g6-rq23-native50m-likes-next-item
source "$project_root/utils/training_queue/queue.sh"
IFS=$'\t' read -r run_name payload <<< "${rows[0]}"
enqueue "$run_name" "G6_RQ23_RECOVERY_JOB_B64=$payload" "WANDB_MODE=offline"
drain
