#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
rows_file=$(mktemp)
trap 'rm -f "$rows_file"' EXIT
python -m experiments.g6_rqkmeans_history.launchers.confirmation_runtime > "$rows_file"
mapfile -t rows < "$rows_file"
if [ "${#rows[@]}" -eq 0 ]; then
    echo "confirmation manifest produced no jobs" >&2
    exit 1
fi
for row in "${rows[@]}"; do
    if [ -z "$row" ] || [[ "$row" != *$'\t'* ]]; then
        echo "confirmation manifest produced a malformed job" >&2
        exit 1
    fi
done
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

TRAINING_QUEUE_SCRIPT=$project_root/experiments/g6_rqkmeans_history/launchers/run_confirmation.py
TRAINING_QUEUE_DATA_GROUP=g6-rq-confirmation-native50m
source "$project_root/utils/training_queue/queue.sh"
for row in "${rows[@]}"; do
    IFS=$'\t' read -r run_name payload <<< "$row"
    enqueue "$run_name" "G6_CONFIRMATION_JOB_B64=$payload" "WANDB_MODE=offline"
done
drain
