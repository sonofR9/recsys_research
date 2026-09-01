#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
if [ "${G6_RQ23_SINGLE_WAVE:-0}" != "1" ]; then
    wave_count=$(python - "$@" <<'PY'
import sys
from experiments.g6_rqkmeans_history.launchers.collision_runtime import (
    cache_safe_waves,
    select_jobs,
)
print(len(cache_safe_waves(select_jobs(sys.argv[1:]))))
PY
)
    for ((wave = 0; wave < wave_count; wave++)); do
        G6_RQ23_SINGLE_WAVE=1 "$0" --wave "$wave" "$@"
    done
    exit 0
fi

rows_file=$(mktemp)
trap 'rm -f "$rows_file"' EXIT
python -m experiments.g6_rqkmeans_history.launchers.collision_runtime "$@" > "$rows_file"
mapfile -t rows < "$rows_file"
if [ "${#rows[@]}" -eq 0 ]; then
    echo "collision manifest produced no jobs" >&2
    exit 1
fi
for row in "${rows[@]}"; do
    if [ -z "$row" ] || [[ "$row" != *$'\t'* ]]; then
        echo "collision manifest produced a malformed job" >&2
        exit 1
    fi
done

if [ "${G6_RQ23_QUEUE_DRY_RUN:-0}" = "1" ]; then
    for row in "${rows[@]}"; do
        IFS=$'\t' read -r run_name _ <<< "$row"
        printf '%s\n' "$run_name"
    done
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

TRAINING_QUEUE_SCRIPT=$project_root/experiments/g6_rqkmeans_history/launchers/run_collision.py
TRAINING_QUEUE_DATA_GROUP=g6-rq23-native50m-likes-next-item
source "$project_root/utils/training_queue/queue.sh"
for row in "${rows[@]}"; do
    IFS=$'\t' read -r run_name payload <<< "$row"
    enqueue "$run_name" "G6_RQ23_JOB_B64=$payload" "WANDB_MODE=offline"
done
drain
