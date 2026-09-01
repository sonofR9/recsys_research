#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$project_root"
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
attempt=${G6_RQ0_PREFLIGHT_ATTEMPT:-$(python -c 'import uuid; print(uuid.uuid4().hex)')}
evidence=${G6_RQ0_PREFLIGHT_EVIDENCE:-$project_root/generated/logs/g6-rq0-preflight-attempts/$attempt.json}
script=$project_root/experiments/g6_rqkmeans_history/launchers/run_preflight.py

(
    TRAINING_QUEUE_SCRIPT=$script
    TRAINING_QUEUE_DATA_GROUP=g6-rq0-native50m-preflight-v5-$attempt-bootstrap
    source "$project_root/utils/training_queue/queue.sh"
    enqueue "g6_rq0_preflight_${attempt}_overhead_128" \
        "G6_RQ0_PREFLIGHT_MODE=overhead" \
        "G6_RQ0_PREFLIGHT_ATTEMPT=$attempt" \
        "G6_RQ0_PREFLIGHT_EVIDENCE=$evidence" \
        "WANDB_MODE=offline"
    drain
)

(
    TRAINING_QUEUE_SCRIPT=$script
    TRAINING_QUEUE_DATA_GROUP=g6-rq0-native50m-preflight-v5-$attempt-probes
    source "$project_root/utils/training_queue/queue.sh"
    for batch_size in 128 256 512 1024 1280; do
        enqueue "g6_rq0_preflight_${attempt}_training_${batch_size}" \
            "G6_RQ0_PREFLIGHT_MODE=training" \
            "G6_RQ0_PREFLIGHT_BATCH=$batch_size" \
            "G6_RQ0_PREFLIGHT_ATTEMPT=$attempt" \
            "G6_RQ0_PREFLIGHT_EVIDENCE=$evidence" \
            "WANDB_MODE=offline"
    done
    enqueue "g6_rq0_preflight_${attempt}_validation_128" \
        "G6_RQ0_PREFLIGHT_MODE=validation" \
        "G6_RQ0_PREFLIGHT_ATTEMPT=$attempt" \
        "G6_RQ0_PREFLIGHT_EVIDENCE=$evidence" \
        "WANDB_MODE=offline"
    drain
)

G6_RQ0_PREFLIGHT_EVIDENCE="$evidence" python -c '
import os
from pathlib import Path
from experiments.g6_rqkmeans_history.analysis.preflight import finalize_preflight

finalize_preflight(Path(os.environ["G6_RQ0_PREFLIGHT_EVIDENCE"]))
'

if [ "${G6_RQ0_PREFLIGHT_PROMOTE:-0}" = "1" ]; then
    G6_RQ0_PREFLIGHT_EVIDENCE="$evidence" python -c '
import os
from pathlib import Path
from experiments.g6_rqkmeans_history.analysis.preflight import promote_preflight

promote_preflight(Path(os.environ["G6_RQ0_PREFLIGHT_EVIDENCE"]))
'
fi
