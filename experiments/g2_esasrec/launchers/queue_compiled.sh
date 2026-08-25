#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 COMPILED_MANIFEST.json" >&2
    exit 2
fi

compiled_manifest=$1
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
compiled_rows=$(
    python -m experiments.g2_esasrec.launchers.compiled "$compiled_manifest"
)
rows=()
if [ -n "$compiled_rows" ]; then
    mapfile -t rows <<< "$compiled_rows"
fi

if [ "${G2_QUEUE_DRY_RUN:-0}" = "1" ]; then
    for row in "${rows[@]}"; do
        IFS=$'\t' read -r kind run_name payload <<< "$row"
        printf '%s\t%s\n' "$kind" "$run_name"
    done
    exit 0
fi

python "$project_root/utils/training_queue/service.py" status --json >/dev/null

queue_kind() {
    local requested_kind=$1
    local script=$2
    local submitted=0
    TRAINING_QUEUE_SCRIPT=$script
    TRAINING_QUEUE_DATA_GROUP=g2-native50m-likes-next-item
    source "$project_root/utils/training_queue/queue.sh"
    for row in "${rows[@]}"; do
        IFS=$'\t' read -r kind run_name payload <<< "$row"
        [ "$kind" = "$requested_kind" ] || continue
        if [ "$requested_kind" = "official" ]; then
            if [ -z "${G2_RECTOOLS_PYTHON:-}" ]; then
                echo "G2_RECTOOLS_PYTHON is required for official jobs" >&2
                return 1
            fi
            enqueue "$run_name" "G2_COMPILED_JOB_B64=$payload" \
                "G2_RECTOOLS_PYTHON=$G2_RECTOOLS_PYTHON"
        else
            enqueue "$run_name" "G2_COMPILED_JOB_B64=$payload" \
                "WANDB_MODE=offline"
        fi
        submitted=1
    done
    [ "$submitted" -eq 0 ] || drain
}

queue_kind local "$project_root/experiments/g2_esasrec/launchers/run_local.py"
queue_kind official "$project_root/experiments/g2_esasrec/launchers/run_official_queued.py"
