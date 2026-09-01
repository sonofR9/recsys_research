#!/usr/bin/env bash
set -u

launcher_dir=$(cd "$(dirname "$0")" && pwd) || exit 1
repo_root=$(cd "$launcher_dir/../../../.." && pwd) || exit 1
cd "$repo_root" || exit 1

list_only=0
if [[ "${1:-}" == "--list" ]]; then
    list_only=1
    shift
fi
if [[ "$#" -lt 1 ]]; then
    echo "usage: $0 [--list] RQ14_BOUNDARY_RUN [...]" >&2
    exit 2
fi

logs=${G1_QUERY_LOGS:-$repo_root/generated/logs}
evidence_dir=${G1_QUERY_EVIDENCE_DIR:-experiments/g1_sasrec_item_ids_likes/evidence}
scratchpad_dir=${G1_QUERY_SCRATCHPAD_DIR:-experiments/g1_sasrec_item_ids_likes/scratchpad}
evidence=$evidence_dir/rq14_query_memory_results.json
python -m experiments.g1_sasrec_item_ids_likes.analysis.rq14_query_memory_report \
    --logs "$logs" --evidence "$evidence_dir" \
    --scratchpad "$scratchpad_dir" >/dev/null || exit 2

mapfile -t candidate_rows < <(
    python - "$evidence" "$@" <<'PY'
import json
from pathlib import Path
import sys

from experiments.g1_sasrec_item_ids_likes.analysis.rq13_rq14_query_candidates import validated_rq14_boundary_candidates

document = json.loads(Path(sys.argv[1]).read_text())
candidates = validated_rq14_boundary_candidates(document, sys.argv[2:])
for candidate in candidates:
    print(f"{candidate.run_name}\t{candidate.treatment}")
PY
)
if [[ "${#candidate_rows[@]}" -ne "$#" ]]; then
    echo "Requested runs are not the exact current RQ14 boundary followups" >&2
    exit 2
fi
if [[ "$list_only" -eq 1 ]]; then
    for row in "${candidate_rows[@]}"; do
        IFS=$'\t' read -r run _ <<< "$row"
        echo "$run"
    done
    exit 0
fi

source "$launcher_dir/../artifacts.sh" || exit 1
export WANDB_MODE=${WANDB_MODE:-offline}
export G1_DATASET_SIZE=500m
export TRAINING_QUEUE_MONITOR_LIGHT_GPUS=${TRAINING_QUEUE_MONITOR_LIGHT_GPUS:-1}
unset G1_MAX_USERS G1_MAX_EPOCHS G1_SEED G1_TRAIN_BATCH_SIZE
unset G1_VAL_BATCH_SIZE G1_VARIANT G1_QUERY_RUN

config=experiments/g1_sasrec_item_ids_likes/configs/rq13_rq14_query_variant.py
TRAINING_QUEUE_SCRIPT=$config
source "${G1_TRAINING_QUEUE_LIBRARY:-utils/training_queue/queue.sh}" || exit 1

enqueued_count=0
skipped_count=0
for row in "${candidate_rows[@]}"; do
    IFS=$'\t' read -r run treatment <<< "$row"
    TRAINING_QUEUE_DATA_GROUP="g1-rq14-${treatment}-500m-seq128"
    directory="$logs/$run"
    verifier_args=("G1_QUERY_RUN=$run")
    artifact_status=0
    g1_require_config_compatible_or_absent "$directory" "$config" \
        "${verifier_args[@]}" || artifact_status=$?
    if [[ "$artifact_status" -eq 0 ]]; then
        echo "=== skipped compatible $run ==="
        skipped_count=$((skipped_count + 1))
        continue
    fi
    [[ "$artifact_status" -eq 1 ]] || exit "$artifact_status"
    enqueue "$run" "${verifier_args[@]}" || exit 1
    enqueued_count=$((enqueued_count + 1))
done

echo "=== rq14 boundary native-500M: enqueued=${enqueued_count}, skipped=${skipped_count} ==="
g1_stop_artifact_verifier
drain || exit 1
