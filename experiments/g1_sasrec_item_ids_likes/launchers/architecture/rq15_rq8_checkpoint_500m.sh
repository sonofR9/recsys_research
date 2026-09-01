#!/usr/bin/env bash
set -u

launcher_dir=$(cd "$(dirname "$0")" && pwd)
source "$launcher_dir/../artifacts.sh"

repo_root=$(cd "$launcher_dir/../../../.." && pwd)
cd "$repo_root"
logs=${G1_RQ15_LOGS:-$repo_root/generated/logs}
export G1_DATASET_SIZE=500m
export WANDB_MODE=${WANDB_MODE:-offline}
export TRAINING_QUEUE_MONITOR_LIGHT_GPUS=${TRAINING_QUEUE_MONITOR_LIGHT_GPUS:-1}
unset G1_MAX_USERS G1_MAX_EPOCHS G1_SEED G1_TRAIN_BATCH_SIZE G1_VAL_BATCH_SIZE
unset G1_VARIANT G1_RQ8_RUN G1_RQ15_RUN G1_RQ15_SOURCE_RUN

config=experiments/g1_sasrec_item_ids_likes/configs/rq15_rq8_checkpoint_variant.py
TRAINING_QUEUE_SCRIPT=$config
source "${G1_TRAINING_QUEUE_LIBRARY:-utils/training_queue/queue.sh}" || exit 1

mapfile -t source_rows < <(
    python - <<'PY'
from experiments.g1_sasrec_item_ids_likes.analysis.rq15_training_candidates import source_candidates
for candidate in source_candidates():
    print(candidate.run_name, candidate.checkpoint_name)
PY
)
if [[ "${#source_rows[@]}" -ne 3 ]]; then
    echo "RQ15 source stage must contain exactly three runs" >&2
    exit 2
fi

enqueued_count=0
skipped_count=0
for row in "${source_rows[@]}"; do
    read -r run checkpoint_name <<< "$row"
    directory="$logs/$run"
    checkpoint="$directory/$checkpoint_name"
    artifact_status=0
    g1_require_config_recipe_compatible_or_absent "$directory" "$config" \
        "G1_RQ15_SOURCE_RUN=$run" || artifact_status=$?
    if [[ "$artifact_status" -eq 0 ]]; then
        if [[ -f "$checkpoint" ]]; then
            echo "=== skipped compatible $run ==="
            skipped_count=$((skipped_count + 1))
            continue
        fi
        g1_archive_artifact "$directory" incomplete || exit 2
        artifact_status=1
    fi
    [[ "$artifact_status" -eq 1 ]] || exit "$artifact_status"
    enqueue "$run" "G1_RQ15_SOURCE_RUN=$run" || exit 1
    enqueued_count=$((enqueued_count + 1))
done
echo "=== rq15 RQ8 checkpoint sources: enqueued=${enqueued_count}, skipped=${skipped_count} ==="
g1_stop_artifact_verifier
drain || exit 1

for row in "${source_rows[@]}"; do
    read -r run checkpoint_name <<< "$row"
    directory="$logs/$run"
    checkpoint="$directory/$checkpoint_name"
    artifact_status=0
    g1_require_config_recipe_compatible_or_absent "$directory" "$config" \
        "G1_RQ15_SOURCE_RUN=$run" || artifact_status=$?
    if [[ "$artifact_status" -ne 0 || ! -f "$checkpoint" ]]; then
        echo "RQ15 source artifact is incomplete or incompatible: $run" >&2
        exit 2
    fi
done
g1_stop_artifact_verifier

python - "$logs" <<'PY'
from pathlib import Path
import sys

from experiments.g1_sasrec_item_ids_likes.analysis.rq15_training_candidates import selected_source_candidate

logs = Path(sys.argv[1])
candidate = selected_source_candidate(logs)
print(f"Selected RQ15 source: {candidate.run_name} ({candidate.checkpoint_path(logs)})")
PY
