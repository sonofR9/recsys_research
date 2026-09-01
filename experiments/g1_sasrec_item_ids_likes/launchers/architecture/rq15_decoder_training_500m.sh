#!/usr/bin/env bash
set -u

pretrained_only=0
if [[ "${1:-}" == "--pretrained-only" ]]; then
    pretrained_only=1
    shift
fi
if [[ "$#" -ne 0 ]]; then
    echo "Usage: $0 [--pretrained-only]" >&2
    exit 2
fi

launcher_dir=$(cd "$(dirname "$0")" && pwd)
source "${G1_RQ15_ARTIFACTS_LIBRARY:-$launcher_dir/../artifacts.sh}"

repo_root=$(cd "$launcher_dir/../../../.." && pwd)
cd "$repo_root"
logs=${G1_RQ15_LOGS:-$repo_root/generated/logs}
export G1_DATASET_SIZE=500m
export WANDB_MODE=${WANDB_MODE:-offline}
export TRAINING_QUEUE_MONITOR_LIGHT_GPUS=${TRAINING_QUEUE_MONITOR_LIGHT_GPUS:-1}
unset G1_MAX_USERS G1_MAX_EPOCHS G1_SEED G1_TRAIN_BATCH_SIZE G1_VAL_BATCH_SIZE
unset G1_VARIANT G1_RQ8_RUN G1_RQ15_RUN

source_row=""
source_config=experiments/g1_sasrec_item_ids_likes/configs/rq15_rq8_checkpoint_variant.py
source_ready=1
mapfile -t source_runs < <(
    python - <<'PY'
from experiments.g1_sasrec_item_ids_likes.analysis.rq15_training_candidates import source_candidates
for candidate in source_candidates():
    print(candidate.run_name, candidate.checkpoint_name)
PY
)
for row in "${source_runs[@]}"; do
    read -r run checkpoint_name <<< "$row"
    directory="$logs/$run"
    if [[ ! -f "$directory/$checkpoint_name" ]]; then
        source_ready=0
        continue
    fi
    source_status=0
    g1_verify_config_recipe_artifact "$directory" "$source_config" \
        "G1_RQ15_SOURCE_RUN=$run" || source_status=$?
    if [[ "$source_status" -eq 1 ]]; then
        source_ready=0
    elif [[ "$source_status" -ne 0 ]]; then
        exit "$source_status"
    fi
done
g1_stop_artifact_verifier
if [[ "$source_ready" -eq 1 ]] && source_row=$(python - "$logs" <<'PY'
from pathlib import Path
import sys

from experiments.g1_sasrec_item_ids_likes.analysis.rq15_training_candidates import selected_source_candidate

logs = Path(sys.argv[1])
candidate = selected_source_candidate(logs)
checkpoint = candidate.checkpoint_path(logs)
print(candidate.run_name, checkpoint)
PY
); then
    read -r source_run checkpoint <<< "$source_row"
    if [[ -n "${G1_RQ15_SOURCE_RUN:-}" && "$G1_RQ15_SOURCE_RUN" != "$source_run" ]]; then
        echo "G1_RQ15_SOURCE_RUN does not match the validation-selected source" >&2
        exit 2
    fi
    if [[ -n "${G1_RQ15_FIRST_STAGE_CHECKPOINT:-}" && "$G1_RQ15_FIRST_STAGE_CHECKPOINT" != "$checkpoint" ]]; then
        echo "G1_RQ15_FIRST_STAGE_CHECKPOINT does not match the validation-selected source" >&2
        exit 2
    fi
else
    source_run=""
    checkpoint=""
fi

config=experiments/g1_sasrec_item_ids_likes/configs/rq15_decoder_training_variant.py
TRAINING_QUEUE_SCRIPT=$config
source "${G1_TRAINING_QUEUE_LIBRARY:-utils/training_queue/queue.sh}" || exit 1

mapfile -t candidates < <(
    python - "$pretrained_only" <<'PY'
import sys

from experiments.g1_sasrec_item_ids_likes.analysis.rq15_training_candidates import launch_initial_candidates

pretrained_only = bool(int(sys.argv[1]))
for candidate in launch_initial_candidates():
    if pretrained_only and candidate.training_method != "pretrained_finetune":
        continue
    print(candidate.training_method, candidate.run_name)
PY
)
expected_count=26
if [[ "$pretrained_only" -eq 1 ]]; then
    expected_count=9
fi
if [[ "${#candidates[@]}" -ne "$expected_count" ]]; then
    echo "RQ15 initial launch must contain exactly $expected_count runs" >&2
    exit 2
fi

enqueued_count=0
skipped_count=0
missing_checkpoint_count=0
for candidate in "${candidates[@]}"; do
    read -r method run <<< "$candidate"
    verifier_args=("G1_RQ15_RUN=$run")
    if [[ "$method" == pretrained_finetune ]]; then
        if [[ -z "$source_run" || ! -f "$checkpoint" ]]; then
            echo "=== skipped missing first-stage checkpoint $run ===" >&2
            missing_checkpoint_count=$((missing_checkpoint_count + 1))
            continue
        fi
        verifier_args+=("G1_RQ15_SOURCE_RUN=$source_run")
        verifier_args+=("G1_RQ15_FIRST_STAGE_CHECKPOINT=$checkpoint")
    fi
    directory="$logs/$run"
    artifact_status=0
    g1_require_config_recipe_compatible_or_absent "$directory" "$config" \
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

echo "=== rq15 decoder training: enqueued=${enqueued_count}, skipped=${skipped_count}, missing_checkpoint=${missing_checkpoint_count} ==="
g1_stop_artifact_verifier
drain || exit 1
if [[ "$missing_checkpoint_count" -gt 0 ]]; then
    echo "Missing compatible RQ8 first-stage checkpoint: $checkpoint" >&2
    echo "Run rq15_rq8_checkpoint_500m.sh, then rerun this launcher." >&2
    exit 2
fi
