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
if [[ "$#" -ne 0 ]]; then
    echo "usage: $0 [--list]" >&2
    exit 2
fi

logs=${G1_RQ15_LOGS:-$repo_root/generated/logs}
evidence=${G1_RQ15_EVIDENCE:-experiments/g1_sasrec_item_ids_likes/evidence/rq15_training_results.json}
mapfile -t candidate_rows < <(
    python - "$evidence" "$logs" 3>&1 1>&2 <<'PY'
import json
import os
from pathlib import Path
import sys

from experiments.g1_sasrec_item_ids_likes.analysis.rq15_training_candidates import validated_required_followup_candidates
from experiments.g1_sasrec_item_ids_likes.analysis.rq15_training_report import collect_report_bundle

try:
    evidence_path = Path(sys.argv[1])
    document = json.loads(evidence_path.read_text())
    evidence_dir = evidence_path.parent
    authoritative = collect_report_bundle(
        Path(sys.argv[2]),
        rq14_results=evidence_dir / "rq14_query_memory_results.json",
        correctness_evidence=evidence_dir / "rq15_training_correctness.json",
        explanation_evidence=evidence_dir / "rq15_training_explanation.json",
    ).evidence
    candidates = validated_required_followup_candidates(
        document,
        authoritative_evidence=authoritative,
    )
except (OSError, RuntimeError, json.JSONDecodeError, ValueError) as error:
    raise SystemExit(str(error))
with os.fdopen(3, "w") as candidate_output:
    for candidate in candidates:
        print(
            f"{candidate.training_method}\t{candidate.stage}\t{candidate.run_name}",
            file=candidate_output,
        )
PY
) || exit 2
if [[ "${#candidate_rows[@]}" -eq 0 ]]; then
    echo "RQ15 evidence contains no canonical follow-up candidates" >&2
    exit 2
fi
for row in "${candidate_rows[@]}"; do
    IFS=$'\t' read -r method stage run extra <<< "$row"
    if [[ -z "$method" || -z "$stage" || -z "$run" || -n "$extra" ]]; then
        echo "Malformed canonical RQ15 follow-up candidate row" >&2
        exit 2
    fi
done
if [[ "$list_only" -eq 1 ]]; then
    for row in "${candidate_rows[@]}"; do
        IFS=$'\t' read -r _ _ run <<< "$row"
        echo "$run"
    done
    exit 0
fi

source "${G1_RQ15_ARTIFACTS_LIBRARY:-$launcher_dir/../artifacts.sh}" || exit 1
source_run=""
checkpoint=""
for row in "${candidate_rows[@]}"; do
    IFS=$'\t' read -r method _ _ <<< "$row"
    [[ "$method" == "pretrained_finetune" ]] || continue
    source_config=experiments/g1_sasrec_item_ids_likes/configs/rq15_rq8_checkpoint_variant.py
    mapfile -t source_rows < <(
        python - <<'PY'
from experiments.g1_sasrec_item_ids_likes.analysis.rq15_training_candidates import source_candidates
for candidate in source_candidates():
    print(candidate.run_name, candidate.checkpoint_name)
PY
    )
    for source_row in "${source_rows[@]}"; do
        read -r candidate_source checkpoint_name <<< "$source_row"
        directory="$logs/$candidate_source"
        if [[ ! -f "$directory/$checkpoint_name" ]] || \
            ! g1_verify_config_recipe_artifact "$directory" "$source_config" \
                "G1_RQ15_SOURCE_RUN=$candidate_source"; then
            echo "Missing or incompatible RQ15 source artifact: $candidate_source" >&2
            g1_stop_artifact_verifier
            exit 2
        fi
    done
    g1_stop_artifact_verifier
    read -r source_run checkpoint < <(
        python - "$logs" <<'PY'
from pathlib import Path
import sys

from experiments.g1_sasrec_item_ids_likes.analysis.rq15_training_candidates import selected_source_candidate

logs = Path(sys.argv[1])
candidate = selected_source_candidate(logs)
checkpoint = candidate.checkpoint_path(logs)
print(candidate.run_name, checkpoint)
PY
    ) || exit 2
    if [[ -n "${G1_RQ15_SOURCE_RUN:-}" && "$G1_RQ15_SOURCE_RUN" != "$source_run" ]]; then
        echo "G1_RQ15_SOURCE_RUN does not match the validation-selected source" >&2
        exit 2
    fi
    if [[ -n "${G1_RQ15_FIRST_STAGE_CHECKPOINT:-}" && "$G1_RQ15_FIRST_STAGE_CHECKPOINT" != "$checkpoint" ]]; then
        echo "G1_RQ15_FIRST_STAGE_CHECKPOINT does not match the validation-selected source" >&2
        exit 2
    fi
    break
done
for row in "${candidate_rows[@]}"; do
    IFS=$'\t' read -r method _ _ <<< "$row"
    if [[ "$method" == "pretrained_finetune" && ! -f "$checkpoint" ]]; then
        echo "Missing compatible first-stage checkpoint: $checkpoint" >&2
        exit 2
    fi
done

export G1_DATASET_SIZE=500m
export WANDB_MODE=${WANDB_MODE:-offline}
export TRAINING_QUEUE_MONITOR_LIGHT_GPUS=${TRAINING_QUEUE_MONITOR_LIGHT_GPUS:-1}
unset G1_MAX_USERS G1_MAX_EPOCHS G1_SEED G1_TRAIN_BATCH_SIZE
unset G1_VAL_BATCH_SIZE G1_VARIANT G1_RQ8_RUN G1_RQ15_RUN G1_RQ15_SOURCE_RUN

config=experiments/g1_sasrec_item_ids_likes/configs/rq15_decoder_training_variant.py
TRAINING_QUEUE_SCRIPT=$config
source "${G1_TRAINING_QUEUE_LIBRARY:-utils/training_queue/queue.sh}" || exit 1

enqueued_count=0
skipped_count=0
for row in "${candidate_rows[@]}"; do
    IFS=$'\t' read -r method stage run <<< "$row"
    verifier_args=("G1_RQ15_RUN=$run")
    if [[ "$method" == "pretrained_finetune" ]]; then
        verifier_args+=("G1_RQ15_SOURCE_RUN=$source_run")
        verifier_args+=("G1_RQ15_FIRST_STAGE_CHECKPOINT=$checkpoint")
    fi
    directory="$logs/$run"
    artifact_status=0
    g1_require_config_compatible_or_absent "$directory" "$config" \
        "${verifier_args[@]}" || artifact_status=$?
    if [[ "$artifact_status" -eq 0 ]]; then
        echo "=== skipped compatible $run ==="
        skipped_count=$((skipped_count + 1))
        continue
    fi
    [[ "$artifact_status" -eq 1 ]] || exit "$artifact_status"
    TRAINING_QUEUE_DATA_GROUP="g1-rq15-${method}-${stage}-500m-seq128"
    enqueue "$run" "${verifier_args[@]}" || exit 1
    enqueued_count=$((enqueued_count + 1))
done

echo "=== rq15 followups native-500M: enqueued=${enqueued_count}, skipped=${skipped_count} ==="
g1_stop_artifact_verifier
drain || exit 1
