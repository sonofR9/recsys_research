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

logs=${G1_QUERY_LOGS:-$repo_root/generated/logs}
evidence_dir=${G1_QUERY_EVIDENCE_DIR:-experiments/g1_sasrec_item_ids_likes/evidence}
scratchpad_dir=${G1_QUERY_SCRATCHPAD_DIR:-experiments/g1_sasrec_item_ids_likes/scratchpad}
evidence=$evidence_dir/rq13_prefix_expansion_results.json
correctness_evidence=$evidence_dir/rq13_prefix_expansion_correctness.json
python -m experiments.g1_sasrec_item_ids_likes.analysis.rq13_prefix_expansion_report \
    --logs "$logs" --evidence "$evidence_dir" \
    --correctness-evidence "$correctness_evidence" \
    --scratchpad "$scratchpad_dir" >/dev/null || exit 2

audit_stage=$(python - "$evidence" <<'PY'
import json
from pathlib import Path
import sys

document = json.loads(Path(sys.argv[1]).read_text())
cap_fit = document.get("cap_fit")
if not isinstance(cap_fit, dict):
    raise SystemExit("RQ13 cap-fit state is absent")
status = cap_fit.get("status")
if status not in {"stage_one_audit_required", "selected_cap_pending"}:
    raise SystemExit(f"RQ13 selected-cap stage is not ready: {status}")
print(status)
PY
) || exit 2
if [[ "$audit_stage" == "stage_one_audit_required" ]]; then
    python -m experiments.g1_sasrec_item_ids_likes.analysis.rq13_prefix_expansion_audit \
        --logs "$logs" --results "$evidence" \
        --output "$correctness_evidence" >/dev/null || exit 2
    python -m experiments.g1_sasrec_item_ids_likes.analysis.rq13_prefix_expansion_report \
        --logs "$logs" --evidence "$evidence_dir" \
        --correctness-evidence "$correctness_evidence" \
        --scratchpad "$scratchpad_dir" >/dev/null || exit 2
fi

mapfile -t candidate_rows < <(
    python - "$evidence" "$correctness_evidence" <<'PY'
import json
from pathlib import Path
import sys

from experiments.g1_sasrec_item_ids_likes.analysis.rq13_prefix_expansion_audit import validate_bound_stage_one_audit
from experiments.g1_sasrec_item_ids_likes.analysis.rq13_rq14_query_candidates import validated_selected_cap_candidates

document = json.loads(Path(sys.argv[1]).read_text())
audit = json.loads(Path(sys.argv[2]).read_text())
validate_bound_stage_one_audit(document, audit)
required = document.get("required_followups")
if not isinstance(required, list):
    raise SystemExit("RQ13 result evidence has no exact followup list")
for candidate in validated_selected_cap_candidates(document, required):
    print(f"{candidate.run_name}\t{candidate.treatment}")
PY
)
if [[ "${#candidate_rows[@]}" -ne 3 ]]; then
    echo "RQ13 selected-cap stage must contain exactly three runs" >&2
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
    TRAINING_QUEUE_DATA_GROUP="g1-rq13-${treatment}-500m-seq128"
    enqueue "$run" "${verifier_args[@]}" || exit 1
    enqueued_count=$((enqueued_count + 1))
done

echo "=== rq13 selected-cap native-500M: enqueued=${enqueued_count}, skipped=${skipped_count} ==="
g1_stop_artifact_verifier
drain || exit 1
