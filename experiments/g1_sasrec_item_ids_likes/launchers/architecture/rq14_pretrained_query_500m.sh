#!/usr/bin/env bash
set -u

mode=initial
if [[ "${1:-}" == "--list" ]]; then
    mode=list
    shift
elif [[ "${1:-}" == "--followups" ]]; then
    mode=followups
    shift
fi
if [[ "$#" -ne 0 ]]; then
    echo "usage: $0 [--list|--followups]" >&2
    exit 2
fi

launcher_dir=$(cd "$(dirname "$0")" && pwd) || exit 1
repo_root=$(cd "$launcher_dir/../../../.." && pwd) || exit 1
cd "$repo_root" || exit 1

if [[ "$mode" == list ]]; then
    python - <<'PY'
from experiments.g1_sasrec_item_ids_likes.analysis.rq14_pretrained_candidates import launch_candidates
for candidate in launch_candidates():
    print(candidate.run_name)
PY
    exit $?
fi

logs=${G1_RQ14_PRETRAINED_LOGS:-$repo_root/generated/logs}
experiment_dir=experiments/g1_sasrec_item_ids_likes
rq15_results=$experiment_dir/evidence/rq15_training_results.json
candidate_only=$experiment_dir/scratchpad/rq14_query_memory_reader_500m.md
correctness=$experiment_dir/evidence/rq14_pretrained_correctness.json

if [[ "$mode" == followups ]]; then
    python -m experiments.g1_sasrec_item_ids_likes.analysis.rq14_pretrained_report \
        --logs "$logs" \
        --rq15-results "$rq15_results" \
        --candidate-only "$candidate_only" \
        --correctness "$correctness" \
        --evidence "$experiment_dir/evidence" \
        --scratchpad "$experiment_dir/scratchpad" >/dev/null || exit 2
fi

mapfile -t candidate_rows < <(
    python - "$mode" "$experiment_dir/evidence/rq14_pretrained_results.json" <<'PY'
import json
from pathlib import Path
import sys

from experiments.g1_sasrec_item_ids_likes.analysis.rq14_pretrained_candidates import candidate_by_run, launch_candidates

if sys.argv[1] == "initial":
    candidates = launch_candidates()
else:
    evidence = json.loads(Path(sys.argv[2]).read_text())
    names = evidence.get("required_boundary_followups")
    if not isinstance(names, list) or not names:
        raise SystemExit("pretrained RQ14 has no required boundary followups")
    if evidence.get("required_followups") != names:
        raise SystemExit("pretrained RQ14 still has non-boundary followups")
    candidates = tuple(candidate_by_run(name) for name in names)
    if any(candidate.stage != "lr_boundary" for candidate in candidates):
        raise SystemExit("pretrained RQ14 evidence contains a non-boundary followup")
for candidate in candidates:
    print(f"{candidate.run_name}\t{candidate.treatment}")
PY
) || exit 2
if [[ "$mode" == initial && "${#candidate_rows[@]}" -ne 9 ]]; then
    echo "Pretrained RQ14 initial launcher must contain exactly nine new cells" >&2
    exit 2
fi

source "$launcher_dir/../artifacts.sh" || exit 1
source_config=$experiment_dir/configs/rq15_rq8_checkpoint_variant.py
mapfile -t source_rows < <(
    python - <<'PY'
from experiments.g1_sasrec_item_ids_likes.analysis.rq15_training_candidates import source_candidates
for candidate in source_candidates():
    print(candidate.run_name, candidate.checkpoint_name)
PY
)
for row in "${source_rows[@]}"; do
    read -r source_candidate checkpoint_name <<< "$row"
    directory=$logs/$source_candidate
    if [[ ! -f "$directory/$checkpoint_name" ]] || \
        ! g1_verify_config_recipe_artifact "$directory" "$source_config" \
            "G1_RQ15_SOURCE_RUN=$source_candidate"; then
        echo "Missing or incompatible RQ15 source artifact: $source_candidate" >&2
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
candidate = selected_source_candidate(Path(sys.argv[1]))
print(candidate.run_name, candidate.checkpoint_path(Path(sys.argv[1])))
PY
) || exit 2

python - "$logs" "$rq15_results" "$candidate_only" "$checkpoint" <<'PY'
from pathlib import Path
import sys
from experiments.g1_sasrec_item_ids_likes.analysis.rq14_pretrained_report import collect_report_bundle, validate_selected_checkpoint
bundle = collect_report_bundle(
    Path(sys.argv[1]),
    rq15_results_path=Path(sys.argv[2]),
    candidate_only_path=Path(sys.argv[3]),
)
if len(bundle.evidence["reused_rq15_cells"]) != 3:
    raise SystemExit("the three exact RQ15 reuse artifacts are not compatible")
validate_selected_checkpoint(bundle.evidence, Path(sys.argv[4]))
PY

export G1_DATASET_SIZE=500m
export WANDB_MODE=${WANDB_MODE:-offline}
export TRAINING_QUEUE_MONITOR_LIGHT_GPUS=${TRAINING_QUEUE_MONITOR_LIGHT_GPUS:-1}
unset G1_MAX_USERS G1_MAX_EPOCHS G1_SEED G1_TRAIN_BATCH_SIZE
unset G1_VAL_BATCH_SIZE G1_VARIANT G1_RQ14_PRETRAINED_RUN

config=$experiment_dir/configs/rq14_pretrained_query_variant.py
TRAINING_QUEUE_SCRIPT=$config
source "${G1_TRAINING_QUEUE_LIBRARY:-utils/training_queue/queue.sh}" || exit 1

enqueued_count=0
skipped_count=0
for row in "${candidate_rows[@]}"; do
    IFS=$'\t' read -r run treatment extra <<< "$row"
    if [[ -z "$run" || -z "$treatment" || -n "$extra" ]]; then
        echo "Malformed pretrained RQ14 candidate row" >&2
        exit 2
    fi
    verifier_args=(
        "G1_RQ14_PRETRAINED_RUN=$run"
        "G1_RQ15_SOURCE_RUN=$source_run"
        "G1_RQ15_FIRST_STAGE_CHECKPOINT=$checkpoint"
    )
    artifact_status=0
    g1_require_config_compatible_or_absent "$logs/$run" "$config" \
        "${verifier_args[@]}" || artifact_status=$?
    if [[ "$artifact_status" -eq 0 ]]; then
        skipped_count=$((skipped_count + 1))
        continue
    fi
    [[ "$artifact_status" -eq 1 ]] || exit "$artifact_status"
    TRAINING_QUEUE_DATA_GROUP="g1-rq14-pretrained-${treatment}-500m-seq128"
    enqueue "$run" "${verifier_args[@]}" || exit 1
    enqueued_count=$((enqueued_count + 1))
done

echo "=== pretrained RQ14 ${mode}: enqueued=${enqueued_count}, skipped=${skipped_count} ==="
g1_stop_artifact_verifier
drain || exit 1
