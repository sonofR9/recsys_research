#!/usr/bin/env bash
set -u

launcher_dir=$(cd "$(dirname "$0")" && pwd)
repo_root=$(cd "$launcher_dir/../../../.." && pwd) || exit 1
logs=${G1_RQ7_LOGS:-$repo_root/generated/logs}
cd "$repo_root" || exit 1
python -m experiments.g1_sasrec_item_ids_likes.analysis.rq7_reinvestigation_selection \
    --logs "$logs" --require-diagnostic-gate || exit $?
source "$launcher_dir/rq7_reinvestigation_stage.sh" || exit 1
g1_rq7_launch_stage initial_candidates 36 500m initial || exit $?
