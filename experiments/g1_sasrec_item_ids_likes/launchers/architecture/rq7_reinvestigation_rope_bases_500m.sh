#!/usr/bin/env bash
set -u

launcher_dir=$(cd "$(dirname "$0")" && pwd)
source "$launcher_dir/rq7_reinvestigation_stage.sh" || exit 1
g1_rq7_launch_stage rope_base_candidates 6 500m rope-base || exit $?
