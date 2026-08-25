#!/usr/bin/env bash
set -u

launcher_dir=$(cd "$(dirname "$0")" && pwd)
source "$launcher_dir/rq7_reinvestigation_stage.sh" || exit 1
g1_rq7_launch_stage bounded_reverse_diagnostic_candidates 4 50m bounded-reverse-diagnostic || exit $?
