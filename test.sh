#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$root"

pytest_command=${PYTEST:-pytest}

if (( $# > 0 )); then
    CUDA_VISIBLE_DEVICES="" "$pytest_command" -m "not slow_gpu" "$@"
    exit $?
fi

jobs=${TEST_JOBS:-16}
if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo "TEST_JOBS must be a positive integer" >&2
    exit 2
fi

e2e_device=${TEST_E2E_DEVICE:-cpu}
if [[ "$e2e_device" != cpu && "$e2e_device" != gpu ]]; then
    echo "TEST_E2E_DEVICE must be cpu or gpu" >&2
    exit 2
fi

e2e_gpu=
if [[ "$e2e_device" == gpu ]]; then
    requested_gpu=${TEST_GPU:-}
    if [[ ! "$requested_gpu" =~ ^[0-9]+$ ]]; then
        echo "GPU E2E override needs a numeric TEST_GPU; using CPU" >&2
    elif ! command -v nvidia-smi >/dev/null; then
        echo "nvidia-smi is unavailable; using CPU for E2E tests" >&2
    elif ! gpu_uuid=$(
        nvidia-smi --id="$requested_gpu" --query-gpu=uuid --format=csv,noheader,nounits
    ) || [[ -z "$gpu_uuid" ]]; then
        echo "GPU $requested_gpu is unavailable; using CPU for E2E tests" >&2
    elif ! gpu_processes=$(
        nvidia-smi --id="$requested_gpu" --query-compute-apps=pid \
            --format=csv,noheader,nounits
    ); then
        echo "Could not inspect GPU $requested_gpu processes; using CPU" >&2
    elif [[ -n "$gpu_processes" ]]; then
        echo "GPU $requested_gpu has a foreign compute process; using CPU" >&2
    elif python3 utils/training_queue/gpu_check.py \
        --gpu "$gpu_uuid" \
        --duration "${TEST_GPU_CHECK_SECONDS:-30}" \
        --interval 1; then
        e2e_gpu=$requested_gpu
    else
        echo "GPU $requested_gpu did not stay lightly used; using CPU" >&2
    fi
fi

test_marker="not slow_gpu and not training_e2e"
test_item_output=$(
    CUDA_VISIBLE_DEVICES="" "$pytest_command" --collect-only -q -m "$test_marker"
)
mapfile -t test_items < <(
    printf '%s\n' "$test_item_output" | sed -n '/\.py::/p'
)
if (( ${#test_items[@]} == 0 )); then
    echo "No test items collected" >&2
    exit 5
fi

e2e_item_output=$(
    CUDA_VISIBLE_DEVICES="" "$pytest_command" --collect-only -q -m "training_e2e"
)
mapfile -t e2e_items < <(
    printf '%s\n' "$e2e_item_output" | sed -n '/\.py::/p'
)
if (( ${#e2e_items[@]} == 0 )); then
    echo "No training E2E test items collected" >&2
    exit 5
fi

if (( jobs > ${#test_items[@]} )); then
    jobs=${#test_items[@]}
fi

work_directory=$(mktemp -d "${TMPDIR:-/tmp}/competition-tests.XXXXXX")

cleanup() {
    local group
    [[ -n "${work_directory:-}" && -d "$work_directory" ]] || return
    for ((group = 0; group < jobs; group++)); do
        rm -f -- \
            "$work_directory/group-$group.paths" \
            "$work_directory/group-$group.log"
    done
    rm -f -- "$work_directory/e2e.log"
    rmdir -- "$work_directory"
}
trap cleanup EXIT

for ((group = 0; group < jobs; group++)); do
    : > "$work_directory/group-$group.paths"
done

for ((item_index = 0; item_index < ${#test_items[@]}; item_index++)); do
    group=$((item_index % jobs))
    printf '%s\n' "${test_items[item_index]}" >> "$work_directory/group-$group.paths"
done

echo "Running CPU tests in $jobs parallel groups"

declare -a process_ids=()
for ((group = 0; group < jobs; group++)); do
    mapfile -t group_files < <(sort "$work_directory/group-$group.paths")
    CUDA_VISIBLE_DEVICES="" TEST_EXPECTED_E2E_DEVICE=cpu \
        "$pytest_command" -q -m "$test_marker" \
        "${group_files[@]}" > "$work_directory/group-$group.log" 2>&1 &
    process_ids[group]=$!
done

e2e_process_id=
if [[ -n "$e2e_gpu" ]]; then
    echo "Running ${#e2e_items[@]} training E2E tests serially on GPU $e2e_gpu"
    gpu_gate_lock="generated/gpu-${gpu_uuid}.lock"
    mkdir -p -- "${gpu_gate_lock%/*}"
    (
        flock -x 9
        CUDA_VISIBLE_DEVICES="$e2e_gpu" TEST_EXPECTED_E2E_DEVICE=cuda \
            "$pytest_command" -q -m "training_e2e" "${e2e_items[@]}"
    ) 9> "$gpu_gate_lock" > "$work_directory/e2e.log" 2>&1 &
    e2e_process_id=$!
fi

status=0
for ((group = 0; group < jobs; group++)); do
    if ! wait "${process_ids[group]}"; then
        status=1
    fi
    echo "=== test group $((group + 1))/$jobs ==="
    sed -n '1,$p' "$work_directory/group-$group.log"
done

if [[ -n "$e2e_process_id" ]]; then
    if ! wait "$e2e_process_id"; then
        status=1
    fi
    echo "=== training E2E tests on GPU $e2e_gpu ==="
    sed -n '1,$p' "$work_directory/e2e.log"
else
    echo "Running ${#e2e_items[@]} training E2E tests serially on CPU"
    if ! CUDA_VISIBLE_DEVICES="" TEST_EXPECTED_E2E_DEVICE=cpu \
        "$pytest_command" -q -m "training_e2e" \
        "${e2e_items[@]}" > "$work_directory/e2e.log" 2>&1; then
        status=1
    fi
    echo "=== training E2E tests on CPU ==="
    sed -n '1,$p' "$work_directory/e2e.log"
fi

exit "$status"
