# Queues enough preprocessing to cover every training slot on every idle GPU.

_queue_depth_script=${BASH_SOURCE[0]%/*}/queue_depth.py
_gpu_check_script=${BASH_SOURCE[0]%/*}/gpu_check.py
_queue_service_script=${BASH_SOURCE[0]%/*}/service.py
_queue_service_state=${TRAINING_QUEUE_SERVICE_STATE_DIR:-generated/training-queue-service}
_queue_timing_index=${_queue_service_state}/timing-history.json
_run_lock_dir=generated/logs/.run-locks
_gpu_check_evidence_dir=${TRAINING_QUEUE_GPU_CHECK_EVIDENCE_DIR:-generated/training-queue-gpu-checks}
_queue_control_python=${TRAINING_QUEUE_CONTROL_PYTHON:-python3}

if [ -z "${TRAINING_QUEUE_SERVICE_CHILD:-}" ] \
    && [ -f "${_queue_service_state}/status.json" ] \
    && python3 "$_queue_service_script" \
        --state-dir "$_queue_service_state" status --json >/dev/null 2>&1; then
    if [ -z "${TRAINING_QUEUE_SCRIPT:-}" ]; then
        echo "TRAINING_QUEUE_SCRIPT must name the experiment script" >&2
        return 1
    fi

    _validate_forwarded_name() {
        local name=$1
        local upper=${name^^}
        if ! [[ "$name" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]]; then
            echo "Invalid forwarded variable name: ${name}" >&2
            return 1
        fi
        case $upper in
            TRAINING_QUEUE_*|DCN_GPU_*|DCN_RUN_*|DCN_PREPARED_MARKER|\
                DCN_TRAINING_RELEASE|DCN_RUNNER_DATA_READY|CUDA_VISIBLE_DEVICES)
                echo "Cannot persist queue-internal variable: ${name}" >&2
                return 1
                ;;
        esac
        if [[ "$upper" =~ (^|_)(TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?|COOKIE)($|_) \
            || "$upper" =~ (^|_)((API|PRIVATE)_KEY|(ACCESS|AUTH|BEARER|REFRESH|ID|OAUTH)_TOKEN)($|_) ]]; then
            case $upper in
                BEGINNING_TOKEN|CLS_TOKEN|PADDING_TOKEN) ;;
                *)
                    echo "Cannot persist secret-like variable: ${name}" >&2
                    return 1
                    ;;
            esac
        fi
    }

    for name in ${TRAINING_QUEUE_FORWARD_ENV:-}; do
        _validate_forwarded_name "$name" || return 1
    done
    for name in ${TRAINING_QUEUE_REQUIRED_FORWARD_ENV:-}; do
        _validate_forwarded_name "$name" || return 1
        if [[ " ${TRAINING_QUEUE_FORWARD_ENV:-} " != *" ${name} "* ]]; then
            echo "Required variable is not registered for forwarding: ${name}" >&2
            return 1
        fi
        if [ ! -v "$name" ] || [ -z "${!name}" ]; then
            echo "Required forwarded variable is unset: ${name}" >&2
            return 1
        fi
    done

    _queue_service_batch=$(python3 "$_queue_service_script" \
        --state-dir "$_queue_service_state" new-batch) || return 1
    _queue_service_sealed=0

    enqueue() {
        local run=$1
        shift
        local -a forwarded=()
        local name
        for name in ${TRAINING_QUEUE_FORWARD_ENV:-}; do
            if [ -v "$name" ]; then
                forwarded+=("${name}=${!name}")
            fi
        done
        if [ -v PYTORCH_CUDA_ALLOC_CONF ]; then
            forwarded+=("PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}")
        fi
        forwarded+=("$@")
        python3 "$_queue_service_script" --state-dir "$_queue_service_state" \
            enqueue-run --batch "$_queue_service_batch" \
            --script "$TRAINING_QUEUE_SCRIPT" --run "$run" \
            --data-group "${TRAINING_QUEUE_DATA_GROUP:-}" -- "${forwarded[@]}"
    }

    _seal_service_batch() {
        [ "$_queue_service_sealed" -eq 0 ] || return 0
        python3 "$_queue_service_script" --state-dir "$_queue_service_state" \
            seal-batch "$_queue_service_batch" || return 1
        _queue_service_sealed=1
    }

    drain() {
        _seal_service_batch || return 1
        python3 "$_queue_service_script" --state-dir "$_queue_service_state" \
            wait-batch "$_queue_service_batch"
    }

    return 0
fi

if ! command -v flock >/dev/null 2>&1; then
    echo "The training queue requires flock for per-run ownership" >&2
    return 1
fi
mkdir -p "$_run_lock_dir" || return 1

_refresh_queue_depth() {
    local measured
    local gpu_count=${#_gpu_uuids[@]}
    local experiment_script=${1:-${_queue_depth_experiment_script:-$TRAINING_QUEUE_SCRIPT}}
    local data_group=${2:-${_queue_depth_data_group:-${TRAINING_QUEUE_DATA_GROUP:-}}}
    [ "$gpu_count" -gt 0 ] || gpu_count=1
    if [ -n "${TRAINING_QUEUE_IN_FLIGHT:-}" ]; then
        measured=$TRAINING_QUEUE_IN_FLIGHT
    elif [ "${_runs_per_gpu:-1}" -gt 1 ]; then
        measured=1
    elif ! measured=$("$_queue_control_python" "$_queue_depth_script" \
        --gpu-count "$gpu_count" \
        --max-depth "${TRAINING_QUEUE_MAX_IN_FLIGHT:-4}" \
        --history-root generated/logs \
        --history-index "$_queue_timing_index" \
        --service-state "$_queue_service_state" \
        --script "$experiment_script" \
        --data-group "$data_group"); then
        echo "Could not calculate the preprocessing queue depth" >&2
        return 1
    fi
    if ! [[ "$measured" =~ ^[1-9][0-9]*$ ]]; then
        echo "TRAINING_QUEUE_IN_FLIGHT must be a positive integer" >&2
        return 1
    fi
    if [ -z "${_queue_depth:-}" ]; then
        echo "=== queue depth per GPU: ${measured} ($((measured - 1)) preprocessing ahead) ==="
    elif [ "$measured" != "$_queue_depth" ]; then
        echo "=== queue depth per GPU adjusted from ${_queue_depth} to ${measured} ==="
    fi
    _queue_depth=$measured
    _queue_depth_experiment_script=$experiment_script
    _queue_depth_data_group=$data_group
}

if [ -z "${TRAINING_QUEUE_SCRIPT:-}" ]; then
    echo "TRAINING_QUEUE_SCRIPT must name the experiment script" >&2
    return 1
fi

if ! _gpu_inventory=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits); then
    echo "Could not inspect GPUs with nvidia-smi" >&2
    return 1
fi
if ! _busy_gpu_uuids=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits); then
    echo "Could not inspect running GPU processes with nvidia-smi" >&2
    return 1
fi

_gpu_requested() {
    local candidate=$1
    local requested
    [ -z "${TRAINING_QUEUE_GPUS:-}" ] && return 0
    for requested in $TRAINING_QUEUE_GPUS; do
        [ "$candidate" = "$requested" ] && return 0
    done
    return 1
}

_gpu_busy() {
    local candidate=$1
    local busy
    while read -r busy; do
        [ "$candidate" = "$busy" ] && return 0
    done <<< "$_busy_gpu_uuids"
    return 1
}

_gpu_uuids=()
_all_gpu_uuids=()
declare -A _gpu_index=()
declare -A _gpu_state=()
while IFS=', ' read -r gpu uuid; do
    _gpu_requested "$gpu" || continue
    _all_gpu_uuids+=("$uuid")
    _gpu_index[$uuid]=$gpu
    if _gpu_busy "$uuid"; then
        _gpu_state[$uuid]=busy
    else
        _gpu_uuids+=("$uuid")
        _gpu_state[$uuid]=available
    fi
done <<< "$_gpu_inventory"

_monitor_light_gpus=${TRAINING_QUEUE_MONITOR_LIGHT_GPUS:-0}
_gpu_retry_seconds=${TRAINING_QUEUE_GPU_RETRY_SECONDS:-60}
_gpu_recheck_seconds=${TRAINING_QUEUE_GPU_RECHECK_SECONDS:-600}
if [ "$_monitor_light_gpus" != 0 ] && [ "$_monitor_light_gpus" != 1 ]; then
    echo "TRAINING_QUEUE_MONITOR_LIGHT_GPUS must be 0 or 1" >&2
    return 1
fi
if ! [[ "$_gpu_recheck_seconds" =~ ^[0-9]+$ ]]; then
    echo "TRAINING_QUEUE_GPU_RECHECK_SECONDS must be a non-negative integer" >&2
    return 1
fi
if ! [[ "$_gpu_retry_seconds" =~ ^[0-9]+$ ]]; then
    echo "TRAINING_QUEUE_GPU_RETRY_SECONDS must be a non-negative integer" >&2
    return 1
fi
if [ "$_monitor_light_gpus" = 1 ]; then
    mkdir -p "$_gpu_check_evidence_dir" || return 1
fi
if [ "${#_gpu_uuids[@]}" -eq 0 ] && [ "$_monitor_light_gpus" != 1 ]; then
    echo "No idle GPUs available${TRAINING_QUEUE_GPUS:+ from TRAINING_QUEUE_GPUS=${TRAINING_QUEUE_GPUS}}" >&2
    return 1
fi

_runs_per_gpu=${TRAINING_QUEUE_RUNS_PER_GPU:-1}
if [ "$_runs_per_gpu" != 1 ]; then
    echo "The training queue allows one simultaneous training run per GPU" >&2
    return 1
fi
_cpu_threads_per_run=${TRAINING_QUEUE_CPU_THREADS_PER_RUN:-}
if [ -n "$_cpu_threads_per_run" ] \
    && ! [[ "$_cpu_threads_per_run" =~ ^[1-9][0-9]*$ ]]; then
    echo "TRAINING_QUEUE_CPU_THREADS_PER_RUN must be a positive integer" >&2
    return 1
fi

_initial_gpu_uuids=("${_gpu_uuids[@]}")
_gpu_uuids=()
_worker_uuids=()
_worker_slots=()
declare -A _worker_running=()
declare -A _gpu_admitted=()
declare -A _gpu_running=()
declare -A _gpu_next_check=()

for uuid in "${_all_gpu_uuids[@]}"; do
    _gpu_admitted[$uuid]=0
    _gpu_running[$uuid]=0
    _gpu_next_check[$uuid]=0
done

_admit_gpu() {
    local uuid=$1
    local slot
    local worker
    _gpu_state[$uuid]=available
    _gpu_next_check[$uuid]=$(($(date +%s) + _gpu_recheck_seconds))
    if [ "${_gpu_admitted[$uuid]}" -eq 0 ]; then
        _gpu_admitted[$uuid]=1
        _gpu_uuids+=("$uuid")
        for ((slot = 0; slot < _runs_per_gpu; slot++)); do
            worker=${#_worker_uuids[@]}
            _worker_uuids+=("$uuid")
            _worker_slots+=("$slot")
            _worker_running[$worker]=0
        done
    fi
}

for uuid in "${_initial_gpu_uuids[@]}"; do
    _gpu_state[$uuid]=pending
    _admit_gpu "$uuid"
done

_refresh_queue_depth || return 1

_barrier_dir=generated/.training-queue-${BASHPID}
_training_release=${_barrier_dir}/start
mkdir -p "$_barrier_dir"
_barrier_released=0
_next_barrier_slot=0

_release_training() {
    if [ "$_barrier_released" -eq 0 ]; then
        touch "$_training_release"
        _barrier_released=1
        echo "=== training released with $(_prepared_count) prepared run(s) ==="
    fi
}
trap _release_training EXIT
if [ "$_runs_per_gpu" -gt 1 ]; then
    _release_training
fi

declare -A _gpu_check_pid=()
declare -A _gpu_check_log=()
declare -A _gpu_check_error=()
_periodic_check_gpu=
_next_recheck_index=0
_draining_queue=0
_runs_enqueued=0

_job_is_running() {
    local target=$1
    local active
    for active in $(jobs -pr); do
        [ "$target" = "$active" ] && return 0
    done
    return 1
}

_start_gpu_check() {
    local uuid=$1
    local message=${2:-monitoring lightly used}
    local gpu=${_gpu_index[$uuid]}
    local gate_lock=generated/gpu-${uuid}.lock
    local -a command=(
        python3 "$_gpu_check_script" --gpu "$uuid"
        --duration "${TRAINING_QUEUE_GPU_CHECK_SECONDS:-30}"
        --interval "${TRAINING_QUEUE_GPU_SAMPLE_SECONDS:-1}"
    )
    [ "${_gpu_state[$uuid]}" = checking ] && return
    if ! mkdir -p "$_gpu_check_evidence_dir"; then
        echo "Could not create GPU check evidence directory: ${_gpu_check_evidence_dir}" >&2
        _run_failed=1
        return 1
    fi
    _gpu_state[$uuid]=checking
    echo "=== ${message} GPU ${gpu} ==="
    if [ "$message" = rechecking ]; then
        _gpu_check_log[$uuid]=
        local error_marker=${_barrier_dir}/${uuid}.gpu-evidence-error
        _gpu_check_error[$uuid]=$error_marker
        (
            check_child=
            attempt_log=
            _stop_periodic_check() {
                if [ -n "$check_child" ] && kill -0 "$check_child" 2>/dev/null; then
                    kill "$check_child" 2>/dev/null || true
                    wait "$check_child" 2>/dev/null || true
                fi
                if [ -n "$attempt_log" ] && [ -e "$attempt_log" ]; then
                    rm -- "$attempt_log"
                fi
                exit 143
            }
            trap _stop_periodic_check TERM INT
            flock -x 9 || exit 1
            while true; do
                local safe_uuid=${uuid//[^a-zA-Z0-9_.-]/_}
                attempt_log=${_gpu_check_evidence_dir}/gpu-check-${safe_uuid}-$(date +%s%N)-${BASHPID}.json
                sleep "${TRAINING_QUEUE_GPU_SETTLE_SECONDS:-2}" 9>&- &
                check_child=$!
                wait "$check_child" || exit
                check_child=
                "${command[@]}" 9>&- > "$attempt_log" 2>&1 &
                check_child=$!
                if wait "$check_child"; then
                    check_child=
                    rm -- "$attempt_log"
                    break
                fi
                check_child=
                if [ ! -s "$attempt_log" ]; then
                    echo "Could not persist rejected GPU check evidence: ${attempt_log}" >&2
                    touch "$error_marker"
                    exit 1
                fi
                attempt_log=
                sleep "$_gpu_retry_seconds" 9>&- &
                check_child=$!
                wait "$check_child" || exit
                check_child=
            done
        ) 9> "$gate_lock" &
    else
        local safe_uuid=${uuid//[^a-zA-Z0-9_.-]/_}
        local log=${_gpu_check_evidence_dir}/gpu-check-${safe_uuid}-$(date +%s%N)-${BASHPID}.json
        _gpu_check_log[$uuid]=$log
        _gpu_check_error[$uuid]=
        "${command[@]}" > "$log" 2>&1 &
    fi
    _gpu_check_pid[$uuid]=$!
}

_sync_gpu_checks() {
    local uuid
    local pid
    for uuid in "${_all_gpu_uuids[@]}"; do
        [ "${_gpu_state[$uuid]}" = checking ] || continue
        pid=${_gpu_check_pid[$uuid]}
        _job_is_running "$pid" && continue
        if wait "$pid" 2>/dev/null; then
            unset '_gpu_check_pid[$uuid]'
            if [ -n "${_gpu_check_log[$uuid]:-}" ]; then
                rm -- "${_gpu_check_log[$uuid]}"
            fi
            unset '_gpu_check_log[$uuid]'
            _gpu_state[$uuid]=pending
            _admit_gpu "$uuid"
            _refresh_queue_depth
            echo "=== admitted lightly used GPU ${_gpu_index[$uuid]} ==="
        else
            unset '_gpu_check_pid[$uuid]'
            if [ -n "${_gpu_check_error[$uuid]:-}" ] \
                && [ -e "${_gpu_check_error[$uuid]}" ]; then
                _run_failed=1
                rm -- "${_gpu_check_error[$uuid]}"
            fi
            if [ -n "${_gpu_check_log[$uuid]:-}" ] \
                && [ ! -s "${_gpu_check_log[$uuid]}" ]; then
                echo "Could not persist rejected GPU check evidence: ${_gpu_check_log[$uuid]}" >&2
                _run_failed=1
            fi
            unset '_gpu_check_log[$uuid]'
            _gpu_state[$uuid]=rejected
            _gpu_next_check[$uuid]=$(($(date +%s) + _gpu_retry_seconds))
            echo "=== kept GPU ${_gpu_index[$uuid]} excluded ==="
        fi
        unset '_gpu_check_error[$uuid]'
        if [ "$_periodic_check_gpu" = "$uuid" ]; then
            _periodic_check_gpu=
        fi
    done
}

_maybe_recheck_gpus() {
    [ "$_monitor_light_gpus" = 1 ] || return
    [ "$_draining_queue" -eq 0 ] || return
    local now=$(date +%s)
    local uuid
    local attempt
    local index

    for uuid in "${_all_gpu_uuids[@]}"; do
        if [ "${_gpu_state[$uuid]}" = rejected ] \
            && [ "${_gpu_next_check[$uuid]}" -le "$now" ]; then
            _start_gpu_check "$uuid"
        fi
    done

    [ "$_runs_enqueued" -gt 0 ] || return
    if [ -n "$_periodic_check_gpu" ]; then
        return
    fi

    for ((attempt = 0; attempt < ${#_all_gpu_uuids[@]}; attempt++)); do
        index=$(((_next_recheck_index + attempt) % ${#_all_gpu_uuids[@]}))
        uuid=${_all_gpu_uuids[$index]}
        if [ "${_gpu_admitted[$uuid]}" -eq 1 ] \
            && [ "${_gpu_state[$uuid]}" = available ] \
            && [ "${_gpu_next_check[$uuid]}" -le "$now" ]; then
            _gpu_state[$uuid]=draining
            _periodic_check_gpu=$uuid
            _next_recheck_index=$(((index + 1) % ${#_all_gpu_uuids[@]}))
            echo "=== draining GPU ${_gpu_index[$uuid]} for recheck ==="
            _start_gpu_check "$uuid" rechecking
            return
        fi
    done
}

_has_pending_monitored_gpu() {
    local uuid
    [ "$_monitor_light_gpus" = 1 ] || return 1
    for uuid in "${_all_gpu_uuids[@]}"; do
        case ${_gpu_state[$uuid]} in
            checking|draining|rejected) return 0 ;;
        esac
    done
    return 1
}

_has_empty_gpu_check() {
    local uuid
    [ "$_monitor_light_gpus" = 1 ] || return 1
    for uuid in "${_all_gpu_uuids[@]}"; do
        if [ "${_gpu_state[$uuid]}" = checking ] \
            && [ "${_gpu_running[$uuid]}" -eq 0 ]; then
            return 0
        fi
    done
    return 1
}

if [ "$_monitor_light_gpus" = 1 ]; then
    for uuid in "${_all_gpu_uuids[@]}"; do
        [ "${_gpu_state[$uuid]}" = busy ] && _start_gpu_check "$uuid"
    done
fi

_remove_barrier() {
    local path
    for path in "$_barrier_dir"/*.ready "$_barrier_dir"/*.failed \
        "$_training_release"; do
        [ -e "$path" ] && rm -- "$path"
    done
    for path in "$_barrier_dir"/runner-data-*.pickle \
        "$_barrier_dir"/runner-data-*.lock \
        "$_barrier_dir"/runner-data-*.ready \
        "$_barrier_dir"/runner-data-*.tmp; do
        [ -e "$path" ] && rm -- "$path"
    done
    rmdir -- "$_barrier_dir"
}

_prepared_count() {
    local markers=("$_barrier_dir"/[0-9]*.ready)
    if [ -e "${markers[0]}" ]; then
        echo "${#markers[@]}"
    else
        echo 0
    fi
}

_failed_count() {
    local markers=("$_barrier_dir"/[0-9]*.failed)
    if [ -e "${markers[0]}" ]; then
        echo "${#markers[@]}"
    else
        echo 0
    fi
}

_wait_for_prepared() {
    local expected=$1
    while [ "$(($(_prepared_count) + $(_failed_count)))" -lt "$expected" ]; do
        sleep 0.1
    done
    if [ "$(_failed_count)" -gt 0 ]; then
        echo "A run failed before the preprocessing buffer was ready" >&2
        _release_training
        return 1
    fi
    _release_training
}

_release_when_full() {
    [ "$_barrier_released" -eq 1 ] && return
    local capacity=$((${#_worker_uuids[@]} * _queue_depth))
    if [ "$_running" -ge "$capacity" ]; then
        _wait_for_prepared 1 || return 1
    fi
    return 0
}

declare -A _pid_worker=()
declare -A _pid_service_job=()
declare -A _pid_service_result_dir=()
_next_worker_index=0
_running=0
_run_failed=0

_reap_finished() {
    local finished_pid=
    local pid
    local active_pid
    local is_running
    local worker
    local exit_code=0
    local running_pids
    _sync_gpu_checks
    running_pids=$(jobs -pr)
    for pid in "${!_pid_worker[@]}"; do
        is_running=0
        while read -r active_pid; do
            [ "$pid" = "$active_pid" ] && is_running=1
        done <<< "$running_pids"
        if [ "$is_running" -eq 0 ]; then
            wait "$pid" || exit_code=$?
            if [ "$exit_code" -ne 0 ]; then
                _run_failed=1
            fi
            finished_pid=$pid
            break
        fi
    done
    [ -n "$finished_pid" ] || return 1
    worker=${_pid_worker[$finished_pid]}
    _worker_running[$worker]=$((_worker_running[$worker] - 1))
    local uuid=${_worker_uuids[$worker]}
    _gpu_running[$uuid]=$((${_gpu_running[$uuid]} - 1))
    local service_job_id=${_pid_service_job[$finished_pid]:-}
    local service_result_dir=${_pid_service_result_dir[$finished_pid]:-}
    if [ -n "$service_job_id" ]; then
        local result_tmp=${service_result_dir}/.${service_job_id}.$BASHPID.tmp
        printf '%s\n' "$exit_code" > "$result_tmp"
        mv -- "$result_tmp" "${service_result_dir}/${service_job_id}.result"
    fi
    unset '_pid_worker[$finished_pid]'
    unset '_pid_service_job[$finished_pid]'
    unset '_pid_service_result_dir[$finished_pid]'
    _running=$((_running - 1))
    _refresh_queue_depth
    _maybe_recheck_gpus
    return 0
}

_reap_one() {
    while ! _reap_finished; do
        sleep 0.1
    done
}

_select_available_worker() {
    local require_empty=$1
    local attempt
    local index
    for ((attempt = 0; attempt < ${#_worker_uuids[@]}; attempt++)); do
        index=$(((_next_worker_index + attempt) % ${#_worker_uuids[@]}))
        [ "${_gpu_state[${_worker_uuids[$index]}]}" = available ] \
            || continue
        [ "${_worker_running[$index]}" -lt "$_queue_depth" ] \
            || continue
        if [ "$require_empty" -eq 1 ] \
            && [ "${_worker_running[$index]}" -ne 0 ]; then
            continue
        fi
        _selected_worker=$index
        _next_worker_index=$(((index + 1) % ${#_worker_uuids[@]}))
        return 0
    done
    return 1
}

_select_worker() {
    while true; do
        _sync_gpu_checks
        while [ "$_running" -gt 0 ] && _reap_finished; do
            :
        done
        _maybe_recheck_gpus
        _select_available_worker 1 && return
        if _has_empty_gpu_check; then
            if [ "$_running" -gt 0 ]; then
                _reap_finished || sleep 0.1
            else
                sleep 0.1
            fi
            continue
        fi
        _select_available_worker 0 && return
        if [ "$_running" -gt 0 ]; then
            _reap_finished || sleep 0.1
        elif _has_pending_monitored_gpu; then
            sleep 0.1
        else
            echo "No eligible GPUs available after monitoring" >&2
            return 1
        fi
    done
}

_enqueue_script() {
    local experiment_script=$1
    local service_job_id=$2
    local service_result_dir=$3
    local data_group=$4
    shift 4
    local run=$1
    shift
    local worker
    local gpu_uuid
    local gpu_index
    local gpu_slot
    local failed_marker
    local prepared_marker
    local runner_data_ready=
    local log="generated/logs/${run}/sweep.log"
    local run_lock="${_run_lock_dir}/${run}.lock"
    _refresh_queue_depth "$experiment_script" "$data_group" || return 1
    _select_worker || return 1
    worker=$_selected_worker
    gpu_uuid=${_worker_uuids[$worker]}
    gpu_index=${_gpu_index[$gpu_uuid]}
    gpu_slot=${_worker_slots[$worker]}
    prepared_marker=${_barrier_dir}/${_next_barrier_slot}.ready
    failed_marker=${_barrier_dir}/${_next_barrier_slot}.failed
    if [ -n "$data_group" ]; then
        local safe_data_group=${data_group//[^a-zA-Z0-9_.-]/_}
        runner_data_ready=${_barrier_dir}/runner-data-${safe_data_group}.ready
    fi
    _next_barrier_slot=$((_next_barrier_slot + 1))
    mkdir -p "$(dirname "$log")"
    echo "=== ${run} queued on GPU ${gpu_index}, slot ${gpu_slot} $(date +%H:%M:%S) ==="
    (
        local run_lock_fd
        exec {run_lock_fd}>"$run_lock" || exit 1
        if ! flock -n "$run_lock_fd"; then
            echo "!!! ${run} is already owned by another process"
            touch "$failed_marker"
            exit 1
        fi
        local -a cpu_environment=()
        if [ -n "$_cpu_threads_per_run" ]; then
            cpu_environment=(
                "OMP_NUM_THREADS=$_cpu_threads_per_run"
                "MKL_NUM_THREADS=$_cpu_threads_per_run"
                "OPENBLAS_NUM_THREADS=$_cpu_threads_per_run"
                "NUMEXPR_NUM_THREADS=$_cpu_threads_per_run"
                "POLARS_MAX_THREADS=$_cpu_threads_per_run"
            )
        fi
        local exit_code=0
        env \
            "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF-expandable_segments:True}" \
            "$@" "${cpu_environment[@]}" CUDA_VISIBLE_DEVICES="$gpu_uuid" \
            DCN_GPU_LOCK_DEVICE="$gpu_uuid" \
            DCN_GPU_LOCK_SLOT="$gpu_slot" \
            DCN_RUN_LOCK_FD="$run_lock_fd" \
            DCN_RUN_LOCK_PATH="$run_lock" \
            DCN_PREPARED_MARKER="$prepared_marker" \
            DCN_TRAINING_RELEASE="$_training_release" \
            DCN_RUNNER_DATA_READY="$runner_data_ready" python -m dcn.main \
            -s "$experiment_script" > "$log" 2>&1 || exit_code=$?
        if [ "$exit_code" -ne 0 ]; then
            echo "!!! ${run} failed, see ${log}"
            touch "$failed_marker"
        elif [ ! -e "$prepared_marker" ]; then
            touch "$failed_marker"
            exit_code=1
        fi
        if [ "$exit_code" -eq 0 ]; then
            if ! "$_queue_control_python" "$_queue_depth_script" \
                --record-timing-log "$log" \
                --history-index "$_queue_timing_index" \
                --script "$experiment_script" --data-group "$data_group"; then
                echo "Could not record queue timing history for ${run}" >&2
            fi
        fi
        exit "$exit_code"
    ) &
    local spawned_pid=$!
    _pid_worker[$spawned_pid]=$worker
    if [ -n "$service_job_id" ]; then
        _pid_service_job[$spawned_pid]=$service_job_id
        _pid_service_result_dir[$spawned_pid]=$service_result_dir
    fi
    _worker_running[$worker]=$((_worker_running[$worker] + 1))
    _gpu_running[$gpu_uuid]=$((${_gpu_running[$gpu_uuid]} + 1))
    _running=$((_running + 1))
    _runs_enqueued=$((_runs_enqueued + 1))
    _release_when_full
}

enqueue() {  # run name, then VAR=value assignments for the run
    _enqueue_script "$TRAINING_QUEUE_SCRIPT" "" "" \
        "${TRAINING_QUEUE_DATA_GROUP:-}" "$@"
}

_service_enqueue() {
    local experiment_script=$1
    local service_job_id=$2
    local service_result_dir=$3
    local data_group=$4
    shift 4
    local running_before=$_running
    local exit_code=0
    _enqueue_script "$experiment_script" "$service_job_id" \
        "$service_result_dir" "$data_group" "$@" || exit_code=$?
    if [ "$exit_code" -ne 0 ] && [ "$_running" -eq "$running_before" ]; then
        local result_tmp=${service_result_dir}/.${service_job_id}.$BASHPID.tmp
        printf '%s\n' "$exit_code" > "$result_tmp"
        mv -- "$result_tmp" "${service_result_dir}/${service_job_id}.result"
    fi
    touch "${_queue_service_state}/acks/${service_job_id}"
}

_service_reap() {
    while _reap_finished; do
        :
    done
}

drain() {
    _draining_queue=1
    if [ "$_barrier_released" -eq 0 ]; then
        if [ "$_running" -gt 0 ]; then
            _wait_for_prepared 1
        else
            _release_training
        fi
    fi
    while [ "$_running" -gt 0 ]; do
        _reap_one
    done
    local uuid
    local pid
    for uuid in "${!_gpu_check_pid[@]}"; do
        pid=${_gpu_check_pid[$uuid]}
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
        fi
        wait "$pid" 2>/dev/null || true
        if [ -n "${_gpu_check_log[$uuid]:-}" ] \
            && [ -e "${_gpu_check_log[$uuid]}" ] \
            && [ ! -s "${_gpu_check_log[$uuid]}" ]; then
            rm -- "${_gpu_check_log[$uuid]}"
        fi
        unset '_gpu_check_log[$uuid]'
        unset '_gpu_check_error[$uuid]'
    done
    _remove_barrier
    if [ "$_run_failed" -ne 0 ]; then
        echo "One or more queued runs failed" >&2
        return 1
    fi
}
