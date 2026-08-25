#!/usr/bin/env bash

_g1_artifact_launcher_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

g1_register_queue_environment() {
    local name
    for name in G1_DATASET_SIZE WANDB_MODE; do
        if [[ " ${TRAINING_QUEUE_FORWARD_ENV:-} " != *" ${name} "* ]]; then
            TRAINING_QUEUE_FORWARD_ENV="${TRAINING_QUEUE_FORWARD_ENV:+${TRAINING_QUEUE_FORWARD_ENV} }${name}"
        fi
        if [[ " ${TRAINING_QUEUE_REQUIRED_FORWARD_ENV:-} " != *" ${name} "* ]]; then
            TRAINING_QUEUE_REQUIRED_FORWARD_ENV="${TRAINING_QUEUE_REQUIRED_FORWARD_ENV:+${TRAINING_QUEUE_REQUIRED_FORWARD_ENV} }${name}"
        fi
    done
    export TRAINING_QUEUE_FORWARD_ENV TRAINING_QUEUE_REQUIRED_FORWARD_ENV
}

g1_register_queue_environment

g1_start_artifact_verifier() {
    [[ -n "${_g1_artifact_verifier_pid:-}" ]] && return
    coproc G1_ARTIFACT_VERIFIER {
        python -u "$_g1_artifact_launcher_dir/verify_artifact.py" --server
    }
    _g1_artifact_verifier_pid=$G1_ARTIFACT_VERIFIER_PID
    _g1_artifact_verifier_read=${G1_ARTIFACT_VERIFIER[0]}
    _g1_artifact_verifier_write=${G1_ARTIFACT_VERIFIER[1]}
}

g1_stop_artifact_verifier() {
    [[ -n "${_g1_artifact_verifier_pid:-}" ]] || return
    exec {_g1_artifact_verifier_write}>&-
    wait "$_g1_artifact_verifier_pid" 2>/dev/null || true
    exec {_g1_artifact_verifier_read}<&-
    unset _g1_artifact_verifier_pid
}

g1_verify_tuning_artifact() {
    local directory=$1
    local dataset_size=$2
    shift 2
    g1_start_artifact_verifier
    {
        printf '%s\t%s' "$directory" "$dataset_size"
        printf '\t%s' "$@"
        printf '\n'
    } >&"$_g1_artifact_verifier_write"
    local response
    if ! IFS= read -r response <&"$_g1_artifact_verifier_read"; then
        echo "Artifact verifier stopped unexpectedly" >&2
        return 2
    fi
    case $response in
        0) return 0 ;;
        1) return 1 ;;
        2$'\t'*) echo "Artifact verifier error: ${response#*$'\t'}" >&2 ; return 2 ;;
        *) echo "Invalid artifact verifier response: $response" >&2 ; return 2 ;;
    esac
}

g1_verify_config_artifact() {
    local directory=$1
    local config_path=$2
    shift 2
    g1_start_artifact_verifier
    {
        printf '%s\tconfig\t%s' "$directory" "$config_path"
        printf '\t%s' "$@"
        printf '\n'
    } >&"$_g1_artifact_verifier_write"
    local response
    if ! IFS= read -r response <&"$_g1_artifact_verifier_read"; then
        echo "Artifact verifier stopped unexpectedly" >&2
        return 2
    fi
    case $response in
        0) return 0 ;;
        1) return 1 ;;
        2$'\t'*) echo "Artifact verifier error: ${response#*$'\t'}" >&2 ; return 2 ;;
        *) echo "Invalid artifact verifier response: $response" >&2 ; return 2 ;;
    esac
}

g1_verify_config_recipe_artifact() {
    local directory=$1
    local config_path=$2
    shift 2
    g1_start_artifact_verifier
    {
        printf '%s\tconfig-recipe\t%s' "$directory" "$config_path"
        printf '\t%s' "$@"
        printf '\n'
    } >&"$_g1_artifact_verifier_write"
    local response
    if ! IFS= read -r response <&"$_g1_artifact_verifier_read"; then
        echo "Artifact verifier stopped unexpectedly" >&2
        return 2
    fi
    case $response in
        0) return 0 ;;
        1) return 1 ;;
        2$'\t'*) echo "Artifact verifier error: ${response#*$'\t'}" >&2 ; return 2 ;;
        *) echo "Invalid artifact verifier response: $response" >&2 ; return 2 ;;
    esac
}

g1_artifact_exists() {
    [[ -e "$1" || -L "$1" ]]
}

g1_classify_tuning_artifact() {
    local directory=$1
    local dataset_size=$2
    shift 2
    g1_start_artifact_verifier
    {
        printf '%s\tclassify-tuning\t%s' "$directory" "$dataset_size"
        printf '\t%s' "$@"
        printf '\n'
    } >&"$_g1_artifact_verifier_write"
    IFS= read -r _g1_artifact_state <&"$_g1_artifact_verifier_read" || return 2
}

g1_classify_config_artifact() {
    local directory=$1
    local config_path=$2
    shift 2
    g1_start_artifact_verifier
    {
        printf '%s\tclassify-config\t%s' "$directory" "$config_path"
        printf '\t%s' "$@"
        printf '\n'
    } >&"$_g1_artifact_verifier_write"
    IFS= read -r _g1_artifact_state <&"$_g1_artifact_verifier_read" || return 2
}

g1_classify_config_recipe_artifact() {
    local directory=$1
    local config_path=$2
    shift 2
    g1_start_artifact_verifier
    {
        printf '%s\tclassify-config-recipe\t%s' "$directory" "$config_path"
        printf '\t%s' "$@"
        printf '\n'
    } >&"$_g1_artifact_verifier_write"
    IFS= read -r _g1_artifact_state <&"$_g1_artifact_verifier_read" || return 2
}

g1_archive_artifact() {
    local directory=$1
    local reason=$2
    case $reason in
        incomplete|incompatible) ;;
        *)
            echo "Invalid artifact archive reason: $reason" >&2
            return 2
            ;;
    esac
    if ! g1_artifact_exists "$directory"; then
        echo "Cannot archive missing artifact: $directory" >&2
        return 2
    fi
    if ! command -v flock >/dev/null 2>&1; then
        echo "Cannot safely archive artifact without flock: $directory" >&2
        return 2
    fi
    local parent
    local name
    parent=$(dirname -- "$directory") || return 2
    name=$(basename -- "$directory") || return 2
    if [[ -z "$name" || "$name" == */* || "$name" != g1_* ]] || \
        [[ $(basename -- "$parent") == old ]]; then
        echo "Invalid artifact path: $directory" >&2
        return 2
    fi
    local lock_dir="$parent/.run-locks"
    local archive_dir="$parent/old"
    local lock_path="$lock_dir/$name.lock"
    mkdir -p "$lock_dir" "$archive_dir" || return 2
    local run_lock_fd
    exec {run_lock_fd}>"$lock_path" || return 2
    if ! flock -n "$run_lock_fd"; then
        echo "Artifact is owned by an active training process: $directory" >&2
        return 2
    fi
    if ! g1_artifact_exists "$directory"; then
        flock -u "$run_lock_fd"
        exec {run_lock_fd}>&-
        echo "Artifact disappeared before it could be archived: $directory" >&2
        return 2
    fi
    local attempt=1
    local archive
    while true; do
        archive=$(printf '%s/%s.%s-%03d' "$archive_dir" "$name" "$reason" "$attempt")
        [[ -e "$archive" || -L "$archive" ]] || break
        attempt=$((attempt + 1))
    done
    if ! mv -- "$directory" "$archive"; then
        flock -u "$run_lock_fd"
        exec {run_lock_fd}>&-
        return 2
    fi
    flock -u "$run_lock_fd"
    exec {run_lock_fd}>&-
    echo "=== archived $reason artifact as old/$(basename "$archive") ==="
}

g1_require_compatible_or_absent() {
    local directory=$1
    local dataset_size=$2
    shift 2
    g1_artifact_exists "$directory" || return 1
    if [[ ! -d "$directory" ]]; then
        echo "Artifact path is not a directory: $directory" >&2
        return 2
    fi
    if [[ ! -L "$directory" ]] && \
        [[ -z "$(find "$directory" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        return 1
    fi
    g1_classify_tuning_artifact "$directory" "$dataset_size" "$@" || return 2
    case $_g1_artifact_state in
        complete) return 0 ;;
        resumable)
            g1_archive_artifact "$directory" incomplete || return 2
            return 1
            ;;
        incompatible)
            g1_archive_artifact "$directory" incompatible || return 2
            return 1
            ;;
        2$'\t'*) echo "Artifact verifier error: ${_g1_artifact_state#*$'\t'}" >&2 ; return 2 ;;
        *) echo "Invalid artifact state: $_g1_artifact_state" >&2 ; return 2 ;;
    esac
}

g1_require_config_compatible_or_absent() {
    local directory=$1
    local config_path=$2
    shift 2
    g1_artifact_exists "$directory" || return 1
    if [[ ! -d "$directory" ]]; then
        echo "Artifact path is not a directory: $directory" >&2
        return 2
    fi
    if [[ ! -L "$directory" ]] && \
        [[ -z "$(find "$directory" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        return 1
    fi
    g1_classify_config_artifact "$directory" "$config_path" "$@" || return 2
    case $_g1_artifact_state in
        complete) return 0 ;;
        resumable)
            g1_archive_artifact "$directory" incomplete || return 2
            return 1
            ;;
        incompatible)
            g1_archive_artifact "$directory" incompatible || return 2
            return 1
            ;;
        2$'\t'*) echo "Artifact verifier error: ${_g1_artifact_state#*$'\t'}" >&2 ; return 2 ;;
        *) echo "Invalid artifact state: $_g1_artifact_state" >&2 ; return 2 ;;
    esac
}

g1_require_config_recipe_compatible_or_absent() {
    local directory=$1
    local config_path=$2
    shift 2
    g1_artifact_exists "$directory" || return 1
    if [[ ! -d "$directory" ]]; then
        echo "Artifact path is not a directory: $directory" >&2
        return 2
    fi
    if [[ ! -L "$directory" ]] && \
        [[ -z "$(find "$directory" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        return 1
    fi
    g1_classify_config_recipe_artifact "$directory" "$config_path" "$@" || return 2
    case $_g1_artifact_state in
        complete) return 0 ;;
        resumable)
            g1_archive_artifact "$directory" incomplete || return 2
            return 1
            ;;
        incompatible)
            g1_archive_artifact "$directory" incompatible || return 2
            return 1
            ;;
        2$'\t'*) echo "Artifact verifier error: ${_g1_artifact_state#*$'\t'}" >&2 ; return 2 ;;
        *) echo "Invalid artifact state: $_g1_artifact_state" >&2 ; return 2 ;;
    esac
}

g1_reuse_first_compatible_artifact() {
    local destination=$1
    local dataset_size=$2
    local pattern=$3
    shift 3
    local source
    local verifier_status
    while IFS= read -r source; do
        verifier_status=0
        g1_verify_tuning_artifact "$source" "$dataset_size" "$@" \
            || verifier_status=$?
        if [[ "$verifier_status" -eq 0 ]]; then
            mkdir -p "$(dirname "$destination")"
            ln -s "$(realpath "$source")" "$destination"
            echo "=== reused $(basename "$source") as $(basename "$destination") ==="
            return 0
        fi
        [[ "$verifier_status" -eq 1 ]] || return "$verifier_status"
    done < <(compgen -G "$pattern" | sort)
    return 1
}
