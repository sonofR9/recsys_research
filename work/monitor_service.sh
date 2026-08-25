#!/usr/bin/env bash
set -euo pipefail

repository_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
runtime_directory="$repository_root/work/.runtime"
log_file="$runtime_directory/monitor.log"
repository_key=$(printf '%s' "$repository_root" | cksum | awk '{print $1}')
session_name="work-monitor-$repository_key"

is_running() {
    tmux has-session -t "=$session_name" 2>/dev/null
}

start() {
    if is_running; then
        echo "work monitor is already running"
        return
    fi
    if [[ -z "${TMUX:-}" ]]; then
        echo "start the work monitor from tmux" >&2
        exit 2
    fi
    local delivery=${WORK_MONITOR_DELIVERY:-input}
    mkdir -p "$runtime_directory"
    local monitor_command
    case "$delivery" in
        input)
            if [[ -z "${TMUX_PANE:-}" ]]; then
                echo "input delivery needs the lead Codex tmux pane" >&2
                exit 2
            fi
            printf -v monitor_command \
                'exec python -m work.workflow monitor --tmux-pane %q >>%q 2>&1' \
                "$TMUX_PANE" "$log_file"
            ;;
        log)
            printf -v monitor_command \
                'exec python -m work.workflow monitor --notify-stdout >>%q 2>&1' \
                "$log_file"
            ;;
        *)
            echo "WORK_MONITOR_DELIVERY must be input or log" >&2
            exit 2
            ;;
    esac
    tmux new-session -d -s "$session_name" -c "$repository_root" \
        "$monitor_command"
    echo "work monitor started with $delivery delivery"
}

stop() {
    if ! is_running; then
        echo "work monitor is not running"
        return
    fi
    tmux kill-session -t "=$session_name"
    echo "work monitor stopped"
}

status() {
    if is_running; then
        echo "work monitor is running"
    else
        echo "work monitor is stopped"
        return 1
    fi
}

case "${1:-}" in
    start) start ;;
    stop) stop ;;
    status) status ;;
    *) echo "usage: $0 {start|stop|status}" >&2; exit 2 ;;
esac
