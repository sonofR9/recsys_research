#!/usr/bin/env bash

set -u

_service_state=$1
_service_queue_dir=${BASH_SOURCE[0]%/*}
export TRAINING_QUEUE_SERVICE_CHILD=1
export TRAINING_QUEUE_SCRIPT=__persistent_service__
export TRAINING_QUEUE_MONITOR_LIGHT_GPUS=${TRAINING_QUEUE_MONITOR_LIGHT_GPUS:-1}

source "${_service_queue_dir}/queue.sh" || exit 1
touch "${_service_state}/engine.ready"

while IFS= read -r command; do
    eval "$command"
done

drain
