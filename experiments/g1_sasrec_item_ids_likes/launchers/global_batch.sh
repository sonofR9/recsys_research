g1_require_global_batch_selection() {
    local repo_root=$1
    local batch_size=$2
    local selection=${G1_GLOBAL_BATCH_SELECTION:-}
    local key embedding_lr deep_lr selected_batch extra
    IFS=: read -r key embedding_lr deep_lr selected_batch extra <<< "$selection"
    if [[ -n "${extra:-}" || "$key" != control/control || \
          -z "${embedding_lr:-}" || -z "${deep_lr:-}" || \
          ! "${selected_batch:-}" =~ ^[1-9][0-9]*$ || \
          "$selected_batch" != "$batch_size" ]]; then
        echo "G1_GLOBAL_BATCH_SELECTION must be control/control:embedding_lr:deep_lr:batch_size" >&2
        return 2
    fi
    local verifier=${G1_TEST_GLOBAL_BATCH_VERIFIER:-}
    if [[ -n "$verifier" ]]; then
        if [[ -z "${G1_TRAINING_QUEUE_LIBRARY:-}" ]]; then
            echo "G1_TEST_GLOBAL_BATCH_VERIFIER requires an injected queue library" >&2
            return 2
        fi
        "$verifier" \
            --embedding-lr "$embedding_lr" \
            --deep-lr "$deep_lr" \
            --batch-size "$selected_batch" || return
    else
        python "$repo_root/experiments/g1_sasrec_item_ids_likes/analysis/verify_global_batch.py" \
            --embedding-lr "$embedding_lr" \
            --deep-lr "$deep_lr" \
            --batch-size "$selected_batch" || return
    fi
    G1_VERIFIED_CONTROL_EMBEDDING_LR=$embedding_lr
    G1_VERIFIED_CONTROL_DEEP_LR=$deep_lr
}

g1_same_number() {
    awk -v left="$1" -v right="$2" 'BEGIN { exit !(left == right) }'
}
