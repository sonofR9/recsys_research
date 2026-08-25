#!/usr/bin/env bash
echo "G1_TEST_GLOBAL_BATCH_VERIFIER" >&2

while [[ $# -gt 0 ]]; do
    case $1 in
        --batch-size)
            batch_size=$2
            shift 2
            ;;
        *) shift 2 ;;
    esac
done

[[ "${batch_size:-}" != 999 ]]
