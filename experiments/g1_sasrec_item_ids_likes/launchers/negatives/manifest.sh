#!/usr/bin/env bash

G1_NEGATIVE_FAMILY_SPECS=(
    fixed_inbatch_global_q_yi2019:neg_fixed_inbatch_global_q_yi2019
    fixed_inbatch_leave_one_out:neg_fixed_inbatch_leave_one_out
    streaming_inbatch_global_q_yi2019:neg_streaming_inbatch_global_q_yi2019
    uniform_random:neg_random
    popularity_random_global_q_yi2019:neg_popularity_random_global_q_yi2019
    uncorrected_inbatch:neg_in_batch_no_logq
    uniform_random_plus_streaming_logq_negative_only:neg_mixed_streaming_logq_negative_only
    uniform_random_plus_fixed_logq_negative_only:neg_mixed_fixed_logq_negative_only
)

G1_NEGATIVE_VALID_FAMILIES=
for spec in "${G1_NEGATIVE_FAMILY_SPECS[@]}"; do
    G1_NEGATIVE_VALID_FAMILIES+=" ${spec%%:*}"
done
G1_NEGATIVE_VALID_FAMILIES=${G1_NEGATIVE_VALID_FAMILIES# }
G1_NEGATIVE_INITIAL_EMBEDDING_LRS="0.008 0.016 0.032"
G1_NEGATIVE_INITIAL_DEEP_LRS="0.003 0.006 0.012"
G1_NEGATIVE_SECONDARY_COUNTS="1024 2048"
G1_NEGATIVE_SECONDARY_ALPHAS="0.0025 0.005 0.02 0.04"
G1_NEGATIVE_SECONDARY_RANDOM_FRACTIONS="0.125 0.25 0.75 0.875"
G1_NEGATIVE_FIRST_EMBEDDING_LR_EXTENSION="0.064"
G1_NEGATIVE_FIRST_DEEP_LR_EXTENSION="0.024"
G1_NEGATIVE_COUNT_EXTENSIONS="4096"
