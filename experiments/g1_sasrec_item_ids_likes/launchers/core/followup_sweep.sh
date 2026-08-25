#!/usr/bin/env bash
set -u

cd "$(dirname "$0")"
export G1_DATASET_SIZE=500m
export G1_SEEDS=${G1_SEEDS:-"0 1 2 3"}
unset G1_MAX_USERS
unset G1_VAL_BATCH_SIZE

./seeds.sh \
    lr_cosine_warmup \
    embedding_lr_1e4 embedding_lr_2e4 embedding_lr_5e4 \
    deep_lr_2e3 deep_lr_3e3 deep_lr_5e3 \
    cosine_ffn_swiglu cosine_ffn_swiglu_matched \
    lr_cosine_cycles2 lr_cosine_cycles4 \
    cosine_pos_learned_reverse cosine_pos_rope cosine_pos_rope_reverse \
    cosine_seq_128 cosine_dropout_0 cosine_heads_gqa cosine_norm_post \
    window_50 \
    time_bins_add time_bins_8 time_bins_16 time_bins_64 \
    time_bins_reverse_rope \
    neg_online_logq neg_random neg_random_offline_logq neg_in_batch_no_logq \
    cosine_dim_32 cosine_dim_128 \
    mup_dim32_lr5e2 mup_dim32_lr1e1 \
    mup_dim128_lr5e2 mup_dim128_lr1e1 \
    selected_quality selected_balanced
