#!/usr/bin/env bash
set -u

cd "$(dirname "$0")"
export G1_DATASET_SIZE=500m
export G1_SEEDS=${G1_SEEDS:-"0 1 2 3"}
unset G1_MAX_USERS
unset G1_VAL_BATCH_SIZE

./seeds.sh \
    lr_cosine lr_linear lr_inverse_sqrt lr_inverse_sqrt_warmup \
    lr_warmup lr_step lr_exponential lr_polynomial lr_wsd \
    cosine_dim_16 cosine_dim_256 \
    cosine_depth_1 cosine_depth_4 \
    cosine_heads_1 cosine_heads_4 cosine_heads_8 \
    cosine_ffn_128 cosine_ffn_512 \
    cosine_seq_50 cosine_seq_200 \
    cosine_dropout_30 cosine_dropout_50 \
    cosine_pos_none cosine_pos_alibi \
    cosine_pos_rope_alibi cosine_pos_rope_reverse_alibi \
    cosine_pos_rope_learned cosine_pos_rope_learned_reverse \
    cosine_pos_rope_reverse_learned \
    cosine_pos_rope_reverse_learned_reverse \
    cosine_pos_learned_alibi cosine_pos_learned_reverse_alibi \
    cosine_pos_rope_learned_reverse_alibi \
    cosine_pos_rope_reverse_learned_alibi \
    cosine_pos_all cosine_pos_reverse_all \
    cosine_norm_rms cosine_norm_batch \
    cosine_norm_all_rms cosine_norm_input_layer cosine_norm_input_rms \
    cosine_norm_no_final cosine_bos \
    window_25 per_layer_embeddings \
    time_rope time_rope_reverse time_log_rope time_log_rope_reverse \
    time_plain_add time_log_add time_log_concat time_bins_concat \
    time_bins_log_rope \
    neg_random neg_in_batch_no_logq \
    neg_mixed_online_logq neg_mixed_offline_logq \
    selected_quality
