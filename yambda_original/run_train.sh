#!/usr/bin/env bash

# alibi=True
# positional_embedding=True
# use_bos_tokens=True
# output_projection=none
# swiglu most probably false. Gemini says need hyperaparameters tuning...

# python -m models.sasrec.train \
#     --exp_name debug \
#     --data_dir /home/sonofr/repos/shad/ysda_recsys/competition/generated/yambda_data \
#     --size 50m \
#     --validate \
#     --train_days 299 \
#     --batch_size 256 \
#     --learning_rate 0.003 \
#     --warmup_ratio 0.1 \
#     --min_lr_ratio 0.1 \
#     --max_seq_len 200 \
#     --embedding_dim 128 \
#     --num_heads 2 \
#     --num_layers 2 \
#     --additional_info "" \
#     --tensorboard_enabled true \
#     --output_projection none \
#     --use_bos_tokens true \
#     --use_alibi true \
#     --use_positional_embedding true \
#     --use_swiglu false \
#     --num_kv_heads 2 \
#     --num_epochs 70  \
#     --seed 42

# python -m models.sasrec.train \
#     --exp_name 5b_v1 \
#     --data_dir /home/sonofr/repos/shad/ysda_recsys/competition/generated/yambda_data \
#     --size 5b \
#     --validate \
#     --train_days 299 \
#     --batch_size 256 \
#     --learning_rate 0.001 \
#     --warmup_ratio 0.08 \
#     --min_lr_ratio 0.06 \
#     --max_seq_len 512 \
#     --embedding_dim 256 \
#     --num_heads 4 \
#     --num_layers 6 \
#     --dropout 0.3 \
#     --additional_info "" \
#     --tensorboard_enabled true \
#     --output_projection none \
#     --use_bos_tokens true \
#     --use_alibi true \
#     --use_positional_embedding true \
#     --use_swiglu true \
#     --num_kv_heads 2 \
#     --num_epochs 80  \
#     --seed 42


python -m models.sasrec.train \
    --exp_name 5b_full \
    --data_dir /home/sonofr/repos/shad/ysda_recsys/competition/generated/yambda_data \
    --size 5b \
    --full_train \
    --batch_size 256 \
    --learning_rate 0.001 \
    --warmup_ratio 0.08 \
    --min_lr_ratio 0.06 \
    --max_seq_len 512 \
    --embedding_dim 256 \
    --num_heads 4 \
    --num_layers 6 \
    --dropout 0.3 \
    --additional_info "" \
    --tensorboard_enabled true \
    --output_projection none \
    --use_bos_tokens true \
    --use_alibi true \
    --use_positional_embedding true \
    --use_swiglu true \
    --num_kv_heads 2 \
    --num_epochs 80  \
    --seed 42
# final template
# python -m models.sasrec.train \
#     --exp_name final_v_debug \
#     --data_dir /home/sonofr/repos/shad/ysda_recsys/competition/generated/yambda_data \
#     --size 5b \
#     --full_train \
