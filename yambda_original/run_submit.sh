#!/usr/bin/env bash


# python -m models.sasrec.eval \
#     --exp_name debug_separate_eval \
#     --data_dir /home/sonofr/repos/shad/ysda_recsys/competition/generated/yambda_data \
#     --size 50m \
#     --interaction likes \
#     --train_days 299 \
#     --batch_size 256 \
#     --max_seq_len 200 \
#     --num_candidates 100 \
#     --device cuda:0

python -m models.sasrec.eval \
    --exp_name 5b_full \
    --data_dir /home/sonofr/repos/shad/ysda_recsys/competition/generated/yambda_data \
    --size 5b \
    --full_train \
    --batch_size 256 \
    --max_seq_len 512 \
    --num_candidates 100 \
    --use_kagglehub \
    --output_path final_pretrained_submission.csv 
