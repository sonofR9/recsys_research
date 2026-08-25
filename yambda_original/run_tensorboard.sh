#!/usr/bin/env bash

cd /home/sonofr/repos/shad/ysda_recsys/competition/yambda_original/checkpoints

tensorboard --logdir .  --host 0.0.0.0 --port=6006