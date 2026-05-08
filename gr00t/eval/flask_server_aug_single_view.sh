#!/usr/bin/env bash

set -e

cd /liujinxin/liyifan/Isaac-GR00T

source /liujinxin/liyifan/Isaac-GR00T/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

python /liujinxin/liyifan/Isaac-GR00T/gr00t/eval/flask_server_aug_single_view.py \
    --model_path /liujinxin/liyifan/Isaac-GR00T/checkpoints/2026-04-24_G1_push_toy_16/checkpoint-20000 \
    --dataset-path /liujinxin/liyifan/Isaac-GR00T/dataset/2026-04-24_G1_push_toy \
    --embodiment-tag UNITREE_G1_29DOF_SINGLE_VIEW \
    --port 9002