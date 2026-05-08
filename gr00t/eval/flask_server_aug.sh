#!/usr/bin/env bash

set -e

cd /liujinxin/liyifan/Isaac-GR00T

source /liujinxin/liyifan/Isaac-GR00T/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

python /liujinxin/liyifan/Isaac-GR00T/gr00t/eval/flask_server_aug.py \
    --model_path /liujinxin/liyifan/Isaac-GR00T/checkpoints/2026-04-15_G1/checkpoint-20000 \
    --dataset-path /liujinxin/liyifan/Isaac-GR00T/dataset/2026-04-15_G1 \
    --embodiment-tag UNITREE_G1_29DOF \
    --port 9002