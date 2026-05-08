#!/usr/bin/env bash

set -e

cd /liujinxin/liyifan/Isaac-GR00T

source /liujinxin/liyifan/Isaac-GR00T/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

python /liujinxin/liyifan/Isaac-GR00T/gr00t/eval/run_gr00t_server.py \
    --model_path /liujinxin/liyifan/Isaac-GR00T/checkpoints/2026-0428_tidy_up_g1/checkpoint-30000 \
    --dataset-path /liujinxin/liyifan/Isaac-GR00T/dataset/2026-0428_tidy_up_g1 \
    --embodiment-tag UNITREE_G1_29DOF_SINGLE_VIEW \
    --port 9002