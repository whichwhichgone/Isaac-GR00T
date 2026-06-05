#!/usr/bin/env bash

set -e

cd /liujinxin/liyifan/Isaac-GR00T

source /liujinxin/liyifan/Isaac-GR00T/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

python /liujinxin/liyifan/Isaac-GR00T/gr00t/eval/run_gr00t_server.py \
    --model_path /liujinxin/liyifan/Isaac-GR00T/checkpoints/G1_real_6D_window_cont_rel/checkpoint-40000 \
    --dataset-path /liujinxin/liyifan/Isaac-GR00T/dataset/G1_real_6D_window_cont_rel \
    --embodiment-tag UNITREE_G1_29DOF \
    --port 9002