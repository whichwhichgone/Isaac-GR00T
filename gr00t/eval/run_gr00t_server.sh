#!/usr/bin/env bash

set -e

cd /liujinxin/liyifan/Isaac-GR00T

source /liujinxin/liyifan/Isaac-GR00T/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

python /liujinxin/liyifan/Isaac-GR00T/gr00t/eval/run_gr00t_server.py \
    --model_path /liujinxin/liyifan/Isaac-GR00T/checkpoints/2026-07-09_pick_cube_bottle_g1_0616-0626_all_other_mixture_3views_h30_dropout_0.3_0.1hand/checkpoint-80000 \
    --embodiment-tag UNITREE_G1_29DOF_HAND \
    --port 9002