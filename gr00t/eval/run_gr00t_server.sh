#!/usr/bin/env bash

set -e

cd /liujinxin/liyifan/Isaac-GR00T

source /liujinxin/liyifan/Isaac-GR00T/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

python /liujinxin/liyifan/Isaac-GR00T/gr00t/eval/run_gr00t_server.py \
    --model_path /liujinxin/liyifan/Isaac-GR00T/checkpoints/2026-07-02_pick_cube_bottle_g1_0616-0626_mixture_3views_only_new_no_history/checkpoint-45000 \
    --embodiment-tag UNITREE_G1_29DOF_HAND_NO_HISTORY \
    --port 9002