#!/usr/bin/env bash

set -e

cd /liujinxin/liyifan/Isaac-GR00T

source /liujinxin/liyifan/Isaac-GR00T/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

python /liujinxin/liyifan/Isaac-GR00T/gr00t/eval/run_gr00t_server.py \
    --model_path /liujinxin/liyifan/Isaac-GR00T/checkpoints/2026-05-24_clean_desk_place_sofa_g1_fast_tune_llm_sampling/checkpoint-40000 \
    --dataset-path /liujinxin/liyifan/Isaac-GR00T/dataset/2026-05-24_clean_desk_place_sofa_g1_fast_sampling \
    --embodiment-tag UNITREE_G1_15X7_MOCAP \
    --port 9002