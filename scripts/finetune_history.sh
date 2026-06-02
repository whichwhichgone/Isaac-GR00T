#!/bin/bash
set -x -e

cd /liujinxin/liyifan/Isaac-GR00T/
source .venv/bin/activate
export NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_API_KEY="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"
export WANDB_ENTITY="liyifansmx-westlake-university"

torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base-model-path /liujinxin/liyifan/Isaac-GR00T/checkpoints/GR00T-N1.6-3B \
    --dataset-path /liujinxin/zhaowei/G1_MOTION/data/G1_real_6D_window \
    --embodiment_tag UNITREE_G1_11X9_MOCAP_HISTORY \
    --num_gpus $NUM_GPUS \
    --output-dir ./checkpoints/G1_real_6D_window_0527 \
    --save_total_limit 5 \
    --save-steps 20000 \
    --max-steps 60000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --reset_state_act_projector \
    --learning_rate 1e-4 \
    --use_wandb \
    --global_batch_size 128 \
    --dataloader_num_workers 6 \
    --color_jitter_params brightness 0.4 contrast 0.5 saturation 0.6 hue 0.1 \
    --state_dropout_prob 0.1 \
    --tune_llm \

    # --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
