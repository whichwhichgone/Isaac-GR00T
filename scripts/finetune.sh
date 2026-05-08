#!/bin/bash
set -x -e

cd /liujinxin/liyifan/Isaac-GR00T/
source .venv/bin/activate

export NUM_GPUS=8
export WANDB_API_KEY="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"
export WANDB_ENTITY="liyifansmx-westlake-university"

torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base-model-path /liujinxin/liyifan/Isaac-GR00T/checkpoints/GR00T-N1.6-3B \
    --dataset-path /liujinxin/liyifan/Isaac-GR00T/dataset/2026-0428_tidy_up_g1 \
    --embodiment_tag UNITREE_G1_29DOF_SINGLE_VIEW \
    --num_gpus $NUM_GPUS \
    --output-dir ./checkpoints/2026-0428_tidy_up_g1 \
    --save_total_limit 5 \
    --save-steps 10000 \
    --max-steps 30000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --use_wandb \
    --global_batch_size 256 \
    --dataloader_num_workers 6 \
    # --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
    --color_jitter_params brightness 0.4 contrast 0.5 saturation 0.6 hue 0.1
