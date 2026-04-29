
set -x -e

export NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=4,5,6,7
export WANDB_API_KEY=83793606f810aa3d385ea5d12dbd352514ac54e1

torchrun --nproc_per_node=$NUM_GPUS --master_port=29501 \
    gr00t/experiment/launch_finetune.py \
    --base_model_path models/GR00T-N1.6-3B \
    --dataset_path /liujinxin/zhaowei/Isaac-GR00T/examples/GR00T-WholeBodyControl/G1_1000_lang_lerobot/ \
    --embodiment_tag UNITREE_G1_29DOF \
    --num_gpus $NUM_GPUS \
    --output_dir logs_output/g1_29dof_finetune_1000_stickman_walk_lang_tunellm \
    --save_total_limit 5 \
    --max_steps 10000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --use_wandb \
    --global_batch_size 256 \
    --dataloader_num_workers 6 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --tune_llm
