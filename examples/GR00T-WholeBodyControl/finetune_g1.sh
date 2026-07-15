
set -x -e

export NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_API_KEY=83793606f810aa3d385ea5d12dbd352514ac54e1

DATASET_PATHS=(
    "/liujinxin/zhaowei/G1_MOTION/data/G1_real_6D_window_cont_rel_hand_sft"
    "/liujinxin/zhaowei/G1_MOTION/data/G1_real_6D_window_cont_rel_hand_sft_p2"
)

torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base_model_path /wangdonglin2/datasets/checkpoint-30000 \
    --dataset_path "${DATASET_PATHS[@]}" \
    --embodiment_tag UNITREE_G1_29DOF \
    --num_gpus $NUM_GPUS \
    --output_dir logs_output/g1_29dof_finetune_real_6D_window_cont_rel_hand_sft_0713 \
    --save_total_limit 3 \
    --max_steps 15000 \
    --save_steps 3000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --use_wandb \
    --global_batch_size 512 \
    --dataloader_num_workers 6 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --random_rotation_angle 8 \
    --crop_fraction 0.9 \
    --state_dropout_prob 0.1 \
    --state_element_dropout_prob 0.3 \
    --stickman_encoder_learning_rate 1e-5 \
    --no-tune_llm
