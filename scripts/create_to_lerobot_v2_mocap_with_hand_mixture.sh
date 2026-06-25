#!/usr/bin/env bash
set -euo pipefail

echo "Checking ffmpeg..."
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg not found. Installing..."
    apt-get update
    apt-get install -y ffmpeg
else
    echo "ffmpeg already installed: $(which ffmpeg)"
    ffmpeg -version | head -n 1
fi
DATE="2026-06-25"
REPO_NAME="pick_cube_bottle_g1_0616-0623"
OUTPUT_PATH="/liujinxin/liyifan/Isaac-GR00T/dataset/${REPO_NAME}"
DATASET_PATH_1="/liujinxin/liyifan/Isaac-GR00T/dataset/${REPO_NAME}"
DATASET_PATH_2="/liujinxin/liyifan/Isaac-GR00T/dataset/G1_hand_window_pick_water_bowl_sink_0609-0610"
DATASET_PATH_3="/liujinxin/liyifan/Isaac-GR00T/dataset/G1_real_6D_window_cont_rel_0518-0604"
EMBODIMENT_TAG_1="UNITREE_G1_29DOF_HAND"
EMBODIMENT_TAG_2="UNITREE_G1_29DOF_HAND"
EMBODIMENT_TAG_3="UNITREE_G1_29DOF"
MODALITY_NAME="modality_window_with_hand"

NUM_GPUS=8
BATCH_PER_GPU=180
GLOBAL_BATCH_SIZE=$((NUM_GPUS * BATCH_PER_GPU))
export WANDB_API_KEY="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"
export WANDB_ENTITY="liyifansmx-westlake-university"
echo "DATE=${DATE}"
echo "REPO_NAME=${REPO_NAME}"
echo "DATASET_PATH_1=${DATASET_PATH_1}"
echo "DATASET_PATH_2=${DATASET_PATH_2}"
echo "DATASET_PATH_3=${DATASET_PATH_3}"
echo "EMBODIMENT_TAG_1=${EMBODIMENT_TAG_1}"
echo "EMBODIMENT_TAG_2=${EMBODIMENT_TAG_2}"
echo "EMBODIMENT_TAG_3=${EMBODIMENT_TAG_3}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"

# cd /liujinxin/liyifan/Isaac-GR00T
# source /liujinxin/conda3/bin/activate dreamzero

# python scripts/convert_to_lerobot_new_with_hand.py \
#     --output_dir "${OUTPUT_PATH}"

# cp "/liujinxin/liyifan/Isaac-GR00T/scripts/${MODALITY_NAME}.json" "${DATASET_PATH_1}/meta/modality.json"

# conda deactivate
# source .venv/bin/activate

# python /liujinxin/liyifan/Isaac-GR00T/gr00t/data/stats.py \
#     --dataset-path "${DATASET_PATH_1}" \
#     --embodiment-tag "${EMBODIMENT_TAG_1}"

ulimit -n 1048576 || true

cd /liujinxin/liyifan/Isaac-GR00T
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node="${NUM_GPUS}" --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base-model-path /liujinxin/liyifan/Isaac-GR00T/checkpoints/GR00T-N1.6-3B \
    --dataset-path-groups "${DATASET_PATH_1}" "${DATASET_PATH_2}" "${DATASET_PATH_3}" \
    --dataset_embodiment_tags "${EMBODIMENT_TAG_1}" "${EMBODIMENT_TAG_2}" "${EMBODIMENT_TAG_3}" \
    --dataset_mix_ratios "3" "1" "2" \
    --num_gpus "${NUM_GPUS}" \
    --output-dir "./checkpoints/${DATE}_${REPO_NAME}_mixture_3views" \
    --save_total_limit 5 \
    --save-steps 10000 \
    --max-steps 80000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --global_batch_size "${GLOBAL_BATCH_SIZE}" \
    --dataloader_num_workers 6 \
    --use_wandb \
    --color_jitter_params brightness 0.4 contrast 0.5 saturation 0.6 hue 0.1 
#     --state_dropout_prob 0.1 \
#     --tune_llm \