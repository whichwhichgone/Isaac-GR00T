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

REPO_NAME="G1_real_6D_window_cont_rel"
DATASET_PATH="/liujinxin/liyifan/Isaac-GR00T/dataset/${REPO_NAME}"
EMBODIMENT_TAG="UNITREE_G1_29DOF"
NUM_GPUS=4
export WANDB_API_KEY="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"
export WANDB_ENTITY="liyifansmx-westlake-university"

echo "REPO_NAME=${REPO_NAME}"
echo "DATASET_PATH=${DATASET_PATH}"
echo "EMBODIMENT_TAG=${EMBODIMENT_TAG}"
echo "NUM_GPUS=${NUM_GPUS}"

cd /liujinxin/liyifan/Isaac-GR00T
# source /liujinxin/conda3/bin/activate dreamzero

# python scripts/convert_to_lerobot_new.py \
#     --output_dir "${DATASET_PATH}"

# cp /liujinxin/liyifan/Isaac-GR00T/scripts/modality.json "${DATASET_PATH}/meta"

# conda deactivate
source .venv/bin/activate

# python /liujinxin/liyifan/Isaac-GR00T/gr00t/data/stats.py \
#     --dataset-path "${DATASET_PATH}" \
#     --embodiment-tag "${EMBODIMENT_TAG}"

export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node="${NUM_GPUS}" --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base-model-path /liujinxin/liyifan/Isaac-GR00T/checkpoints/GR00T-N1.6-3B \
    --dataset-path "${DATASET_PATH}" \
    --embodiment_tag "${EMBODIMENT_TAG}" \
    --num_gpus "${NUM_GPUS}" \
    --output-dir "./checkpoints/${REPO_NAME}" \
    --save_total_limit 5 \
    --save-steps 20000 \
    --max-steps 40000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --global_batch_size 128 \
    --dataloader_num_workers 6 \
    --color_jitter_params brightness 0.4 contrast 0.5 saturation 0.6 hue 0.1 \
    --state_dropout_prob 0.1 \
    --tune_llm