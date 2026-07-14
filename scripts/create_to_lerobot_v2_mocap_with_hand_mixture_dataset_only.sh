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
DATE="2026-07-13"
CHECKPOINT_NAME="finetune"

REPO_NAME2="0626_pick_cube_bottle_g1"
REPO_NAME3="0629_pick_cube_bottle_g1"
REPO_NAME4="0710_pick_cube_bottle_g1_fix_2"
REPO_NAME5="0710_pick_cube_bottle_g1_medium_leg_1"
REPO_NAME6="0710_pick_cube_bottle_g1_raise_leg_once_2"

DATASET_PATH2="/liujinxin/dataset/piper/G1/${REPO_NAME2}"
DATASET_PATH3="/liujinxin/dataset/piper/G1/${REPO_NAME3}"
DATASET_PATH4="/liujinxin/dataset/piper/G1/${REPO_NAME4}"
DATASET_PATH5="/liujinxin/dataset/piper/G1/${REPO_NAME5}"
DATASET_PATH6="/liujinxin/dataset/piper/G1/${REPO_NAME6}"
OUTPUT_PATH2="/liujinxin/liyifan/Isaac-GR00T/dataset/${REPO_NAME2}"
OUTPUT_PATH3="/liujinxin/liyifan/Isaac-GR00T/dataset/${REPO_NAME3}"
OUTPUT_PATH4="/liujinxin/liyifan/Isaac-GR00T/dataset/${REPO_NAME4}"
OUTPUT_PATH5="/liujinxin/liyifan/Isaac-GR00T/dataset/${REPO_NAME5}"
OUTPUT_PATH6="/liujinxin/liyifan/Isaac-GR00T/dataset/${REPO_NAME6}"


EMBODIMENT_TAG_1="UNITREE_G1_29DOF_HAND"

MODALITY_NAME2="modality_window_with_hand"

NUM_GPUS=8
BATCH_PER_GPU=180
GLOBAL_BATCH_SIZE=$((NUM_GPUS * BATCH_PER_GPU))
export WANDB_API_KEY="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"
export WANDB_ENTITY="liyifansmx-westlake-university"

echo "EMBODIMENT_TAG_1=${EMBODIMENT_TAG_1}"

echo "NUM_GPUS=${NUM_GPUS}"
echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"

# dataset 1
cd /liujinxin/liyifan/Isaac-GR00T
source /liujinxin/conda3/bin/activate dreamzero

python scripts/convert_to_lerobot_new_with_hand.py \
    --output_dir "${OUTPUT_PATH2}"  \
    --input_dirs "${DATASET_PATH2}"

cp "/liujinxin/liyifan/Isaac-GR00T/scripts/${MODALITY_NAME2}.json" "${OUTPUT_PATH2}/meta/modality.json"

conda deactivate
source .venv/bin/activate

python /liujinxin/liyifan/Isaac-GR00T/gr00t/data/stats.py \
    --dataset-path "${OUTPUT_PATH2}" \
    --embodiment-tag "${EMBODIMENT_TAG_1}"


# dataset 2
cd /liujinxin/liyifan/Isaac-GR00T
source /liujinxin/conda3/bin/activate dreamzero

python scripts/convert_to_lerobot_new_with_hand.py \
    --output_dir "${OUTPUT_PATH3}"  \
    --input_dirs "${DATASET_PATH3}"

cp "/liujinxin/liyifan/Isaac-GR00T/scripts/${MODALITY_NAME2}.json" "${OUTPUT_PATH3}/meta/modality.json"

conda deactivate
source .venv/bin/activate

python /liujinxin/liyifan/Isaac-GR00T/gr00t/data/stats.py \
    --dataset-path "${OUTPUT_PATH3}" \
    --embodiment-tag "${EMBODIMENT_TAG_1}"


# dataset 3
cd /liujinxin/liyifan/Isaac-GR00T
source /liujinxin/conda3/bin/activate dreamzero

python scripts/convert_to_lerobot_new_with_hand.py \
    --output_dir "${OUTPUT_PATH4}"  \
    --input_dirs "${DATASET_PATH4}"

cp "/liujinxin/liyifan/Isaac-GR00T/scripts/${MODALITY_NAME2}.json" "${OUTPUT_PATH4}/meta/modality.json"

conda deactivate
source .venv/bin/activate

python /liujinxin/liyifan/Isaac-GR00T/gr00t/data/stats.py \
    --dataset-path "${OUTPUT_PATH4}" \
    --embodiment-tag "${EMBODIMENT_TAG_1}"


# dataset 4
cd /liujinxin/liyifan/Isaac-GR00T
source /liujinxin/conda3/bin/activate dreamzero

python scripts/convert_to_lerobot_new_with_hand.py \
    --output_dir "${OUTPUT_PATH5}"  \
    --input_dirs "${DATASET_PATH5}"

cp "/liujinxin/liyifan/Isaac-GR00T/scripts/${MODALITY_NAME2}.json" "${OUTPUT_PATH5}/meta/modality.json"

conda deactivate
source .venv/bin/activate

python /liujinxin/liyifan/Isaac-GR00T/gr00t/data/stats.py \
    --dataset-path "${OUTPUT_PATH5}" \
    --embodiment-tag "${EMBODIMENT_TAG_1}"


# dataset 5
cd /liujinxin/liyifan/Isaac-GR00T
source /liujinxin/conda3/bin/activate dreamzero

python scripts/convert_to_lerobot_new_with_hand.py \
    --output_dir "${OUTPUT_PATH6}"  \
    --input_dirs "${DATASET_PATH6}"

cp "/liujinxin/liyifan/Isaac-GR00T/scripts/${MODALITY_NAME2}.json" "${OUTPUT_PATH6}/meta/modality.json"

conda deactivate
source .venv/bin/activate

python /liujinxin/liyifan/Isaac-GR00T/gr00t/data/stats.py \
    --dataset-path "${OUTPUT_PATH6}" \
    --embodiment-tag "${EMBODIMENT_TAG_1}"


ulimit -n 1048576 || true

cd /liujinxin/liyifan/Isaac-GR00T
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --nproc_per_node="${NUM_GPUS}" --master_port=29600 \
    gr00t/experiment/launch_finetune.py \
    --base-model-path /liujinxin/liyifan/Isaac-GR00T/checkpoints/2026-07-09_pick_cube_bottle_g1_0616-0626_all_other_mixture_3views_h30_dropout_0.3_0.1hand/checkpoint-80000 \
    --dataset-path-groups "${OUTPUT_PATH2},${OUTPUT_PATH3},${OUTPUT_PATH4},${OUTPUT_PATH5},${OUTPUT_PATH6}" \
    --dataset_embodiment_tags "${EMBODIMENT_TAG_1}"\
    --dataset_mix_ratios "1"\
    --num_gpus "${NUM_GPUS}" \
    --output-dir "./checkpoints/${DATE}_${CHECKPOINT_NAME}_mixture_3views_h30_dropout_0.3" \
    --save_total_limit 5 \
    --save-steps 5000 \
    --max-steps 30000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 3e-5 \
    --global_batch_size "${GLOBAL_BATCH_SIZE}" \
    --dataloader_num_workers 6 \
    --use_wandb \
    --color_jitter_params brightness 0.4 contrast 0.5 saturation 0.6 hue 0.1 \
    --random_rotation_angle 10 \
    --state_dropout_prob 0.3 
#     --tune_llm \    