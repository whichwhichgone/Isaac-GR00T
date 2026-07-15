
set -x -e

export CUDA_VISIBLE_DEVICES=0

uv run python gr00t/eval/run_gr00t_server.py \
    --embodiment_tag UNITREE_G1_29DOF \
    --model_path /wangdonglin2/datasets/checkpoint-30000 \
    --device cuda:0 \
    --host 0.0.0.0 \
    --port 9002
