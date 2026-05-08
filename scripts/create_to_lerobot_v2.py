      
#!/usr/bin/env python3
"""
Convert G1 motion pkl files to LeRobot v2 format dataset.

Uses the official LeRobot API (LeRobotDataset.create / add_frame / save_episode).

Usage:
    python convert_to_lerobot_v2.py --input_dir G1_motion_data --output_dir G1_lerobot_v2

Dependencies:
    pip install lerobot imageio imageio-ffmpeg
"""

import json
import os
import sys
import glob
import shutil
import argparse
# import joblib
import numpy as np
from pathlib import Path
from PIL import Image
from multiprocessing import Pool
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# from visualize_stickman import load_stickman

sys.path.insert(0,"/liujinxin/liyifan/Isaac-GR00T/third_party/lerobot-main")
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


FPS = 50
IMAGE_SHAPE = (224, 224, 3)
CAMERAS = ["head_img", "left_wrist_img", "right_wrist_img"]
REPO_ID = "lerobot_datasets/g1_motion"
# CAMERAS_MAP = {"head_img": "front", "left_wrist_img": "left", "right_wrist_img": "right"}
CAMERAS_MAP = {"head_img": "front", "left_wrist_img": "left", "right_wrist_img": "right"}

MODALITY = "/liujinxin/liyifan/Isaac-GR00T/scripts/modality.json"
DATASET_DIR = [
               "/liujinxin/dataset/piper/G1/0420_toy_g1"
               ]
REPO_NAME = "2026-04-24_G1_push_toy"  # Name of the output dataset, also used for the Hugging Face Hub
OUTPUT_DIR = LEROBOT_HOME/REPO_NAME

FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (36,),
        "names": (
            ["root_trans_x", "root_trans_y", "root_trans_z"]
            + ["root_rot_w", "root_rot_x", "root_rot_y", "root_rot_z"]
            + [f"joint_{i}" for i in range(29)]
        ),
    },
    "action": {
        "dtype": "float32",
        "shape": (36,),
        "names": (
            ["root_trans_x", "root_trans_y", "root_trans_z"]
            + ["root_rot_w", "root_rot_x", "root_rot_y", "root_rot_z"]
            + [f"joint_{i}" for i in range(29)]
        ),
    },
    "task_vis_stickman": {
        "dtype": "float32",
        "shape": (18,),
        "names": [
            f"{node}_{axis}"
            for node in ["pelvis", "L_foot", "R_foot", "spine2", "L_wrist", "R_wrist"]
            for axis in ["x", "y", "z"]
        ],
    },
    **{
        f"observation.images.{cam}": {
            "dtype": "video",
            "shape": IMAGE_SHAPE,
            "names": ["height", "width", "channel"],
        }
        for cam in CAMERAS
    },
}

PLACEHOLDER_IMAGE = np.zeros(IMAGE_SHAPE, dtype=np.uint8)

def load_image(image_path, image_type):
    image = Image.open(image_path)

    if image.mode != 'RGB':
        image = image.convert('RGB')
    width, height = image.size
    aspect_ratio = width / height
    target_size = IMAGE_SHAPE[:2]

    if aspect_ratio > 1:
        new_width = target_size[0]
        new_height = int(target_size[0] / aspect_ratio)
    else:
        new_height = target_size[1]
        new_width = int(target_size[1] * aspect_ratio)

    resized_image = image.resize((new_width, new_height), Image.BILINEAR)

    canvas = Image.new("RGB", target_size, (255, 255, 255))
    canvas.paste(
        resized_image,
        ((target_size[0] - new_width) // 2, (target_size[1] - new_height) // 2)
    )

    return np.array(canvas)


def get_episode(data:dict, root_path: str):

    def get_state(data, idx):
        root_rot = np.asarray(data[idx]["imu"], dtype=np.float32) # (1, 4)
        root_trans = np.zeros(3, dtype=np.float32)
        joint_pos = np.asarray(data[idx]["body_joint"], dtype=np.float32)
        state = np.concatenate([root_trans, root_rot, joint_pos], axis=0)
        return state

    episode = []
    task = data[0]["task"][0]
    for idx in range(len(data)-1):
        step = data[idx]
        frame = {}

        for lerobot_key, g1_key in CAMERAS_MAP.items():
            image_path = os.path.join(root_path, step[g1_key])
            image = load_image(image_path, g1_key)
            frame[f"observation.images.{lerobot_key}"] = image

        state = get_state(data, idx)
        action = get_state(data, idx+1)
        frame["observation.state"] = state
        frame["action"] = action
        frame["task_vis_stickman"] = np.zeros(18, dtype=np.float32)
        frame["task"] = task
        episode.append(frame)
    return episode

def process_file(args):
    root, file = args
    file_path = os.path.join(root,file)
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            episode = get_episode(data, root)
            return episode
    except Exception as e:
        print( f"打开文件 {file_path} 时出错: {e}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--modality_path", default=MODALITY)
    parser.add_argument("--fps", type=int, default=FPS)
    args = parser.parse_args()

    output_dir = args.output_dir
    fps = args.fps
    modality_path = args.modality_path

    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        robot_type="unitree_g1_29dof",
        fps=fps,
        features=FEATURES,
        root=output_path,
        image_writer_threads=10,
        image_writer_processes=8,
    )

    ep_idx = 0
    file_pool = []
    for input_dir in DATASET_DIR:
        json_files = sorted(glob.glob(os.path.join(input_dir, "**/data.json"), recursive=True))

        for json_path in json_files:
            root_path = os.path.dirname(json_path)
            file = os.path.basename(json_path)
            file_pool.append((root_path,file))
    print(f"Found{len(file_pool)} episodes")

    with Pool(processes=12) as pool:
        for episode in pool.imap_unordered(process_file, file_pool, chunksize=2):
            if not episode or isinstance(episode, str):
                print(f"跳过无效结果:{episode}")
                continue
            task = episode[0]["task"]
            for frame in episode:
                frame.pop("task", None)
                dataset.add_frame(frame)
            print('#####step ', task)
            dataset.save_episode(task=task)
    dataset.consolidate(run_compute_stats=True)

    meta_path = os.path.join(output_dir, "meta")
    shutil.copy(modality_path, meta_path)
   