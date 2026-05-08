#!/usr/bin/env python3
"""
Convert G1 motion json files to LeRobot v2 format dataset.

This version stores concat action in component-major order:

    base_translation[t+1:t+K], base_rotation[t+1:t+K], left_leg[t+1:t+K], ...

instead of time-major order:

    full_state[t+1], full_state[t+2], ..., full_state[t+K]

This is important for GR00T modality mapping, because modality keys should keep the
same physical component names as state keys, e.g. base_translation/base_rotation/left_leg.
"""

import json
import os
import sys
import glob
import shutil
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from multiprocessing import Pool

sys.path.insert(0, "/liujinxin/liyifan/Isaac-GR00T/third_party/lerobot-main")
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


# 原始数据假设约 120Hz；STRIDE=6 表示每 6 个原始 step 对应一个图像/训练样本，约 20Hz。
FPS = 20
IMAGE_SHAPE = (224, 224, 3)
CAMERAS = ["head_img"]
REPO_ID = "lerobot_datasets/g1_motion"
# CAMERAS_MAP = {"head_img": "front", "left_wrist_img": "left", "right_wrist_img": "right"}
CAMERAS_MAP = {"head_img": "front"}

MODALITY = "/liujinxin/liyifan/Isaac-GR00T/scripts/modality.json"
DATASET_DIR = [
    "/liujinxin/dataset/piper/G1/0428_tidy_up_g1",
]
REPO_NAME = "2026-0428_tidy_up_g1_downsampling_concat"
OUTPUT_DIR = LEROBOT_HOME / REPO_NAME

# 原始 120Hz 下，图像/训练样本间隔 6 个 step。
STRIDE = 6

# 原始动作下采样倍率：120Hz -> 60Hz。
DOWNSAMPLE = 2

# 每个 LeRobot frame 的 action 是未来 CONCAT_ACTION_STEPS 个 60Hz state/action 拼成的一个 108-D action token。
# STRIDE=6, DOWNSAMPLE=2 时，CONCAT_ACTION_STEPS=3。
CONCAT_ACTION_STEPS = STRIDE // DOWNSAMPLE

STATE_DIM = 36
ACTION_DIM = STATE_DIM * CONCAT_ACTION_STEPS

STATE_FEATURE_NAMES = (
    ["root_trans_x", "root_trans_y", "root_trans_z"]
    + ["root_rot_w", "root_rot_x", "root_rot_y", "root_rot_z"]
    + [f"joint_{i}" for i in range(29)]
)

# G1 29DoF state/action layout:
#   base_translation: 3
#   base_rotation:    4
#   left_leg:         6
#   right_leg:        6
#   waist:            3
#   left_arm:         7
#   right_arm:        7
# Total: 36
ACTION_COMPONENT_SLICES = {
    "base_translation": slice(0, 3),
    "base_rotation": slice(3, 7),
    "left_leg": slice(7, 13),
    "right_leg": slice(13, 19),
    "waist": slice(19, 22),
    "left_arm": slice(22, 29),
    "right_arm": slice(29, 36),
}

# 这个顺序必须和你的 modality.json 里 action modality_keys 的顺序一致。
ACTION_COMPONENT_ORDER = [
    "base_translation",
    "base_rotation",
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "right_arm",
]

assert sum(ACTION_COMPONENT_SLICES[k].stop - ACTION_COMPONENT_SLICES[k].start for k in ACTION_COMPONENT_ORDER) == STATE_DIM


def build_component_major_action_names() -> list[str]:
    """Build 108 unique names in the same order as build_component_major_action()."""
    names = []
    for component_name in ACTION_COMPONENT_ORDER:
        sl = ACTION_COMPONENT_SLICES[component_name]
        component_base_names = STATE_FEATURE_NAMES[sl]
        for future_i in range(CONCAT_ACTION_STEPS):
            for base_name in component_base_names:
                names.append(f"{component_name}_future{future_i + 1}_{base_name}")
    assert len(names) == ACTION_DIM, f"action names should be {ACTION_DIM}, got {len(names)}"
    return names


ACTION_FEATURE_NAMES = build_component_major_action_names()

FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (STATE_DIM,),
        "names": STATE_FEATURE_NAMES,
    },
    "action": {
        "dtype": "float32",
        "shape": (ACTION_DIM,),
        "names": ACTION_FEATURE_NAMES,
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

    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.size
    aspect_ratio = width / height
    target_size = IMAGE_SHAPE[:2]  # (target_h, target_w) according to IMAGE_SHAPE
    target_h, target_w = target_size

    # 保持原始逻辑：等比例缩放 + 白边 padding 到 224x224。
    # PIL resize 需要传入 (width, height)。
    if aspect_ratio > 1:
        new_width = target_w
        new_height = int(target_w / aspect_ratio)
    else:
        new_height = target_h
        new_width = int(target_h * aspect_ratio)

    resized_image = image.resize((new_width, new_height), Image.BILINEAR)

    canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
    canvas.paste(
        resized_image,
        ((target_w - new_width) // 2, (target_h - new_height) // 2),
    )

    return np.asarray(canvas, dtype=np.uint8)


def get_episode(
    data: list[dict],
    root_path: str,
    stride: int = STRIDE,
    downsample: int = DOWNSAMPLE,
):
    if not data or len(data) <= stride:
        raise ValueError(f"Data length is {len(data)}, is not enough!")

    if stride % downsample != 0:
        raise ValueError(
            f"stride must be divisible by downsample, got stride={stride}, downsample={downsample}"
        )

    concat_action_steps = stride // downsample
    if concat_action_steps != CONCAT_ACTION_STEPS:
        raise ValueError(
            f"concat_action_steps={concat_action_steps} does not match global "
            f"CONCAT_ACTION_STEPS={CONCAT_ACTION_STEPS}"
        )

    # 原始 120Hz -> downsample=2 后约 60Hz。
    data_ds = data[::downsample]

    if len(data_ds) <= concat_action_steps:
        raise ValueError(
            f"Downsampled data length is {len(data_ds)}, "
            f"not enough for concat_action_steps={concat_action_steps}"
        )

    def get_state(sequence, idx):
        root_rot = np.asarray(sequence[idx]["imu"], dtype=np.float32).reshape(-1)
        root_trans = np.zeros(3, dtype=np.float32)
        joint_pos = np.asarray(sequence[idx]["body_joint"], dtype=np.float32).reshape(-1)

        assert root_rot.shape[0] == 4, f"imu should have 4 values, got {root_rot.shape}"
        assert joint_pos.shape[0] == 29, f"body_joint should have 29 values, got {joint_pos.shape}"

        state = np.concatenate([root_trans, root_rot, joint_pos], axis=0).astype(np.float32)
        assert state.shape[0] == STATE_DIM, f"state should be {STATE_DIM}-dim, got {state.shape}"

        return state

    def build_component_major_action(action_chunk: np.ndarray) -> np.ndarray:
        """
        Convert action_chunk from [K, 36] to [K * 36] in component-major order.

        Input, time-major matrix:
            action_chunk = [state_{t+1}, state_{t+2}, ..., state_{t+K}]  # [K, 36]

        Output, component-major flat vector:
            [
              base_translation_{t+1}, ..., base_translation_{t+K},
              base_rotation_{t+1},    ..., base_rotation_{t+K},
              left_leg_{t+1},         ..., left_leg_{t+K},
              ...
            ]

        This avoids modality keys such as base_translation0/base_translation1.
        Your modality.json should use physical component keys only:
            base_translation, base_rotation, left_leg, right_leg, waist, left_arm, right_arm
        with enlarged action ranges.
        """
        assert action_chunk.ndim == 2, f"action_chunk should be 2-D, got {action_chunk.shape}"
        assert action_chunk.shape == (CONCAT_ACTION_STEPS, STATE_DIM), (
            f"action_chunk should have shape ({CONCAT_ACTION_STEPS}, {STATE_DIM}), "
            f"got {action_chunk.shape}"
        )

        parts = []
        for component_name in ACTION_COMPONENT_ORDER:
            sl = ACTION_COMPONENT_SLICES[component_name]
            component_seq = action_chunk[:, sl]  # [K, component_dim]
            parts.append(component_seq.reshape(-1))  # [K * component_dim]

        action_label = np.concatenate(parts, axis=0).astype(np.float32)
        assert action_label.shape == (ACTION_DIM,), (
            f"action_label should have shape ({ACTION_DIM},), got {action_label.shape}"
        )
        return action_label

    def get_action_chunk(sequence, start_idx, concat_action_steps):
        """
        action 是未来若干个 next-state 的 component-major concat。

        如果 stride=6, downsample=2，则 concat_action_steps=3：
            observation.state = state[start_idx]
            action = concat_component_major(
                state[start_idx + 1],
                state[start_idx + 2],
                state[start_idx + 3],
            )

        对应到原始 120Hz 索引是：
            state_raw[t + 2], state_raw[t + 4], state_raw[t + 6]
        """
        last_idx = start_idx + concat_action_steps

        if last_idx >= len(sequence):
            raise IndexError(
                f"Action chunk out of range: start_idx={start_idx}, "
                f"last_idx={last_idx}, len(data)={len(sequence)}"
            )

        action_chunk = [
            get_state(sequence, i)
            for i in range(start_idx + 1, start_idx + concat_action_steps + 1)
        ]
        action_chunk = np.stack(action_chunk, axis=0).astype(np.float32)  # [K, 36]

        return build_component_major_action(action_chunk)

    def build_frame(idx, task):
        step = data_ds[idx]
        frame = {}

        for lerobot_key, g1_key in CAMERAS_MAP.items():
            if g1_key not in step:
                raise KeyError(
                    f"Missing image key {g1_key} at idx={idx}, available keys={list(step.keys())}"
                )

            image_path = os.path.join(root_path, step[g1_key])

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            image = load_image(image_path, g1_key)
            frame[f"observation.images.{lerobot_key}"] = image

        # 当前图像时刻对应的 state，不拼未来 state。
        frame["observation.state"] = get_state(data_ds, idx)  # [36]

        # 未来 CONCAT_ACTION_STEPS 个 60Hz next-state action，按 component-major 拼成 [108]。
        frame["action"] = get_action_chunk(
            sequence=data_ds,
            start_idx=idx,
            concat_action_steps=concat_action_steps,
        )

        frame["task_vis_stickman"] = np.zeros(18, dtype=np.float32)
        frame["task"] = task

        return frame

    raw_task = data[0].get("task", "unknown task")
    task = raw_task[0] if isinstance(raw_task, list) and len(raw_task) > 0 else raw_task

    episode = []

    # 在下采样后的 60Hz 序列上，每 concat_action_steps=3 个 step 生成一个 20Hz 训练样本。
    # 每个样本的 action 是未来 3 个 60Hz state/action 的 component-major concat。
    for idx in range(0, len(data_ds) - concat_action_steps, concat_action_steps):
        episode.append(build_frame(idx, task))

    return episode


def process_file(args):
    root, file = args
    file_path = os.path.join(root, file)
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        episode = get_episode(data, root)
        return episode
    except Exception as e:
        print(f"打开文件 {file_path} 时出错: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--modality_path", default=MODALITY)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--num_processes", type=int, default=12)
    args = parser.parse_args()

    output_dir = args.output_dir
    fps = args.fps
    modality_path = args.modality_path

    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        robot_type="unitree_g1_29dof_single_view",
        fps=fps,
        features=FEATURES,
        root=output_path,
        image_writer_threads=10,
        image_writer_processes=8,
    )

    file_pool = []
    for input_dir in DATASET_DIR:
        json_files = sorted(glob.glob(os.path.join(input_dir, "**/data.json"), recursive=True))

        for json_path in json_files:
            root_path = os.path.dirname(json_path)
            file = os.path.basename(json_path)
            file_pool.append((root_path, file))

    print(f"Found {len(file_pool)} episodes")
    print(f"FPS={fps}, CONCAT_ACTION_STEPS={CONCAT_ACTION_STEPS}, ACTION_DIM={ACTION_DIM}")
    print("Action component ranges in flat action vector:")
    cursor = 0
    for component_name in ACTION_COMPONENT_ORDER:
        comp_dim = ACTION_COMPONENT_SLICES[component_name].stop - ACTION_COMPONENT_SLICES[component_name].start
        next_cursor = cursor + comp_dim * CONCAT_ACTION_STEPS
        print(f"  {component_name}: [{cursor}, {next_cursor})")
        cursor = next_cursor

    with Pool(processes=args.num_processes) as pool:
        for episode in pool.imap_unordered(process_file, file_pool, chunksize=2):
            if not episode or isinstance(episode, str):
                print(f"跳过无效结果: {episode}")
                continue

            task = episode[0]["task"]
            for frame in episode:
                frame.pop("task", None)
                dataset.add_frame(frame)

            print("#####step ", task)
            dataset.save_episode(task=task)

    dataset.consolidate(run_compute_stats=True)

    meta_path = os.path.join(output_dir, "meta")
    shutil.copy(modality_path, meta_path)
