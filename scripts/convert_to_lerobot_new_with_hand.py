#!/usr/bin/env python3
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image
import numpy as np
import tyro
import sys
sys.path.append("/liujinxin/liyifan/Isaac-GR00T/third_party/lerobot-main")
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

SOURCE_JSON_NAME = "data_root_relative_6D.json"
OUTPUT_JSON_NAME = "data_root_relative_6D_window_cont.json"

BODY_KEYS = ("imu", "body_joint", "mocap", "hand_cmd", "hand_state")
SELECT_11_INDICES = [
    0,  # pelvis/root,       original 29 index: root/floating base
    2,  # left_knee_link,    original 29 index: 3
    3,  # left_foot_link,    original 29 index: 5
    6,  # right_knee_link,   original 29 index: 9
    7,  # right_foot_link,   original 29 index: 11
    9,  # left_shoulder_link, original 29 index: 16
    10,  # left_elbow_link,    original 29 index: 18
    11,  # left_wrist/hand,    original 29 index: 21
    12,  # right_shoulder_link, original 29 index: 23
    13,  # right_elbow_link,    original 29 index: 25
    14,  # right_wrist/hand,    original 29 index: 28
]


def _load_frame_images(
    ep_dir: Path,
    frame: dict[str, Any],
    placeholder: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and decode the three camera images for one frame (runs in a worker thread)."""

    def _load(src_key: str) -> np.ndarray:
        path_str = frame.get(src_key)
        if path_str is None:
            return placeholder
        return np.asarray(
            Image.open(ep_dir / Path(path_str)).convert("RGB").resize(image_size),
            dtype=np.uint8,
        )

    return _load("front_head"), _load("left_hand"), _load("right_hand")


def _collect_episodes(input_dirs: list[Path]) -> list[Path]:
    episodes: list[Path] = []
    for input_dir in input_dirs:
        if not input_dir.is_dir():
            print(f"[WARNING] Input directory not found, skipping: {input_dir}")
            continue

        found = sorted(path for path in input_dir.glob("episode_*") if path.is_dir())
        valid = [episode for episode in found if (episode / SOURCE_JSON_NAME).exists()]
        print(f"{input_dir}: {len(valid)} episode(s) found")
        episodes.extend(valid)

    return episodes


def _states_to_vector(states: list[dict[str, Any]]) -> np.ndarray:
    vectors = []
    for state in states:
        imu = np.asarray(state["imu"], dtype=np.float64).reshape(-1)
        body_joint = np.asarray(state["body_joint"], dtype=np.float64).reshape(-1)
        hand_state = np.asarray(state["hand_state"], dtype=np.float64).reshape(-1)
        vectors.append(np.concatenate([imu, body_joint, hand_state], axis=0))
    return np.concatenate(vectors, axis=0)


def _action_to_vector(actions: list[dict[str, Any]]) -> np.ndarray:
    vectors = []
    for action in actions:
        hand_cmd = np.asarray(action["hand_cmd"], dtype=np.float64)
        mocap = np.asarray(action["mocap"], dtype=np.float64)
        mocap_velocity = mocap[:3]
        mocap_relative = mocap[3:].reshape(-1, 9)
        mocap_relative = mocap_relative[SELECT_11_INDICES]
        mocap_rel_xyz = mocap_relative[:, :3].flatten()
        mocap_rel_rot = mocap_relative[:, 3:].flatten()
        mocap_rel_final = np.concatenate([mocap_velocity, mocap_rel_xyz, mocap_rel_rot], axis=0)
        vectors.append(np.concatenate([mocap_rel_final, hand_cmd], axis=0))
    return np.concatenate(vectors, axis=0)


def split_front_and_body(
    frames: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    list_front: list[dict[str, Any]] = []
    list_body: list[dict[str, Any]] = []
    seen_front: set[str] = set()

    for frame_idx, frame in enumerate(frames):
        if "front_head" not in frame:
            raise ValueError(f"Frame {frame_idx} is missing the 'front_head' key")
        if "left_hand" not in frame:
            raise ValueError(f"Frame {frame_idx} is missing the 'left_hand' key")
        if "right_hand" not in frame:
            raise ValueError(f"Frame {frame_idx} is missing the 'right_hand' key")
        if "task" not in frame:
            raise ValueError(f"Frame {frame_idx} is missing the 'task' key")
        for key in BODY_KEYS:
            if key not in frame:
                raise ValueError(f"Frame {frame_idx} is missing the '{key}' key")

        front_head = frame["front_head"]
        left_hand = frame["left_hand"]
        right_hand = frame["right_hand"]
        task = frame["task"]
        if front_head not in seen_front:
            seen_front.add(front_head)
            list_front.append({"front_head": front_head, "left_hand": left_hand, "right_hand": right_hand, "task": task, "body_index": frame_idx})

        list_body.append({key: frame[key] for key in BODY_KEYS})

    return list_front, list_body


def build_windowed_frames(
    list_front: list[dict[str, Any]],
    list_body: list[dict[str, Any]],
    body_fps: int,
    front_fps: int,
    interval_num: int,
) -> list[dict[str, Any]]:
    new_list: list[dict[str, Any]] = []

    for t, front_frame in enumerate(list_front):
        center = int(front_frame["body_index"])
        left = center - interval_num
        right = center + interval_num

        if left < 0:
            left_pad = [list_body[0]] * (-left)
            left = 0
        else:
            left_pad = []

        if right >= len(list_body):
            right_pad = [list_body[-1]] * (right - len(list_body) + 1)
            right = len(list_body) - 1
        else:
            right_pad = []

        states = left_pad + list_body[left : center + 1]
        action = list_body[center : right + 1] + right_pad

        states = _states_to_vector(states)
        action = _action_to_vector(action)
        new_list.append(
            {
                "front_head": front_frame["front_head"],
                "left_hand": front_frame["left_hand"],
                "right_hand": front_frame["right_hand"],
                "states": states,
                "action": action,
                "task": front_frame["task"],
            }
        )
    return new_list


def load_windowed_episode(ep_dir: Path, args: "DataSetArgs") -> list[dict[str, Any]]:
    with (ep_dir / SOURCE_JSON_NAME).open("r", encoding="utf-8") as f:
        raw_frames = json.load(f)

    if not isinstance(raw_frames, list):
        raise ValueError(f"Expected a list in {ep_dir / SOURCE_JSON_NAME}")

    list_front, list_body = split_front_and_body(raw_frames)
    return build_windowed_frames(
        list_front=list_front,
        list_body=list_body,
        body_fps=args.body_fps,
        front_fps=args.fps,
        interval_num=args.interval_num,
    )


def convert(args: "DataSetArgs") -> None:
    episode_dirs = _collect_episodes(args.input_dirs)
    print(f"Total: {len(episode_dirs)} episode(s)\n")

    output_path = Path(args.output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type=args.robot_type,
        fps=args.fps,
        features=args.features,
        root=output_path,
        image_writer_threads=args.image_writer_threads,
    )
    dataset.meta.info["num_state_action_per_frame"] = args.interval_num + 1

    for ep_idx, ep_dir in enumerate(episode_dirs):
        print(f"[{ep_idx + 1}/{len(episode_dirs)}] {ep_dir}")
        windowed_frames = load_windowed_episode(ep_dir, args)

        if args.save_window_json:
            json_frames = [
                {
                    **frame,
                    "states": frame["states"].tolist(),
                    "action": frame["action"].tolist(),
                }
                for frame in windowed_frames
            ]
            with (ep_dir / OUTPUT_JSON_NAME).open("w", encoding="utf-8") as f:
                json.dump(json_frames, f, ensure_ascii=False, indent=2)

        image_size = (args.image_shape[1], args.image_shape[0])
        instruction = None
        with ThreadPoolExecutor(max_workers=args.image_loader_threads) as loader:
            img_futures = [
                loader.submit(
                    _load_frame_images,
                    ep_dir,
                    frame,
                    args.placeholder_image,
                    image_size,
                )
                for frame in windowed_frames
            ]

            for frame, img_future in zip(windowed_frames, img_futures):
                head_img, left_wrist_img, right_wrist_img = img_future.result()
                states = frame.get("states", None)
                action = frame.get("action", None)
                task = frame.get("task", None)
                instruction = task[0]
            
                assert states is not None, f"states missing in {ep_dir}"
                assert action is not None, f"action missing in {ep_dir}"
                assert task is not None and isinstance(task, list), (
                    f"task missing or invalid in {ep_dir}"
                )

                formatted_frame = {
                    "observation.state": states,
                    "observation.images.head_img": head_img,
                    "observation.images.left_wrist_img": left_wrist_img,
                    "observation.images.right_wrist_img": right_wrist_img,
                    "action": action,
                    "task_vis_stickman": np.zeros((900,), dtype=np.float64),  # placeholder
                }
                dataset.add_frame(formatted_frame)

        dataset.save_episode(task=instruction)

    print(f"\nDone! {len(episode_dirs)} episodes saved to {output_path}")


@dataclass
class DataSetArgs:
    input_dirs: list[Path] = field(
        default_factory=lambda: [
            Path("/liujinxin/dataset/piper/G1/0629_pick_cube_bottle_g1"),
            Path("/liujinxin/dataset/piper/G1/0630_pick_cube_bottle_g1_sink"),
        ]
    )
    """One or more directories, each containing episode_* subfolders."""

    output_dir: Path = Path("/liujinxin/liyifan/Isaac-GR00T/dataset/G1_hand_window_pick_water_bowl_sink_0609-0610")
    """Output directory for the LeRobot v2 dataset."""

    fps: int = 20
    """Frame rate of front images and the generated LeRobot dataset."""

    body_fps: int = 50
    """Frame rate of body_joint / imu / mocap in data_root_relative_6D.json."""

    repo_id: str = "lerobot_datasets/g1_real_6d_window_cont_rel"
    """LeRobot repo ID written into meta/info.json."""

    robot_type: str = "unitree_g1_29dof_hand"
    """Robot type tag written into meta/info.json."""

    image_writer_threads: int = 8
    """Number of threads for writing images."""

    image_loader_threads: int = 8
    """Number of threads for loading images from disk."""

    image_shape: tuple[int, int, int] = (256, 256, 3)
    """(H, W, C) shape for the front camera image."""

    cameras: list[str] = field(
        default_factory=lambda: ["head_img", "left_wrist_img", "right_wrist_img"]
    )
    """Camera keys used to build observation.images.* features."""

    data_mode: str = "continuous"

    save_window_json: bool = False
    """Also write data_standard_6D_window.json into each episode directory."""

    def __post_init__(self):
        self.placeholder_image = np.zeros(self.image_shape, dtype=np.uint8)
        assert self.body_fps >= self.fps, (
            "body_fps usually are greater than or equal to fps, check your data."
        )

        if self.data_mode in ["continuous"]:
            self.interval_num = int(self.body_fps * 1) - 1         # use 1s history and 1s future, -1 just for counting convenience
        else:
            self.interval_num = (self.body_fps // self.fps) - 2
            if self.interval_num < 0:
                self.interval_num = 0

        self.features = {
            "observation.state": {
                "dtype": "float64",
                "shape": ((self.interval_num + 1) * (6 + 29 + 12),),
                "names": None,
            },
            "action": {
                "dtype": "float64",
                "shape": ((self.interval_num + 1) * (3 + 11 * 9 + 12),),
                "names": None,
            },
            "task_vis_stickman": {
                "dtype": "float64",
                "shape": (900,),
                "names": None,
            },
            **{
                f"observation.images.{cam}": {
                    "dtype": "video",
                    "shape": self.image_shape,
                    "names": ["height", "width", "channel"],
                }
                for cam in self.cameras
            },
        }


if __name__ == "__main__":
    args = tyro.cli(DataSetArgs)
    if not args.input_dirs:
        raise ValueError("Provide at least one input directory via --input-dirs.")

    convert(args)
