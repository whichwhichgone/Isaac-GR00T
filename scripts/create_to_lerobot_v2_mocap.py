#!/usr/bin/env python3

import json
import os
import sys
import glob
import shutil
import logging
from dataclasses import dataclass, field
from pathlib import Path
from multiprocessing import Pool
from typing import Any, Iterator

import tyro
import numpy as np
from PIL import Image

# =========================
# LeRobot path
# =========================
LEROBOT_SRC = "/liujinxin/liyifan/Isaac-GR00T/third_party/lerobot-main"
sys.path.insert(0, LEROBOT_SRC)

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


# =========================
# Global constants
# =========================
IMAGE_SHAPE = (256, 256, 3)
STRIDE = 1
CAMERAS = [
    "head_img",
    "left_wrist_img",
    "right_wrist_img",
]

# LeRobot key -> your json image key
CAMERAS_MAP = {
    "head_img": "front",
    "left_wrist_img": "left",
    "right_wrist_img": "right",
}

MOCAP_LIST = [0, 2, 3, 6, 7, 9, 10, 11, 12, 13, 14]

FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (35,),
        "names": (
            [f"root_rot_6d_{i}" for i in range(6)]
            + [f"joint_{i}" for i in range(29)]
        ),
    },
    "action": {
        "dtype": "float32",
        "shape": (102,),
        "names": (
            ["root", 
             "left_hip_roll_joint", "left_knee_joint", "left_ankle_roll_joint", "left_ankle_roll_joint",
             "right_hip_roll_joint", "right_knee_joint", "right_ankle_roll_joint", "right_ankle_roll_joint",
             "left_shoulder_roll_joint", "left_elbow_joint", "left_wrist_yaw_joint",
             "right_shoulder_roll_joint", "right_elbow_joint", "right_wrist_yaw_joint"
             ]
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


# =========================
# Config
# =========================
@dataclass
class Config:
    dataset_dir: list[Path] = field(
        default_factory=lambda: [
            Path("/liujinxin/dataset/piper/G1/0518_clean_desk_place_sofa_g1_B"),
            Path("/liujinxin/dataset/piper/G1/0518_clean_desk_place_sofa_g1_fast"),
            Path("/liujinxin/dataset/piper/G1/0519_clean_desk_place_sofa_g1_B"),
            Path("/liujinxin/dataset/piper/G1/0520_clean_desk_place_sofa_g1_B"),
            Path("/liujinxin/dataset/piper/G1/0521_clean_desk_place_sofa_g1_B"),
            Path("/liujinxin/dataset/piper/G1/0525_clean_desk_place_sofa_g1_B"),
            Path("/liujinxin/dataset/piper/G1/0526_clean_desk_place_sofa_g1_B"),
            Path("/liujinxin/dataset/piper/G1/0527_clean_desk_place_sofa_g1_B"),
            Path("/liujinxin/dataset/piper/G1/0528_clean_desk_place_sofa_g1_B"),
            Path("/liujinxin/dataset/piper/G1/0529_clean_desk_place_sofa_g1_B"),
            Path("/liujinxin/dataset/piper/G1/0601_clean_desk_place_sofa_g1_B"),
            Path("/liujinxin/dataset/piper/G1/0602_clean_desk_place_sofa_g1_B"),
            Path("/liujinxin/dataset/piper/G1/0603_clean_desk_place_sofa_g1_B"),
            Path("/liujinxin/dataset/piper/G1/0604_clean_desk_place_sofa_g1_B"),
        ]
    )

    modality: str = "/liujinxin/liyifan/Isaac-GR00T/scripts/modality.json"

    repo_id: str = "lerobot_datasets/g1_motion"   # 数据集所在文件夹
    repo_name: str = "0518-0604_clean_desk_place_sofa_g1_fast_root_rel"     # 输出的lerobot数据集名称

    robot_type: str = "unitree_g1_29dof"
    fps: int = 50

    output_dir: str | None = None

    use_relative: bool = True

    force: bool = True
    allow_missing_images: bool = False

    num_workers: int = 12
    chunksize: int = 2

    image_writer_threads: int = 4
    image_writer_processes: int = 2


# =========================
# Logger
# =========================
def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("g1_to_lerobot")


# =========================
# DataCollector
# =========================
class DataCollector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = logging.getLogger("g1_to_lerobot")

    def _get_joint_state(self, step: dict[str, Any]) -> np.ndarray:
        """
        构造 35 维 state:

            root_rotation:   6
            body_joint: 29

        最终：
            state.shape = (35,)
        """
        if "imu" not in step:
            raise KeyError("Missing key: imu")

        if "body_joint" not in step:
            raise KeyError("Missing key: body_joint")

        root_rot = np.asarray(step["imu"], dtype=np.float32).reshape(-1)

        joint_pos = np.asarray(step["body_joint"], dtype=np.float32).reshape(-1)

        if joint_pos.shape != (29,):
            raise ValueError(
                f"Invalid body_joint shape: {joint_pos.shape}, expected (29,)"
            )

        state = np.concatenate(
            [
                root_rot,
                joint_pos,
            ],
            axis=0,
        ).astype(np.float32)

        if state.shape != (35,):
            raise ValueError(f"Invalid state shape: {state.shape}, expected (35,)")

        return state

    def _get_task(self, data: list[dict[str, Any]], json_path: str) -> str:
        if len(data) == 0:
            raise ValueError(f"Empty episode: {json_path}")

        if "task" not in data[0]:
            raise KeyError(f"Missing task in first step: {json_path}")

        task = data[0]["task"]

        if isinstance(task, list):
            if len(task) == 0:
                raise ValueError(f"Empty task list in: {json_path}")
            task = task[0]

        if not isinstance(task, str):
            task = str(task)

        return task

    def _get_episode(self, json_path: str) -> list[dict[str, Any]]:
        """
        读取一个 data.json，构造一个 episode。

        每个 frame 包含：
            observation.images.head_img
            observation.images.left_wrist_img
            observation.images.right_wrist_img
            observation.state
            action
            task_vis_stickman
            task
        """
        stride = STRIDE
        root_path = os.path.dirname(json_path)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not data or len(data) <= stride:
            raise ValueError(f"Data length is {len(data)}, is not enough!")

        if not isinstance(data, list):
            raise ValueError(f"JSON root should be list: {json_path}")

        if len(data) < 2:
            raise ValueError(f"Episode too short: {json_path}, len={len(data)}")

        task = self._get_task(data, json_path)

        episode = []

        mocap_data = []
        for idx in range(len(data)):
            step = data[idx]
            if "mocap" not in step:
                raise KeyError(f"Missing key: mocap in {json_path}, step={idx}")
            mocap = np.asarray(step["mocap"], dtype=np.float32)
            root_delta = mocap[:3]
            mocap_without_root = mocap[3:].reshape(15, 9)[MOCAP_LIST]
            mocap_without_root = mocap_without_root.reshape(-1)
            mocap_with_root = np.concatenate([root_delta, mocap_without_root])
            mocap_data.append(mocap_with_root)
        mocap_data = np.asarray(mocap_data, dtype=np.float32)

        if mocap_data.ndim != 2 or mocap_data.shape[-1] != 102:
            raise ValueError(f"Expected mocap data shape (T, 102), got {mocap_data.shape}")

        for idx in range(0, len(data) - stride, stride):
            step = data[idx]

            frame = {}

            for lerobot_key, g1_key in CAMERAS_MAP.items():
                if g1_key not in step:
                    image_path = None
                else:
                    image_path = os.path.join(root_path, step[g1_key])

                frame[f"__image_path__.{lerobot_key}"] = image_path

            state = self._get_joint_state(step)

            action = mocap_data[idx + stride]

            frame["observation.state"] = state
            frame["action"] = action
            frame["task_vis_stickman"] = np.zeros(18, dtype=np.float32)
            frame["task"] = task

            episode.append(frame)

        return episode

    def _get_json_list(self) -> list[str]:
        json_files = []

        for input_dir in self.cfg.dataset_dir:
            pattern = os.path.join(input_dir, "**", "data_root_relative_6D.json")
            found = sorted(glob.glob(pattern, recursive=True))
            json_files.extend(found)

        json_files = sorted(json_files)
        return json_files

    def collector(self) -> Iterator[tuple[str, list[dict[str, Any]] | None, str | None]]:
        """
        流式返回 episode。

        yield:
            json_path, episode, error

        注意：
            这里不会把所有 episode 都存进一个 results 列表。
            Pool 每处理完一个 episode，主进程就可以立刻拿到。
        """
        json_files = self._get_json_list()

        self.logger.info("Found %d json files", len(json_files))

        if len(json_files) == 0:
            return

        if self.cfg.num_workers <= 1:
            for json_path in json_files:
                yield _process_one_json((self.cfg, json_path))
            return

        worker_args = [(self.cfg, json_path) for json_path in json_files]

        with Pool(processes=self.cfg.num_workers) as pool:
            for result in pool.imap_unordered(
                _process_one_json,
                worker_args,
                chunksize=self.cfg.chunksize,
            ):
                yield result


def _process_one_json(
    args: tuple[Config, str],
) -> tuple[str, list[dict[str, Any]] | None, str | None]:
    """
    Pool worker 函数。

    重要：
        worker 里面只做数据预处理。
        不要在 worker 里面创建 LeRobotDataset。
        不要在 worker 里面 dataset.add_frame()。
    """
    cfg, json_path = args

    try:
        collector = DataCollector(cfg)

        episode = collector._get_episode(json_path)
        return json_path, episode, None

    except Exception as e:
        return json_path, None, repr(e)


# =========================
# LerobotBuilder
# =========================
class LerobotBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = logging.getLogger("g1_to_lerobot")

    def _load_image(self, image_path: str) -> np.ndarray:
        if image_path is None or not os.path.exists(image_path):
            return PLACEHOLDER_IMAGE.copy()

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        target_h, target_w = IMAGE_SHAPE[:2]

        resized = image.resize((target_h, target_w), Image.BILINEAR)

        image_array = np.asarray(resized, dtype=np.uint8)

        if image_array.shape != IMAGE_SHAPE:
            raise ValueError(
                f"Invalid image shape: {image_array.shape}, expected {IMAGE_SHAPE}"
            )

        return image_array

    def _attach_images(
        self,
        frame: dict[str, Any],
    ) -> dict[str, Any]:
        """
        在主进程 add_frame 前加载图像。

        worker 只返回图像路径，避免把大尺寸 image ndarray
        通过 multiprocessing pickle 回传到主进程。
        """
        for lerobot_key in CAMERAS:
            path_key = f"__image_path__.{lerobot_key}"
            image_key = f"observation.images.{lerobot_key}"

            image_path = frame.pop(path_key, None)

            if image_path is None:
                image = PLACEHOLDER_IMAGE.copy()
            else:
                image = self._load_image(image_path)

            frame[image_key] = image

        return frame

    def _get_output_dir(self) -> Path:
        if self.cfg.output_dir is not None:
            return Path(self.cfg.output_dir)

        return LEROBOT_HOME / self.cfg.repo_name

    def _prepare_output_dir(self, output_dir: Path) -> None:
        if output_dir.exists():
            if not self.cfg.force:
                raise FileExistsError(
                    f"Output dir already exists: {output_dir}\n"
                    f"Use --force if you want to remove it."
                )

            self.logger.warning("Removing existing output dir: %s", output_dir)
            shutil.rmtree(output_dir)

    def _copy_modality(self, output_dir: Path) -> None:
        if not self.cfg.modality:
            return

        modality_path = Path(self.cfg.modality)

        if not modality_path.exists():
            self.logger.warning("Modality file not found, skip: %s", modality_path)
            return

        meta_dir = output_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        dst_path = meta_dir / modality_path.name
        shutil.copy2(modality_path, dst_path)

        self.logger.info("Copied modality file to: %s", dst_path)

    def builder(self) -> None:
        output_dir = self._get_output_dir()
        self._prepare_output_dir(output_dir)

        self.logger.info("Creating LeRobotDataset...")
        self.logger.info("repo_id: %s", self.cfg.repo_id)
        self.logger.info("robot_type: %s", self.cfg.robot_type)
        self.logger.info("fps: %s", self.cfg.fps)
        self.logger.info("output_dir: %s", output_dir)
        self.logger.info("use_relative: %s", self.cfg.use_relative)


        dataset = LeRobotDataset.create(
            repo_id=self.cfg.repo_id,
            robot_type=self.cfg.robot_type,
            fps=self.cfg.fps,
            features=FEATURES,
            root=output_dir,
            image_writer_threads=self.cfg.image_writer_threads,
            image_writer_processes=self.cfg.image_writer_processes,
        )

        collector = DataCollector(self.cfg)

        success_count = 0
        failed_count = 0
        total_frames = 0

        for json_path, episode, error in collector.collector():
            if episode is None:
                failed_count += 1
                self.logger.warning(
                    "Skip invalid episode: %s | error=%s",
                    json_path,
                    error,
                )
                continue

            if len(episode) == 0:
                failed_count += 1
                self.logger.warning("Skip empty episode: %s", json_path)
                continue

            task = episode[0]["task"]

            for frame in episode:
                frame.pop("task", None)
                frame = self._attach_images(frame)
                dataset.add_frame(frame)

            dataset.save_episode(task=task)

            success_count += 1
            total_frames += len(episode)

            self.logger.info(
                "Saved episode %d | frames=%d | task=%s | source=%s",
                success_count,
                len(episode),
                task,
                json_path,
            )

        if success_count == 0:
            raise RuntimeError("No valid episodes were converted.")


        self._copy_modality(output_dir)

        self.logger.info("Done.")
        self.logger.info("Success episodes: %d", success_count)
        self.logger.info("Failed episodes: %d", failed_count)
        self.logger.info("Total frames: %d", total_frames)
        self.logger.info("Output dir: %s", output_dir)


# =========================
# Main
# =========================
if __name__ == "__main__":
    setup_logger()

    cfg = tyro.cli(Config)

    builder = LerobotBuilder(cfg)
    builder.builder()