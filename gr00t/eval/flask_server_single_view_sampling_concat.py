#!/usr/bin/env python3
"""
HTTP wrapper server for GR00T that is compatible with the user's existing client.

Supported request styles:
1) JSON body from the existing client:
   {"observation": {"video": ..., "state": ..., "language": ...}}
2) Multipart/form-data fallback:
   - json / instruction
   - img_scene / front / img_static
   - img_hand_left / left / img_gripper
   - right_hand / right
   - state / state.txt

Response format:
{
  "action": ...,
  "timing": {"total_ms": ..., "infer_ms": ...}
}
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
import os
import time
import traceback
from typing import Any

from flask import Flask, jsonify, request
import numpy as np
from PIL import Image
import torch
import tyro

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.replay_policy import ReplayPolicy

from PIL import Image

MODALITY_FILE  = "/liujinxin/liyifan/Isaac-GR00T/dataset/2026-04-24_G1_push_toy_downsamling_concat/meta/modality.json"
JSON_FILE_PATH = "/liujinxin/dataset/piper/G1/0428_tidy_up_g1/episode_6/data.json"
DEFAULT_MODEL_SERVER_PORT = 5555
IMAGE_SHAPE = (224, 224, 3)
FIELD_DTYPES = {
    "video": {
        "ego_view": np.uint8
    },
    "state": {
        "base_translation": np.float32,
        "base_rotation": np.float32,
        "left_leg": np.float32,
        "right_leg": np.float32,
        "waist": np.float32,
        "left_arm": np.float32,
        "right_arm": np.float32,
    },
}
OUTPUT_FIELD = [
        "left_leg",
        "right_leg",
        "waist",
        "left_arm",
        "right_arm",
        "base_rotation",
        ]

CONCAT_FACTOR = 3

index = 0
warm_up = 0
@dataclass
class ServerConfig:
    """Configuration for running the GR00T HTTP inference server."""

    # GR00T policy configs
    model_path: str | None = None
    """Path to the model checkpoint directory."""

    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    """Embodiment tag."""

    device: str = "cuda"
    """Device to run the model on."""

    # Replay policy configs
    dataset_path: str | None = None
    """Path to the dataset for replay trajectory."""

    modality_config_path: str | None = None
    """Path to the modality configuration file."""

    execution_horizon: int | None = None
    """Policy execution horizon during inference."""

    # Server configs
    host: str = "0.0.0.0"
    """Host address for the server."""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server."""

    strict: bool = True
    """Whether to enforce strict input and output validation."""

    use_sim_policy_wrapper: bool = False
    """Whether to use the sim policy wrapper."""

def _normalize_policy_output(result: Any) -> np.ndarray:
    if isinstance(result, dict):
        action = result.get("actions")
        if action is None:
            action = result.get("action")
        if action is None:
            raise ValueError(f"Model returned dict without 'actions' or 'action': {result.keys()}")
        return np.asarray(action)
    return np.asarray(result)


def create_policy(config: ServerConfig):
    if config.model_path is not None and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    if config.model_path is not None:
        policy = Gr00tPolicy(
            embodiment_tag=config.embodiment_tag,
            model_path=config.model_path,
            device=config.device,
            strict=config.strict,
        )
    elif config.dataset_path is not None:
        if config.modality_config_path is None:
            from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

            modality_configs = MODALITY_CONFIGS[config.embodiment_tag.value]
        else:
            with open(config.modality_config_path, "r", encoding="utf-8") as f:
                modality_configs = json.load(f)
        policy = ReplayPolicy(
            dataset_path=config.dataset_path,
            modality_configs=modality_configs,
            execution_horizon=config.execution_horizon,
            strict=config.strict,
        )
    else:
        raise ValueError("Either model_path or dataset_path must be provided")

    if config.use_sim_policy_wrapper:
        from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

        policy = Gr00tSimPolicyWrapper(policy)

    return policy

def load_image(image_path, image_type):
    image = Image.open(image_path)

    if image.mode != 'RGB':
        image = image.convert('RGB')
    if image_type == "left_wrist_view":
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

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
    result = np.array(canvas)
    return result[None,None,...]

def create_app(policy) -> Flask:
    app = Flask(__name__)

    with open(JSON_FILE_PATH, 'r') as f:
        json_data = json.load(f)
        total_indices = len(json_data)

    @app.route("/ping", methods=["GET"])
    def ping():
        return jsonify({"status": "ok", "message": "Server is alive"})

    @app.route("/predict", methods=["POST"])
    def predict():
        global index
        global warm_up
        start_total = time.perf_counter_ns()
        try:
            payload = request.get_json(silent=True)
            data = payload["observation"]

            for group, fields in FIELD_DTYPES.items():
                if group not in data or not isinstance(data[group], dict):
                    continue
                for key, dtype in fields.items():
                    if key in data[group] and data[group][key] is not None:
                        data[group][key] = np.asarray(data[group][key], dtype=dtype)

            if index == 0 and warm_up == 0:
                index = 0
                warm_up = 1
            else:
                index = index + 60
            if index > total_indices - 1:
                index = 0
                print('Reset to first frame images')
            root_dir = os.path.dirname(JSON_FILE_PATH)
            image_index = index // 6
            image_map = {
            "ego_view" : f"{root_dir}/images/front_head/{image_index:05d}.png"}

            for k, v in image_map.items():
                data['video'][k] = load_image(v, k)
            
            # data['language']['annotation.human.task_description']=data['language']['annotation.human.task_description'][0]
            start_inf = time.perf_counter_ns()
            with torch.inference_mode():
                result = policy.get_action(data)
            end_inf = time.perf_counter_ns()

            actions = _normalize_policy_output(result)[0]
            end_total = time.perf_counter_ns()

            total_ms = (end_total - start_total) / 1e6
            infer_ms = (end_inf - start_inf) / 1e6

            action_list = []
            _, t, _ = actions['base_translation'].shape
            for seq in range(t):
                for i in range(CONCAT_FACTOR):
                    action = []
                    for v in OUTPUT_FIELD:
                        joint_concat = actions[v].squeeze(0)
                        dim_concat = joint_concat.shape[-1]
                        dim = dim_concat // CONCAT_FACTOR
                        start = i * dim
                        end = (i + 1) * dim
                        joint = joint_concat[seq,start:end]
                        action.append(joint)
                    action = np.concatenate(action, axis=-1)
                    action_list.append(action)
            result = np.stack(action_list, axis=0)
            print(f"[INFO] action shape: {result.shape}")
            print(f"[INFO] total time: {total_ms:.2f} ms, model time: {infer_ms:.2f} ms")

            return jsonify(
                {
                    "action": result.tolist(),
                    "timing": {
                        "total_ms": total_ms,
                        "infer_ms": infer_ms,
                    },
                }
            )
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    return app


def main(config: ServerConfig):
    print("Starting GR00T HTTP inference server...")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Dataset path: {config.dataset_path}")
    print(f"  Device: {config.device}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")
    print(f"  Strict: {config.strict}")

    policy = create_policy(config)
    app = create_app(policy)
    app.run(host=config.host, port=config.port)


if __name__ == "__main__":
    config = tyro.cli(ServerConfig)
    main(config)
