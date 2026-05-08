      
import argparse
import os
import time
import numpy as np
import torch
from flask import Flask, request, jsonify
import cv2
from openpi.training import config as _config
from openpi.shared import download
from openpi.policies import policy_config as _policy_config
import json

index = 0
signal = 0

def to_numpy(x, dtype=None):
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def ensure_video_5d(arr, name):
    """
    期望: (B, T, H, W, C)
    常见输入:
      - (H, W, C) -> (1, 1, H, W, C)
      - (T, H, W, C) -> (1, T, H, W, C)
      - (B, T, H, W, C) -> 原样
    """
    arr = to_numpy(arr, np.uint8)

    if arr.ndim == 3:
        arr = arr[None, None, ...]
    elif arr.ndim == 4:
        arr = arr[None, ...]
    elif arr.ndim == 5:
        pass
    else:
        raise ValueError(f"{name} must be 3D/4D/5D, got shape={arr.shape}")

    return arr


def ensure_state_btd(arr, name, dim):
    """
    期望: (B, T, D)
    常见输入:
      - (D,) -> (1, 1, D)
      - (T, D) -> (1, T, D)
      - (B, T, D) -> 原样
    """
    arr = to_numpy(arr, np.float32)

    if arr.ndim == 1:
        if arr.shape[0] != dim:
            raise ValueError(f"{name} last dim mismatch, expected {dim}, got {arr.shape}")
        arr = arr[None, None, :]
    elif arr.ndim == 2:
        if arr.shape[-1] != dim:
            raise ValueError(f"{name} last dim mismatch, expected {dim}, got {arr.shape}")
        arr = arr[None, :, :]
    elif arr.ndim == 3:
        if arr.shape[-1] != dim:
            raise ValueError(f"{name} last dim mismatch, expected {dim}, got {arr.shape}")
    else:
        raise ValueError(f"{name} must be 1D/2D/3D, got shape={arr.shape}")

    return arr


def normalize_observation(obs: dict):
    if "video" not in obs:
        raise ValueError("Missing observation.video")
    if "state" not in obs:
        raise ValueError("Missing observation.state")
    if "language" not in obs:
        raise ValueError("Missing observation.language")

    video_in = obs["video"]
    state_in = obs["state"]
    language_in = obs["language"]

    video = {
        "ego_view": ensure_video_5d(video_in["ego_view"], "video.ego_view"),
        "left_wrist_view": ensure_video_5d(video_in["left_wrist_view"], "video.left_wrist_view"),
        "right_wrist_view": ensure_video_5d(video_in["right_wrist_view"], "video.right_wrist_view"),
    }

    state = {
        "base_translation": ensure_state_btd(state_in["base_translation"], "state.base_translation", 3),
        "base_rotation": ensure_state_btd(state_in["base_rotation"], "state.base_rotation", 4),
        "left_leg": ensure_state_btd(state_in["left_leg"], "state.left_leg", 6),
        "right_leg": ensure_state_btd(state_in["right_leg"], "state.right_leg", 6),
        "waist": ensure_state_btd(state_in["waist"], "state.waist", 3),
        "left_arm": ensure_state_btd(state_in["left_arm"], "state.left_arm", 7),
        "right_arm": ensure_state_btd(state_in["right_arm"], "state.right_arm", 7),
    }

    # language 这里保持你现在发送端的结构
    if "annotation.human.task_description" not in language_in:
        raise ValueError("Missing language.annotation.human.task_description")

    language = {
        "annotation.human.task_description": language_in["annotation.human.task_description"]
    }

    return {
        "video": video,
        "state": state,
        "language": language,
    }


def action_to_jsonable(action_dict: dict):
    out = {}
    for k, v in action_dict.items():
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        else:
            v = np.asarray(v)
        out[k] = v.tolist()
    return out


class LLMRobotServer:
    def __init__(self, config_name: str, checkpoint_path: str):
        config = _config.get_config(config_name)
        checkpoint_dir = download.maybe_download(checkpoint_path)
        self.policy = _policy_config.create_trained_policy(config, checkpoint_dir)


def create_app(server: LLMRobotServer):
    app = Flask(__name__)

    with open("/liujinxin/dataset/piper/0401_test_unitree/episode_98/data.json", "r") as f:
        data = json.load(f)
        total_indices = len(data)
    
    @app.route("/ping", methods=["GET"])
    def ping():
        return jsonify({"status": "ok"})

    @app.route("/predict", methods=["POST"])
    def predict():
        global index
        global signal
        start_time = time.perf_counter_ns()

        try:
            payload = request.get_json(force=True)
            if payload is None:
                raise ValueError("Request JSON is empty")

            if "observation" not in payload:
                raise ValueError("Missing 'observation' field")

            observation = normalize_observation(payload["observation"])

            observation['state'] = np.concatenate(
                [
                    observation['state']['left_leg'].squeeze(),
                    observation['state']['right_leg'].squeeze(),
                    observation['state']['waist'].squeeze(),
                    observation['state']['left_arm'].squeeze(),
                    observation['state']['right_arm'].squeeze(),
                    observation['state']['base_rotation'].squeeze(),
                ],
                axis=-1
            )
            # data = {
            #     "prompt": "take three steps forward, then turn left and take one step before waving right hand",
            #     "state": observation['state'],
            #     "front_head": observation['video']['ego_view'].squeeze(),
            #     "left_hand": observation['video']['left_wrist_view'].squeeze(),
            #     "right_hand": observation['video']['right_wrist_view'].squeeze(),
            # }

            # with open("debug_g1_data.json", "w") as f:
            #     json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in data.items()}, f)

            # read image of frame 1 of in advance.
            if index == 0 and signal < 5:
                index = 0
                signal += 1
            # else:
            #     index += 1
                print("Now signal is ", signal)
                print("Use first frame images")
                
            else:
                index = index + 50
                print(f"Use frame {index}")
            
            if index > total_indices - 1:
                index = 0
                print("Reset to first frame images")
            front_head_path = f"/liujinxin/dataset/piper/0401_test_unitree/episode_98/images/front_head/{index:05d}.png"
            left_hand_path = f"/liujinxin/dataset/piper/0401_test_unitree/episode_98/images/left_hand/{index:05d}.png"
            right_hand_path = f"/liujinxin/dataset/piper/0401_test_unitree/episode_98/images/right_hand/{index:05d}.png"
            front_head = cv2.cvtColor(cv2.imread(front_head_path), cv2.COLOR_BGR2RGB)
            left_hand = cv2.cvtColor(cv2.imread(left_hand_path), cv2.COLOR_BGR2RGB)
            right_hand = cv2.cvtColor(cv2.imread(right_hand_path), cv2.COLOR_BGR2RGB)

            data = {
                "prompt": "take three steps forward, then turn left and take one step before waving right hand",
                "state": torch.from_numpy(observation['state']),
                "front_head": torch.from_numpy(front_head),
                "left_hand": torch.from_numpy(left_hand),
                "right_hand": torch.from_numpy(right_hand),
            }

            result = server.policy.infer(data)

            # 兼容两种常见返回：
            # 1) result 直接是 action dict
            # 2) result = {"actions": ..., "policy_timing": ...}
            if isinstance(result, dict) and "actions" in result:
                actions = result["actions"]
                policy_timing = result.get("policy_timing", {})
            else:
                actions = result
                policy_timing = {}

            # if isinstance(actions, (list, tuple)):
            #     # 如果是 list/tuple，默认取第一个
            #     action0 = actions[0]
            # else:
            #     action0 = actions

            end_time = time.perf_counter_ns()

            return jsonify({
                # "action": action_to_jsonable(action0),
                "action": actions.tolist(),
                "timing": {
                    "total_ms": (end_time - start_time) / 1e6,
                    "infer_ms": policy_timing.get("infer_ms", None),
                }
            })

        except Exception as e:
            return jsonify({
                "error": str(e),
            }), 400

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--config-name", type=str, default="pi05_piper_sweater_box_bz64")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="/liujinxin/zbh/openpi/checkpoints/pi05_piper_sweater_box_bz64/pi05_piper_sweater_box_bz64/74000",
    )
    args = parser.parse_args()

    server = LLMRobotServer(
        config_name=args.config_name,
        checkpoint_path=args.checkpoint_path,
    )
    app = create_app(server)
    app.run(host=args.host, port=args.port, threaded=True)