      
from dataclasses import dataclass
import json
import sys
import numpy as np
import tyro
import requests
import os 


@dataclass
class ClientConfig:
    # host: str = "172.16.78.10"
    host: str = "127.0.0.1"
    port: int = 9002
    timeout_ms: int = 150000
    json_file_path: str = "/liujinxin/liyifan/Isaac-GR00T/scripts/open_loop_eval_1_actions.json"
    task_description: str = "take three steps forward, then turn left and take one step before waving right hand"
    frame_index: int = 0
    save_action_path: str = None


def _to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def get_action_via_http(observation: dict, host: str, port: int, timeout_ms: int = 15000):
    url = f"http://{host}:{port}/predict"
    payload = {
        "observation": _to_serializable(observation)
    }

    resp = requests.post(url, json=payload, timeout=timeout_ms / 1000.0)
    resp.raise_for_status()

    result = resp.json()
    if "action" not in result:
        raise ValueError(f"Server response missing 'action': {result}")

    return np.asarray(result["action"], dtype=np.float32), result


def load_open_loop_actions_json(json_file_path: str) -> list:
    with open(json_file_path, "r", encoding="utf-8") as f:
        data_json = json.load(f)

    if not isinstance(data_json, list) or len(data_json) == 0:
        raise ValueError(f"JSON 内容为空或格式不对: {json_file_path}")

    return data_json


def convert_open_loop_json_to_actions_json_data(data_json: list) -> dict:
    """
    输入每帧格式:
        [root_xyz(3), imu_wxyz(4), body_joint(29)]

    输出:
        {
            "base_translation": (1, T, 3),
            "base_rotation":    (1, T, 4),
            "left_leg":         (1, T, 6),
            "right_leg":        (1, T, 6),
            "waist":            (1, T, 3),
            "left_arm":         (1, T, 7),
            "right_arm":        (1, T, 7),
        }
    """
    actions_json_data = {
        "base_translation": [],
        "base_rotation": [],
        "left_leg": [],
        "right_leg": [],
        "waist": [],
        "left_arm": [],
        "right_arm": [],
    }

    for i, step in enumerate(data_json):
        arr = np.asarray(step, dtype=np.float32).reshape(-1)
        if arr.shape[0] != 36:
            raise ValueError(f"第 {i} 帧 shape 错误，期望 (36,) 实际 {arr.shape}")

        xyz = arr[:3]
        imu = arr[3:7]
        body_joint = arr[7:]

        actions_json_data["base_translation"].append(xyz)
        actions_json_data["base_rotation"].append(imu)
        actions_json_data["left_leg"].append(body_joint[:6])
        actions_json_data["right_leg"].append(body_joint[6:12])
        actions_json_data["waist"].append(body_joint[12:15])
        actions_json_data["left_arm"].append(body_joint[15:22])
        actions_json_data["right_arm"].append(body_joint[22:29])

    for k in actions_json_data:
        actions_json_data[k] = np.asarray(actions_json_data[k], dtype=np.float32)[None, ...]

    return actions_json_data


def build_observation_from_actions_json_data(
    actions_json_data: dict,
    task_description: str,
    frame_index: int = 0,
):
    def take_frame(x, name, expected_dim, frame_index):
        arr = np.asarray(x, dtype=np.float32)

        if arr.ndim == 3:
            # (1, T, D)
            if arr.shape[0] != 1 or arr.shape[2] != expected_dim:
                raise ValueError(f"{name} shape 错误: {arr.shape}")
            T = arr.shape[1]
            if not (0 <= frame_index < T):
                raise IndexError(f"{name}: frame_index={frame_index} 越界, T={T}")
            return arr[:, frame_index:frame_index + 1, :]

        if arr.ndim == 2:
            # (T, D)
            if arr.shape[1] != expected_dim:
                raise ValueError(f"{name} shape 错误: {arr.shape}")
            T = arr.shape[0]
            if not (0 <= frame_index < T):
                raise IndexError(f"{name}: frame_index={frame_index} 越界, T={T}")
            return arr[None, frame_index:frame_index + 1, :]

        if arr.ndim == 1:
            # (D,) -> (1,1,D)
            if arr.shape[0] != expected_dim:
                raise ValueError(f"{name} shape 错误: {arr.shape}")
            return arr[None, None, :]

        raise ValueError(f"{name} 不支持的 shape: {arr.shape}")

    state = {
        "base_translation": take_frame(actions_json_data["base_translation"], "base_translation", 3, frame_index),
        "base_rotation": take_frame(actions_json_data["base_rotation"], "base_rotation", 4, frame_index),
        "left_leg": take_frame(actions_json_data["left_leg"], "left_leg", 6, frame_index),
        "right_leg": take_frame(actions_json_data["right_leg"], "right_leg", 6, frame_index),
        "waist": take_frame(actions_json_data["waist"], "waist", 3, frame_index),
        "left_arm": take_frame(actions_json_data["left_arm"], "left_arm", 7, frame_index),
        "right_arm": take_frame(actions_json_data["right_arm"], "right_arm", 7, frame_index),
    }

    video = {
        "ego_view": np.zeros((1, 1, 224, 224, 3), dtype=np.uint8),
        "left_wrist_view": np.zeros((1, 1, 224, 224, 3), dtype=np.uint8),
        "right_wrist_view": np.zeros((1, 1, 224, 224, 3), dtype=np.uint8),
    }

    observation = {
        "video": video,
        "state": state,
        "language": {
            "annotation.human.task_description": [[task_description]]
        }
    }
    return observation


if __name__ == "__main__":
    config = tyro.cli(ClientConfig)
    while True:
        try:
            # 1. ping 服务
            r = requests.get(f"http://{config.host}:{config.port}/ping", timeout=3)
            r.raise_for_status()
            print("Server is alive!")

            # 2. 读取 json
            # --- debug ---
            # if os.path.exists("/home/binghong/Documents/openpi_g1_deploy/play_actions_2.json"):
            #     config.json_file_path = "/home/binghong/Documents/openpi_g1_deploy/play_actions_2.json"
            # --- debug ---

            data_json = load_open_loop_actions_json(config.json_file_path)
            print(f"loaded json: {config.json_file_path}")
            print(f"num_frames: {len(data_json)}")

            # 3. 处理成 actions_json_data
            actions_json_data = convert_open_loop_json_to_actions_json_data(data_json)
            print("parsed actions_json_data:")
            for k, v in actions_json_data.items():
                print(f"  {k}: shape={v.shape}")

            # 4. 取某一帧构造 observation
            # --- debug ---
            # if os.path.exists("/home/binghong/Documents/openpi_g1_deploy/play_actions_2.json"):
            #     config.frame_index = actions_json_data['base_translation'].shape[1] - 1
            # --- debug ---

            observation = build_observation_from_actions_json_data(
                actions_json_data=actions_json_data,
                task_description=config.task_description,
                frame_index=config.frame_index,
            )
            print(f"build observation success, frame_index={config.frame_index}")

            # 5. 发请求并接收 action
            action_arr, full_result = get_action_via_http(
                observation=observation,
                host=config.host,
                port=config.port,
                timeout_ms=config.timeout_ms,
            )

            print("=" * 60)
            print("receive action success")
            print("action shape:", action_arr.shape)
            print("action dtype:", action_arr.dtype)
            print(action_arr)
            print("elapsed_total_time_ms:", full_result["timing"]['total_ms'])
            print("elapsed_infer_ms:", full_result["timing"]['infer_ms'])
            print("=" * 60)

            # --- debug ---
            # new_action_arr = np.concatenate([action_arr[:, -4:], action_arr[:, :-4]], axis=1)
            # # pad 0
            # prefix = np.zeros((50, 3))
            # new_arr = np.concatenate([prefix, new_action_arr], axis=1)

            # save_path = "/home/binghong/Documents/openpi_g1_deploy/play_actions_2.json"
            # if os.path.exists(save_path):
            #     with open(save_path, "r", encoding="utf-8") as f:
            #         existing_data = json.load(f)
            # else:
            #     existing_data = []
            # existing_data.extend(new_arr[0:20, :].tolist())
            # with open(save_path, "w", encoding="utf-8") as f:
            #     json.dump(existing_data, f, ensure_ascii=False, indent=2)
            # --- debug ---


            # 6. 可选保存
            if config.save_action_path is not None:
                if config.save_action_path.endswith(".npy"):
                    np.save(config.save_action_path, action_arr)
                elif config.save_action_path.endswith(".json"):
                    with open(config.save_action_path, "w", encoding="utf-8") as f:
                        json.dump(_to_serializable(full_result), f, ensure_ascii=False, indent=2)
                else:
                    raise ValueError("save_action_path 只支持 .npy 或 .json")
                print(f"saved to: {config.save_action_path}")

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
