import json
import copy
import numpy as np
from scipy.ndimage import gaussian_filter1d


input_json_path = "/liujinxin/dataset/piper/G1/0428_tidy_up_g1/episode_6/data.json"
output_json_path = "/liujinxin/liyifan/Isaac-GR00T/episode_6_gaussian_filtered_0.5.json"

# 高斯滤波参数
# sigma 越大，动作越平滑，但动作幅度会被削弱
SIGMA_BODY = 0.5
SIGMA_HAND = 0.8

# 是否滤波 imu
# imu 通常是四元数，不建议直接滤波
FILTER_IMU = True
SIGMA_IMU = 0.5


def gaussian_filter_sequence(data, key, sigma, expected_dim=None):
    """
    对 data 中指定 key 的时间序列做高斯滤波。

    data: list[dict]
    key: 例如 "body_joint" / "hand_joint"
    sigma: 高斯滤波强度
    expected_dim: 期望维度，例如 body_joint=29, hand_joint=14
    """

    values = []

    for i, frame in enumerate(data):
        if key not in frame:
            raise KeyError(f"Frame {i} does not contain key: {key}")

        arr = np.asarray(frame[key], dtype=np.float32).reshape(-1)

        if expected_dim is not None and arr.shape[0] != expected_dim:
            raise ValueError(
                f"Frame {i}, key={key}, expected dim {expected_dim}, "
                f"but got {arr.shape[0]}"
            )

        values.append(arr)

    values = np.stack(values, axis=0)  # shape: (T, D)

    print(f"{key} before filter shape:", values.shape)

    # 沿时间维 axis=0 做高斯滤波
    filtered_values = gaussian_filter1d(
        values,
        sigma=sigma,
        axis=0,
        mode="nearest"
    )

    return filtered_values


with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

if not isinstance(data, list):
    raise TypeError("JSON root should be a list of frames.")

print("num frames:", len(data))

filtered_data = copy.deepcopy(data)

# 滤波 body_joint: (T, 29)
body_filtered = gaussian_filter_sequence(
    data,
    key="body_joint",
    sigma=SIGMA_BODY,
    expected_dim=29
)

# 滤波 hand_joint: (T, 14)
hand_filtered = gaussian_filter_sequence(
    data,
    key="hand_joint",
    sigma=SIGMA_HAND,
    expected_dim=14
)

for i in range(len(filtered_data)):
    filtered_data[i]["body_joint"] = body_filtered[i].astype(float).tolist()
    filtered_data[i]["hand_joint"] = hand_filtered[i].astype(float).tolist()


# 可选：滤波 imu
# 如果 imu 是四元数，滤波后必须重新归一化
if FILTER_IMU:
    imu_filtered = gaussian_filter_sequence(
        data,
        key="imu",
        sigma=SIGMA_IMU,
        expected_dim=4
    )

    # 四元数重新归一化
    norm = np.linalg.norm(imu_filtered, axis=1, keepdims=True)
    imu_filtered = imu_filtered / np.clip(norm, 1e-8, None)

    for i in range(len(filtered_data)):
        filtered_data[i]["imu"] = imu_filtered[i].astype(float).tolist()


with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=4)

print("Saved filtered json to:", output_json_path)