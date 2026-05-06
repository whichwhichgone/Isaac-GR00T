from dataclasses import dataclass
from time import sleep
import os
import sys
import json
import time
import threading
from collections import deque
from multiprocessing import Array

import cv2
import numpy as np
import tyro
import requests
import pinocchio as pin

sys.path.append(os.getcwd())
sys.path.insert(0, "/home/unitree/unitree_sdk2_python")
sys.path.insert(0, "/home/unitree/xr_teleoperate")

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotations
import cyclonedds.idl.types as types

from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.sub import Subscriber, DataReader
from cyclonedds.pub import Publisher, DataWriter
from cyclonedds.core import Qos, Policy

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from teleop.robot_control.robot_arm import G1_29_ArmController
from utils.cameras_imageio_v1 import ThreadedVideoCapture



# =========================
# DDS config
# =========================
DOMAIN_ID = 1

# 输出：发给真机的 15 点 mocap
TOPIC_NAME_OUT = "MocapUE5G115Topic"


class RobotStateProvider:
    def __init__(self):
        ChannelFactoryInitialize(0, "eth0")
        self.arm_ctrl = G1_29_ArmController(False, False, False)
        self.dual_hand_state_array = Array("d", 14, lock=False)
        self.cap = ThreadedVideoCapture("front_head")

    def read_step(self):
        imu = np.array(self.arm_ctrl.get_base_orientation_quat(), dtype=np.float32).reshape(4)
        body_joint = np.array(self.arm_ctrl.get_current_motor_q(), dtype=np.float32)[:29]
        hand_joint = np.array(self.dual_hand_state_array, dtype=np.float32).reshape(14)

        ret, frame = self.cap.read()
        front = frame if ret else None

        return {
            "imu": imu,
            "body_joint": body_joint,
            "hand_joint": hand_joint,
            "front": front,
            "timestamp": time.time(),
        }


# =========================
# DDS output message
# =========================
@dataclass
@annotations.final
@annotations.autoid("sequential")
class MocapUE5G115Msg(idl.IdlStruct, typename="MocapUE5G115Msg"):
    fps: types.float32
    timestamp: types.int64
    xyz: types.array[types.float32, 15 * 3]
    wxyz: types.array[types.float32, 15 * 4]
    fingers: types.array[types.float32, 15]


class MocapUE5G115MsgPublisher:
    def __init__(self):
        self.participant = DomainParticipant(domain_id=DOMAIN_ID)
        self.qos = Qos(
            Policy.History.KeepLast(depth=4),
        )
        self.topic = Topic(self.participant, TOPIC_NAME_OUT, MocapUE5G115Msg, qos=self.qos)
        self.publisher = Publisher(self.participant)
        self.writer = DataWriter(self.publisher, self.topic)

    def publish(self, fps, xyz_15x3, wxyz_15x4, fingers_15=None):
        xyz_15x3 = np.asarray(xyz_15x3, dtype=np.float32).reshape(15, 3)
        wxyz_15x4 = np.asarray(wxyz_15x4, dtype=np.float32).reshape(15, 4)

        if fingers_15 is None:
            fingers_15 = np.zeros(15, dtype=np.float32)
        else:
            fingers_15 = np.asarray(fingers_15, dtype=np.float32).reshape(15)

        msg = MocapUE5G115Msg(
            fps=np.float32(fps),
            timestamp=np.int64(time.time_ns()),
            xyz=xyz_15x3.reshape(-1).tolist(),
            wxyz=wxyz_15x4.reshape(-1).tolist(),
            fingers=fingers_15.tolist(),
        )
        self.writer.write(msg)


# =========================
# config
# =========================
@dataclass
class ClientConfig:
    host: str = "192.168.123.154"    # "172.16.78.10"
    port: int = 5001
    timeout_ms: int = 15000
    api_token: str  = None
    task_description: str = ""
    infer_interval: float = 0.1

    urdf_path: str = "/home/unitree/xr_teleoperate/assets/g1/g1_body29_hand14.urdf"
    model_dir: str = "/home/unitree/xr_teleoperate/assets/g1/"
    
    send_fps: float = 120


# =========================
# converter
# =========================
class G1_29_BodyPose7WithRoot:
    """
    Input:
        single frame: (36,)   = [root_xyz(3) | 29 joint angles | 4-dim imu quat_wxyz]
        multi frame : (T, 36)

    Output:
        single frame: (30, 7)
        multi frame : (T, 30, 7)
    """

    def __init__(self, urdf_path, model_dir):
        self.urdf_path = urdf_path
        self.model_dir = model_dir

        self.robot = pin.RobotWrapper.BuildFromURDF(self.urdf_path, self.model_dir)
        self.data = self.robot.model.createData()

        self._init_body29_joint_ids()

    def _init_body29_joint_ids(self):
        self.body29_joint_names = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",

            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",

            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",

            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",

            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]

        self.body29_joint_ids = [
            self.robot.model.getJointId(name) for name in self.body29_joint_names
        ]

        if self.robot.model.nq != 43:
            raise ValueError(
                f"Expected full model nq == 43, got {self.robot.model.nq}. "
                "This mapper is written for g1_body29_hand14.urdf."
            )

    def _body29_to_full_model_q(self, q_29):
        q_29 = np.asarray(q_29, dtype=np.float64).reshape(-1)
        if q_29.shape[0] != 29:
            raise ValueError(f"Expected q_29 shape (29,), got {q_29.shape}")

        q_full = np.zeros(43, dtype=np.float64)
        q_full[0:22] = q_29[0:22]
        q_full[29:36] = q_29[22:29]
        return q_full

    @staticmethod
    def _normalize_quaternion_wxyz(quat_wxyz):
        quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
        norm = np.linalg.norm(quat_wxyz)
        if norm < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return quat_wxyz / norm

    def _quat_wxyz_to_yaw(self, quat_wxyz):
        quat_wxyz = self._normalize_quaternion_wxyz(quat_wxyz)
        w, x, y, z = quat_wxyz
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return yaw

    def _single_frame_pose7(self, q_29, imu_quat_wxyz, root_xyz, quat_first=True, yaw_ref=None):
        q_full = self._body29_to_full_model_q(q_29)
        imu_quat_wxyz = self._normalize_quaternion_wxyz(imu_quat_wxyz)
        root_xyz = np.asarray(root_xyz, dtype=np.float64).reshape(3)

        pin.forwardKinematics(self.robot.model, self.data, q_full)

        pose7 = np.zeros((30, 7), dtype=np.float64)

        yaw = self._quat_wxyz_to_yaw(imu_quat_wxyz)
        yaw_rel = yaw if yaw_ref is None else (yaw - yaw_ref)

        cy = np.cos(yaw_rel)
        sy = np.sin(yaw_rel)
        R_root = np.array([
            [cy, -sy, 0.0],
            [sy,  cy, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        T_root = pin.SE3(R_root, root_xyz)

        quat_xyzw_root = pin.Quaternion(R_root).coeffs()
        quat_wxyz_root = np.array(
            [quat_xyzw_root[3], quat_xyzw_root[0], quat_xyzw_root[1], quat_xyzw_root[2]],
            dtype=np.float64,
        )

        if quat_first:
            pose7[0] = np.concatenate([quat_wxyz_root, root_xyz], axis=0)
        else:
            pose7[0] = np.concatenate([root_xyz, quat_wxyz_root], axis=0)

        for i, joint_id in enumerate(self.body29_joint_ids, start=1):
            local_joint_pose = self.data.oMi[joint_id]
            global_joint_pose = T_root * local_joint_pose

            xyz = global_joint_pose.translation.copy()
            quat_xyzw = pin.Quaternion(global_joint_pose.rotation).coeffs()
            quat_wxyz = np.array(
                [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
                dtype=np.float64,
            )

            if quat_first:
                pose7[i] = np.concatenate([quat_wxyz, xyz], axis=0)
            else:
                pose7[i] = np.concatenate([xyz, quat_wxyz], axis=0)

        return pose7

    def get_current_body29_pose7_with_root(self, x, quat_first=True):
        x = np.asarray(x, dtype=np.float64)

        if x.ndim == 1:
            if x.shape[0] != 36:
                raise ValueError(f"Expected single-frame input shape (36,), got {x.shape}")

            root_xyz = x[:3]
            q_29 = x[3:32]
            imu_quat_wxyz = x[32:36]

            return self._single_frame_pose7(
                q_29=q_29,
                imu_quat_wxyz=imu_quat_wxyz,
                root_xyz=root_xyz,
                quat_first=quat_first,
                yaw_ref=None,
            )

        if x.ndim == 2:
            if x.shape[1] != 36:
                raise ValueError(f"Expected multi-frame input shape (T, 36), got {x.shape}")

            T = x.shape[0]
            out = np.zeros((T, 30, 7), dtype=np.float64)
            yaw_ref = self._quat_wxyz_to_yaw(x[0, 32:36])

            for t in range(T):
                root_xyz = x[t, :3]
                q_29 = x[t, 3:32]
                imu_quat_wxyz = x[t, 32:36]

                out[t] = self._single_frame_pose7(
                    q_29=q_29,
                    imu_quat_wxyz=imu_quat_wxyz,
                    root_xyz=root_xyz,
                    quat_first=quat_first,
                    yaw_ref=yaw_ref,
                )

            return out

        raise ValueError(f"Expected input ndim 1 or 2, got ndim={x.ndim}")


# =========================
# pose7 -> 15 points
# =========================
# SELECTED_INDICES = [
#     0,
#     3, 4, 5, 6,
#     9, 10, 11, 12,
#     18, 19, 22,
#     25, 26, 29,
# ]

SELECTED_INDICES = [
    0, 
    2, 4, 6, 6, 
    8, 10, 12, 12,
    17, 19, 22,
    24, 26, 29,
]


def pick_15_points_from_pose7(frame_30x7):
    frame_30x7 = np.asarray(frame_30x7)

    if frame_30x7.shape != (30, 7):
        raise ValueError(f"Expected single pose7 frame shape (30, 7), got {frame_30x7.shape}")

    selected = frame_30x7[SELECTED_INDICES]
    wxyz = selected[:, 0:4]
    xyz = selected[:, 4:7]
    return xyz, wxyz


# =========================
# queue between inference and sender
# =========================
class Pose15Queue:
    def __init__(self, maxlen=300):
        self._lock = threading.Lock()
        self._queue = deque(maxlen=maxlen)
        self._last_xyz = np.zeros((15, 3), dtype=np.float32)
        self._last_wxyz = np.zeros((15, 4), dtype=np.float32)
        self._last_wxyz[:, 0] = 1.0

    def put(self, xyz_15x3, wxyz_15x4):
        xyz_15x3 = np.asarray(xyz_15x3, dtype=np.float32).reshape(15, 3)
        wxyz_15x4 = np.asarray(wxyz_15x4, dtype=np.float32).reshape(15, 4)
        with self._lock:
            self._queue.append((xyz_15x3.copy(), wxyz_15x4.copy()))
            self._last_xyz = xyz_15x3.copy()
            self._last_wxyz = wxyz_15x4.copy()

    def get_next_or_last(self):
        with self._lock:
            if len(self._queue) > 0:
                xyz_15x3, wxyz_15x4 = self._queue.popleft()
                self._last_xyz = xyz_15x3.copy()
                self._last_wxyz = wxyz_15x4.copy()
                return xyz_15x3, wxyz_15x4
            return self._last_xyz.copy(), self._last_wxyz.copy()

    def clear(self):
        with self._lock:
            self._queue.clear()
            
    
    def size(self):
        with self._lock:
            return len(self._queue)

    def empty(self): 
        return self.size() == 0
    
    def is_full(self):
        with self._lock:
            return len(self._queue) >= self._queue.maxlen


# =========================
# send thread
# =========================
class MocapSenderThread(threading.Thread):
    def __init__(self, pose_queue: Pose15Queue, fps: float):
        super().__init__(daemon=True)
        self.pose_queue = pose_queue
        self.fps = fps
        self.publisher = MocapUE5G115MsgPublisher()
        self.running = True

    def run(self):
        dt = 1.0 / self.fps
        fingers_15 = np.zeros(15, dtype=np.float32)

        while self.running:
            t0 = time.time()

            xyz_15x3, wxyz_15x4 = self.pose_queue.get_next_or_last()

            self.publisher.publish(
                fps=self.fps,
                xyz_15x3=xyz_15x3,
                wxyz_15x4=wxyz_15x4,
                fingers_15=fingers_15,
            )

            elapsed = time.time() - t0
            sleep(max(0.0, dt - elapsed))

    def stop(self):
        self.running = False




def load_init_pose15_from_step_json(json_file_path: str, converter: G1_29_BodyPose7WithRoot):
    with open(json_file_path, "r") as f:
        data_json = json.load(f)

    if len(data_json) == 0:
        raise ValueError(f"JSON file is empty: {json_file_path}")

    step = data_json[0]
    x36 = step_to_x36(step)

    pose7 = converter.get_current_body29_pose7_with_root(
        x36,
        quat_first=True,
    )  # (30, 7)

    xyz_15x3, wxyz_15x4 = pick_15_points_from_pose7(pose7)
    return xyz_15x3.astype(np.float32), wxyz_15x4.astype(np.float32)


def step_to_x36(step: dict) -> np.ndarray:
    def squeeze_to_1d(x, name, expected_dim):
        arr = np.asarray(x, dtype=np.float64)

        # (1, D) -> (D,)
        if arr.ndim == 2:
            if arr.shape[0] != 1:
                raise ValueError(f"{name} shape error: expected (1, {expected_dim}), got {arr.shape}")
            arr = arr[0]

        # (D,) 保持不变
        elif arr.ndim == 1:
            pass
        else:
            raise ValueError(f"{name} shape error: unsupported shape {arr.shape}")

        if arr.shape[0] != expected_dim:
            raise ValueError(f"{name} shape error: expected ({expected_dim},), got {arr.shape}")

        return arr

    base_translation = squeeze_to_1d(step["base_translation"], "base_translation", 3)
    base_rotation = squeeze_to_1d(step["base_rotation"], "base_rotation", 4)
    left_leg = squeeze_to_1d(step["left_leg"], "left_leg", 6)
    right_leg = squeeze_to_1d(step["right_leg"], "right_leg", 6)
    waist = squeeze_to_1d(step["waist"], "waist", 3)
    left_arm = squeeze_to_1d(step["left_arm"], "left_arm", 7)
    right_arm = squeeze_to_1d(step["right_arm"], "right_arm", 7)

    q29 = np.concatenate(
        [left_leg, right_leg, waist, left_arm, right_arm],
        axis=0,
    )  # (29,)

    x36 = np.concatenate(
        [base_translation, q29, base_rotation],
        axis=0,
    )  # (36,)

    if x36.shape != (36,):
        raise ValueError(f"x36 shape error: expected (36,), got {x36.shape}")

    return x36



# ======================================================================================
def get_action_via_http_files_from_step(
    step: dict,
    task_description: str,
    host: str,
    port: int,
    timeout_ms: int = 15000,
    action_dim: int = 33,
    action_dtype=np.float64,
):
    """
    直接从 robot step 发送 multipart（不经过 observation 组装/拆分）
    step 预期字段:
      - "imu": (4,)
      - "body_joint": (29,)
      - "front": HxWx3 或 None
    """
    url = f"http://{host}:{port}/predict"

    # 1) state33 = q29 + imu_wxyz
    body_joint = np.asarray(step["body_joint"], dtype=np.float32).reshape(29)
    imu = np.asarray(step["imu"], dtype=np.float32).reshape(4)
    state33 = np.concatenate([body_joint, imu], axis=0).astype(np.float32)  # (33,)

    # 2) 图像准备
    front = step.get("front", None)
    if front is None:
        front_head = np.zeros((240, 424, 3), dtype=np.uint8)
    else:
        front_head = cv2.resize(np.asarray(front), (240, 424))
        if front_head.dtype != np.uint8:
            if np.issubdtype(front_head.dtype, np.floating) and front_head.max() <= 1.0:
                front_head = front_head * 255.0
            front_head = np.clip(front_head, 0, 255).astype(np.uint8)

    # 没有腕部相机时给空图占位
    left_hand = np.zeros((240, 320, 3), dtype=np.uint8)
    right_hand = np.zeros((240, 320, 3), dtype=np.uint8)


    files = {
        "json": json.dumps({"instruction": task_description}),
        "front_head": ("front_head", front_head.tobytes(), "application/octet-stream"),
        "left_hand": ("left_hand", left_hand.tobytes(), "application/octet-stream"),
        "right_hand": ("right_hand", right_hand.tobytes(), "application/octet-stream"),
        "state": ("state", state33.tobytes(), "application/octet-stream"),
    }

    resp = requests.post(url, files=files, timeout=timeout_ms / 1000.0)
    resp.raise_for_status()

    # 兼容 JSON 返回
    content_type = resp.headers.get("Content-Type", "")
    if "application/json" in content_type:
        result = resp.json()
        if "action" not in result:
            raise ValueError(f"Server response missing 'action': {result}")
        return np.asarray(result["action"], dtype=np.float32)

    # 兼容 bytes 返回
    action_flat = np.frombuffer(resp.content, dtype=action_dtype)
    if action_flat.size == 0:
        raise ValueError("Server returned empty action bytes")
    if action_flat.size % action_dim != 0:
        raise ValueError(
            f"Invalid action bytes length: {action_flat.size}, action_dim={action_dim}"
        )

    return action_flat.reshape(1, -1, action_dim).astype(np.float32)
# ======================================================================================



def action_array_to_x36_seq(action_arr, current_base_translation=None):
    """
    输入:
        action_arr:
            - (1, T, 33)
            - (T, 33)
            - (33,)   也兼容，按单帧处理

        33维定义:
            [q29(29), imu_wxyz(4)]

    输出:
        x36_seq: (T, 36)
            [root_xyz(3), q29(29), imu_wxyz(4)]
    """
    arr = np.asarray(action_arr, dtype=np.float32)

    if arr.ndim == 3:
        # (1, T, 33)
        if arr.shape[0] != 1 or arr.shape[2] != 33:
            raise ValueError(f"Expected shape (1, T, 33), got {arr.shape}")
        arr = arr[0]   # -> (T, 33)

    elif arr.ndim == 2:
        # (T, 33)
        if arr.shape[1] != 33:
            raise ValueError(f"Expected shape (T, 33), got {arr.shape}")

    elif arr.ndim == 1:
        # (33,) -> (1, 33)
        if arr.shape[0] != 33:
            raise ValueError(f"Expected shape (33,), got {arr.shape}")
        arr = arr[None, :]

    else:
        raise ValueError(f"Unsupported action shape: {arr.shape}")

    T = arr.shape[0]
    q29 = arr[:, :29]
    imu_wxyz = arr[:, 29:33]
    
    if current_base_translation is None:
        base_translation = np.zeros((T, 3), dtype=np.float32)
    else:
        cur = np.asarray(current_base_translation, dtype=np.float32)

        if cur.ndim == 1:
            if cur.shape[0] != 3:
                raise ValueError(f"current_base_translation must have shape (3,), got {cur.shape}")
            base_translation = np.repeat(cur[None, :], T, axis=0)

        elif cur.ndim == 2:
            if cur.shape != (T, 3):
                raise ValueError(f"current_base_translation must have shape ({T}, 3), got {cur.shape}")
            base_translation = cur

        else:
            raise ValueError(
                f"current_base_translation must have shape (3,) or (T,3), got {cur.shape}"
            )

    x36_seq = np.concatenate([base_translation, q29, imu_wxyz], axis=1)

    if x36_seq.shape[1] != 36:
        raise ValueError(f"x36_seq shape error: {x36_seq.shape}")

    return x36_seq


# =========================
# main
# =========================
if __name__ == "__main__":
    # 1) 读取命令行配置并初始化各模块（机器人状态、运动学转换器、发送队列）
    config = tyro.cli(ClientConfig)

    robot_state = RobotStateProvider()

    converter = G1_29_BodyPose7WithRoot(
        urdf_path=config.urdf_path,
        model_dir=config.model_dir,
    )

    pose_queue = Pose15Queue(maxlen=300)

    # 2) 先放入一帧初始姿态，确保发送线程启动后机器人有可用姿态可发
    default_xyz_15x3, default_wxyz_15x4 = load_init_pose15_from_step_json(
        "/home/unitree/xr_teleoperate/init_action_for_toy.json",
        converter,
    )
    print(f"default_xyz_15x3: {default_xyz_15x3.shape}, default_wxyz_15x4: {default_wxyz_15x4.shape}")
    pose_queue.put(default_xyz_15x3, default_wxyz_15x4)

    sender_thread = MocapSenderThread(
        pose_queue=pose_queue,
        fps=config.send_fps,
    )
    sender_thread.start()

    print("Sender thread started.")
    print("Robot will first receive default standing pose.")
    executed_first_action = False

    try:
        # 3) 主循环：按固定 infer_interval 周期拉取状态并请求下一段动作
        while True:
            loop_t0 = time.time()

            # 3.1 先做一次轻量读取，避免在状态缺失时进入后续网络/转换流程
            step = robot_state.read_step()
            if step is None:
                sleep(config.infer_interval)
                continue

            try:
                # 3.2 发送队列未清空时，说明上一段动作还在发，等待其自然消费完成
                if not pose_queue.empty():
                    sleep(0.01)
                    continue

                # 3.3 即将发起推理前再取一次最新状态，降低控制链路时延
                step = robot_state.read_step()
                if step is None:
                    sleep(0.01)
                    continue
                
                print(f"pose_queue.size(): {pose_queue.size()}, pose_queue.empty(): {pose_queue.empty()}")
                # 3.4 把当前 step 通过 HTTP 送到策略服务，拿回 (T, 33) 动作序列
                action_arr = get_action_via_http_files_from_step(
                    step,
                    config.task_description,
                    host=config.host,
                    port=config.port,
                    timeout_ms=config.timeout_ms,
                )[0]
                print(f"action_arr: {action_arr.shape}")

                
                action_arr = np.asarray(action_arr, dtype=np.float32)
                print("=" * 60)
                print("action_arr shape:", action_arr.shape)
                print("=" * 60)
                
                # 3.5 当前流程不使用策略给出的 base_translation，统一补全为 0
                T = action_arr.shape[0]
                zero_base_translation = np.zeros((T, 3), dtype=np.float32)

                # 3.6 把每帧动作从 33 维拼成 x36: [root_xyz(3), q29(29), imu(4)]
                x36_seq = action_array_to_x36_seq(
                    action_arr,
                    current_base_translation=zero_base_translation,
                )

                # 3.7 用 Pinocchio 做正运动学，得到每帧 30 个点的 pose7
                pose7_seq = converter.get_current_body29_pose7_with_root(
                    x36_seq,
                    quat_first=True,
                )

                if pose7_seq.ndim != 3 or pose7_seq.shape[1:] != (30, 7):
                    raise ValueError(f"pose7_seq shape error: expected (T, 30, 7), got {pose7_seq.shape}")
                

                print(f"pose_queue.size()11: {pose_queue.size()}, pose_queue.empty(): {pose_queue.empty()}")
                # 3.8 清掉旧动作，避免队列中残留历史轨迹导致“混播”
                pose_queue.clear()
                
                # 3.9 把 30 点裁成协议需要的 15 点，并逐帧压入发送队列
                for t in range(pose7_seq.shape[0]):
                    # 若队列打满，短暂让出 CPU，等待发送线程消费
                    while pose_queue.is_full():
                        time.sleep(0.001)

                    frame = pose7_seq[t]
                    xyz_15x3, wxyz_15x4 = pick_15_points_from_pose7(frame)
                    # 对齐目标坐标系高度偏置
                    xyz_15x3[:, 2] += 0.81

                    pose_queue.put(xyz_15x3, wxyz_15x4)
                
                # time.sleep(5)  # 确保动作已经开始发出

                print("=" * 80)
                print("x36_seq shape:", x36_seq.shape)
                print("pose7_seq shape:", pose7_seq.shape)
                print("queued frames:", pose7_seq.shape[0])

                # 仅执行第一段 action_arr：等待发送队列消费完成后退出程序
                if not executed_first_action:
                    executed_first_action = True
                    while not pose_queue.empty():
                        sleep(0.01)
                    break

            except Exception as e:
                print(f"Inference/convert error: {e}")

            # 3.10 节流：控制主循环周期，避免过快请求策略服务
            elapsed = time.time() - loop_t0
            sleep(max(0.0, config.infer_interval - elapsed))

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        # 4) 退出时优雅停线程，避免后台 DDS 发送残留
        sender_thread.stop()
        sender_thread.join(timeout=1.0)
        print("Sender thread stopped.")