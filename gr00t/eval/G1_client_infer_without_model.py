from collections import deque
from dataclasses import dataclass
import json
import os
import pickle
import sys
import threading
import time
from time import sleep

import numpy as np
import pinocchio as pin
import tyro
from PIL import Image

import cv2

sys.path.append(os.getcwd())
from utils.cameras_imageio_v1 import ThreadedVideoCapture
from cyclonedds.core import Policy, Qos
from cyclonedds.domain import DomainParticipant
import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotations
import cyclonedds.idl.types as types
from cyclonedds.pub import DataWriter, Publisher
from cyclonedds.sub import DataReader, Subscriber
from cyclonedds.topic import Topic
from gr00t import server_client


# =========================
# DDS config
# =========================
DOMAIN_ID = 1

# 输入：真机当前姿态
TOPIC_NAME_IN = "WR/BodyPose"

# 输出：发给真机的 15 点 mocap
TOPIC_NAME_OUT = "MocapUE5G115Topicvla"

SEND_FPS = 10  # 120.0

CAMERAS_MAP = {"ego_view": "front", "left_wrist_view": "left", "right_wrist_view": "right"}

# =========================
# DDS input message
# =========================
@dataclass
@annotations.final
@annotations.autoid("sequential")
class WR_GAE_BodyPose_Msg(idl.IdlStruct, typename="WR_GAE_BodyPose_Msg"):
    fps: types.float32
    xyz: types.array[types.float32, 45]
    wxyz: types.array[types.float32, 4]
    q: types.array[types.float32, 29]
    timestamp: types.int64


class BodyPoseSubscriber:
    def __init__(self):
        self.participant = DomainParticipant(domain_id=DOMAIN_ID)
        self.qos = Qos(
            Policy.History.KeepLast(depth=4),
        )
        self.topic = Topic(self.participant, TOPIC_NAME_IN, WR_GAE_BodyPose_Msg, qos=self.qos)
        self.subscriber = Subscriber(self.participant)
        self.reader = DataReader(self.subscriber, self.topic)

    def subscribe(self):
        samples = self.reader.read()
        if not samples:
            return None
        return samples[-1]


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
# gr00t config
# =========================
@dataclass
class ClientConfig:
    host: str = "172.16.78.10"
    port: int = 36367
    timeout_ms: int = 15000
    api_token: str = None
    task_description: str = ""
    infer_interval: float = 0.1

    urdf_path: str = (
        "/media/mpz/d5f7a2a2-7dfb-4053-8e51-ee6943e25306/assets_xr/g1/g1_body29_hand14.urdf"
    )
    model_dir: str = "/media/mpz/d5f7a2a2-7dfb-4053-8e51-ee6943e25306/assets_xr/g1/"

    # stickman_path: str = (
    #     "/media/mpz/d5f7a2a2-7dfb-4053-8e51-ee6943e25306/20260429g1/traj_0_stickman.npy"
    # )
    stickman_path = None
    gt_state_path: str = "/media/mpz/d5f7a2a2-7dfb-4053-8e51-ee6943e25306/0506g1/episode_6.json" # "/media/mpz/d5f7a2a2-7dfb-4053-8e51-ee6943e25306/traj_0_states_0429.json"

    debug_mode : bool = True # False


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
    
    def _quat_wxyz_to_R(self, quat_wxyz):
        """
        quat_wxyz: [w, x, y, z]
        return: 3x3 rotation matrix
        """
        quat_wxyz = self._normalize_quaternion_wxyz(quat_wxyz)
        w, x, y, z = quat_wxyz

        # pinocchio Quaternion 输入是 [x, y, z, w]
        quat_xyzw = np.array([x, y, z, w], dtype=np.float64)
        return pin.Quaternion(quat_xyzw).toRotationMatrix()

    def _single_frame_pose7(self, q_29, imu_quat_wxyz, root_xyz,
                            quat_first=True):
        q_full = self._body29_to_full_model_q(q_29)
        imu_quat_wxyz = self._normalize_quaternion_wxyz(imu_quat_wxyz)
        root_xyz = np.asarray(root_xyz, dtype=np.float64).reshape(3)

        pin.forwardKinematics(self.robot.model, self.data, q_full)

        pose7 = np.zeros((30, 7), dtype=np.float64)

        # 关键：永远使用完整 IMU 朝向，不再 R_ref.T @ R_imu。
        # 这样 pitch/roll 不会被第一帧抵消，弯腰时 root 上下朝向会保留。
        R_root = self._quat_wxyz_to_R(imu_quat_wxyz)

        T_root = pin.SE3(R_root, root_xyz)

        quat_xyzw_root = pin.Quaternion(R_root).coeffs()
        quat_wxyz_root = np.array(
            [quat_xyzw_root[3], quat_xyzw_root[0], quat_xyzw_root[1], quat_xyzw_root[2]],
            dtype=np.float64,
        )
        quat_wxyz_root = self._normalize_quaternion_wxyz(quat_wxyz_root)

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
            quat_wxyz = self._normalize_quaternion_wxyz(quat_wxyz)

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
            )

        if x.ndim == 2:
            if x.shape[1] != 36:
                raise ValueError(f"Expected multi-frame input shape (T, 36), got {x.shape}")

            T = x.shape[0]
            out = np.zeros((T, 30, 7), dtype=np.float64)

            for t in range(T):
                root_xyz = x[t, :3]
                q_29 = x[t, 3:32]
                imu_quat_wxyz = x[t, 32:36]

                out[t] = self._single_frame_pose7(
                    q_29=q_29,
                    imu_quat_wxyz=imu_quat_wxyz,
                    root_xyz=root_xyz,
                    quat_first=quat_first,
                )

            return out

        raise ValueError(f"Expected input ndim 1 or 2, got ndim={x.ndim}")


# =========================
# pose7 -> 15 points
# =========================

# the original 15 points selection
# SELECTED_INDICES = [
#     0,
#     3, 4, 5, 6,
#     9, 10, 11, 12,
#     18, 19, 22,
#     25, 26, 29,
# ]

SELECTED_INDICES = [
    0,
    2,
    4,
    6,
    6,
    8,
    10,
    12,
    12,
    17,
    19,
    22,
    24,
    26,
    29,
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

    def clear(self, reset_last=False):
        with self._lock:
            self._queue.clear()

            if reset_last:
                self._last_xyz = np.zeros((15, 3), dtype=np.float32)
                self._last_wxyz = np.zeros((15, 4), dtype=np.float32)
                self._last_wxyz[:, 0] = 1.0

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

# =========================
# Get image observation
# =========================
class CameraSubscriber:
    def __init__(self):
        self.running = True
        self.caps = {
            "front_head": ThreadedVideoCapture("front_head"),
        }
        self.save_dir = "./image_debug"
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.cam_thread.start()

    def _camera_loop(self):
        while self.running:
            for name, cap in self.caps.items():
                ret, frame = cap.read()
                if ret:
                    with self.frame_lock:
                        self.latest_frame = frame.copy()
            sleep(1 / 20)
    
    def get_frame(self, save_image: bool = False):
        with self.frame_lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()

        if save_image and frame is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, "front_head.png")
            cv2.imwrite(save_path, frame)

        return frame
    

    def stop(self):
        self.running = False
        self.cam_thread.join(timeout=1.0)

        for cap in self.caps.values():
            if hasattr(cap, "release"):
                cap.release()


# =========================
# helpers
# =========================
def load_image(image_input, target_size: tuple):
    if isinstance(image_input, str):
        image = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, np.ndarray):
        image = np.asarray(image_input)

        if image.ndim != 3 or image_input.shape[-1] != 3:
            raise ValueError(f"Expected frame shape (H, W, 3), got {image.shape}")
        
        image = image.astype(np.uint8)
        image = image[:,:,::-1]
        image = Image.fromarray(image).convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    width, height = image.size
    aspect_ratio = width / height

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


def build_observation_from_msg(msg: WR_GAE_BodyPose_Msg, task_description: str, frame=None, stickman_path=None):
    xyz_np = np.array(msg.xyz, dtype=np.float32)
    q_np = np.array(msg.q, dtype=np.float32)
    imu_np = np.array(msg.wxyz, dtype=np.float32)

    if xyz_np.shape != (45,):
        raise ValueError(f"xyz shape error: expected (45,), got {xyz_np.shape}")
    if q_np.shape != (29,):
        raise ValueError(f"q shape error: expected (29,), got {q_np.shape}")
    if imu_np.shape != (4,):
        raise ValueError(f"wxyz shape error: expected (4,), got {imu_np.shape}")

    left_leg = q_np[0:6]
    right_leg = q_np[6:12]
    waist = q_np[12:15]
    left_arm = q_np[15:22]
    right_arm = q_np[22:29]

    base_rotation = imu_np
    state = {
        "base_translation": np.array([0.0, 0.0, 0.0], dtype=np.float32)[None, None, :],
        "base_rotation": base_rotation[None, None, :].astype(np.float32),
        "left_leg": left_leg[None, None, :].astype(np.float32),
        "right_leg": right_leg[None, None, :].astype(np.float32),
        "waist": waist[None, None, :].astype(np.float32),
        "left_arm": left_arm[None, None, :].astype(np.float32),
        "right_arm": right_arm[None, None, :].astype(np.float32),
    }
    if frame is None:
        video = {
            "ego_view": np.zeros((1, 1, 224, 224, 3), dtype=np.uint8),
            "left_wrist_view": np.zeros((1, 1, 224, 224, 3), dtype=np.uint8),
            "right_wrist_view": np.zeros((1, 1, 224, 224, 3), dtype=np.uint8),
        }
    else:
        video = {
            "ego_view": load_image(np.asarray(frame), (224,224)),
            "left_wrist_view": np.zeros((1, 1, 224, 224, 3), dtype=np.uint8),
            "right_wrist_view": np.zeros((1, 1, 224, 224, 3), dtype=np.uint8), 
        }
    if stickman_path is None:
        stickman_np = np.zeros((1, 1, 900), dtype=np.float32)
    else:
        stickman_np = np.load(stickman_path)
        if len(stickman_np.shape) == 1:
            stickman_np = stickman_np[None, None, :]
        stickman_np = stickman_np.astype(np.float32)

    observation = {
        "video": video,
        "state": state,
        "language": {"annotation.human.task_description": [[task_description]]},
        "stickman": {"annotation.human.stickman": stickman_np},
    }
    print(f"The current observation is: {observation['language']}, {stickman_np}")
    return observation


def build_observation_from_step(step: dict, task_description: str, json_path : str, stickman_path=None):
    
    if "base_rotation" not in step:
        step_new = {
            "step": 0,
            "base_translation": [
                [
                    0.0,
                    0.0,
                    0.0
                ]
            ],
            "base_rotation": [
                step["imu"]
            ],
            "left_leg": [
                step["body_joint"][0:6]
            ],
            "right_leg": [
                step["body_joint"][6:12]
            ],
            "waist": [
                step["body_joint"][12:15]
            ],
            "left_arm": [
                step["body_joint"][15:22]
            ],
            "right_arm": [
                step["body_joint"][22:29]
            ]
        }
        step = step_new
        print(f"step_new.keys(): {step_new.keys()}")
    
    def to_btd(x, name):
        arr = np.asarray(x, dtype=np.float32)

        # (1, D) -> (1, 1, D)
        if arr.ndim == 2:
            return arr[:, None, :]

        # (D,) -> (1, 1, D)
        if arr.ndim == 1:
            return arr[None, None, :]

        # 已经是 (1,1,D)
        if arr.ndim == 3:
            return arr

        raise ValueError(f"{name} shape error: {arr.shape}")
    
    state = {
        "base_translation": to_btd(step["base_translation"], "base_translation"),
        "base_rotation": to_btd(step["base_rotation"], "base_rotation"),
        "left_leg": to_btd(step["left_leg"], "left_leg"),
        "right_leg": to_btd(step["right_leg"], "right_leg"),
        "waist": to_btd(step["waist"], "waist"),
        "left_arm": to_btd(step["left_arm"], "left_arm"),
        "right_arm": to_btd(step["right_arm"], "right_arm"),
    }

    video = {}
    for gr00t_key, g1_key in CAMERAS_MAP.items():
        video[gr00t_key] = np.zeros((1, 1, 224, 224, 3), dtype=np.uint8)
        if g1_key in step:
            root_path = os.path.dirname(json_path)
            image_path = os.path.join(root_path, step[g1_key])
            video[gr00t_key] = load_image(image_path, (224,224))

    if stickman_path is None:
        stickman_np = np.zeros((1, 1, 900), dtype=np.float32)
    else:
        stickman_np = np.load(stickman_path)
        if len(stickman_np.shape) == 1:
            stickman_np = stickman_np[None, None, :]
        stickman_np = stickman_np.astype(np.float32)

    observation = {
        "video": video,
        "state": state,
        "language": {"annotation.human.task_description": [[task_description]]},
        "stickman": {"annotation.human.stickman": stickman_np},
    }
    print(f"The current observation is: {observation['language']}, {stickman_np}")
    return observation


def action0_to_x36_seq(action0: dict) -> np.ndarray:
    """
    action[0] -> (T, 36)
    expected input per key: (1, T, D) or (T, D)
    """

    if "base_translation" not in action0:
        _, T, _ = action0["base_rotation"].shape
        action0["base_translation"] = np.zeros((1, T, 3), dtype=np.float32)

    def normalize_seq(x, name):
        arr = np.asarray(x, dtype=np.float64)

        if arr.ndim == 3:
            # (1, T, D)
            return arr[0]

        if arr.ndim == 2:
            # (T, D)
            return arr

        if arr.ndim == 1:
            # (D,) -> (1, D)
            return arr[None, :]

        raise ValueError(f"Unsupported ndim for {name}: shape={arr.shape}")

    base_translation = normalize_seq(action0["base_translation"], "base_translation")
    base_rotation = normalize_seq(action0["base_rotation"], "base_rotation")
    left_leg = normalize_seq(action0["left_leg"], "left_leg")
    right_leg = normalize_seq(action0["right_leg"], "right_leg")
    waist = normalize_seq(action0["waist"], "waist")
    left_arm = normalize_seq(action0["left_arm"], "left_arm")
    right_arm = normalize_seq(action0["right_arm"], "right_arm")

    T = base_translation.shape[0]
    expected_shapes = {
        "base_translation": (T, 3),
        "base_rotation": (T, 4),
        "left_leg": (T, 6),
        "right_leg": (T, 6),
        "waist": (T, 3),
        "left_arm": (T, 7),
        "right_arm": (T, 7),
    }
    actual_shapes = {
        "base_translation": base_translation.shape,
        "base_rotation": base_rotation.shape,
        "left_leg": left_leg.shape,
        "right_leg": right_leg.shape,
        "waist": waist.shape,
        "left_arm": left_arm.shape,
        "right_arm": right_arm.shape,
    }

    for k in expected_shapes:
        if actual_shapes[k] != expected_shapes[k]:
            raise ValueError(
                f"{k} shape mismatch: expected {expected_shapes[k]}, got {actual_shapes[k]}"
            )

    q29 = np.concatenate(
        [left_leg, right_leg, waist, left_arm, right_arm],
        axis=1,
    )  # (T, 29)

    x36_seq = np.concatenate(
        [base_translation, q29, base_rotation],
        axis=1,
    )  # (T, 36)

    if x36_seq.ndim != 2 or x36_seq.shape[1] != 36:
        raise ValueError(f"x36_seq shape error: expected (T, 36), got {x36_seq.shape}")

    return x36_seq, x36_seq.shape[0]


def normalize_quat_wxyz(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def nlerp_quat_wxyz(q0, q1, alpha):
    q0 = normalize_quat_wxyz(q0)
    q1 = normalize_quat_wxyz(q1)

    # 保证走最短路径
    if np.dot(q0, q1) < 0.0:
        q1 = -q1

    q = (1.0 - alpha) * q0 + alpha * q1
    return normalize_quat_wxyz(q)


def interpolate_pose7_frame_pair(
    frame0_30x7: np.ndarray, frame1_30x7: np.ndarray, alpha: float
) -> np.ndarray:
    frame0 = np.asarray(frame0_30x7, dtype=np.float64)
    frame1 = np.asarray(frame1_30x7, dtype=np.float64)

    if frame0.shape != (30, 7) or frame1.shape != (30, 7):
        raise ValueError(f"Expected both frames to be (30,7), got {frame0.shape}, {frame1.shape}")

    quat0 = frame0[:, 0:4]
    quat1 = frame1[:, 0:4]
    xyz0 = frame0[:, 4:7]
    xyz1 = frame1[:, 4:7]

    xyz_interp = (1.0 - alpha) * xyz0 + alpha * xyz1

    quat_interp = np.zeros((30, 4), dtype=np.float64)
    for j in range(30):
        quat_interp[j] = nlerp_quat_wxyz(quat0[j], quat1[j], alpha)

    interp_frame = np.concatenate([quat_interp, xyz_interp], axis=1)
    return interp_frame.astype(np.float32)


def interpolate_pose7_pairwise(pose7_seq: np.ndarray, num_interp: int = 3) -> np.ndarray:
    """
    在相邻 pose7 帧之间插值

    输入:
        pose7_seq: (T, 30, 7)
            每行格式 [qw, qx, qy, qz, x, y, z]

        num_interp:
            每两帧之间插入多少帧

    输出:
        out_seq: (T_new, 30, 7)

    规则:
        x0, (插值), x1, (插值), x2, ...
    """
    pose7_seq = np.asarray(pose7_seq, dtype=np.float64)

    if pose7_seq.ndim != 3 or pose7_seq.shape[1:] != (30, 7):
        raise ValueError(f"Expected (T,30,7), got {pose7_seq.shape}")

    T = pose7_seq.shape[0]

    if T <= 1 or num_interp <= 0:
        return pose7_seq.astype(np.float32)

    out = []

    for i in range(T - 1):
        frame0 = pose7_seq[i]
        frame1 = pose7_seq[i + 1]

        # 先放原帧
        out.append(frame0.astype(np.float32))

        # 插值帧
        for k in range(1, num_interp + 1):
            alpha = k / (num_interp + 1)
            interp_frame = interpolate_pose7_frame_pair(frame0, frame1, alpha)
            out.append(interp_frame)

    # 最后一帧补上
    out.append(pose7_seq[-1].astype(np.float32))

    return np.stack(out, axis=0).astype(np.float32)


def normalize_quat_wxyz(q, eps=1e-12):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm < eps:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def quat_mul_wxyz(q1, q2):
    """
    Hamilton product.
    q = q1 * q2

    q1, q2: [w, x, y, z]
    """
    q1 = normalize_quat_wxyz(q1)
    q2 = normalize_quat_wxyz(q2)

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    q = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )

    return normalize_quat_wxyz(q)


def canonicalize_quat_wxyz(q, ref=None):
    """
    统一四元数符号，避免 q 和 -q 数值跳变。
    """
    q = normalize_quat_wxyz(q)

    if ref is not None:
        ref = normalize_quat_wxyz(ref)
        if np.dot(ref, q) < 0:
            q = -q
    else:
        if q[0] < 0:
            q = -q

    return normalize_quat_wxyz(q)


def quat_inv_wxyz(q):
    """
    四元数逆。
    q: [w, x, y, z]
    对单位四元数，逆就是共轭。
    """
    q = normalize_quat_wxyz(q)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def convert_rel_to_abs(q_rel, q_init):
    """
    将模型输出的相对 IMU chunk 还原为绝对 IMU chunk。

    训练时:
        q_rel[t] = inv(q_abs[t - 1]) * q_abs[t]

    推理时:
        q_abs[t] = q_abs[t - 1] * q_rel[t]

    Args:
        q_rel:
            shape:
                (T, 4)

        q_init:
            当前 observation.state["base_rotation"]
            shape 通常是:
                (4,)

    Returns:
        q_abs:
            shape 和 q_rel 保持一致
    """
    q_rel_arr = np.asarray(q_rel, dtype=np.float64)

    q_rel_seq = q_rel_arr

    q_init_arr = np.asarray(q_init, dtype=np.float64)

    if q_init_arr.shape[-1] != 4:
        raise ValueError(f"q_init shape error: expected last dim 4, got {q_init_arr.shape}")

    q_prev = q_init_arr.reshape(-1, 4)[-1]
    q_prev = canonicalize_quat_wxyz(q_prev)

    q_abs_seq = []

    for t in range(q_rel_seq.shape[0]):
        delta_q = canonicalize_quat_wxyz(q_rel_seq[t])

        # 关键：训练时 rel = inv(prev) * cur
        # 所以推理时 cur = prev * rel
        q_cur = quat_mul_wxyz(q_prev, delta_q)

        # 保持四元数符号连续
        q_cur = canonicalize_quat_wxyz(q_cur, ref=q_prev)

        q_abs_seq.append(q_cur.astype(np.float32))
        q_prev = q_cur

    q_abs_seq = np.stack(q_abs_seq, axis=0).astype(np.float32)

    return q_abs_seq


def get_action_from_json(data, start_idx=0, horizon=None):
    """
    从 json 中构造 action chunk。

    返回:
        action:
            base_translation: (T, 3)
            base_rotation:    (T, 4)  # 相对 IMU, rel = inv(prev) * cur
            left_leg:         (T, 6)
            right_leg:        (T, 6)
            waist:            (T, 3)
            left_arm:         (T, 7)
            right_arm:        (T, 7)
    """
    episode_len = len(data)

    expected_key = {
        "left_leg": (0, 6),
        "right_leg": (6, 12),
        "waist": (12, 15),
        "left_arm": (15, 22),
        "right_arm": (22, 29),
    }

    if episode_len < 2:
        raise ValueError(f"episode_len should be >= 2, got {episode_len}")

    if start_idx < 0:
        start_idx = 0

    if horizon is None:
        end_idx = episode_len - 1
    else:
        end_idx = min(start_idx + horizon, episode_len - 1)

    if start_idx >= end_idx:
        raise ValueError(
            f"No enough data to build action: start_idx={start_idx}, end_idx={end_idx}, episode_len={episode_len}"
        )

    action = {key: [] for key in expected_key.keys()}
    action["base_translation"] = []
    action["base_rotation"] = []

    for idx in range(start_idx, end_idx):
        prev_step = data[idx]
        cur_step = data[idx + 1]

        body_joint = np.asarray(cur_step["body_joint"], dtype=np.float64).reshape(-1)
        if body_joint.shape[0] != 29:
            raise ValueError(f"body_joint should be 29-dim, got {body_joint.shape}")

        for key, (s, e) in expected_key.items():
            joint_part = body_joint[s:e]
            action[key].append(joint_part)

        action["base_translation"].append(np.zeros(3, dtype=np.float64))

        cur_imu = np.asarray(cur_step["imu"], dtype=np.float64).reshape(4)
        prev_imu = np.asarray(prev_step["imu"], dtype=np.float64).reshape(4)

        cur_imu = canonicalize_quat_wxyz(cur_imu)
        prev_imu = canonicalize_quat_wxyz(prev_imu)

        rel_imu = quat_mul_wxyz(quat_inv_wxyz(prev_imu), cur_imu)
        rel_imu = canonicalize_quat_wxyz(rel_imu)

        action["base_rotation"].append(rel_imu)

    for key, value in action.items():
        action[key] = np.stack(value, axis=0).astype(np.float32)

    return action
    

# =========================
# main
# =========================
if __name__ == "__main__":
    data_json = None

    config = tyro.cli(ClientConfig)
    body_pose = BodyPoseSubscriber()

    converter = G1_29_BodyPose7WithRoot(
        urdf_path=config.urdf_path,
        model_dir=config.model_dir,
    )

    pose_queue = Pose15Queue(maxlen=300)

    sender_thread = MocapSenderThread(
        pose_queue=pose_queue,
        fps=SEND_FPS,
    )
    sender_thread.start()
    print("Sender thread started.")

    with open(config.gt_state_path, "r") as f:
        data_json = json.load(f)
        episode_len = len(data_json)

    try:
        while True:
            loop_t0 = time.time()

            msg = body_pose.subscribe()
            if msg is None:
                print("No body pose message received, waiting...")
                sleep(config.infer_interval)
                continue
            try:
                # 先等上一段动作发完
                if not pose_queue.empty():
                    print(f"Pose queue is not empty (size={pose_queue.size()}), waiting ...")
                    sleep(config.infer_interval)
                    continue
                frame = None
                observation = build_observation_from_msg(
                    msg, config.task_description, frame, stickman_path=config.stickman_path
                )    

                action0 = get_action_from_json(data_json)

                action0['base_rotation'] = convert_rel_to_abs(action0['base_rotation'], observation["state"]["base_rotation"])
                x36_seq, chunk_size = action0_to_x36_seq(action0)

                pose7_seq = converter.get_current_body29_pose7_with_root(x36_seq)

                # print(f"pose7_seq shape before interp: {pose7_seq.shape}")

                # # 再在 pose7 空间插值（参考第二、三份）
                # pose7_seq = interpolate_pose7_pairwise(pose7_seq, num_interp=12)
                # print(f"pose7_seq shape after interp: {pose7_seq.shape}")

                if pose7_seq.ndim != 3 or pose7_seq.shape[1:] != (30, 7):
                    raise ValueError(
                        f"pose7_seq shape error: expected (T, 30, 7), got {pose7_seq.shape}"
                    )

                # 清掉旧动作，避免残留
                pose_queue.clear()
                for t in range(pose7_seq.shape[0]):
                    # if t>100:
                    #     break

                    while pose_queue.is_full():
                        sleep(0.001)

                    frame = pose7_seq[t]  # (30, 7)
                    xyz_15x3, wxyz_15x4 = pick_15_points_from_pose7(frame)
                    xyz_15x3[:, 2] += 0.81
                    pose_queue.put(xyz_15x3, wxyz_15x4)
                # time.sleep(5)  # 确保动作已经开始发出

                print("=" * 80)
                print("x36_seq shape:", x36_seq.shape)
                print("pose7_seq shape:", pose7_seq.shape)
                print("queued frames:", pose7_seq.shape[0])

            except Exception as e:
                print(f"Inference/convert error: {e}")
                print(f"error file:{e.__traceback__.tb_frame.f_globals['__file__']}")
                print(f"error line:{e.__traceback__.tb_lineno}")

            elapsed = time.time() - loop_t0
            sleep(max(0.0, config.infer_interval - elapsed))

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        sender_thread.stop()
        sender_thread.join(timeout=1.0)
        print("Sender thread stopped.")

    