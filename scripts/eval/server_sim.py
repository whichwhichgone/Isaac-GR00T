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


sys.path.append(os.getcwd())

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

SEND_FPS = 100  # 120.0


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
    port: int = 34701
    timeout_ms: int = 15000
    api_token: str = None
    task_description: str = ""
    infer_interval: float = 0.1

    urdf_path: str = (
        "/media/mpz/d5f7a2a2-7dfb-4053-8e51-ee6943e25306/assets_xr/g1/g1_body29_hand14.urdf"
    )
    model_dir: str = "/media/mpz/d5f7a2a2-7dfb-4053-8e51-ee6943e25306/assets_xr/g1/"

    stickman_path: str = (
        "/media/mpz/d5f7a2a2-7dfb-4053-8e51-ee6943e25306/20260429g1/traj_0_stickman.npy"
    )
    gt_state_path: str = "/media/mpz/d5f7a2a2-7dfb-4053-8e51-ee6943e25306/traj_0_states_0429.json"


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

    def _quat_wxyz_to_R(self, quat_wxyz):
        """
        quat_wxyz: [w, x, y, z]
        return: 3x3 rotation matrix
        """
        quat_wxyz = self._normalize_quaternion_wxyz(quat_wxyz)
        w, x, y, z = quat_wxyz

        # pinocchio Quaternion 输入是 xyzw
        quat_xyzw = np.array([x, y, z, w], dtype=np.float64)
        return pin.Quaternion(quat_xyzw).toRotationMatrix()

    def _single_frame_pose7(self, q_29, imu_quat_wxyz, root_xyz, R_ref=None):
        q_full = self._body29_to_full_model_q(q_29)
        imu_quat_wxyz = self._normalize_quaternion_wxyz(imu_quat_wxyz)
        root_xyz = np.asarray(root_xyz, dtype=np.float64).reshape(3)

        pin.forwardKinematics(self.robot.model, self.data, q_full)

        pose7 = np.zeros((30, 7), dtype=np.float64)

        # 关键修改：
        # 以前这里只取 yaw，所以 pitch/roll 被丢掉，弯腰时 root 始终水平。
        # 现在直接使用完整 imu quaternion，保留 yaw/pitch/roll。
        R_imu = self._quat_wxyz_to_R(imu_quat_wxyz)

        # 多帧时，用第一帧作为参考坐标系；单帧时直接用当前 imu。
        if R_ref is None:
            R_root = R_imu
        else:
            R_root = R_ref.T @ R_imu

        T_root = pin.SE3(R_root, root_xyz)

        quat_xyzw_root = pin.Quaternion(R_root).coeffs()
        quat_wxyz_root = np.array(
            [quat_xyzw_root[3], quat_xyzw_root[0], quat_xyzw_root[1], quat_xyzw_root[2]],
            dtype=np.float64,
        )
        quat_wxyz_root = self._normalize_quaternion_wxyz(quat_wxyz_root)
        pose7[0] = np.concatenate([quat_wxyz_root, root_xyz], axis=0)

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
            pose7[i] = np.concatenate([quat_wxyz, xyz], axis=0)

        return pose7

    def get_current_body29_pose7_with_root(self, x):
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
                R_ref=None,
            )

        if x.ndim == 2:
            if x.shape[1] != 36:
                raise ValueError(f"Expected multi-frame input shape (T, 36), got {x.shape}")

            T = x.shape[0]
            out = np.zeros((T, 30, 7), dtype=np.float64)

            # 用第一帧完整朝向作为参考，而不是只用 yaw_ref
            R_ref = self._quat_wxyz_to_R(x[0, 32:36])

            for t in range(T):
                root_xyz = x[t, :3]
                q_29 = x[t, 3:32]
                imu_quat_wxyz = x[t, 32:36]

                out[t] = self._single_frame_pose7(
                    q_29=q_29,
                    imu_quat_wxyz=imu_quat_wxyz,
                    root_xyz=root_xyz,
                    R_ref=R_ref,
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


# =========================
# helpers
# =========================
def build_observation_from_msg(msg: WR_GAE_BodyPose_Msg, task_description: str, stickman_path=None):
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
        "base_rotation": base_rotation[None, None, :].astype(np.float32),
        "left_leg": left_leg[None, None, :].astype(np.float32),
        "right_leg": right_leg[None, None, :].astype(np.float32),
        "waist": waist[None, None, :].astype(np.float32),
        "left_arm": left_arm[None, None, :].astype(np.float32),
        "right_arm": right_arm[None, None, :].astype(np.float32),
    }

    video = {
        "ego_view": np.zeros((1, 1, 224, 224, 3), dtype=np.uint8),
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


def build_observation_from_step(step: dict, task_description: str, stickman_path=None):
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
        "base_rotation": to_btd(step["base_rotation"], "base_rotation"),
        "left_leg": to_btd(step["left_leg"], "left_leg"),
        "right_leg": to_btd(step["right_leg"], "right_leg"),
        "waist": to_btd(step["waist"], "waist"),
        "left_arm": to_btd(step["left_arm"], "left_arm"),
        "right_arm": to_btd(step["right_arm"], "right_arm"),
    }

    video = {
        "ego_view": np.zeros((1, 1, 224, 224, 3), dtype=np.uint8),
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

    return x36_seq


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


# =========================
# main
# =========================
if __name__ == "__main__":
    data_json = None
    config = tyro.cli(ClientConfig)
    body_pose = BodyPoseSubscriber()

    client = server_client.PolicyClient(
        host=config.host,
        port=config.port,
        timeout_ms=config.timeout_ms,
        api_token=config.api_token,
    )

    if client.ping():
        print("Server is alive!")
    else:
        print("Failed to connect to the server.")
        sys.exit(1)

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
                    continue

                observation = build_observation_from_msg(
                    msg, config.task_description, stickman_path=config.stickman_path
                )

                with open(config.gt_state_path, "r") as f:
                    data_json = json.load(f)
                index_data_json = 0

                if index_data_json < 1:
                    print(f"Using gt state data index step: {index_data_json}.")
                    if index_data_json >= len(data_json):
                        print("index_data_json >= len(data_json)")
                        exit(0)

                    step = data_json[index_data_json]
                    observation_new = build_observation_from_step(
                        step, config.task_description, stickman_path=config.stickman_path
                    )

                    observation["state"] = observation_new["state"]
                    index_data_json += 1

                action = client.get_action(observation)
                action0 = action[0]
                x36_seq = action0_to_x36_seq(action0)
                pose7_seq = converter.get_current_body29_pose7_with_root(x36_seq)

                print(f"pose7_seq shape before interp: {pose7_seq.shape}")

                # 再在 pose7 空间插值（参考第二、三份）
                pose7_seq = interpolate_pose7_pairwise(pose7_seq, num_interp=12)
                print(f"pose7_seq shape after interp: {pose7_seq.shape}")

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
                        time.sleep(0.001)

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
