"""
Unitree G1 + Dex3 数据采集（工业稳定版 - 中文修复版）
"""

import os
import cv2
import json
import time
import datetime
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor

# ⭐ NEW: PIL 中文支持
from PIL import Image, ImageDraw, ImageFont

from utils.cameras_imageio_v1 import ThreadedVideoCapture

import sys
sys.path.insert(0, "/home/unitree/unitree_sdk2_python")
sys.path.insert(0, "/home/unitree/xr_teleoperate")

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from teleop.robot_control.robot_arm import G1_29_ArmController
from multiprocessing import Array

ChannelFactoryInitialize(0, "eth0")


# =========================
# utils
# =========================
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def save_image_file(image_path, image_data):
    cv2.imwrite(image_path, image_data)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ⭐ NEW: 中文绘制函数（关键）
def put_text_cn(img, text, pos, font_size=24, color=(255, 255, 255)):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    # Linux 常见字体（你机器一般有）
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    draw.text(pos, text, font=font, fill=color)
    return np.array(img_pil)


# =========================
# main
# =========================
class DataCollector:
    def __init__(self, args):

        self.fps = args["fps"]
        self.ratio = self.fps // 20

        self.root_path = create_folder(args["root_path"])
        self.task = args["task"]

        self.episode_count = self._init_episode()

        self.data_list = []
        self.mode = "WAITING"

        self.action_count = 0
        self.img_count = 0

        self.latest_frame = None
        self.frame_lock = threading.Lock()

        self.running = True

        # =========================
        #  language (保留你逻辑不变)
        # =========================
        self.lang = "zh"

        self.I18N = {
            "en": {
                # "mode": "Mode",
                "episode": "Episode",
                "action": "Action Count",
                "image": "Image Count",
                "fps": "FPS Target",
                "ratio": "Image Ratio",
                "data": "Data Len",
                "status": "Status",
                "recording": "RECORDING",
                "idle": "IDLE",
                "time": "Time"
            },
            "zh": {
                # "mode": "模式",
                "episode": "回合",
                "action": "动作数",
                "image": "图片数",
                "fps": "目标帧率",
                "ratio": "采样比例",
                "data": "数据量",
                "status": "状态",
                "recording": "采集中",
                "idle": "空闲",
                "time": "时间"
            }
        }

        print("[INFO] init robot")
        self.arm_ctrl = G1_29_ArmController(False, False, False)
        self.dual_hand_state_array = Array("d", 14, lock=False)

        cv2.namedWindow("ada")
        cv2.namedWindow("front_head")

        print("[INFO] init camera")
        self.caps = {
            "front_head": ThreadedVideoCapture("front_head"),
        }

        self.cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.cam_thread.start()

        time.sleep(2)

        self.executor = ThreadPoolExecutor(max_workers=4)

    # =========================
    def toggle_lang(self):
        self.lang = "en" if self.lang == "zh" else "zh"
        print(f"[INFO] language -> {self.lang}")

    # =========================
    def _init_episode(self):
        max_id = -1
        for item in os.listdir(self.root_path):
            if item.startswith("episode_"):
                try:
                    idx = int(item.split("_")[1])
                    max_id = max(max_id, idx)
                except:
                    pass
        return max_id + 1

    # =========================
    def _camera_loop(self):
        while self.running:
            for name, cap in self.caps.items():
                ret, frame = cap.read()
                if ret:
                    with self.frame_lock:
                        self.latest_frame = frame.copy()
            time.sleep(1 / 20)

    # =========================
    def _get_joints(self):
        return np.array(self.arm_ctrl.get_current_motor_q())[:29]

    # =========================
    def _save_json(self):
        if not self.data_list:
            return

        path = os.path.join(self.episode_path, "data.json")
        with open(path, "w") as f:
            json.dump(self.data_list, f, cls=NumpyEncoder, indent=4)

    # =========================
    def _start(self):
        print(f"[INFO] Starting episode_{self.episode_count} ...")
        self.episode_path = create_folder(
            os.path.join(self.root_path, f"episode_{self.episode_count}")
        )

        create_folder(os.path.join(self.episode_path, "images", "front_head"))

        self.data_list = []
        self.action_count = 0
        self.img_count = 0

        self.mode = "COLLECTING"

    def _stop(self):
        self._save_json()
        self.episode_count = self._init_episode()
        print(f"[INFO] Saved episode_{self.episode_count-1}")
        self.mode = "WAITING"

    def _delete_last(self):
        import shutil
        last = os.path.join(self.root_path, f"episode_{self.episode_count-1}")
        if os.path.exists(last):
            shutil.rmtree(last)

        self.episode_count = self._init_episode()

    # =========================
    def _collect_data_only(self):
        imu = np.array(self.arm_ctrl.get_base_orientation_quat())
        body_joint = self._get_joints()
        hand_joint = np.array(self.dual_hand_state_array)

        with self.frame_lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()

        if frame is None:
            return

        if self.action_count % self.ratio == 0:
            img_idx = self.img_count

            image_dir = os.path.join(self.episode_path, "images", "front_head")
            image_path = os.path.join(image_dir, f"{img_idx:05d}.png")

            self.executor.submit(save_image_file, image_path, frame)

            self.current_image_path = os.path.relpath(image_path, self.episode_path)
            self.img_count += 1

        self.data_list.append({
            "front": getattr(self, "current_image_path", None),
            "body_joint": body_joint,
            "hand_joint": hand_joint,
            "imu": imu,
            "task": [self.task],
            "timestamp": time.time(),
        })

        self.action_count += 1

    # =========================
    def run(self):

        print("[INFO] running...")

        target_dt = 1.0 / self.fps
        last = time.time()

        count = 0
        fps_start = time.time()

        while True:

            now = time.time()
            dt = now - last

            if dt < target_dt:
                time.sleep(target_dt - dt)

            last = time.time()

            if count == 100:
                now = time.time()
                print(f"[REAL FPS] {count / (now - fps_start):.2f}")
                count = 0
                fps_start = now

            # ================= UI =================
            ui = np.zeros((400, 800, 3), np.uint8)

            t = datetime.datetime.now().strftime("%H:%M:%S")
            L = self.I18N[self.lang]

            # ⭐ 用 PIL 替换 cv2.putText（支持中文）
            ui = put_text_cn(ui, f"{L['time']}: {t}", (20, 40))
            # ui = put_text_cn(ui, f"{L['mode']}: {self.mode}", (20, 80))
            ui = put_text_cn(ui, f"{L['episode']}: {self.episode_count}", (20, 80))
            ui = put_text_cn(ui, f"{L['action']}: {self.action_count}", (20, 120))
            ui = put_text_cn(ui, f"{L['image']}: {self.img_count}", (20, 160))
            ui = put_text_cn(ui, f"{L['fps']}: {self.fps}", (20, 200))
            ui = put_text_cn(ui, f"{L['ratio']}: 1/{self.ratio}", (20, 240))
            ui = put_text_cn(ui, f"{L['data']}: {len(self.data_list)}", (20, 280))

            status = L["recording"] if self.mode == "COLLECTING" else L["idle"]
            ui = put_text_cn(ui, f"{L['status']}: {status}", (20, 320), color=(0, 0, 255))

            ui = put_text_cn(ui, f"Lang: {self.lang} (L切换)", (500, 40), color=(255, 255, 0))

            cv2.imshow("ada", ui)

            with self.frame_lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()

            if frame is not None:
                cv2.imshow("front_head", frame)

            key = cv2.waitKey(1) & 0xFF

            # ================= NEW =================
            if key == ord("l"):
                self.toggle_lang()

            if key == ord("1"):
                if self.mode == "WAITING":
                    self._start()
                else:
                    self._stop()

            elif key == ord("2"):
                if self.mode == "WAITING":
                    self._delete_last()

            elif key == ord("3"):
                break

            if self.mode == "COLLECTING":
                self._collect_data_only()
                count += 1

        self.running = False
        self.cleanup()

    def cleanup(self):
        self.running = False
        self.executor.shutdown(wait=True)
        cv2.destroyAllWindows()


# =========================
if __name__ == "__main__":

    args = {
        "root_path": "/home/unitree/xr_teleoperate/datasets_wr1/g1/0507_tidy_up_g1",
        "fps": 120,
        "task": "Put the things on the desk into the storage box",
    }

    DataCollector(args).run()