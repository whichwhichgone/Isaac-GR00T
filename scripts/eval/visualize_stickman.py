import os
import time

import matplotlib


matplotlib.use("Agg")
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np


# Local index mapping
PELVIS, LEFT_FOOT, RIGHT_FOOT, SPINE2, LEFT_WRIST, RIGHT_WRIST = range(6)

# Bone connections (local indices)
BONE_PAIRS = [
    (PELVIS, SPINE2),  # torso
    (SPINE2, LEFT_WRIST),  # left arm
    (SPINE2, RIGHT_WRIST),  # right arm
    (PELVIS, LEFT_FOOT),  # left leg
    (PELVIS, RIGHT_FOOT),  # right leg
]

POINT_COLORS = ["red", "purple", "orange", "gold", "blue", "green"]
POINT_LABELS = ["pelvis", "L_foot", "R_foot", "spine2", "L_wrist", "R_wrist"]
POINT_SIZES = [60, 60, 60, 140, 60, 60]


def make_bone_segments(xyz: np.ndarray) -> np.ndarray:
    return np.stack([xyz[[i for i, _ in BONE_PAIRS]], xyz[[j for _, j in BONE_PAIRS]]], axis=1)


def load_stickman(path: str) -> np.ndarray:
    data = np.load(path)  # (900,)
    return data.reshape(50, 18)  # (50, 18)


class StickmanVisualizer:
    def __init__(self, kp_xyz: np.ndarray, title: str = "", fps: float = 50.0):
        self.kp_xyz = kp_xyz  # (T, 6, 3)
        self.num_frames = kp_xyz.shape[0]
        self.title = title
        self.play_fps = fps
        self.current_frame = 0
        self.is_paused = False
        self.is_closed = False

        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        self.bones = Line3DCollection(
            make_bone_segments(self.kp_xyz[0]),
            colors="black",
            linewidths=2.0,
            alpha=0.8,
        )
        self.ax.add_collection3d(self.bones)
        self.scatters = [
            self.ax.scatter([], [], [], c=c, s=s, label=l, zorder=5)
            for c, l, s in zip(POINT_COLORS, POINT_LABELS, POINT_SIZES)
        ]
        self.ax.legend(loc="upper right", fontsize=8)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", lambda e: setattr(self, "is_closed", True))

    def _on_key(self, event):
        k = (event.key or "").lower()
        if k == " ":
            self.is_paused = not self.is_paused
        elif k == "r":
            self.current_frame = 0
        elif k in [".", ">"]:
            self.is_paused = True
            self.current_frame = (self.current_frame + 1) % self.num_frames
            self._render(force_draw=True)
        elif k in [",", "<"]:
            self.is_paused = True
            self.current_frame = (self.current_frame - 1) % self.num_frames
            self._render(force_draw=True)
        elif k in ["q", "escape"]:
            self.is_closed = True
            plt.close(self.fig)

    def _render(self, force_draw=False):
        frame_idx = self.current_frame % self.num_frames
        xyz = self.kp_xyz[frame_idx]  # (6, 3)

        # Update bone segments
        self.bones.set_segments(make_bone_segments(xyz))

        # Update keypoint scatters
        for idx, sc in enumerate(self.scatters):
            sc._offsets3d = ([xyz[idx, 0]], [xyz[idx, 1]], [xyz[idx, 2]])

        # Update axis limits
        center = np.mean(xyz, axis=0)
        self.ax.set_xlim([center[0] - 1.0, center[0] + 1.0])
        self.ax.set_ylim([center[1] - 1.0, center[1] + 1.0])
        self.ax.set_zlim([0.0, 2.0])
        self.ax.set_title(
            f"{self.title}\nFrame {frame_idx}/{self.num_frames}  fps={self.play_fps:.0f}\n"
            f"[Space] pause  [R] reset  [,/.] step  [Q] quit"
        )

        if force_draw:
            self.fig.canvas.draw()
        else:
            self.fig.canvas.draw_idle()
        try:
            self.fig.canvas.flush_events()
        except Exception:
            plt.pause(0)

    def run(self):
        plt.ion()
        plt.show(block=False)
        next_tick = time.perf_counter()
        period = 1.0 / self.play_fps

        while not self.is_closed:
            if self.is_paused:
                self._render(force_draw=False)
                time.sleep(0.01)
                next_tick = time.perf_counter() + period
                continue

            now = time.perf_counter()
            if now < next_tick:
                time.sleep(next_tick - now)

            self._render(force_draw=False)
            self.current_frame = (self.current_frame + 1) % self.num_frames
            next_tick += period

        plt.ioff()

    def save_gif(self, output_path: str):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        def update(frame_idx):
            self.current_frame = frame_idx
            self._render(force_draw=True)
            return [self.bones, *self.scatters]

        anim = FuncAnimation(
            self.fig,
            update,
            frames=self.num_frames,
            interval=1000.0 / self.play_fps,
            blit=False,
            repeat=True,
        )
        anim.save(output_path, writer=PillowWriter(fps=self.play_fps))
        plt.close(self.fig)


if __name__ == "__main__":
    npy_path = "open_loop_eval/traj_0_stickman.npy"
    gif_path = "open_loop_eval/traj_0_stickman.gif"
    kp_xyz = load_stickman(npy_path).reshape(-1, 6, 3)  # (50, 6, 3)
    viz = StickmanVisualizer(kp_xyz, title=npy_path, fps=10.0)
    viz.save_gif(gif_path)
    print(f"Saved stickman GIF to {gif_path}")
