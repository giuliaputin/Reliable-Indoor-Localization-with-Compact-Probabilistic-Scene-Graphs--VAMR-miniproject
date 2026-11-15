"""RIO10 sequence loader and metadata utilities (WP1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import yaml


@dataclass
class Frame:
    index: int
    frame_id: str
    rgb: np.ndarray                 # HxWx3 uint8 (RGB)
    depth: np.ndarray | None        # HxW float32 in meters (None for RGB-only queries)
    T_world_cam: np.ndarray | None  # 4x4 camera-to-world pose
    K: np.ndarray                   # 3x3 intrinsics


def load_rio10_intrinsics(seq_dir: str | Path) -> tuple[np.ndarray, int, int]:
    """Parse RIO10's per-sequence camera.yaml into (K, width, height)."""
    cam_yaml = Path(seq_dir) / "camera.yaml"
    if not cam_yaml.is_file():
        raise FileNotFoundError(f"Missing RIO10 camera.yaml in {seq_dir}")
    with cam_yaml.open("r") as f:
        cfg = yaml.safe_load(f)["camera_intrinsics"]

    fx, fy, cx, cy = cfg["model"]
    K = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return K, int(cfg["width"]), int(cfg["height"])


def write_3rscan_info_txt(seq_dir: str | Path, overwrite: bool = False) -> Path:
    """Generate a 3RScan-compatible _info.txt from RIO10's camera.yaml."""
    seq_path = Path(seq_dir)
    info_path = seq_path / "_info.txt"
    if info_path.exists() and not overwrite:
        return info_path

    K, width, height = load_rio10_intrinsics(seq_path)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    mat_str = f"{fx:.8f} 0 {cx:.8f} 0 0 {fy:.8f} {cy:.8f} 0 0 0 1 0 0 0 0 1"

    content = (
        f"m_colorWidth = {width}\n"
        f"m_colorHeight = {height}\n"
        f"m_depthWidth = {width}\n"
        f"m_depthHeight = {height}\n"
        f"m_calibrationColorIntrinsic = {mat_str}\n"
        f"m_calibrationDepthIntrinsic = {mat_str}\n"
    )
    info_path.write_text(content)
    return info_path


class RIO10Sequence:
    """Iterates frames of a RIO10 sequence yielding RGB, depth, camera pose, and intrinsics."""

    def __init__(self, seq_dir: str | Path, load_depth: bool = True, depth_scale: float = 1000.0) -> None:
        self.seq_dir = Path(seq_dir)
        self.load_depth = load_depth
        self.depth_scale = depth_scale
        self.K, self.width, self.height = load_rio10_intrinsics(self.seq_dir)

        rgb_files = sorted(self.seq_dir.glob("frame-*.color.jpg"))
        self.frame_ids = [p.name.split(".")[0].split("-")[1] for p in rgb_files]

    def __len__(self) -> int:
        return len(self.frame_ids)

    def get_frame(self, idx: int) -> Frame:
        fid = self.frame_ids[idx]
        bgr = cv2.imread(str(self.seq_dir / f"frame-{fid}.color.jpg"), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Missing color frame {fid} in {self.seq_dir}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        depth = None
        if self.load_depth:
            depth_path = self.seq_dir / f"frame-{fid}.rendered.depth.png"
            if depth_path.is_file():
                raw_d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                depth = raw_d.astype(np.float32) / self.depth_scale

        pose = None
        pose_path = self.seq_dir / f"frame-{fid}.pose.txt"
        if pose_path.is_file():
            pose = np.loadtxt(pose_path, dtype=np.float64).reshape(4, 4)

        return Frame(
            index=idx,
            frame_id=fid,
            rgb=rgb,
            depth=depth,
            T_world_cam=pose,
            K=self.K.copy(),
        )

    def __iter__(self) -> Iterator[Frame]:
        for idx in range(len(self)):
            yield self.get_frame(idx)
