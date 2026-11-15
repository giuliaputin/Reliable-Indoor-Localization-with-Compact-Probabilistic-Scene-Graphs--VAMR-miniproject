"""RIO10 sequence loader.

A RIO10 sequence directory (e.g. ``scene01/extracted/seq01/seq01_01``) contains::

    camera.yaml                      intrinsics (fx fy cx cy), width, height, 3 distortion coeffs
    frame-XXXXXX.color.jpg           RGB, 960x540 (H x W, portrait)
    frame-XXXXXX.rendered.depth.png  uint16 depth in millimetres, rendered from the scan mesh
    frame-XXXXXX.pose.txt            4x4 camera-to-world (T_world_cam), camera optical frame
                                     (x right, y down, z forward); world is z-up

Test sequences have no pose files; ``T_world_cam`` is then ``None``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

DEPTH_SCALE = 1000.0  # uint16 depth units per metre

_FRAME_RE = re.compile(r"frame-(\d+)\.color\.jpg$")


@dataclass
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: list[float]

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)


@dataclass
class Frame:
    index: int
    rgb: np.ndarray                 # HxWx3 uint8 (RGB order)
    depth: np.ndarray | None        # HxW uint16 millimetres (None if not loaded)
    T_world_cam: np.ndarray | None  # 4x4 camera-to-world pose
    K: np.ndarray                   # 3x3 intrinsics

    @property
    def depth_m(self) -> np.ndarray | None:
        return None if self.depth is None else self.depth.astype(np.float32) / DEPTH_SCALE


def load_camera(seq_dir: Path) -> Camera:
    """Parse camera.yaml (simple fixed format, avoids a yaml dependency)."""
    text = (seq_dir / "camera.yaml").read_text()

    def scalar(key: str) -> float:
        return float(re.search(rf"{key}:\s*([-\d.eE]+)", text).group(1))

    def vector(key: str) -> list[float]:
        return [float(v) for v in re.search(rf"{key}:\s*\[([^\]]*)\]", text).group(1).split(",")]

    fx, fy, cx, cy = vector("model")
    return Camera(int(scalar("width")), int(scalar("height")), fx, fy, cx, cy, vector("distortion"))


class RIO10Sequence:
    def __init__(self, seq_dir: str | Path, load_depth: bool = True) -> None:
        self.seq_dir = Path(seq_dir)
        self.load_depth = load_depth
        self.camera = load_camera(self.seq_dir)
        ids = []
        for p in self.seq_dir.iterdir():
            m = _FRAME_RE.match(p.name)
            if m:
                ids.append(int(m.group(1)))
        self.frame_ids = sorted(ids)

    def __len__(self) -> int:
        return len(self.frame_ids)

    def _path(self, idx: int, suffix: str) -> Path:
        return self.seq_dir / f"frame-{idx:06d}.{suffix}"

    def pose(self, idx: int) -> np.ndarray | None:
        p = self._path(idx, "pose.txt")
        if not p.exists():
            return None
        T = np.loadtxt(p).reshape(4, 4)
        return T if np.all(np.isfinite(T)) else None

    def frame(self, idx: int) -> Frame:
        import cv2  # local import: only needed when reading images

        bgr = cv2.imread(str(self._path(idx, "color.jpg")), cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"cannot read color image for frame {idx} in {self.seq_dir}")
        depth = None
        if self.load_depth:
            depth = cv2.imread(str(self._path(idx, "rendered.depth.png")), cv2.IMREAD_UNCHANGED)
            if depth is None or depth.dtype != np.uint16:
                raise IOError(f"bad depth image for frame {idx} in {self.seq_dir}")
        return Frame(idx, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depth, self.pose(idx), self.camera.K)

    def __iter__(self) -> Iterator[Frame]:
        for idx in self.frame_ids:
            yield self.frame(idx)
