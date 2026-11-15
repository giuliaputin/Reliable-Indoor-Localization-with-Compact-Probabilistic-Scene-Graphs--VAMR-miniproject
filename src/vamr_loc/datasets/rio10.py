"""RIO10 sequence loader (WP1 step 4).

Iterates frames of a RIO10 sequence and yields RGB, depth, camera pose and intrinsics.
The file format and pose convention must be checked against the downloaded data
(see configs/dataset/rio10.yaml).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass
class Frame:
    index: int
    rgb: np.ndarray            # HxWx3 uint8
    depth: np.ndarray | None   # HxW float32 in meters (None for RGB-only queries)
    T_world_cam: np.ndarray | None  # 4x4 camera-to-world pose
    K: np.ndarray              # 3x3 intrinsics


class RIO10Sequence:
    def __init__(self, seq_dir: str | Path, load_depth: bool = True) -> None:
        self.seq_dir = Path(seq_dir)
        self.load_depth = load_depth

    def __len__(self) -> int:
        raise NotImplementedError("TODO WP1 step 4")

    def __iter__(self) -> Iterator[Frame]:
        raise NotImplementedError("TODO WP1 step 4")
