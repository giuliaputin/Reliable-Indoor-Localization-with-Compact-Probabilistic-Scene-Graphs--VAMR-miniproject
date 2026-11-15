"""Convert a RIO10 mapping sequence into Hydra input (WP1 step 4).

Usage:
    python scripts/rio10_to_hydra.py data/raw/rio10/mapping/seq01_01 data/processed/seq01_01

Writes RGB, depth, camera_info and the supplied GT poses (as odometry/TF) in a
format the hydra_ros data loader accepts (rosbag or image folder, decide in step 1).
"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seq_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    raise NotImplementedError(f"TODO WP1 step 4: convert {args.seq_dir} → {args.out_dir}")


if __name__ == "__main__":
    main()
