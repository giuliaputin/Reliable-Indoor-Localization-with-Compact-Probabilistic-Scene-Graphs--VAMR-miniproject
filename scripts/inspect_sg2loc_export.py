#!/usr/bin/env python3
"""CLI wrapper to inspect and visualize exported SG2Loc scene graphs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vamr_loc.export.inspect_export import inspect_sg2loc_scene


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect exported SG2Loc scene graph & BVH map")
    parser.add_argument("--export-dir", default="data/exports/sg2loc", type=Path)
    parser.add_argument("--map-seq", default="seq01_01", type=str)
    parser.add_argument(
        "--out-fig",
        default="outputs/exports/seq01_01_dinov2_views.png",
        type=Path,
        help="Path to save the 2D top-1 object view contact sheet",
    )
    parser.add_argument("--gui", action="store_true", help="Launch interactive Open3D 3D viewer")
    args = parser.parse_args()

    inspect_sg2loc_scene(
        export_dir=args.export_dir,
        map_seq=args.map_seq,
        out_fig=args.out_fig,
        show_gui=args.gui,
    )


if __name__ == "__main__":
    main()
