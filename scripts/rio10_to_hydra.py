"""Convert a RIO10 sequence into a ROS 2 bag (mcap) that Hydra can consume.

Runs inside the hydra-dev container (needs rclpy / rosbag2_py from ROS 2 Jazzy):

    python3 scripts/rio10_to_hydra.py <seq_dir> <out_bag> [--stride 1] [--max-frames N] [--fps 10]

Topics written (frame ids in parentheses):
    /rio10/camera/color/image_raw    sensor_msgs/Image rgb8         (rio10_camera)
    /rio10/camera/depth/image_raw    sensor_msgs/Image 16UC1, mm    (rio10_camera)
    /rio10/camera/color/camera_info  sensor_msgs/CameraInfo         (rio10_camera)
    /tf                              world -> rio10_body (GT pose), rio10_body -> rio10_camera (identity)

RIO10 poses are camera-to-world in the optical frame (x right, y down, z forward) with a z-up
world, which is exactly the ROS optical-frame convention, so they are published unchanged.
RIO10 has no timestamps; frames are stamped at a fixed rate (--fps).
Frames without a valid pose are skipped.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vamr_loc.datasets.rio10 import RIO10Sequence  # noqa: E402

import rosbag2_py  # noqa: E402
from builtin_interfaces.msg import Time  # noqa: E402
from geometry_msgs.msg import TransformStamped  # noqa: E402
from rclpy.serialization import serialize_message  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402

WORLD, BODY, CAMERA = "world", "rio10_body", "rio10_camera"
NS = "/rio10/camera"
TOPICS = {
    f"{NS}/color/image_raw": "sensor_msgs/msg/Image",
    f"{NS}/depth/image_raw": "sensor_msgs/msg/Image",
    f"{NS}/color/camera_info": "sensor_msgs/msg/CameraInfo",
    "/tf": "tf2_msgs/msg/TFMessage",
}
T0_NS = 100 * 10**9  # start sim time at 100 s (avoids t=0 edge cases)
TF_LEAD_NS = 20 * 10**6  # bag receive time of /tf precedes the frame by 20 ms


def rot_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix -> (x, y, z, w), numerically robust (Shepperd)."""
    U, _, Vt = np.linalg.svd(R)  # re-orthonormalise
    R = U @ Vt
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return tuple(q / np.linalg.norm(q))


def stamp(t_ns: int) -> Time:
    return Time(sec=t_ns // 10**9, nanosec=t_ns % 10**9)


def transform(t_ns: int, parent: str, child: str, T: np.ndarray) -> TransformStamped:
    msg = TransformStamped()
    msg.header.stamp = stamp(t_ns)
    msg.header.frame_id = parent
    msg.child_frame_id = child
    msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = map(float, T[:3, 3])
    q = rot_to_quat(T[:3, :3])
    msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w = q
    return msg


def image(t_ns: int, arr: np.ndarray, encoding: str) -> Image:
    msg = Image()
    msg.header.stamp = stamp(t_ns)
    msg.header.frame_id = CAMERA
    msg.height, msg.width = arr.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = 0
    arr = np.ascontiguousarray(arr)
    msg.step = arr.strides[0]
    msg.data = arr.tobytes()
    return msg


def camera_info(t_ns: int, seq: RIO10Sequence) -> CameraInfo:
    c = seq.camera
    msg = CameraInfo()
    msg.header.stamp = stamp(t_ns)
    msg.header.frame_id = CAMERA
    msg.width, msg.height = c.width, c.height
    # Depth is rendered from the mesh with a pinhole model; the RGB distortion is small
    # (|k| < 0.04) and is ignored, matching how Hydra uses only K.
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0] * 5
    msg.k = [c.fx, 0.0, c.cx, 0.0, c.fy, c.cy, 0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [c.fx, 0.0, c.cx, 0.0, 0.0, c.fy, c.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seq_dir", type=Path)
    ap.add_argument("out_bag", type=Path)
    ap.add_argument("--stride", type=int, default=1, help="use every n-th frame")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N written frames (0 = all)")
    ap.add_argument("--fps", type=float, default=10.0, help="synthetic frame rate of the output stream")
    ap.add_argument("--max-depth", type=float, default=0.0, help="zero out depth beyond this (m), 0 = keep")
    args = ap.parse_args()

    seq = RIO10Sequence(args.seq_dir)
    ids = seq.frame_ids[:: args.stride]
    print(f"{args.seq_dir}: {len(seq)} frames, using {len(ids)} (stride {args.stride})", flush=True)

    if args.out_bag.exists():
        shutil.rmtree(args.out_bag)
    args.out_bag.parent.mkdir(parents=True, exist_ok=True)
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(args.out_bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    for name, typ in TOPICS.items():
        writer.create_topic(rosbag2_py.TopicMetadata(id=0, name=name, type=typ, serialization_format="cdr"))

    dt_ns = int(1e9 / args.fps)
    identity = np.eye(4)
    written = skipped = 0
    for idx in ids:
        frame = seq.frame(idx)
        if frame.T_world_cam is None:
            skipped += 1
            continue
        t = T0_NS + written * dt_ns
        depth = frame.depth
        if args.max_depth > 0:
            depth = depth.copy()
            depth[depth > args.max_depth * 1000.0] = 0
        tf = TFMessage(transforms=[transform(t, WORLD, BODY, frame.T_world_cam), transform(t, BODY, CAMERA, identity)])
        # TF is recorded slightly before the images so it is always available at lookup time.
        writer.write("/tf", serialize_message(tf), t - TF_LEAD_NS)
        writer.write(f"{NS}/color/camera_info", serialize_message(camera_info(t, seq)), t)
        writer.write(f"{NS}/depth/image_raw", serialize_message(image(t, depth, "16UC1")), t)
        writer.write(f"{NS}/color/image_raw", serialize_message(image(t, frame.rgb, "rgb8")), t)
        written += 1
        if written % 200 == 0:
            print(f"  {written}/{len(ids)} frames", flush=True)
        if args.max_frames and written >= args.max_frames:
            break
    del writer  # closes the bag
    print(f"wrote {written} frames ({skipped} skipped without pose) to {args.out_bag}", flush=True)


if __name__ == "__main__":
    main()
