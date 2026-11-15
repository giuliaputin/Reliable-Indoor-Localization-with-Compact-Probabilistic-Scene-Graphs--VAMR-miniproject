#!/usr/bin/env bash
# Build a Hydra 3D scene graph from one RIO10 sequence, using the running hydra-dev container.
#
# Run from WSL (or from Windows via: wsl.exe -e bash -lc 'bash "<repo>/scripts/build_map.sh" seq01_01'):
#   bash scripts/build_map.sh <sequence> [--stride N] [--max-frames N] [--rate R] [--fps F]
#   e.g. bash scripts/build_map.sh seq01_01 --max-frames 50     # quick test
#        bash scripts/build_map.sh seq01_01                     # full sequence
#
# Steps (all inside hydra-dev):
#   1. copy repo scripts/src/configs to /root/hydra_ws/vamr
#   2. RIO10 frames + GT poses  -> ROS 2 bag            (scripts/rio10_to_hydra.py)
#   3. ADE20K segmentation      -> bag with label images (semantic_inference closed_set_rosbag_writer)
#   4. play bag with --clock into Hydra                  (configs/hydra/rio10.launch.yaml)
#   5. Hydra saves on exit; copy output to data/maps/<sequence>/
set -euo pipefail

SEQ="${1:?usage: build_map.sh <sequence> [--stride N] [--max-frames N] [--rate R] [--fps F]}"
shift
STRIDE=1 MAX_FRAMES=0 RATE=1.0 FPS=10 REUSE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --reuse) REUSE=1; shift ;;   # skip conversion/segmentation if the semantics bag exists
    --stride) STRIDE="$2"; shift 2 ;;
    --max-frames) MAX_FRAMES="$2"; shift 2 ;;
    --rate) RATE="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac
done

CONTAINER="${HYDRA_CONTAINER:-hydra-dev}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCENE_NUM="${SEQ:3:2}"                                   # seq01_01 -> 01
SEQ_DIR="/root/data/scene${SCENE_NUM}/extracted/seq${SCENE_NUM}/${SEQ}"
WORK=/root/hydra_ws/vamr
TAG="${SEQ}"; [[ "$MAX_FRAMES" != 0 ]] && TAG="${SEQ}_test${MAX_FRAMES}"
BAG="$WORK/bags/$TAG/raw"
SEM_BAG="$WORK/bags/$TAG/semantics"
MAP_DIR="$WORK/maps/$TAG"
OUT_DIR="$ROOT/data/maps/$TAG"
LOG_DIR="$ROOT/outputs/logs"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/build_map_${TAG}.log"

dexec() { docker exec "$CONTAINER" bash -c "source /opt/ros/jazzy/setup.bash && source /root/hydra_ws/install/setup.bash && $*"; }

echo "[build_map] $SEQ  stride=$STRIDE max_frames=$MAX_FRAMES rate=$RATE fps=$FPS  log=$LOG"
docker inspect -f '{{.State.Running}}' "$CONTAINER" | grep -q true || { echo "container $CONTAINER is not running" >&2; exit 1; }
dexec "test -f $SEQ_DIR/camera.yaml" || { echo "sequence not found in container: $SEQ_DIR" >&2; exit 1; }

echo "[1/5] sync repo code into $CONTAINER:$WORK"
dexec "mkdir -p $WORK && rm -rf $WORK/scripts $WORK/src $WORK/configs"
for d in scripts src configs; do docker cp "$ROOT/$d" "$CONTAINER:$WORK/$d"; done

{
if [[ "$REUSE" == 1 ]] && dexec "test -f $SEM_BAG/metadata.yaml"; then
  echo "[2-3/5] reusing existing bags in $WORK/bags/$TAG"
else
  echo "[2/5] convert RIO10 -> bag"
  dexec "python3 $WORK/scripts/rio10_to_hydra.py $SEQ_DIR $BAG --stride $STRIDE --max-frames $MAX_FRAMES --fps $FPS"

  echo "[3/5] semantic segmentation (ADE20K efficientvit_seg_l2, TensorRT; first run builds the engine)"
  dexec "ros2 run semantic_inference_ros closed_set_rosbag_writer $BAG -o $SEM_BAG -f \
    -t /rio10/camera/color/image_raw:/rio10/camera/labels/image_raw" | tr '\r' '\n' | grep -v '^Processing bag'
  dexec "rm -rf $BAG"   # semantics bag contains a full copy of all topics
fi

echo "[4/5] run Hydra"
dexec "rm -rf $MAP_DIR && mkdir -p $MAP_DIR"
# Launch Hydra in the background inside the container; it exits (and saves) once the bag clock stops.
dexec "setsid nohup ros2 launch $WORK/configs/hydra/rio10.launch.yaml config_dir:=$WORK/configs/hydra log_path:=$MAP_DIR \
  > $MAP_DIR/hydra_launch.log 2>&1 & echo \$! > $MAP_DIR/launch.pid"
for _ in $(seq 60); do  # wait until Hydra waits for camera info (i.e. is initialised)
  dexec "grep -q 'Waiting for CameraInfo' $MAP_DIR/hydra_launch.log" && break; sleep 2
done
sleep 3
if ! dexec "kill -0 \$(cat $MAP_DIR/launch.pid) 2>/dev/null"; then
  dexec "cat $MAP_DIR/hydra_launch.log"; echo "ERROR: Hydra launch died" >&2; exit 1
fi
dexec "grep -h 'Waiting for CameraInfo' $MAP_DIR/hydra_launch.log | head -1" || true
dexec "ros2 bag play $SEM_BAG --clock 50 --rate $RATE --delay 2"
echo "bag finished, waiting for Hydra to save"
for i in $(seq 180); do
  dexec "kill -0 \$(cat $MAP_DIR/launch.pid) 2>/dev/null" || break
  if [[ $i == 90 ]]; then echo "Hydra still running, sending SIGINT"; dexec "kill -INT \$(cat $MAP_DIR/launch.pid)" || true; fi
  sleep 2
done
dexec "tail -n 30 $MAP_DIR/hydra_launch.log"

echo "[5/5] copy results to $OUT_DIR"
mkdir -p "$OUT_DIR"; rm -rf "${OUT_DIR:?}"/*
docker cp "$CONTAINER:$MAP_DIR/." "$OUT_DIR"
ls -la "$OUT_DIR" "$OUT_DIR/backend" 2>/dev/null || true
test -s "$OUT_DIR/backend/dsg.json" || test -s "$OUT_DIR/backend/dsg_with_mesh.json" \
  || { echo "ERROR: no scene graph produced" >&2; exit 1; }
echo "[build_map] done: $OUT_DIR"
} 2>&1 | tee "$LOG"
exit "${PIPESTATUS[0]}"
