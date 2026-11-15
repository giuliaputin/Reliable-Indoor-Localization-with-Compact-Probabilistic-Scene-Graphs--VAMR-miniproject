#!/usr/bin/env bash
# Export a Hydra 3D scene graph into SG2Loc's 3RScan-compatible format (WP1).
#
# Usage:
#   bash scripts/export_to_sg2loc.sh [MAP_SEQ] [QUERY_SEQ]
#   Example: bash scripts/export_to_sg2loc.sh seq01_01 seq01_02

set -euo pipefail

MAP_SEQ="${1:-seq01_01}"
QUERY_SEQ="${2:-seq01_02}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

CONFIG="$ROOT/configs/export/sg2loc.yaml"
HYDRA_PY="${HYDRA_PYTHON:-$HOME/hydra_env/bin/python3}"
SG2LOC_PY="${SG2LOC_PYTHON:-$HOME/sg2loc_env/bin/python3}"

MAP_DIR="$ROOT/data/maps/$MAP_SEQ"
NPZ_PATH="$MAP_DIR/hydra_extracted.npz"
LABEL_SPACE="/home/aina/hydra_ws/src/hydra/config/label_spaces/ade20k_full_label_space.yaml"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

echo "=== [1/2] Extracting Hydra scene graph ($MAP_SEQ) ==="
"$HYDRA_PY" -m vamr_loc.export.hydra_dsg \
  --dsg "$MAP_DIR/backend/dsg_with_mesh.json" \
  --mesh "$MAP_DIR/backend/mesh.ply" \
  --label-space "$LABEL_SPACE" \
  --out "$NPZ_PATH"

echo "=== [2/2] Building BVH tree, 3D scene graph, & DINOv2 embeddings ==="
"$SG2LOC_PY" -m vamr_loc.export.sg2loc_exporter \
  --config "$CONFIG" \
  --hydra-npz "$NPZ_PATH" \
  --map-seq "$MAP_SEQ" \
  --query-seqs "$QUERY_SEQ"
