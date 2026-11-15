#!/usr/bin/env bash
# Run Hydra in the container on one converted mapping sequence (WP1 step 6).
# Usage: bash scripts/build_map.sh <sequence>   e.g. seq01_01
set -euo pipefail

SEQ="${1:?usage: build_map.sh <sequence>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

docker compose -f "$ROOT/docker/hydra/docker-compose.yml" run --rm hydra bash -lc "
  echo '[TODO] launch hydra_ros with configs/hydra/rio10.yaml on data/processed/$SEQ'
  echo '       and save the scene graph to data/maps/$SEQ/'
"
