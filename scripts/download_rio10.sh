#!/usr/bin/env bash
# Download selected RIO10 scenes into data/raw/rio10 (WP1 step 3).
# RIO10 requires accepting its terms of use: see https://github.com/WaldJohannaU/RIO10
# for the official download script and URLs, then fill in DOWNLOAD_CMD below.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/data/raw/rio10"
SCENES=("scene01")   # start small

for scene in "${SCENES[@]}"; do
  echo "[TODO] download $scene → $OUT/_download/$scene"
  # DOWNLOAD_CMD "$scene" "$OUT/_download/$scene"
  # Then move seqXX_01 → $OUT/mapping/ and the other sequences → $OUT/query/
done
