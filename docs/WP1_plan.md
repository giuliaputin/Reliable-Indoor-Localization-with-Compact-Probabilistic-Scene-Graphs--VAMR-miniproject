# WP1 — Reference maps and localization baseline

**Goal:** build Hydra 3D scene graphs from RIO10 mapping recordings using the
supplied poses and pretrained segmentation. Export object geometry and visual
descriptors, then set up an SG2Loc localization baseline, with SceneGraphLoc
for candidate-scene retrieval. Mapping and query recordings stay strictly
separate.

**Setup decisions:** Hydra runs in Docker inside WSL2 with GPU passthrough
(`docker/hydra/`). The dataset is RIO10 (`data/raw/rio10/`).

---

## Step 1 — Environment
- [ ] Start WSL2 (Ubuntu) and Docker Desktop (Settings → Resources → WSL integration enabled).
- [ ] Check GPU in containers: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`.
- [ ] Check which ROS distro upstream `hydra_ros` currently targets (ROS 2 Jazzy on recent
      releases, ROS 1 Noetic on older ones) and set `ROS_DISTRO` / `HYDRA_ROS_REF` in
      `docker/hydra/Dockerfile`. Pin a commit for reproducibility.
- [ ] `docker compose -f docker/hydra/docker-compose.yml build`.
- [ ] If RViz is needed: check that WSLg GUI forwarding works (`DISPLAY`, `/tmp/.X11-unix` mount).

## Step 2 — Hydra smoke test
- [ ] Run Hydra on its official demo data (uHumans2) following the upstream README.
- [ ] Confirm a scene graph and mesh are produced and saved under `data/maps/demo/`.

## Step 3 — RIO10 data
- [ ] Download 1–2 scenes with `scripts/download_rio10.sh` (start small: e.g. `scene01`).
- [ ] Layout: mapping sequence `seqXX_01` → `data/raw/rio10/mapping/`, other sequences → `data/raw/rio10/query/`.
- [ ] Note: RIO10 test-sequence GT poses are held out for the online benchmark. Use the
      validation sequences (with GT) for local evaluation.
- [ ] Inspect the frame format (color, depth, pose files, camera intrinsics) and record
      intrinsics, depth scale and pose convention (camera-to-world?) in `configs/dataset/rio10.yaml`.

## Step 4 — Conversion to Hydra input
- [ ] Implement `src/vamr_loc/datasets/rio10.py` (iterate frames: RGB, depth, pose, K).
- [ ] Implement `scripts/rio10_to_hydra.py`: write a rosbag (or use the hydra_ros data loader)
      with RGB, depth, camera_info and **GT poses as odometry/TF**.
- [ ] Sanity check: back-project depth with GT poses → fused point cloud should be consistent.

## Step 5 — Semantic segmentation
- [ ] Use a pretrained 2D segmentation model compatible with Hydra
      (MIT-SPARK `semantic_inference`, or precomputed label images).
- [ ] Define the label → Hydra category mapping in `configs/hydra/`.

## Step 6 — Build reference maps
- [ ] `scripts/build_map.sh <scene>` runs Hydra in the container on the mapping sequence.
- [ ] Outputs in `data/maps/<scene>/` (DSG json, mesh). Visual check in RViz / Open3D.

## Step 7 — Export objects
- [ ] Parse the Hydra DSG (`spark_dsg` Python bindings) in `src/vamr_loc/mapping/`.
- [ ] Export per object: id, class, centroid, bounding box, point/mesh subset, visual descriptor
      → `data/exports/<scene>/objects.json` (+ arrays as `.npz`).

## Step 8 — Localization baseline
- [ ] Add SG2Loc [2] and SceneGraphLoc [5] as git submodules in `third_party/`.
- [ ] SceneGraphLoc: retrieve candidate scenes for each RGB query.
- [ ] SG2Loc: estimate the 6-DoF pose against the candidate map. Use **query sequences only**.
- [ ] Log translation and rotation errors and confidence per query to `outputs/wp1_baseline/`.

## Deliverables of WP1
- Reproducible Hydra Docker image
- Hydra maps for selected RIO10 scenes
- Object export format + exporter
- Baseline localization results (pose errors, confidences)
