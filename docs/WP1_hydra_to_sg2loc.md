# WP1: Hydra Scene Graph to SG2Loc Export Pipeline

This document describes how Hydra 3D scene graphs (`dsg_with_mesh.json` + `mesh.ply`)
are converted into the 3RScan-compatible format consumed by **SG2Loc** and **SceneGraphLoc**.

## 1. Environment & Directory Setup

Two isolated Python virtual environments are used on the host:
- `~/hydra_env`: Contains `spark_dsg` and Open3D for reading Hydra's `dsg_with_mesh.json`.
- `~/sg2loc_env`: Contains PyTorch (CUDA), `numba-cuda`, `poselib`, `torch_geometric`, and SG2Loc dependencies.

Expected repository layout (symlinks are gitignored under `data/`):
- `third_party/sg2loc` -> SG2Loc codebase (with `checkpoints/pretrained.pth.tar`)
- `data/raw/rio10/mapping/seq01_01` -> RIO10 reference RGB-D sequence
- `data/raw/rio10/query/seq01_02` -> RIO10 validation query sequence
- `data/maps/seq01_01/backend/` -> Hydra `dsg_with_mesh.json` and `mesh.ply`
- `data/exports/sg2loc/` -> Exported SG2Loc `scenes/` and `files/` tree

## 2. Schema Translation (Hydra -> SG2Loc)

| Component | Hydra Format | SG2Loc Format (`data/exports/sg2loc/`) |
| :--- | :--- | :--- |
| **Floor vertices** | `G.mesh.get_labels() == 3` (ADE20k `floor`) | `obj_ids.npy == 1` (used by `Scan3RParticleFilter.floor_mask`) |
| **Object instances** | `DsgLayers.OBJECTS` (`category_id >= 0`) | `obj_ids.npy = category_id + 2` (`>= 2`; `0` = background) |
| **Semantics** | ADE20k class names | NYU40 IDs (`1..40`) via `configs/export/sg2loc.yaml` |
| **3D Geometry** | `mesh.ply` (`vertices`, `triangles`, `vertex_colors`) | Flattened BVH tree (`files/bvh_trees_0.5/<seq>/`) + FPS point clouds (`64, 128, 256, 512`) |
| **Scene Graph** | Object centroids & bounding boxes | Pairwise geometric edges (`left`, `right`, `front`, `behind`, `above`, `below`, `close_by`, `none`) + 41-D edge BoW & 164-D attribute BoW in `files/orig/data/<seq>.pkl` |
| **Visual Descriptors** | RGB frames + `T_world_cam` poses | Top-10 projecting frames per object, 3-level multi-scale crops, `dinov2_vitg14` CLS embeddings (`1536`-D) in `files/Features3D/obj_dinov2_top10_l3/<seq>.pkl` |
| **Intrinsics** | `camera.yaml` (`540x960`) | Converted automatically to 3RScan `_info.txt` |

## 3. Running & Inspecting the Export

```bash
# Run the full two-stage export (defaults: MAP_SEQ=seq01_01, QUERY_SEQ=seq01_02)
bash scripts/export_to_sg2loc.sh seq01_01 seq01_02

# Inspect exported tensors, save 2D view contact sheet, and open 3D BVH/scene-graph viewer
~/sg2loc_env/bin/python3 scripts/inspect_sg2loc_export.py --map-seq seq01_01 --gui
```
---

### 12. `README.md` & Editable Install Verification

```bash
cat << 'EOF' > README.md
# Reliable Indoor Localization with Compact Probabilistic Scene Graphs
```

Objects reconstructed from the same camera poses can share geometric errors.
If localization ignores these dependencies, its estimates can be overconfident.
This project studies which dependencies a compact 3D scene graph must keep to
support reliable indoor localization.

- **Mapping:** RGB-D recordings -> [Hydra](https://github.com/MIT-SPARK/Hydra) 3D scene graphs
- **Localization:** RGB query images -> SG2Loc baseline, extended with shared map uncertainty
- **Data:** [RIO10](https://github.com/WaldJohannaU/RIO10) (mapping and query recordings are kept separate)

## Work packages

| WP | Goal | Status |
|----|------|--------|
| WP1 | Reference maps (Hydra) + SG2Loc export & localization baseline | Export pipeline complete ([docs/WP1_hydra_to_sg2loc.md](docs/WP1_hydra_to_sg2loc.md)); query localization next |
| WP2 | Characterize independent vs. shared (grouped) map perturbations | – |
| WP3 | Compact probabilistic map model (shared transforms + residuals) in localization scoring | – |
| WP4 | Evaluation & ablations (RIO10, controlled perturbations, calibration, memory, runtime) | – |

## Repository layout
configs/          dataset, Hydra, and SG2Loc export configuration (YAML)
data/             datasets, maps, and SG2Loc exports (gitignored, see data/README.md)
raw/rio10/      original RIO10 sequences: mapping/ and query/
processed/      converted Hydra inputs
maps/           Hydra outputs (dsg_with_mesh.json, mesh.ply)
exports/        exported SG2Loc BVH trees, scene graphs, and DINOv2 descriptors
docker/hydra/     Docker image for Hydra (ROS) with GPU support
docs/             technical documentation, plans, and references
scripts/          download, conversion, map-building, export, and inspection entry points
src/vamr_loc/     Python package (datasets, mapping, export, localization)
third_party/      external code (SG2Loc, SceneGraphLoc)
outputs/          logs, figures, results (gitignored)

## Quick start

```bash
# 1. Install local package in editable mode
source ~/sg2loc_env/bin/activate
pip install --no-deps -e .

# 2. Export Hydra scene graph (seq01_01) to SG2Loc format
bash scripts/export_to_sg2loc.sh seq01_01 seq01_02

# 3. Inspect exported scene graph, 2D DINOv2 projections, and 3D BVH mesh
python3 scripts/inspect_sg2loc_export.py --map-seq seq01_01 --gui
```

For full details on the Hydra -> SG2Loc schema translation, environment setup, and data links, see docs/WP1_hydra_to_sg2loc.md.

## References
See docs/references.md
