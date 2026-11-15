# Reliable Indoor Localization with Compact Probabilistic Scene Graphs

Objects reconstructed from the same camera poses can share geometric errors.
If localization ignores these dependencies, its estimates can be overconfident.
This project studies which dependencies a compact 3D scene graph must keep to
support reliable indoor localization.

- **Mapping:** RGB-D recordings → [Hydra](https://github.com/MIT-SPARK/Hydra) 3D scene graphs
- **Localization:** RGB query images → SG2Loc baseline, extended with shared map uncertainty
- **Data:** [RIO10](https://github.com/WaldJohannaU/RIO10) (mapping and query recordings are kept separate)

## Work packages

| WP | Goal | Status |
|----|------|--------|
| WP1 | Reference maps (Hydra) + localization baseline (SG2Loc, SceneGraphLoc retrieval) | in progress, see [docs/WP1_plan.md](docs/WP1_plan.md) |
| WP2 | Characterize independent vs. shared (grouped) map perturbations | – |
| WP3 | Compact probabilistic map model (shared transforms + residuals) in localization scoring | – |
| WP4 | Evaluation & ablations (RIO10, controlled perturbations, calibration, memory, runtime) | – |

## Repository layout

```
configs/          dataset + Hydra configuration (YAML)
data/             datasets and generated maps (gitignored, see data/README.md)
  raw/rio10/      original RIO10 sequences: mapping/ and query/
  processed/      converted Hydra inputs
  maps/           Hydra outputs (scene graphs, meshes)
  exports/        object geometry + descriptors for localization
docker/hydra/     Docker image for Hydra (ROS) with GPU support
docs/             plans, notes, references
scripts/          download, conversion and map-building entry points
src/vamr_loc/     Python package (datasets, mapping, export, localization)
third_party/      external code (SG2Loc, SceneGraphLoc) as git submodules
outputs/          logs, figures, results (gitignored)
```

## Quick start

Requirements: Windows with WSL2 + Docker Desktop (WSL2 backend) and an NVIDIA GPU,
or a native Linux machine with Docker and the NVIDIA Container Toolkit.

```bash
# Python tooling (host or WSL)
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Hydra container (see docker/hydra/README.md)
docker compose -f docker/hydra/docker-compose.yml build
docker compose -f docker/hydra/docker-compose.yml run --rm hydra

# Data (see data/README.md)
bash scripts/download_rio10.sh
```

## References

See [docs/references.md](docs/references.md).
