# Hydra Docker image

Runs Hydra (`hydra_ros`) in a ROS container with NVIDIA GPU access.

## Prerequisites (Windows)
1. WSL2 with Ubuntu (`wsl -l -v`)
2. Docker Desktop, WSL2 backend, Ubuntu integration enabled
3. Recent NVIDIA Windows driver (GPU passthrough to WSL is automatic)
4. Test: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`

Run all commands from a WSL shell. Cloning the repo into the WSL filesystem
(e.g. `~/vamr`) makes I/O much faster than `/mnt/c/...`.

## Build & run
```bash
docker compose -f docker/hydra/docker-compose.yml build
docker compose -f docker/hydra/docker-compose.yml run --rm hydra
```
Inside the container, `data/`, `configs/`, `scripts/` and `outputs/` are mounted under `/project`.

## Before the first build
Check the upstream hydra_ros README for the supported ROS distro and the
dependency `.repos` file, then set `ROS_DISTRO` / `HYDRA_ROS_REF` in
`docker-compose.yml` and pin a commit.
