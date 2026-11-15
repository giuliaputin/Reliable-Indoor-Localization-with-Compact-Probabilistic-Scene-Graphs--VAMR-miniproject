# Data

Everything in this folder is **gitignored** except this README and the `.gitkeep` files.

```
data/
  raw/rio10/
    mapping/     reference RGB-D recordings (seqXX_01)  → Hydra maps
    query/       other recordings of the same scenes     → RGB localization queries
  processed/     converted Hydra inputs (rosbags / loader format)
  maps/          Hydra outputs per scene (scene graph, mesh)
  exports/       per-object geometry + descriptors for localization
```

**Rule:** mapping and query recordings are never mixed. Maps are built only from `mapping/`.
Localization uses only RGB images from `query/`.

## RIO10
- Project page / download: https://github.com/WaldJohannaU/RIO10 (requires accepting the terms of use)
- 10 scenes, several recordings each. `seqXX_01` is the mapping recording.
  GT poses for test sequences are held out by the online benchmark, so use the
  validation sequences for local evaluation.
- Download with `scripts/download_rio10.sh` (edit the scene list first).
