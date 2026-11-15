"""Step 2: Export extracted Hydra scene graphs into SG2Loc's 3RScan directory structure."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from scipy.spatial import ConvexHull
from torchvision import transforms as tvf
from tqdm import tqdm

from vamr_loc.datasets.rio10 import write_3rscan_info_txt

# 3DSSG spatial relationship vocabulary IDs (41-dim, 0 = none)
REL_NONE = 0
REL_LEFT = 2
REL_RIGHT = 3
REL_FRONT = 4
REL_BEHIND = 5
REL_CLOSE_BY = 6
REL_ABOVE = 27
REL_BELOW = 28


def _resolve_path(repo_root: Path, p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (repo_root / path).resolve()


def _import_sg2loc(sg2loc_repo: Path):
    if not sg2loc_repo.exists():
        raise FileNotFoundError(
            f"SG2Loc repository not found at {sg2loc_repo}. "
            "Symlink or clone SG2Loc into third_party/sg2loc."
        )
    if str(sg2loc_repo) not in sys.path:
        sys.path.insert(0, str(sg2loc_repo))

    from sg2loc.preprocessing.bvh_build import build_bvh, flatten_bvh
    from sg2loc.preprocessing.dinov2_utils import DinoV2ExtractFeatures
    from sg2loc.scan3r import utils as scan3r
    from sg2loc.utils import common, point_cloud

    return build_bvh, flatten_bvh, DinoV2ExtractFeatures, scan3r, common, point_cloud


def link_rio10_sequences(
    raw_rio10_dir: Path,
    scenes_dir: Path,
    map_seq: str,
    query_seqs: list[str],
) -> None:
    """Symlink RIO10 mapping/query folders into <export_dir>/scenes/ and generate _info.txt."""
    scenes_dir.mkdir(parents=True, exist_ok=True)
    for seq in [map_seq, *query_seqs]:
        candidates = [
            raw_rio10_dir / "mapping" / seq,
            raw_rio10_dir / "query" / seq,
            raw_rio10_dir / seq,
        ]
        src_seq = next((c for c in candidates if c.exists()), None)
        if src_seq is None:
            raise FileNotFoundError(f"Could not find RIO10 sequence '{seq}' under {raw_rio10_dir}")

        dst_scene = scenes_dir / seq
        dst_scene.mkdir(parents=True, exist_ok=True)
        dst_seq = dst_scene / "sequence"
        if not dst_seq.exists():
            os.symlink(str(src_seq.resolve()), str(dst_seq))
            print(f"[SG2LocExport] Linked {dst_seq} -> {src_seq}")

        write_3rscan_info_txt(dst_seq)


def build_bvh_and_scene_cache(
    vertices: np.ndarray,
    triangles: np.ndarray,
    vertex_colors: np.ndarray,
    vertex_obj_ids: np.ndarray,
    nyu40_per_vert: np.ndarray,
    scenes_dir: Path,
    files_dir: Path,
    map_seq: str,
    query_seqs: list[str],
    bvh_dir_name: str,
    max_tris_per_leaf: int,
    build_bvh_fn,
    flatten_bvh_fn,
) -> None:
    """Write scenes/<seq>/{data.npy,pcl_center.npy} and files/<bvh_dir_name>/<map_seq>/*.npy."""
    structured = np.empty(
        len(vertices),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("objectId", "<i2"), ("globalId", "<i2")],
    )
    structured["x"] = vertices[:, 0].astype(np.float32)
    structured["y"] = vertices[:, 1].astype(np.float32)
    structured["z"] = vertices[:, 2].astype(np.float32)
    structured["objectId"] = vertex_obj_ids.astype(np.int16)
    structured["globalId"] = nyu40_per_vert.astype(np.int16)
    pcl_center = np.mean(vertices, axis=0).astype(np.float64)

    for seq in [map_seq, *query_seqs]:
        scene_dir = scenes_dir / seq
        scene_dir.mkdir(parents=True, exist_ok=True)
        np.save(scene_dir / "data.npy", structured)
        np.save(scene_dir / "pcl_center.npy", pcl_center)

    bvh_dir = files_dir / bvh_dir_name / map_seq
    bvh_dir.mkdir(parents=True, exist_ok=True)

    print("[SG2LocExport] Building BVH raycasting tree over Hydra mesh...")
    root = build_bvh_fn(
        triangles.astype(np.int32),
        vertices.astype(np.float64),
        max_triangles_per_leaf=max_tris_per_leaf,
    )
    nodes, leaf_triangles, triangle_orig_indices = [], [], []
    flatten_bvh_fn(root, nodes, leaf_triangles, triangle_orig_indices)

    num_tris = len(triangles)
    uv_coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    uv_indices = np.tile(np.array([0, 1, 2], dtype=np.int64), (num_tris, 1))

    np.save(bvh_dir / "vertices.npy", vertices.astype(np.float64))
    np.save(bvh_dir / "bvh_nodes.npy", np.asarray(nodes, dtype=np.float64))
    np.save(bvh_dir / "bvh_triangles.npy", np.asarray(leaf_triangles, dtype=np.int32))
    np.save(bvh_dir / "obj_ids.npy", vertex_obj_ids.astype(np.int32))
    np.save(bvh_dir / "uv_coords.npy", uv_coords)
    np.save(bvh_dir / "uv_indices.npy", uv_indices)
    np.save(bvh_dir / "triangle_orig_indices.npy", np.asarray(triangle_orig_indices, dtype=np.int64))
    np.save(bvh_dir / "vertex_colors.npy", vertex_colors.astype(np.uint8))
    print(f"[SG2LocExport] BVH built: {len(nodes)} nodes, {len(leaf_triangles)} tris -> {bvh_dir}")


def build_scene_graph_pkl(
    vertices: np.ndarray,
    vertex_obj_ids: np.ndarray,
    objects_meta: list[dict],
    files_dir: Path,
    map_seq: str,
    query_seqs: list[str],
    sg_cfg: dict,
    ade20k_to_nyu40: dict[str, int],
    common_mod,
    point_cloud_mod,
) -> tuple[list[int], dict[int, int]]:
    """Build files/orig/data/<scan_id>.pkl matching SceneGraphLoc's tensor schema."""
    pc_resolutions = list(sg_cfg["pc_resolutions"])
    default_nyu40 = int(sg_cfg["default_nyu40_id"])
    min_pts = int(sg_cfg.get("min_obj_vertices", 20))

    object_points: dict[int, list] = {r: [] for r in pc_resolutions}
    objects_ids: list[int] = []
    global_objects_ids: list[int] = []
    bary_centers: list[list[float]] = []
    obj_id_to_nyu40: dict[int, int] = {}

    for obj in objects_meta:
        oid = int(obj["sg_obj_id"])
        pts = vertices[vertex_obj_ids == oid]
        if len(pts) < min_pts:
            continue
        nyu40_id = int(ade20k_to_nyu40.get(obj["ade_name"].lower(), default_nyu40))
        try:
            hull = ConvexHull(pts)
            center = np.mean(hull.points[hull.vertices], axis=0)
        except Exception:
            center = np.mean(pts, axis=0)

        for r in pc_resolutions:
            object_points[r].append(point_cloud_mod.pcl_farthest_sample(pts.copy(), r))

        bary_centers.append(center.tolist())
        objects_ids.append(oid)
        global_objects_ids.append(nyu40_id)
        obj_id_to_nyu40[oid] = nyu40_id

    for r in pc_resolutions:
        object_points[r] = np.asarray(object_points[r], dtype=np.float64)

    n_objs = len(objects_ids)
    if n_objs < 2:
        raise ValueError(f"Need at least 2 objects for SG2Loc scene graph, found {n_objs}")

    object_id2idx = {oid: idx for idx, oid in enumerate(objects_ids)}
    centers_arr = np.asarray(bary_centers, dtype=np.float64)

    v_thresh = float(sg_cfg["vertical_thresh_m"])
    c_thresh = float(sg_cfg["cardinal_max_dist_m"])
    close_thresh = float(sg_cfg["close_by_max_dist_m"])

    triples, pairs, edges_cat = [], [], []
    outgoing_deg = np.zeros(n_objs, dtype=np.int32)

    for i in range(n_objs):
        for j in range(n_objs):
            if i == j:
                continue
            diff = centers_arr[i] - centers_arr[j]
            dist = float(np.linalg.norm(diff))
            if abs(diff[2]) > v_thresh and abs(diff[2]) > max(abs(diff[0]), abs(diff[1])):
                rel_id = REL_ABOVE if diff[2] > 0 else REL_BELOW
            elif abs(diff[0]) >= abs(diff[1]) and dist < c_thresh:
                rel_id = REL_RIGHT if diff[0] > 0 else REL_LEFT
            elif dist < c_thresh:
                rel_id = REL_FRONT if diff[1] > 0 else REL_BEHIND
            elif dist < close_thresh:
                rel_id = REL_CLOSE_BY
            else:
                rel_id = REL_NONE

            triples.append([objects_ids[i], objects_ids[j], rel_id])
            pairs.append([objects_ids[i], objects_ids[j]])
            edges_cat.append(rel_id)
            if rel_id != REL_NONE:
                outgoing_deg[i] += 1

    root_obj_idx = int(np.argmax(outgoing_deg))
    root_obj_id = np.int64(objects_ids[root_obj_idx])
    rel_trans = centers_arr[root_obj_idx] - centers_arr
    edges = np.array([[object_id2idx[p[0]], object_id2idx[p[1]]] for p in pairs], dtype=np.int64)

    bow_edge = np.zeros((n_objs, 41), dtype=np.float64)
    for (s_idx, _), (_, _, rel_id) in zip(edges, triples):
        if 0 <= rel_id < 41:
            bow_edge[s_idx, rel_id] += 1.0

    bow_attr = np.zeros((n_objs, 164), dtype=np.float64)
    for idx, nyu_id in enumerate(global_objects_ids):
        if 1 <= nyu_id <= 40:
            bow_attr[idx, nyu_id] = 1.0

    out_dir = files_dir / "orig/data"
    out_dir.mkdir(parents=True, exist_ok=True)
    for seq in [map_seq, *query_seqs]:
        sg_dict = {
            "scan_id": seq,
            "objects_id": np.asarray(objects_ids, dtype=np.int64),
            "global_objects_id": np.asarray(global_objects_ids, dtype=np.int64),
            "objects_cat": np.asarray(global_objects_ids, dtype=np.int64),
            "triples": triples,
            "pairs": pairs,
            "edges": edges,
            "obj_points": object_points,
            "objects_count": n_objs,
            "edges_count": len(edges),
            "object_id2idx": object_id2idx,
            "object_attributes": [[] for _ in range(n_objs)],
            "edges_cat": edges_cat,
            "rel_trans": rel_trans,
            "root_obj_id": root_obj_id,
            "bow_vec_object_attr_feats": bow_attr,
            "bow_vec_object_edge_feats": bow_edge,
        }
        common_mod.write_pkl_data(sg_dict, str(out_dir / f"{seq}.pkl"))

    print(f"[SG2LocExport] Wrote 3D scene graph ({n_objs} objects, {len(edges)} edges) -> {out_dir}")
    return objects_ids, obj_id_to_nyu40


def generate_object_visual_embeddings(
    vertices: np.ndarray,
    vertex_obj_ids: np.ndarray,
    objects_ids: list[int],
    scenes_dir: Path,
    files_dir: Path,
    map_seq: str,
    query_seqs: list[str],
    vis_cfg: dict,
    DinoV2ExtractFeaturesCls,
    scan3r_mod,
    common_mod,
) -> None:
    """Project 3D object vertices into reference frames and extract multi-level DINOv2 embeddings."""
    out_dir = files_dir / vis_cfg["dir_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pkl = out_dir / f"{map_seq}.pkl"

    intr = scan3r_mod.load_intrinsics(str(scenes_dir), map_seq, type="color")
    K = intr["intrinsic_mat"]
    W, H = int(intr["width"]), int(intr["height"])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    step = int(vis_cfg["frame_subsample_step"])
    topk = int(vis_cfg["topk_views"])
    crop_levels = int(vis_cfg["crop_levels"])
    exp_ratio = float(vis_cfg["crop_expansion_ratio"])
    rotate_270 = bool(vis_cfg["rotate_270"])

    frame_idxs = scan3r_mod.load_frame_idxs(str(scenes_dir), map_seq, skip=step)
    poses_wc = scan3r_mod.load_frame_poses(str(scenes_dir), map_seq, frame_idxs, type="matrix")

    obj_pts_map = {oid: vertices[vertex_obj_ids == oid] for oid in objects_ids}
    obj_votes: dict[int, dict[str, int]] = {oid: {} for oid in objects_ids}
    obj_boxes: dict[int, dict[str, tuple[int, int, int, int]]] = {oid: {} for oid in objects_ids}

    for fidx in frame_idxs:
        T_wc = poses_wc[fidx]
        if not np.isfinite(T_wc).all():
            continue
        T_cw = np.linalg.inv(T_wc)
        R_cw, t_cw = T_cw[:3, :3], T_cw[:3, 3]

        for oid, pts_w in obj_pts_map.items():
            pts_c = pts_w @ R_cw.T + t_cw
            valid = pts_c[:, 2] > 0.2
            if valid.sum() < 10:
                continue
            pts_c = pts_c[valid]
            u = fx * (pts_c[:, 0] / pts_c[:, 2]) + cx
            v = fy * (pts_c[:, 1] / pts_c[:, 2]) + cy
            in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            vis_count = int(in_img.sum())
            if vis_count < 10:
                continue
            u_vis, v_vis = u[in_img], v[in_img]
            x1, y1 = int(np.floor(u_vis.min())), int(np.floor(v_vis.min()))
            x2, y2 = int(np.ceil(u_vis.max())) + 1, int(np.ceil(v_vis.max())) + 1
            if (x2 - x1) < 8 or (y2 - y1) < 8:
                continue
            obj_votes[oid][fidx] = vis_count
            obj_boxes[oid][fidx] = (max(0, x1), max(0, y1), min(W, x2), min(H, y2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SG2LocExport] Extracting multi-level {vis_cfg['model_name']} embeddings on {device}...")
    extractor = DinoV2ExtractFeaturesCls(
        vis_cfg["model_name"],
        int(vis_cfg["desc_layer"]),
        vis_cfg["desc_facet"],
        use_cls=True,
        device=device,
    )
    base_tf = tvf.Compose(
        [tvf.ToTensor(), tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    obj_image_votes_topK: dict[int, list[str]] = {}
    obj_visual_emb: dict[int, dict[str, np.ndarray]] = {}

    for oid in tqdm(objects_ids, desc="Object DINOv2"):
        votes = obj_votes[oid]
        top_frames = sorted(votes, key=votes.get, reverse=True)[:topk]
        if not top_frames:
            top_frames = [frame_idxs[0]]
            obj_boxes[oid][frame_idxs[0]] = (0, 0, W, H)

        obj_image_votes_topK[oid] = top_frames
        obj_visual_emb[oid] = {}

        for fidx in top_frames:
            img_path = scenes_dir / map_seq / "sequence" / f"frame-{fidx}.color.jpg"
            img = Image.open(img_path).convert("RGB")
            x1, y1, x2, y2 = obj_boxes[oid][fidx]

            if rotate_270:
                img_proc = img.transpose(Image.ROTATE_270)
                bx1, by1, bx2, by2 = H - y2, x1, H - y1, x2
            else:
                img_proc = img
                bx1, by1, bx2, by2 = x1, y1, x2, y2
            pw, ph = img_proc.size

            crops = []
            for level in range(crop_levels):
                ex = int(abs(bx2 - bx1) * exp_ratio) * level
                ey = int(abs(by2 - by1) * exp_ratio) * level
                cx1, cy1 = max(0, bx1 - ex), max(0, by1 - ey)
                cx2, cy2 = min(pw, bx2 + ex), min(ph, by2 + ey)
                if cx2 <= cx1 or cy2 <= cy1:
                    cx1, cy1, cx2, cy2 = 0, 0, pw, ph
                crop = img_proc.crop((cx1, cy1, cx2, cy2)).resize((224, 224), Image.BICUBIC)
                crops.append(base_tf(crop))

            batch = torch.stack(crops, dim=0).to(device)
            with torch.no_grad():
                ret = extractor(batch)
                mean_emb = ret[:, 0, :].mean(dim=0).cpu().numpy().astype(np.float32)
            obj_visual_emb[oid][fidx] = mean_emb

    payload = {"obj_visual_emb": obj_visual_emb, "obj_image_votes_topK": obj_image_votes_topK}
    common_mod.write_pkl_data(payload, str(out_pkl))
    for q_seq in query_seqs:
        common_mod.write_pkl_data(payload, str(out_dir / f"{q_seq}.pkl"))
    print(f"[SG2LocExport] Saved DINOv2 object embeddings -> {out_pkl}")


def register_in_metadata(
    files_dir: Path,
    map_seq: str,
    query_seqs: list[str],
    obj_id_to_nyu40: dict[int, int],
    common_mod,
) -> None:
    """Register map and query scans in files/3RScan.json and files/objects.json."""
    rscan_path = files_dir / "3RScan.json"
    objs_path = files_dir / "objects.json"
    if not rscan_path.exists() or not objs_path.exists():
        raise FileNotFoundError(
            f"Missing 3RScan.json or objects.json in {files_dir}. "
            "Unpack SG2Loc's provided dataset metadata into <export_dir>/files/ first."
        )

    for p in (rscan_path, objs_path):
        bak = p.with_suffix(p.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(p, bak)

    rscan_data = common_mod.load_json(str(rscan_path))
    rscan_data = [e for e in rscan_data if e.get("reference") != map_seq]
    rscan_data.append(
        {
            "reference": map_seq,
            "scans": [
                {"reference": q_seq, "transform": np.eye(4).flatten().tolist()}
                for q_seq in query_seqs
            ],
        }
    )
    with rscan_path.open("w") as f:
        json.dump(rscan_data, f, indent=2)

    objs_data = common_mod.load_json(str(objs_path))
    all_seqs = {map_seq, *query_seqs}
    objs_data["scans"] = [s for s in objs_data["scans"] if s.get("scan") not in all_seqs]
    scan_objs_list = [
        {"id": str(oid), "nyu40": str(nyu40)} for oid, nyu40 in sorted(obj_id_to_nyu40.items())
    ]
    for seq in [map_seq, *query_seqs]:
        objs_data["scans"].append({"scan": seq, "objects": scan_objs_list})
    with objs_path.open("w") as f:
        json.dump(objs_data, f, indent=2)

    print(f"[SG2LocExport] Registered {map_seq} & {query_seqs} in 3RScan.json and objects.json")


def export_hydra_to_sg2loc(
    config_path: str | Path,
    hydra_npz_path: str | Path,
    map_seq: str = "seq01_01",
    query_seqs: list[str] | None = None,
    repo_root: str | Path | None = None,
) -> None:
    """Run the full Hydra-to-SG2Loc conversion pipeline."""
    query_seqs = query_seqs or ["seq01_02"]
    config_path = Path(config_path).resolve()
    root = Path(repo_root).resolve() if repo_root else config_path.parents[2]

    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)

    paths_cfg = cfg["paths"]
    raw_rio10_dir = _resolve_path(root, paths_cfg["raw_rio10_dir"])
    export_dir = _resolve_path(root, paths_cfg["export_dir"])
    sg2loc_repo = _resolve_path(root, paths_cfg["sg2loc_repo"])
    scenes_dir = export_dir / "scenes"
    files_dir = export_dir / "files"

    build_bvh_fn, flatten_bvh_fn, DinoV2Cls, scan3r_mod, common_mod, pc_mod = _import_sg2loc(sg2loc_repo)

    data = np.load(hydra_npz_path, allow_pickle=True)
    vertices = data["vertices"]
    triangles = data["triangles"]
    vertex_colors = data["vertex_colors"]
    vertex_obj_ids = data["vertex_obj_ids"]
    objects_meta = json.loads(str(data["objects_json"]))

    ade_map = {k.lower(): int(v) for k, v in cfg["ade20k_to_nyu40"].items()}
    default_nyu = int(cfg["scene_graph"]["default_nyu40_id"])
    nyu40_per_vert = np.zeros(len(vertices), dtype=np.int16)
    nyu40_per_vert[vertex_obj_ids == 1] = 2  # floor
    for obj in objects_meta:
        oid = int(obj["sg_obj_id"])
        nyu40_per_vert[vertex_obj_ids == oid] = ade_map.get(obj["ade_name"].lower(), default_nyu)

    link_rio10_sequences(raw_rio10_dir, scenes_dir, map_seq, query_seqs)
    build_bvh_and_scene_cache(
        vertices,
        triangles,
        vertex_colors,
        vertex_obj_ids,
        nyu40_per_vert,
        scenes_dir,
        files_dir,
        map_seq,
        query_seqs,
        cfg["bvh"]["dir_name"],
        int(cfg["bvh"]["max_triangles_per_leaf"]),
        build_bvh_fn,
        flatten_bvh_fn,
    )
    sg_cfg = {**cfg["scene_graph"], "min_obj_vertices": cfg["extraction"]["min_obj_vertices"]}
    objects_ids, obj_id_to_nyu40 = build_scene_graph_pkl(
        vertices,
        vertex_obj_ids,
        objects_meta,
        files_dir,
        map_seq,
        query_seqs,
        sg_cfg,
        ade_map,
        common_mod,
        pc_mod,
    )
    generate_object_visual_embeddings(
        vertices,
        vertex_obj_ids,
        objects_ids,
        scenes_dir,
        files_dir,
        map_seq,
        query_seqs,
        cfg["visual_embeddings"],
        DinoV2Cls,
        scan3r_mod,
        common_mod,
    )
    register_in_metadata(files_dir, map_seq, query_seqs, obj_id_to_nyu40, common_mod)
    print(f"\n[WP1 Complete] Exported {map_seq} -> {export_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Hydra .npz bundle to SG2Loc format")
    parser.add_argument("--config", default="configs/export/sg2loc.yaml", type=Path)
    parser.add_argument("--hydra-npz", required=True, type=Path)
    parser.add_argument("--map-seq", default="seq01_01", type=str)
    parser.add_argument("--query-seqs", nargs="+", default=["seq01_02"])
    args = parser.parse_args()

    export_hydra_to_sg2loc(
        config_path=args.config,
        hydra_npz_path=args.hydra_npz,
        map_seq=args.map_seq,
        query_seqs=args.query_seqs,
    )


if __name__ == "__main__":
    main()
