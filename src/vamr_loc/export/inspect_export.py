"""Diagnostics and 2D/3D visualization for exported SG2Loc scene graphs."""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from vamr_loc.datasets.rio10 import load_rio10_intrinsics

REL_NAMES = {
    0: "none",
    2: "left",
    3: "right",
    4: "front",
    5: "behind",
    6: "close_by",
    27: "above",
    28: "below",
}


def inspect_sg2loc_scene(
    export_dir: str | Path,
    map_seq: str = "seq01_01",
    out_fig: str | Path | None = "outputs/exports/seq01_01_dinov2_views.png",
    show_gui: bool = False,
) -> None:
    """Print scene graph summary, save top-1 DINOv2 view contact sheet, and optionally open 3D GUI."""
    export_dir = Path(export_dir)
    bvh_dir = export_dir / "files/bvh_trees_0.5" / map_seq
    sg_pkl = export_dir / "files/orig/data" / f"{map_seq}.pkl"
    vis_pkl = export_dir / "files/Features3D/obj_dinov2_top10_l3" / f"{map_seq}.pkl"
    seq_dir = export_dir / "scenes" / map_seq / "sequence"
    classes_txt = export_dir / "files/scannet40_classes.txt"

    nyu_names: dict[int, str] = {}
    if classes_txt.is_file():
        for line in classes_txt.read_text().splitlines():
            if line.strip():
                idx, name = line.strip().split(None, 1)
                nyu_names[int(idx)] = name

    vertices = np.load(bvh_dir / "vertices.npy")
    triangles = np.load(bvh_dir / "bvh_triangles.npy")
    obj_ids = np.load(bvh_dir / "obj_ids.npy")
    with sg_pkl.open("rb") as f:
        sg = pickle.load(f)
    with vis_pkl.open("rb") as f:
        vis = pickle.load(f)

    print(f"=== SG2Loc Export Summary: {map_seq} ===")
    print(f"Mesh: {len(vertices)} vertices, {len(triangles)} triangles, {int((obj_ids == 1).sum())} floor verts")
    print(f"Objects: {sg['objects_count']} | Total edges: {sg['edges_count']} | Root ID: {sg['root_obj_id']}")

    for i, (oid, cat) in enumerate(zip(sg["objects_id"], sg["objects_cat"])):
        oid_int = int(oid)
        top_frames = vis["obj_image_votes_topK"][oid_int]
        emb = vis["obj_visual_emb"][oid_int][top_frames[0]]
        print(
            f"  Obj {oid_int:2d} | NYU40: {nyu_names.get(int(cat), str(cat)):<15s} | "
            f"Pts[512]: {sg['obj_points'][512][i].shape} | "
            f"Views: {len(top_frames):2d} (best: frame-{top_frames[0]}) | "
            f"DINOv2 norm: {np.linalg.norm(emb):.2f}"
        )

    non_none = [(s, o, REL_NAMES.get(r, str(r))) for s, o, r in sg["triples"] if r != 0]
    print(f"Non-empty spatial edges: {len(non_none)} / {len(sg['triples'])}")

    if out_fig is not None:
        out_fig = Path(out_fig)
        out_fig.parent.mkdir(parents=True, exist_ok=True)
        K, _, _ = load_rio10_intrinsics(seq_dir)
        n_objs = len(sg["objects_id"])
        cols = 5
        rows = int(np.ceil(n_objs / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4.5 * rows))
        axes = np.atleast_1d(axes).flatten()

        for idx, (oid, cat) in enumerate(zip(sg["objects_id"], sg["objects_cat"])):
            oid_int = int(oid)
            best_frame = vis["obj_image_votes_topK"][oid_int][0]
            img = Image.open(seq_dir / f"frame-{best_frame}.color.jpg")
            T_wc = np.loadtxt(seq_dir / f"frame-{best_frame}.pose.txt")
            T_cw = np.linalg.inv(T_wc)

            pts_w = vertices[obj_ids == oid_int]
            pts_c = pts_w @ T_cw[:3, :3].T + T_cw[:3, 3]
            valid = pts_c[:, 2] > 0.2
            pts_c = pts_c[valid]
            u = K[0, 0] * (pts_c[:, 0] / pts_c[:, 2]) + K[0, 2]
            v = K[1, 1] * (pts_c[:, 1] / pts_c[:, 2]) + K[1, 2]

            ax = axes[idx]
            ax.imshow(img)
            if len(u) > 0:
                x1, y1, x2, y2 = u.min(), v.min(), u.max(), v.max()
                ax.add_patch(
                    patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="lime", facecolor="none")
                )
            ax.set_title(f"Obj {oid_int}: {nyu_names.get(int(cat), str(cat))}\n(frame-{best_frame})", fontsize=10)
            ax.axis("off")

        for idx in range(n_objs, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        plt.savefig(out_fig, dpi=150)
        plt.close(fig)
        print(f"Saved 2D object view contact sheet -> {out_fig}")

    if show_gui:
        import open3d as o3d

        rng = np.random.default_rng(42)
        palette = {0: [0.85, 0.85, 0.85], 1: [0.55, 0.65, 0.75]}
        for oid in sg["objects_id"]:
            palette[int(oid)] = rng.uniform(0.15, 0.95, size=3).tolist()

        vert_colors = np.array([palette.get(int(oid), [0.8, 0.8, 0.8]) for oid in obj_ids])
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.vertex_colors = o3d.utility.Vector3dVector(vert_colors)
        mesh.compute_vertex_normals()

        geometries = [mesh]
        centroids = {}
        root_id = int(sg["root_obj_id"])
        for oid in sg["objects_id"]:
            oid_int = int(oid)
            c = vertices[obj_ids == oid_int].mean(axis=0)
            centroids[oid_int] = c
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.08 if oid_int == root_id else 0.05)
            sphere.translate(c)
            sphere.paint_uniform_color([1.0, 0.85, 0.0] if oid_int == root_id else [1.0, 0.1, 0.1])
            geometries.append(sphere)

        line_pts, line_idxs = [], []
        for s, o, r in sg["triples"]:
            if r != 0 and s in centroids and o in centroids and s < o:
                idx0 = len(line_pts)
                line_pts.extend([centroids[s], centroids[o]])
                line_idxs.append([idx0, idx0 + 1])

        if line_idxs:
            lines = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(line_pts),
                lines=o3d.utility.Vector2iVector(line_idxs),
            )
            lines.paint_uniform_color([0.1, 0.9, 0.2])
            geometries.append(lines)

        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"SG2Loc Exported BVH Map & Scene Graph ({map_seq})",
            width=1280,
            height=800,
        )
