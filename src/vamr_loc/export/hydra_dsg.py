"""Step 1: Extract Hydra DynamicSceneGraph & mesh into a portable .npz bundle.

Runs inside the Hydra Python environment (where `spark_dsg` is installed) so the
downstream SG2Loc exporter does not depend on ROS or Spark-DSG bindings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml


def extract_hydra_scene_graph(
    dsg_file: str | Path,
    mesh_file: str | Path,
    label_space_file: str | Path,
    out_npz: str | Path,
    floor_ade20k_label: int = 3,
    min_obj_vertices: int = 20,
) -> Path:
    """Export Hydra mesh geometry, floor mask, and object instances to `.npz`."""
    import spark_dsg as dsg  # Imported lazily so other modules don't require spark_dsg

    dsg_file = Path(dsg_file)
    mesh_file = Path(mesh_file)
    label_space_file = Path(label_space_file)
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    with label_space_file.open("r") as f:
        label_cfg = yaml.safe_load(f)
    id_to_name = {item["label"]: item["name"] for item in label_cfg.get("label_names", [])}

    G = dsg.DynamicSceneGraph.load(str(dsg_file))
    mesh = o3d.io.read_triangle_mesh(str(mesh_file))

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    vertex_colors = (np.asarray(mesh.vertex_colors) * 255.0).clip(0, 255).astype(np.uint8)
    sem_labels = np.asarray(G.mesh.get_labels(), dtype=np.int32)

    if len(vertices) != len(sem_labels):
        raise ValueError(f"Vertex count mismatch: mesh has {len(vertices)}, DSG has {len(sem_labels)}")

    # SG2Loc convention: 0 = undefined/background, 1 = floor, >=2 = object instances
    vertex_obj_ids = np.zeros(len(vertices), dtype=np.int32)
    vertex_obj_ids[sem_labels == floor_ade20k_label] = 1

    objects_meta = []
    for node in sorted(G.get_layer(dsg.DsgLayers.OBJECTS).nodes, key=lambda n: n.id.category_id):
        attrs = node.attributes
        raw_id = int(node.id.category_id)
        sg_obj_id = raw_id + 2
        ade_label = int(attrs.semantic_label)
        ade_name = id_to_name.get(ade_label, "otherprop")

        mesh_idxs = np.asarray(attrs.mesh_connections, dtype=np.int64)
        mesh_idxs = mesh_idxs[(mesh_idxs >= 0) & (mesh_idxs < len(vertices))]
        if len(mesh_idxs) < min_obj_vertices:
            continue

        vertex_obj_ids[mesh_idxs] = sg_obj_id
        objects_meta.append(
            {
                "sg_obj_id": sg_obj_id,
                "hydra_raw_id": raw_id,
                "ade_label": ade_label,
                "ade_name": ade_name,
                "position": attrs.position.tolist(),
                "bbox_center": attrs.bounding_box.world_P_center.tolist(),
                "bbox_dims": attrs.bounding_box.dimensions.tolist(),
                "bbox_R": attrs.bounding_box.world_R_center.tolist(),
                "num_verts": int(len(mesh_idxs)),
            }
        )

    np.savez(
        out_npz,
        vertices=vertices,
        triangles=triangles,
        vertex_colors=vertex_colors,
        vertex_obj_ids=vertex_obj_ids,
        sem_labels=sem_labels,
        objects_json=json.dumps(objects_meta),
    )
    print(
        f"[HydraExtract] Saved {out_npz}: {len(vertices)} verts, {len(triangles)} tris, "
        f"{int((vertex_obj_ids == 1).sum())} floor verts, {len(objects_meta)} objects."
    )
    return out_npz


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Hydra DSG + mesh into portable .npz")
    parser.add_argument("--dsg", required=True, type=Path, help="Path to dsg_with_mesh.json")
    parser.add_argument("--mesh", required=True, type=Path, help="Path to mesh.ply")
    parser.add_argument("--label-space", required=True, type=Path, help="ADE20k label_space.yaml")
    parser.add_argument("--out", required=True, type=Path, help="Output .npz path")
    parser.add_argument("--floor-label", type=int, default=3, help="ADE20k floor label ID")
    parser.add_argument("--min-verts", type=int, default=20, help="Minimum object vertex count")
    args = parser.parse_args()

    extract_hydra_scene_graph(
        dsg_file=args.dsg,
        mesh_file=args.mesh,
        label_space_file=args.label_space,
        out_npz=args.out,
        floor_ade20k_label=args.floor_label,
        min_obj_vertices=args.min_verts,
    )


if __name__ == "__main__":
    main()
