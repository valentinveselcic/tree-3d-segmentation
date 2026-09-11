import os
import numpy as np
from plyfile import PlyData, PlyElement

# ── Configuration ────────────────────────────────────────────────────────────
model     = "pointnet2"
exp_name  = f"{model}_scratch-combined"          # matches args.exp_name in main.py
split     = "val"

pred_dir      = f"pointMLP/part_segmentation/checkpoints/{exp_name}/predictions/{split}" 
test_dir      = f"pointMLP/data/combined/{split}" 
output_ply_dir = f"pointMLP/data/predictions/{exp_name}"

os.makedirs(output_ply_dir, exist_ok=True)

# ── Process each scene ───────────────────────────────────────────────────────
scene_names    = sorted(os.listdir(test_dir))
n_scenes       = len(scene_names)
trees_processed = 0

print(f"[INFO] Scenes found: {n_scenes}")

for scene_name in scene_names:
    trees_processed += 1
    print(f"[INFO] Processing: {scene_name}  ({trees_processed}/{n_scenes})")

    coord_path = os.path.join(pred_dir, scene_name, "coord_processed.npy")
    color_path = os.path.join(test_dir, scene_name, "color.npy")  # won't match — skip
    pred_path  = os.path.join(pred_dir, scene_name, "pred.npy")

    if not os.path.exists(pred_path):
        print(f"[WARN] No prediction found for: {scene_name} — skipping")
        continue

    coord = np.load(coord_path).astype(np.float32)   # (N, 3)
    color = np.load(color_path).astype(np.float32)   # (N, 3) float [0,1] or [0,255]
    preds = np.load(pred_path).astype(np.int32)       # (N,)

    # Normalise color to [0, 255] uint8
    if color.max() <= 1.0:
        color = (color * 255).clip(0, 255)
    color = color.astype(np.uint8)

    print(f"  Points: {coord.shape[0]}  |  Unique labels: {np.unique(preds)}")

    vertex_dtype = [
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
    (f'{model}_pred', 'i4'),
    ]

    vertex_array = np.empty(coord.shape[0], dtype=vertex_dtype)
    vertex_array['x']             = coord[:, 0]
    vertex_array['y']             = coord[:, 1]
    vertex_array['z']             = coord[:, 2]
    vertex_array[f'{model}_pred'] = preds

    el          = PlyElement.describe(vertex_array, 'vertex')
    output_path = os.path.join(output_ply_dir, f"{scene_name}_pred.ply")
    PlyData([el], text=True).write(output_path)

print(f"[INFO] Done. PLY files written to: {output_ply_dir}")