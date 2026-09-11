import os
import numpy as np
from plyfile import PlyData, PlyElement


model = "SpUNet"

pred_dir = f"exp/custom_trees/{model}-scratch-combined/result" 

test_dir = "data/BRANCH_v2/numpy_dataset/combined/val"

output_ply_dir = f"data/predictions/{model}-scratch-combined"

os.makedirs(output_ply_dir, exist_ok=True)

number_of_trees = len(os.listdir(test_dir))
trees_processed = 0
print(f"[INFO] Number of trees: {number_of_trees}")
for tree_folder in os.listdir(test_dir):
    print(f"[INFO] Processing: {tree_folder}, processed: {trees_processed}/{number_of_trees}")
    trees_processed += 1

    coord = np.load(os.path.join(test_dir, tree_folder, "coord.npy"))
    color = np.load(os.path.join(test_dir, tree_folder, "color.npy"))
    
    pred_path = os.path.join(pred_dir, f"{tree_folder}_pred.npy") 
    
    if not os.path.exists(pred_path):
        print(f"[WARN] No pred for: {tree_folder}")
        continue
        
    predictions = np.load(pred_path)
    
    print(f"[INFO] Predictions for {tree_folder}: {predictions.shape[0]}, np.unique: {np.unique(predictions)}")
    vertices = []
    for i in range(coord.shape[0]):
        vertices.append((
            coord[i, 0], coord[i, 1], coord[i, 2],
            int(color[i, 0]), int(color[i, 1]), int(color[i, 2]),
            predictions[i]  
        ))
        
    vertex_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        (f'{model}_pred', 'i4') 
    ]
    
    vertex_array = np.array(vertices, dtype=vertex_dtype)
    el = PlyElement.describe(vertex_array, 'vertex')
    
    output_path = os.path.join(output_ply_dir, f"{tree_folder}_pred.ply")
    PlyData([el], text=True).write(output_path)

print(f"[INFO] Done! Output dir: {output_ply_dir}.")