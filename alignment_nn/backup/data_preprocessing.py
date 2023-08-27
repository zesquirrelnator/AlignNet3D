import numpy as np
import torch
import open3d as o3d
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from alignment_utils import create_training_pairs

def load_and_preprocess_data(source_pcd_path, target_obj_path, num_pairs):
    # Function to visualize pairs
    def visualize_original_pair(source, target):
        source.paint_uniform_color([1, 0, 0]) # Paint the source red
        target.paint_uniform_color([0, 1, 0]) # Paint the target green
        o3d.visualization.draw_geometries([source, target]) # Draw both point clouds

    # Load point clouds
    source_point_cloud = o3d.io.read_point_cloud(source_pcd_path)
    target_mesh = o3d.io.read_triangle_mesh(target_obj_path)
    target_point_cloud = target_mesh.sample_points_uniformly(len(source_point_cloud.points))

    visualize_original_pair(source_point_cloud, target_point_cloud)

    ## Convert point clouds to NumPy arrays
    source_points = np.asarray(source_point_cloud.points)
    target_points = np.asarray(target_point_cloud.points)

    print("Source points shape:", source_points.shape) # should print (N, 3)
    print("Target points shape:", target_points.shape) # should print (N, 3)

    #Create training pairs
    training_pairs = create_training_pairs(target_point_cloud, source_point_cloud, num_pairs)
    
    # Separate transformed points, targets, and transformation parameters
    X_pairs = []
    y_params = []
    for pair in training_pairs:
        X_pairs.append((pair[0], pair[1]))
        y_params.append(pair[2])

    # Convert to NumPy arrays
    X_np = np.array(X_pairs)  # Shape: (num_pairs, 2, num_points, 3)
    y_np = np.array(y_params)

    # Convert to PyTorch tensors
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)

    return X, y