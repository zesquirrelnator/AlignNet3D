import open3d as o3d
from alignment_loss import CustomAlignmentLoss
import torch
import numpy as np
import os
from alignment_utils import create_training_pairs


def load_point_cloud(file_path):
    if file_path.endswith('.obj'):
        mesh = o3d.io.read_triangle_mesh(file_path)
        pcd = mesh.sample_points_poisson_disk(number_of_points=20000)
    else:
        pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def create_origin_sphere():
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1)
    mesh_sphere.paint_uniform_color([0, 0, 1])  # Blue for origin
    return mesh_sphere

def visualize_point_clouds(pcd1, pcd2, title=""):
    origin_sphere = create_origin_sphere()

    # Set colors for both point clouds
    pcd1.paint_uniform_color([1, 0, 0])  # Red
    pcd2.paint_uniform_color([0, 1, 0])  # Green

    # Visualize
    o3d.visualization.draw_geometries([pcd1, pcd2, origin_sphere], window_name=title, width=800, height=600)

def main():
    data_folder = '../data/loss_test/'
    source_file = 'true_bed.pcd'
    target_file = 'true_model.obj'
    alignment_loss_module = CustomAlignmentLoss()

    source_path = os.path.join(data_folder, source_file)
    target_path = os.path.join(data_folder, target_file)

    source_pcd = load_point_cloud(source_path)
    target_pcd = load_point_cloud(target_path)

    # Compute loss without any random change
    no_change_tensor = torch.tensor(np.concatenate([np.array([1]), np.zeros(3), np.ones(3), np.zeros(3)])).float().reshape(1, 10)
    alignment_loss_no_change = alignment_loss_module(no_change_tensor, no_change_tensor)
    print("Loss without random change:", alignment_loss_no_change.item())

    visualize_point_clouds(source_pcd, target_pcd, title="Original Pair")

    # Compute loss after random changes
    training_pairs = create_training_pairs(source_pcd, target_pcd, num_pairs=1)
    for j, (transformed_points, target_points, transformation_params) in enumerate(training_pairs):
        transformed_pcd = o3d.geometry.PointCloud()
        transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)

        # Prepare tensors for loss computation
        source_tensor = torch.tensor(transformation_params).float().reshape(1, 10) # Updated for quaternion
        target_tensor = torch.tensor(np.concatenate([np.array([1]), np.zeros(3), np.ones(3), np.zeros(3)])).float().reshape(1, 10)

        # Compute alignment loss
        alignment_loss = alignment_loss_module(source_tensor, target_tensor)
        print("Loss after random change:", alignment_loss.item())

        visualize_point_clouds(transformed_pcd, target_pcd, title=f"Transformation {j + 1}")

if __name__ == "__main__":
    main()
