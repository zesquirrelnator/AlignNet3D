from scipy.spatial.transform import Rotation as R
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
import open3d as o3d
import torch
import os

def random_rotation(points):
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    random_rotation = R.random()
    rotation = R.from_euler('xyz', random_rotation.as_euler('xyz'), degrees=False)
    rotated_points = rotation.apply(points_centered)
    return rotated_points + centroid, random_rotation.as_euler('xyz')

def random_scaling(points):
    scaling_factors = np.random.uniform(0.5, 2, size=3)
    scaled_points = points * scaling_factors
    return scaled_points, scaling_factors

def random_translation(points):
    translation_vector = np.random.uniform(-3, 3, size=3)
    return points + translation_vector, translation_vector  # Translation without centering

def create_training_pairs(source, target, num_pairs=100):
    training_pairs = []
    for _ in range(num_pairs):
        scaled_points, scaling_factors = random_scaling(np.asarray(source.points))
        translated_points, translation_vector = random_translation(scaled_points)
        rotated_points, rotation_angles = random_rotation(translated_points)
        transformation_params = np.concatenate([scaling_factors, translation_vector, rotation_angles]) # Updated order
        training_pairs.append((rotated_points, np.asarray(target.points), transformation_params))
    return training_pairs

def apply_transformations(source, transformations):
    # Split transformations into translation, rotation, and scaling components
    translation = transformations[:, :3]
    rotation_angles = transformations[:, 3:6]
    # Exponentiate the scaling part to get back to the normal space
    scaling = transformations[:, 6:]

    # Convert rotation angles to matrices
    rotation_angles_cpu = rotation_angles.detach().cpu().numpy()
    rotation_matrices_np = np.array([R.from_euler('xyz', angles).as_matrix() for angles in rotation_angles_cpu])
    rotation_matrices = torch.tensor(rotation_matrices_np, device=rotation_angles.device).float()

    # Step 3: Apply scaling
    scaled_source = source * scaling.unsqueeze(-1)

    # Step 4: Apply translation
    translated_source = scaled_source + translation.unsqueeze(-1)

    # Step 5: Apply rotation - Transposing to apply rotation
    rotated_source = translated_source.permute(0, 2, 1)

    # Multiply by rotation matrices
    rotated_source = torch.bmm(rotated_source, rotation_matrices)

    # Transpose back to original shape
    transformed_source = rotated_source.permute(0, 2, 1)

    return transformed_source