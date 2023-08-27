from scipy.spatial.transform import Rotation as R
import numpy as np

def random_rotation(points):
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    random_rotation = R.random()
    rotation = R.from_euler('xyz', random_rotation.as_euler('xyz'), degrees=False)
    rotated_points = rotation.apply(points_centered)
    return rotated_points + centroid, random_rotation.as_euler('xyz')

def random_scaling(points):
    scaling_factors = np.random.uniform(0.4, 1.6, size=3)
    scaled_points = points * scaling_factors
    return scaled_points, scaling_factors  # Scaling without centering

def random_translation(points):
    translation_vector = np.random.uniform(-2, 2, size=3)
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