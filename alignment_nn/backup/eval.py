import torch
import numpy as np
from model import AlignmentModel
from data_preprocessing import load_and_preprocess_data
from alignment_loss import CustomAlignmentLoss
from torch.utils.data import TensorDataset, DataLoader
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R

def visualize_alignment(source_points, target_points, title):
    source_points = source_points.T  # Transpose the array to shape (N, 3)
    target_points = target_points.T  # Transpose the array to shape (N, 3)

    # Now create the point clouds
    source_cloud = o3d.geometry.PointCloud()
    target_cloud = o3d.geometry.PointCloud()

    source_cloud.points = o3d.utility.Vector3dVector(source_points)
    target_cloud.points = o3d.utility.Vector3dVector(target_points)

    # You can set the colors if needed
    source_cloud.paint_uniform_color([1, 0, 0])  # Red for source
    target_cloud.paint_uniform_color([0, 0, 1])  # Blue for target

    # Visualize the point clouds
    o3d.visualization.draw_geometries([source_cloud, target_cloud], window_name=title, width=800, height=600)


# Set the paths for the source and target data
source_pcd_path = '../data/test3/source.pcd'
target_obj_path = '../data/test3/target.obj'

# Load and preprocess the data
X, y = load_and_preprocess_data(source_pcd_path, target_obj_path, num_pairs=10)

# Assuming that X is a numpy array with shape (283738, 3, 3)
# and y is also a numpy array with shape (283738, 3)
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Create DataLoader
dataset = TensorDataset(X_tensor[:, 0], X_tensor[:, 1], y_tensor)
dataloader = DataLoader(dataset, batch_size=20, shuffle=False, num_workers=5)

# Load the trained model
model_path = 'alignment_model.pth'
model = AlignmentModel()
model.load_state_dict(torch.load(model_path))
model.eval()

# Initialize loss function
loss_function = CustomAlignmentLoss()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
print(torch.version.cuda)
model.to(device)

def apply_and_visualize_single_transformation(source, transformations, transformation_type):
    if transformation_type == 'translation':
        translated_source = source + transformations[:, :3].unsqueeze(-1)
        visualize_alignment(translated_source.cpu().numpy()[0], batch_X_target.cpu().numpy()[0], "Translation")

    elif transformation_type == 'rotation':
        rotation_angles = transformations[:, 3:6]
        rotation_angles_cpu = rotation_angles.detach().cpu().numpy()
        rotation_matrices_np = np.array([R.from_euler('xyz', angles).as_matrix() for angles in rotation_angles_cpu])
        rotation_matrices = torch.tensor(rotation_matrices_np, device=rotation_angles.device).float()
        rotated_source = torch.bmm(source.permute(0, 2, 1), rotation_matrices)
        visualize_alignment(rotated_source.permute(0, 2, 1).cpu().numpy()[0], batch_X_target.cpu().numpy()[0], "Rotation")

    elif transformation_type == 'scaling':
        scaling_factors = transformations[:, 6:].unsqueeze(-1)
        scaled_source = source * scaling_factors
        visualize_alignment(scaled_source.cpu().numpy()[0], batch_X_target.cpu().numpy()[0], "Scaling")


def apply_transformations(source, transformations):
    # Split transformations into translation, rotation, and scaling components
    translation = transformations[:, :3]
    rotation_angles = transformations[:, 3:6]
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

# Evaluation loop
total_loss = 0.0

# Inside the evaluation loop:
for batch_X_source, batch_X_target, batch_y in dataloader:
    batch_X_source = batch_X_source.transpose(1, 2).to(device)
    batch_X_target = batch_X_target.transpose(1, 2).to(device)
    batch_y = batch_y.to(device)

    # Visualize the initial alignment (before any transformations)
    visualize_alignment(batch_X_source.cpu().numpy()[0], batch_X_target.cpu().numpy()[0], "Initial Alignment")

    with torch.no_grad():
        predictions = model(batch_X_source, batch_X_target)

        # Extract the scaling factors
        scaling_factors = predictions[:, 6:]
        
        # Print them to see what they are
        print("Scaling factors for this batch:", scaling_factors.cpu().numpy())

        # Apply and visualize each type of transformation separately
        apply_and_visualize_single_transformation(batch_X_source, predictions, 'translation')
        apply_and_visualize_single_transformation(batch_X_source, predictions, 'rotation')
        apply_and_visualize_single_transformation(batch_X_source, predictions, 'scaling')

        # calculate
        transformed_source = apply_transformations(batch_X_source, predictions)
        loss = loss_function(transformed_source, batch_X_target)
        total_loss += loss.item()

        # Print the loss for this batch
        print(f"Batch evaluation loss: {loss.item()}")

        # Visualize the alignment (if desired)
        visualize_alignment(transformed_source.cpu().numpy()[0], batch_X_target.cpu().numpy()[0], f"Evaluation loss: {loss.item()}")

# Print the average loss
average_loss = total_loss / len(dataloader)
print(f"Average evaluation loss: {average_loss}")
