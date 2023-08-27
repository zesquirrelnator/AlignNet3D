import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import open3d as o3d
from model import PointNetLKAlignment
from data_preprocessing import load_and_preprocess_all_pairs
from alignment_loss import CustomAlignmentLoss
from scipy.spatial.transform import Rotation as R
from alignment_utils import apply_transformations


class CustomTransformationOptimizer:
    def __init__(self, model, lr=0.001):
        self.translation_params = list(model.output_layer_translation.parameters())
        self.rotation_params = list(model.output_layer_rotation.parameters())
        self.scaling_params = list(model.output_layer_scaling.parameters())

        self.translation_opt = torch.optim.Adam(self.translation_params, lr=lr, weight_decay=1e-5)
        self.rotation_opt = torch.optim.Adam(self.rotation_params, lr=lr, weight_decay=1e-5)
        self.scaling_opt = torch.optim.Adam(self.scaling_params, lr=lr, weight_decay=1e-5)

    def step(self):
        # Step each optimizer
        self.translation_opt.step()
        self.rotation_opt.step()
        self.scaling_opt.step()

        # Apply constraints to rotation (wrap within [0, 360])
        for p in self.rotation_params:
            p.data = p.data % 360

    def zero_grad(self):
        self.translation_opt.zero_grad()
        self.rotation_opt.zero_grad()
        self.scaling_opt.zero_grad()


# Here, we change the paths to the directory rather than specific file paths
directory_path = '../data'

# Load and preprocess the data
X, y = load_and_preprocess_all_pairs(directory_path, num_pairs_per_pair=1000)

# Convert the tensors to NumPy arrays
y_np = y.numpy()

# Separate source and target point clouds directly from the PyTorch tensor
X_source = X[:, 0]
X_target = X[:, 1]

y = torch.tensor(y_np, dtype=torch.float32)

# Create DataLoader
dataset = TensorDataset(X_source, X_target, y)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=5)

model = PointNetLKAlignment()    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
print(torch.version.cuda)
model.to(device)

# Loss and optimizer
loss_function = CustomAlignmentLoss()
optimizer = CustomTransformationOptimizer(model, lr=0.001)

# TensorBoard writer
writer = SummaryWriter()

# Training loop
best_loss = float('inf')  # Best loss so far
epochs_without_improvement = 0  # Counter for epochs without improvement
max_epochs_without_improvement = 10  # Threshold for early stopping

# Helper function to visualize alignment
def visualize_alignment(source, target, title=""):
    source_cloud = o3d.geometry.PointCloud()
    target_cloud = o3d.geometry.PointCloud()

    source_cloud.points = o3d.utility.Vector3dVector(source)
    target_cloud.points = o3d.utility.Vector3dVector(target)

    # Set colors for both point clouds
    source_cloud.paint_uniform_color([1, 0, 0])  # Source in red
    target_cloud.paint_uniform_color([0, 1, 0])  # Target in green

    # Visualize
    o3d.visualization.draw_geometries([source_cloud, target_cloud], window_name=title, width=800, height=600)


for epoch in range(10000):  # Number of epochs
    epoch_loss = 0
    for batch_X_source, batch_X_target, batch_y in dataloader:
        batch_X_source = batch_X_source.transpose(1, 2)
        batch_X_target = batch_X_target.transpose(1, 2)
        batch_X_source, batch_X_target, batch_y = batch_X_source.to(device), batch_X_target.to(device), batch_y.to(device)

        # Forward pass (without autocast)
        transformations = model(batch_X_source, batch_X_target)

        # Apply transformations to source and target point clouds
        transformed_source = apply_transformations(batch_X_source, transformations)

        # Compute loss using the transformed point clouds and ground truth transformations
        loss = loss_function(transformed_source, batch_X_target)

        # Backward pass
        loss.backward()

        # Update the optimizer
        optimizer.step()

        # Clear gradients
        optimizer.zero_grad()

        epoch_loss += loss.item()

    # Average the loss for this epoch
    epoch_loss /= len(dataloader)

    # Log loss to TensorBoard
    writer.add_scalar('Training loss', epoch_loss, epoch)

    print(f'Epoch {epoch + 1}, Loss: {epoch_loss}')

    # Check for improvement
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    # If no improvement for the defined number of epochs, stop training
    if epochs_without_improvement >= max_epochs_without_improvement:
        print(f'No improvement in loss for {max_epochs_without_improvement} epochs, stopping training.')
        break

# Save the model
torch.save(model.state_dict(), 'alignment_model.pth')

# Close TensorBoard writer
writer.close()

print('Training completed. Model saved.')