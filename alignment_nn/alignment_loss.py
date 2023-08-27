import torch
from torch import nn
from pytorch3d.loss import chamfer_distance

class CustomAlignmentLoss(nn.Module):
    def __init__(self, transformation_weight=0.01):
        super(CustomAlignmentLoss, self).__init__()
        self.transformation_weight = transformation_weight

    def forward(self, transformed_source, transformed_target):
        # Ensure the tensors have the same data type
        transformed_source = transformed_source.to(dtype=torch.float32)
        transformed_target = transformed_target.to(dtype=torch.float32)

        # Calculate Chamfer Distance
        transformation_loss, _ = chamfer_distance(transformed_source, transformed_target)

        # Apply the weight to the transformation loss
        weighted_transformation_loss = self.transformation_weight * transformation_loss

        # Print the weighted transformation loss
        print(f"Transformation loss: {weighted_transformation_loss.item()}")

        return weighted_transformation_loss
