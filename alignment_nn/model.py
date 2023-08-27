import torch
import torch.nn as nn
import torch.nn.init as init
from alignment_utils import apply_transformations

# PointNet basic block
class PointNetBlock(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.3):
        super(PointNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, output_size, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.conv3(x)))
        return x

class PointNetLKAlignment(nn.Module):
    def __init__(self, dropout_rate=0.3, num_iterations=3):
        super(PointNetLKAlignment, self).__init__()
        
        self.pointnet_encoder = PointNetBlock(3, 128, dropout_rate)
        self.num_iterations = num_iterations
        
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        
        self.output_layer_translation = nn.Linear(64, 3)
        self.output_layer_rotation = nn.Linear(64, 3)
        self.output_layer_scaling = nn.Linear(64, 3)

    def forward(self, source, target):
        for _ in range(self.num_iterations):
            source_encoded = self.pointnet_encoder(source)
            target_encoded = self.pointnet_encoder(target)
            source_encoded = torch.max(source_encoded, 2, keepdim=True)[0]
            target_encoded = torch.max(target_encoded, 2, keepdim=True)[0]
            x = torch.cat([source_encoded, target_encoded], dim=1).squeeze(2)
            x = self.fc_layers(x)

            translation_output = self.output_layer_translation(x)
            rotation_output = self.output_layer_rotation(x)
            scaling_output = torch.exp(self.output_layer_scaling(x))

            transformations = torch.cat([translation_output, rotation_output, scaling_output], dim=1)
            source = apply_transformations(source, transformations)
        
        return transformations