import torch
import torch.nn as nn
import torch.nn.init as init

# PointNet basic block
class PointNetBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(PointNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, output_size, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x

# Custom Alignment Model using PointNet blocks
class AlignmentModel(nn.Module):
    def __init__(self):
        super(AlignmentModel, self).__init__()
        self.pointnet1 = PointNetBlock(3, 128)
        self.pointnet2 = PointNetBlock(3, 128)
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.output_layer_translation = nn.Linear(64, 3)
        self.output_layer_rotation = nn.Linear(64, 3)
        self.output_layer_scaling = nn.Linear(64, 3)

    def forward(self, source, target):
        source = self.pointnet1(source) 
        target = self.pointnet2(target)
        source = torch.max(source, 2, keepdim=True)[0]
        target = torch.max(target, 2, keepdim=True)[0]
        x = torch.cat([source, target], dim=1).squeeze(2)
        x = self.fc_layers(x)

        translation_output = self.output_layer_translation(x)
        rotation_output = self.output_layer_rotation(x)
        scaling_output = torch.exp(self.output_layer_scaling(x))

        return torch.cat([translation_output, rotation_output, scaling_output], dim=1)