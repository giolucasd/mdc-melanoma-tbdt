import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Simple Convolutional Neural Network for binary melanoma classification.
    
    Architecture:
    - Input: 3-channel images (224x224)
    - Conv layers with ReLU activations and max pooling
    - Fully connected layers for classification
    - Output: binary classification (0: benign, 1: malignant)
    """
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Fully connected layers
        # After 4 pooling layers: 224 -> 112 -> 56 -> 28 -> 14
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc_bn1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(512, 128)
        self.fc_bn2 = nn.BatchNorm1d(128)
        
        self.fc3 = nn.Linear(128, 1)  # Binary classification (sigmoid output)
    
    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc_bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.fc_bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x
