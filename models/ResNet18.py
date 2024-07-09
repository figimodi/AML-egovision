import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18(nn.Module):
    def __init__(self, num_classes=20, num_channels=16, dropout_prob=.3):
        super().__init__()
        self.resnet18 = resnet18(pretrained=True)
        self.num_channels = num_channels
        self.resnet18.conv1 = nn.Conv2d(self.num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) 
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs), # Added an extra Linear layer before the final layer
            nn.ReLU(), # Activation function
            nn.Dropout(p=dropout_prob), # Dropout layer
            nn.Linear(num_ftrs, num_classes) # Final classification layer
        )

    def forward(self, x):
        x = x.squeeze(2)
        x = self.resnet18(x)
        return x, {} 
