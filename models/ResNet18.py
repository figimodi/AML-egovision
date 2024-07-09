import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18(nn.Module):
    def __init__(self, num_classes=20, num_channels=16):
        super().__init__()
        self.resnet18 = resnet18(pretrained=False)
        self.num_channels = num_channels
        self.resnet18.conv1 = nn.Conv2d(self.num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) 
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = x.squeeze(2)
        x = self.resnet18(x)
        return x, {} 
