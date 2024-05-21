import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.resnet18 = resnet18(pretrained=False)  # Load pre-trained ResNet-18
        num_ftrs = self.resnet18.fc.in_features  # Get the number of input features to the last layer
        self.resnet18.fc = nn.Linear(num_ftrs * 16, num_classes)  # Modify last layer for 16 image features

    def forward(self, x):
        batch_size, num_images, channels, height, width = x.size()
        x = x.view(-1, channels, height, width) 
        x = self.resnet18(x)
        x = x.view(batch_size, -1)  
        return x
