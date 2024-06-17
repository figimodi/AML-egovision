import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18(nn.Module):
    def __init__(self, num_classes=20, num_images_per_sample=16):
        super().__init__()
        self.resnet18 = resnet18(pretrained=False)
        self.num_images_per_sample = num_images_per_sample
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        batch_size, num_images, channels, height, width = x.size()
        x = x.view(-1, channels, height, width)  # Flatten for individual processing
        x = self.resnet18(x)
        x = x.view(batch_size, self.num_images_per_sample, -1)
        x = torch.mean(x, dim=1)  # Average across the images in the sample
        return x, {} 
