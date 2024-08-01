from torch import nn
import torch
import torch.nn.functional as F


class LeNet5_CACO(nn.Module):
    def __init__(self, num_classes=20):
        super(LeNet5_CACO, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1)
        
        # Placeholder for the first fully connected layer, will be dynamically adjusted
        self.fc1 = nn.Linear(1, 64)  # Initial placeholder
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Assuming the input shape is (batch, channels, 1, width, height)
        x = x.squeeze(2)  # Remove the singleton dimension: shape (batch, channels, width, height)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        # Dynamically compute the size of the flattened tensor and adjust the fully connected layer
        if not hasattr(self, '_fc1_input_size') or self._fc1_input_size != x.size(1):
            self._fc1_input_size = x.size(1)
            self.fc1 = nn.Linear(self._fc1_input_size, 64).to(x.device)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x, {}
