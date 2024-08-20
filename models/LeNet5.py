import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils as torch_utils

class LeNet5(nn.Module):

    # network structure
    def __init__(self, num_classes=20):
        super(LeNet5, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=6, kernel_size=3, padding=2)  # Change input channels to 16
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        self.fc1 = nn.Linear(96, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_classes)

    def forward(self, x):
        x = x.squeeze(2)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(x.size(0), -1)
        
        # if not hasattr(self, '_fc1_input_size') or self._fc1_input_size != x.size(1):
        #     self._fc1_input_size = x.size(1)
        #     self.fc1 = nn.Linear(self._fc1_input_size, 120).to(x.device)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x, {}