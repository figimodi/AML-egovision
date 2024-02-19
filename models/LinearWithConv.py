from torch import nn


class LWC(nn.Module):
    def __init__(self, num_classes, input_size=1024, hidden_size=512):
        super().__init__()
        """
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size*32, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
        self.classifier = nn.Sequential(
            self.fc1,
            self.relu,
            self.fc2,
            self.relu,
            self.fc3
        )

    def forward(self, x):
        out = x.reshape(-1, 1, 1024)
        out = self.conv1(out)
        out = self.relu(out)
        
        out = out.reshape(-1, 32*1024)
        out = self.classifier(out)
        
        return out, {}
