from torch import nn


class LWP(nn.Module):
    def __init__(self, num_classes, input_size=1024, hidden_size=512):
        super().__init__()
        """
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """
        
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        self.classifier = nn.Sequential(
            self.pool1,
            self.fc1,
            self.relu,
            self.fc2
        )

    def forward(self, x):
        return self.classifier(x), {}
