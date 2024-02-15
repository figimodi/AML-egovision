from torch import nn


class Classifier(nn.Module):
    def __init__(self, num_classes, input_size=1024, hidden_size=512):
        super().__init__()
        """
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """
        
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        self.classifier = lambda x: self.fc2(self.relu(self.fc1(x)))

    def forward(self, x):
        return self.classifier(x), {}
