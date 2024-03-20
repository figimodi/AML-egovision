from torch import nn
from torchvision import ops

class MLP(nn.Module):
    def __init__(self, num_classes, input_size=1024, hidden_size=512):
        super().__init__()

        # using torchvision built in MLP
        self.clf = ops.MLP(in_channels=input_size, hidden_channels=[hidden_size, hidden_size, num_classes], dropout=0.5)

        # self.relu = nn.ReLU()
        # self.fc1 = nn.Linear(input_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, num_classes)
        # self.sigmoid = nn.Sigmoid()
        
        # self.classifier = nn.Sequential(
        #     self.fc1,
        #     self.relu,
        #     self.fc2,
        #     self.relu,
        #     self.fc3,
        # )

        # TODO: add more layers

        self.classifier = self.clf

    def forward(self, x):
        return self.classifier(x), {}
