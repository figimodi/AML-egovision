from torch import nn
import torch


class LSTMClassifier(nn.Module):
    def __init__(self, num_classes, input_size=1024, hidden_size=512):
        super().__init__()
        """
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """
        
        self.conv1 = nn.Conv1d(1, 32, 3, 1, 1)
        
        self.lstm = nn.LSTM(input_size*32, hidden_size, 1, batch_first=True)
        
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_size, num_classes)
        
        self.h0 = lambda x: torch.zeros(1, hidden_size).to(x.device)
        self.c0 = lambda x: torch.zeros(1, hidden_size).to(x.device)

    def forward(self, x):
        out = x.reshape(-1, 1, 1024)
        out = self.conv1(out)
        out = self.relu(out)
        x = out.reshape(-1, 32*1024)
        
        out, _ = self.lstm(x, (self.h0(x), self.c0(x)))
        out = self.relu(out)
        out = self.fc1(out)
        
        return out, {}
