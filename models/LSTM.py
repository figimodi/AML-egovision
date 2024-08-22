from torch import nn
import torch


class LSTM(nn.Module):
    def __init__(self, num_classes=20, input_size=1024, hidden_size=512):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.reshape((x.shape[0], 5, 1024))

        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc1(out)
        
        return out, {}
