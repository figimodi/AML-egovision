from torch import nn
import torch


class LSTM(nn.Module):
    def __init__(self, num_classes, input_size=1024, hidden_size=512):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, 1, batch_first=True, dropout=0.4)
        self.fc1 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.reshape((x.shape[0], 5, 1024))

        # initialize the hidden state.
        h0 = torch.zeros(1, x.shape[0], 512).to(x.device)
        c0 = torch.zeros(1, x.shape[0], 512).to(x.device)
        state = (h0, c0)

        out, state = self.lstm(x, state)
        out = self.fc1(out[:, -1, :])
        
        return out, {}
