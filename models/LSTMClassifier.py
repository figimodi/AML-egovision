from torch import nn
import torch


class LSTMClassifier(nn.Module):
    def __init__(self, num_classes, input_size=1024, hidden_size=512):
        super().__init__()
        """
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """
        
        # TODO: try dropout
        self.lstm = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
        
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.reshape((x.shape[0], 5, 1024))

        # initialize the hidden state.
        h0 = torch.zeros(1, x.shape[0], 512).to(x.device)
        c0 = torch.zeros(1, x.shape[0], 512).to(x.device)
        state = (h0, c0)

        out, state = self.lstm(x, state)
        out = self.relu(out[:, -1, :])
        out = self.fc1(out)
        
        return out, {}
