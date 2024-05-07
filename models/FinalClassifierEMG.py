from torch import nn

class FinalClassifierEMG(nn.Module):
    def __init__(self, input_size=16, hidden_size1=5, hidden_size2=50, dropout=0.2, num_classes=20):
        super().__init__()
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2

        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.dense = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = self.dropout(out[:, -1, :])
        out = self.dense(out)
        
        return out, {}
