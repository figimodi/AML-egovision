from torch import nn, cat

class FinalClassifierEMG(nn.Module):
    def __init__(self, input_size=16, hidden_size=50, num_layers=2, num_classes=20, dropout_prob=.2):
        super(FinalClassifierEMG, self).__init__()

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, self.hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.hidden_size , num_classes)  # *2 for bidirectional output

    def forward(self, x):
        out, _ = self.lstm(x)
        
        out = out[:, -1, :] 

        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc(out)
        return out, {}
