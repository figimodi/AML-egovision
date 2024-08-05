from torch import nn, cat

class FinalClassifierEMG_L2(nn.Module):
    def __init__(self, input_size=16, hidden_size1=5, hidden_size2=50, num_layers=1, num_classes=20, dropout_prob=.2):
        super(FinalClassifierEMG_L2, self).__init__()

        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2

        self.lstm1 = nn.LSTM(input_size, self.hidden_size1, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1, self.hidden_size2, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.hidden_size2 , num_classes)  # *2 for bidirectional output

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        
        out = out[:, -1, :] 

        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc(out)
        return out, {}
