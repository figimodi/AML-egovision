from torch import nn
import torch

class EMGClassifier(nn.Module):
    def __init__(self, input_size=16, hidden_size=50, num_layers=1, num_classes=20, dropout_prob=.2):
        super(EMGClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        out, _ = self.lstm(x)        
        out = out[:, -1, :] 

        return out, {}

class RGBClassifier(nn.Module):
    def __init__(self, num_classes=20, input_size=1024, hidden_size=512):
        super(RGBClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, 1, batch_first=True)

    def forward(self, x):
        x = x.reshape((x.shape[0], 5, 1024))
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        
        return out, {}

class FusionClassifier(nn.Module):
    def __init__(self, num_classes=20):
        super(FusionClassifier, self).__init__()
       
        self.EMGClassifier = EMGClassifier()
        self.LSTMClassifier = RGBClassifier()

        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(512+50, num_classes)

    def forward(self, emg, rgb):

        emg = self.EMGClassifier(emg)
        rgb = self.LSTMClassifier(rgb)

        conc = torch.cat((emg[0], rgb[0]), dim=1)
        x = self.fc(conc)

        return x, {}
