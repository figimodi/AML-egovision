from torch import nn
import torch
import torch.nn.functional as F


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

class LeNet5(nn.Module):
    def __init__(self, num_classes=20):
        super(LeNet5, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=6, kernel_size=3, padding=2)  # Change input channels to 16
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        self.fc1 = nn.Linear(1, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x):
        x = x.squeeze(2)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(x.size(0), -1)
        
        if not hasattr(self, '_fc1_input_size') or self._fc1_input_size != x.size(1):
            self._fc1_input_size = x.size(1)
            self.fc1 = nn.Linear(self._fc1_input_size, 120).to(x.device)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x

class FusionClassifierEMGRGBspecto(nn.Module):
    def __init__(self, num_classes=20):
        super(FusionClassifierEMGRGBspecto, self).__init__()
       
        self.EMGClassifier = EMGClassifier()
        self.RGBClassifier = RGBClassifier()
        self.LeNet5 = LeNet5()

        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(84+512+50, num_classes)

    def forward(self, emg, rgb, specto):

        emg = self.EMGClassifier(emg)
        rgb = self.RGBClassifier(rgb)
        specto = self.LeNet5(specto)

        conc = torch.cat((emg[0], rgb[0], specto), dim=1)
        x = self.fc(conc)

        return x, {}
