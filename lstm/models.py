import torch
import torch.nn as nn


class CNN_1LSTM(nn.Module):
    def __init__(self, num_cnn_layers, num_classes, in_channels, lstm_hidden_size=128, fc_dropout=0.2):
        super().__init__()
        self.cnn_layers = nn.ModuleList()
        out_channels = 64
        for i in range(num_cnn_layers):
            self.cnn_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2)
                )
            )
            in_channels = out_channels
            out_channels *= 2

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=lstm_hidden_size,
            num_layers=2,           
            batch_first=True,
            bidirectional=True,     
            dropout=0.4
        )
        self.fc1 = nn.Linear(lstm_hidden_size * 2, 128)
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for cnn_block in self.cnn_layers:
            x = cnn_block(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        hidden_cat = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        x = self.relu(self.fc1(hidden_cat))
        x = self.fc_dropout(x)
        x = self.fc2(x)
        return x

class CNN_2LSTM(nn.Module):
    def __init__(self, num_cnn_layers, num_classes, in_channels):
        super().__init__()
        self.cnn_layers = nn.ModuleList()
        out_channels = 64
        for i in range(num_cnn_layers):
            self.cnn_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2)
                )
            )
            in_channels = out_channels
            out_channels *= 2
        
        self.lstm1 = nn.LSTM(
            input_size=in_channels, hidden_size=128, num_layers=3,
            batch_first=True, bidirectional=False, dropout=0.4
        )
        self.lstm2 = nn.LSTM(
            input_size=in_channels, hidden_size=128, num_layers=1,
            batch_first=True, bidirectional=True, dropout=0.4
        )
        self.fc1 = nn.Linear(128 + 128*2, 128) # 768
        self.fc_dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for cnn_block in self.cnn_layers:
            x = cnn_block(x)
        x = x.permute(0, 2, 1)
    
        _, (h1, _) = self.lstm1(x)
        _, (h2, _) = self.lstm2(x)
        h1_last = h1[-1,:,:]
        h2_cat = torch.cat((h2[-2,:,:], h2[-1,:,:]), dim=1)
        combined_hidden = torch.cat((h1_last, h2_cat), dim=1)
        
        x = self.relu(self.fc1(combined_hidden))
        x = self.fc_dropout(x)
        x = self.fc2(x)
        return x



class CNN(nn.Module):
    def __init__(self, in_channels, num_classes, dropout=0.3):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            # block 1
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # block 2
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # block 3
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.feature_extractor(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


