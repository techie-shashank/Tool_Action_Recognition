import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):

    def __init__(self, input_channels, time_steps, num_classes, hidden_size=64, num_layers=2):
        super(LSTMClassifier, self).__init__()

        self.encoder_output_size = hidden_size
        self.encoder = nn.LSTM(input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
       )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward_encoder_only(self, x):
        lstm_out, _ = self.encoder(x)
        out = lstm_out[:, -1, :]
        return out

    def forward(self, x):
        lstm_out, _ = self.encoder(x)  # output shape: (batch_size, time_steps, hidden_size)
        out = lstm_out[:, -1, :]    # Take output from last time step
        result = self.fc(out)
        return result
