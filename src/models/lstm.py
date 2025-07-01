import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_channels, time_steps, num_classes, hidden_size=64, num_layers=2):
        super(LSTMClassifier, self).__init__()

        self.input_channels = input_channels
        self.encoder_output_size = hidden_size * 2

        # Channel-wise learnable attention
        self.channel_attention = nn.Parameter(torch.ones(input_channels))

        # LSTM Encoder
        self.encoder = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Fully Connected Classifier Head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward_encoder_only(self, x):
        # Apply channel-wise attention
        attn = torch.sigmoid(self.channel_attention)  # Shape: (input_channels,)
        x = x * attn  # Broadcasting over batch and time dimensions
        lstm_out, _ = self.encoder(x)
        out = lstm_out[:, -1, :]  # Last time step output
        return out

    def forward(self, x):
        # Apply channel-wise attention
        attn = torch.sigmoid(self.channel_attention)  # Shape: (input_channels,)
        x = x * attn  # Broadcasting over batch and time dimensions

        lstm_out, _ = self.encoder(x)  # LSTM expects shape: (batch, time, input_channels)
        out = lstm_out[:, -1, :]  # Take output from last time step
        result = self.fc(out)
        return result

    def l1_regularization(self):
        return torch.sum(torch.abs(self.channel_attention))
