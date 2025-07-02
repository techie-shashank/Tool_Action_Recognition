import torch
import torch.nn as nn
import json
import os

class LSTMClassifier(nn.Module):
    def __init__(self, input_channels, time_steps, num_classes, hidden_size=64, num_layers=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        
        # Load hyperparameters from config (available?)
        self._load_hyperparameters()
        
        # Override with provided parameters or use config values
        self.hidden_size = getattr(self, 'config_hidden_size', hidden_size)
        self.num_layers = getattr(self, 'config_num_layers', num_layers)
        self.dropout = getattr(self, 'config_dropout', dropout)
        
        self.input_channels = input_channels
        self.encoder_output_size = self.hidden_size * 2

        # Channel-wise learnable attention
        self.channel_attention = nn.Parameter(torch.ones(input_channels))

        # LSTM Encoder
        self.encoder = nn.LSTM(
            input_size=input_channels,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )

        # Fully Connected Classifier Head with configurable architecture
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, num_classes)
        )

    def _load_hyperparameters(self):
        """Load hyperparameters from config file if available."""
        try:
            config_path = os.path.join(os.path.dirname(__file__), '../config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                model_params = config.get('model_params', {}).get('lstm', {})
                self.config_hidden_size = model_params.get('hidden_size', 64)
                self.config_num_layers = model_params.get('num_layers', 2)
                self.config_dropout = model_params.get('dropout', 0.3)
        except:
            pass  # Use default values if config loading fails

    def forward_encoder_only(self, x):
        # Apply channel-wise attention
        attn = torch.sigmoid(self.channel_attention)
        x = x * attn
        lstm_out, _ = self.encoder(x)
        out = lstm_out[:, -1, :]
        return out

    def forward(self, x):
        # Apply channel-wise attention
        attn = torch.sigmoid(self.channel_attention)
        x = x * attn

        lstm_out, _ = self.encoder(x)
        out = lstm_out[:, -1, :]
        result = self.fc(out)
        return result

    def l1_regularization(self):
        return torch.sum(torch.abs(self.channel_attention))