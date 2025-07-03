import torch.nn as nn
from pytorch_tcn import TCN
import json
import os


class TCNClassifier(nn.Module):
    def __init__(self, input_channels, time_steps, num_classes, 
                 num_channels=None, kernel_size=4, dropout=0.2):
        super(TCNClassifier, self).__init__()
        
        # Load hyperparameters from config if available
        self._load_hyperparameters()
        
        # Override with provided parameters or use config values
        self.num_channels = getattr(self, 'config_num_channels', num_channels or [64, 64, 64])
        self.kernel_size = getattr(self, 'config_kernel_size', kernel_size)
        self.dropout = getattr(self, 'config_dropout', dropout)
        
        self.encoder_output_size = self.num_channels[-1]
        
        self.encoder = TCN(
            num_inputs=input_channels,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            causal=True,
            use_norm='weight_norm',
            activation='relu',
            input_shape='NLC',
            output_projection=None
        )

        # Output classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.num_channels[-1], 128),
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
                
                model_params = config.get('model_params', {}).get('tcn', {})
                self.config_num_channels = model_params.get('num_channels', [64, 64, 64])
                self.config_kernel_size = model_params.get('kernel_size', 4)
                self.config_dropout = model_params.get('dropout', 0.2)
        except:
            pass

    def forward_encoder_only(self, x):
        y = self.encoder(x)
        y = y[:, -1, :]
        return y

    def forward(self, x):
        y = self.encoder(x)
        y = y[:, -1, :]
        return self.classifier(y)