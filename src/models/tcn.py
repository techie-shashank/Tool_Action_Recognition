import torch.nn as nn
from pytorch_tcn import TCN

class TCNClassifier(nn.Module):
    def __init__(self, input_channels, time_steps, num_classes):
        super(TCNClassifier, self).__init__()

        self.encoder_output_size = 64
        self.encoder = TCN(
            num_inputs=input_channels,
            num_channels=[64, 64, self.encoder_output_size],
            kernel_size=4,
            dropout=0.2,
            causal=True,
            use_norm='weight_norm',
            activation='relu',
            input_shape='NLC',
            output_projection=None
        )

        # Output classifier head
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),  # 64 is the last TCN channel
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward_encoder_only(self, x):
        y = self.encoder(x)  # Output: (B, T, C_out)
        y = y[:, -1, :]
        return y

    def forward(self, x):
        y = self.encoder(x)             # Output: (B, T, C_out)
        y = y[:, -1, :]              # Take last timestep
        return self.classifier(y)
