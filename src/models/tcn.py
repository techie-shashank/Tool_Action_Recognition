import torch.nn as nn
from pytorch_tcn import TCN

class TCNClassifier(nn.Module):
    def __init__(self, input_channels, time_steps, num_classes):
        super(TCNClassifier, self).__init__()

        self.tcn = TCN(
            num_inputs=input_channels,
            num_channels=[64, 64, 64],
            kernel_size=4,
            dropout=0.2,
            causal=True,
            use_norm='weight_norm',
            activation='relu',
            input_shape='NCL',
            output_projection=None
        )

        # Output classifier head
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),  # 64 is the last TCN channel
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)      # Convert to (B, C, T)
        y = self.tcn(x)             # Output: (B, C_out, T)
        y = y[:, :, -1]             # Take last timestep
        return self.classifier(y)
