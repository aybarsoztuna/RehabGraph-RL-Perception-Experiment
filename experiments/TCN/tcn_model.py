import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network for motion regression"""
    def __init__(self, num_inputs=75, num_channels=[64, 128, 64], kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size, 
                         padding=(kernel_size-1)*dilation_size, dilation=dilation_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        
        self.network = nn.Sequential(*layers)
        self.regressor = nn.Linear(num_channels[-1], 1)
        
    def forward(self, x):
        # x shape: (batch, features) -> (batch, features, 1) for Conv1d
        x = x.unsqueeze(2)
        x = self.network(x)
        x = x.mean(dim=2)  # global average pooling
        return self.regressor(x)

# Example usage
if __name__ == "__main__":
    model = TemporalConvNet()
    print("✅ TCN Model created successfully")
    print(model)
