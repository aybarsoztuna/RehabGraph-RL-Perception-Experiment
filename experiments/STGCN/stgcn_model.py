import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class STGCN(nn.Module):
    """Spatio-Temporal Graph Convolutional Network"""
    def __init__(self, num_nodes=25, in_features=3, hidden_features=64):
        super().__init__()
        self.num_nodes = num_nodes
        
        # Spatial Graph Convolution
        self.gcn1 = GCNConv(in_features, hidden_features)
        self.gcn2 = GCNConv(hidden_features, hidden_features)
        
        # Temporal Convolution
        self.tcn = nn.Conv1d(hidden_features, hidden_features, kernel_size=3, padding=1)
        
        # Output layer
        self.regressor = nn.Linear(hidden_features, 1)
        
    def forward(self, x, edge_index):
        # x shape: (batch, num_nodes, features) -> (batch*num_nodes, features)
        batch_size = x.size(0)
        x = x.view(-1, x.size(-1))
        
        # Spatial GCN
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        
        # Reshape back and apply temporal conv
        x = x.view(batch_size, self.num_nodes, -1)
        x = x.transpose(1, 2)  # (batch, features, nodes)
        x = F.relu(self.tcn(x))
        x = x.mean(dim=2)  # Global average pooling
        
        return self.regressor(x)

# Example usage
if __name__ == "__main__":
    model = STGCN()
    print("✅ ST-GCN Model created successfully")
    print(model)
