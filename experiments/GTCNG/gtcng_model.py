"""
GTCN-G: Gated Temporal Convolutional Network with Graph Attention Residuals

Based on: Xu, T., Wen, Z., Zhao, X., Hu, Q., Li, Y., & Liu, C. (2025).
GTCN-G: A Residual Graph-Temporal Fusion Network for Imbalanced Intrusion Detection.
IEEE TrustCom 2025.

Adapted for upper-limb motion regression in stroke rehabilitation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class GatedTCN(nn.Module):
    """
    Gated Temporal Convolutional Network (G-TCN)
    Uses a learned sigmoid gate to modulate the output of a standard 1D convolution.
    """
    
    def __init__(self, input_dim, hidden_dim, kernel_size=3, dilation=1):
        super().__init__()
        
        # Primary convolution
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size, 
                               padding=(kernel_size-1)*dilation, dilation=dilation)
        
        # Gate convolution (sigmoid gate)
        self.conv2 = nn.Conv1d(input_dim, hidden_dim, kernel_size,
                               padding=(kernel_size-1)*dilation, dilation=dilation)
        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: (batch, features, time)
        h1 = self.tanh(self.conv1(x))
        h2 = self.sigmoid(self.conv2(x))
        return h1 * h2


class MultiScaleGatedTCN(nn.Module):
    """
    Multi-scale Gated TCN with residual connections.
    Captures temporal patterns at different scales.
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers=3, kernel_size=3):
        super().__init__()
        
        self.layers = nn.ModuleList()
        in_dim = input_dim
        
        for i in range(num_layers):
            dilation = 2 ** i
            self.layers.append(
                GatedTCN(in_dim, hidden_dim, kernel_size, dilation)
            )
            in_dim = hidden_dim
            
        self.residual_conv = nn.Conv1d(input_dim, hidden_dim, 1)
        
    def forward(self, x):
        # x: (batch, features, time)
        residual = self.residual_conv(x)
        
        for layer in self.layers:
            x = layer(x)
            
        # Residual connection
        return x + residual


class GraphAttentionResidual(nn.Module):
    """
    Graph Attention Network with residual connections.
    Preserves original node features through residual connections.
    """
    
    def __init__(self, in_features, hidden_dim, num_heads=4):
        super().__init__()
        
        self.gat1 = GATConv(in_features, hidden_dim, heads=num_heads)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1)
        
        # Residual projection for original features
        self.residual_proj = nn.Linear(in_features, hidden_dim)
        
    def forward(self, x, edge_index):
        # x: (batch * joints, features)
        # edge_index: (2, edges)
        
        # GAT layers
        h1 = F.relu(self.gat1(x, edge_index))
        h2 = self.gat2(h1, edge_index)
        
        # Residual connection: preserve original features
        residual = self.residual_proj(x)
        
        return F.relu(h2 + residual)


class GTCNG(nn.Module):
    """
    GTCN-G: Gated Temporal Convolutional Network with Graph Attention Residuals
    
    This implementation adapts the GTCN-G architecture from intrusion detection
    to upper-limb motion regression in stroke rehabilitation.
    
    Core components:
    1. Multi-scale Gated TCN for temporal feature extraction
    2. Graph Attention Residuals for spatial feature extraction
    3. Feature fusion through concatenation
    4. Regression head for posture deviation prediction
    """
    
    def __init__(self, num_joints=25, in_features=3, temporal_dim=75, 
                 hidden_dim=64, num_heads=4):
        super().__init__()
        
        self.num_joints = num_joints
        self.temporal_dim = temporal_dim
        
        # Input projection
        self.input_proj = nn.Linear(in_features, hidden_dim)
        
        # Temporal branch: Multi-scale Gated TCN
        self.gated_tcn = MultiScaleGatedTCN(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=3
        )
        
        # Spatial branch: Graph Attention Residual
        self.graph_attention = GraphAttentionResidual(
            in_features=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )
        
        # Fusion and regression
        self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x, edge_index):
        """
        Args:
            x: Input tensor (batch, joints, features)
            edge_index: Anatomical graph connectivity (2, edges)
        
        Returns:
            Regression output (batch, 1)
        """
        batch_size, num_joints, in_feat = x.shape
        
        # Input projection
        x_proj = self.input_proj(x)  # (batch, joints, hidden_dim)
        
        # === Temporal Branch ===
        # Reshape to (batch, features, joints) for Conv1d
        x_temporal = x_proj.permute(0, 2, 1)  # (batch, hidden_dim, joints)
        
        # Apply Gated TCN
        h_temporal = self.gated_tcn(x_temporal)  # (batch, hidden_dim, joints)
        h_temporal = h_temporal.permute(0, 2, 1)  # (batch, joints, hidden_dim)
        
        # === Spatial Branch ===
        # Flatten for GCN: (batch * joints, hidden_dim)
        h_spatial_flat = h_temporal.reshape(-1, h_temporal.size(-1))
        
        # Apply Graph Attention Residual
        h_spatial = self.graph_attention(h_spatial_flat, edge_index)
        
        # Reshape back: (batch, joints, hidden_dim)
        h_spatial = h_spatial.view(batch_size, num_joints, -1)
        
        # Pool across joints
        h_spatial_pooled = h_spatial.mean(dim=1)  # (batch, hidden_dim)
        
        # === Temporal Pooling ===
        # Average across time/joints dimension
        h_temporal_pooled = h_temporal.mean(dim=1)  # (batch, hidden_dim)
        
        # === Fusion ===
        h_fused = torch.cat([h_temporal_pooled, h_spatial_pooled], dim=-1)
        h_fused = F.relu(self.fusion_layer(h_fused))
        
        # === Regression ===
        output = self.regressor(h_fused)
        
        return output


def get_upper_limb_edge_index(num_joints=25):
    """Create anatomical edge indices for upper-limb joints"""
    edges = []
    for i in range(num_joints - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


if __name__ == "__main__":
    # Test the model
    print("=" * 60)
    print("GTCN-G: Gated Temporal Convolutional Network with Graph Attention Residuals")
    print("Based on Xu et al., 2025 - IEEE TrustCom 2025")
    print("=" * 60)
    
    model = GTCNG(num_joints=25, in_features=3, temporal_dim=3, hidden_dim=64)
    
    batch_size = 4
    x = torch.randn(batch_size, 25, 3)
    edge_index = get_upper_limb_edge_index(25)
    
    output = model(x, edge_index)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print("\n✅ GTCN-G model created successfully")
