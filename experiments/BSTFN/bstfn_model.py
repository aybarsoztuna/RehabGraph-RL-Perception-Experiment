"""
BSTFN: Bayesian Spatio-Temporal Fusion Network

Based on: Meng, Z., Meng, Q., Liu, K., Chen, Z., & Feng, S. (2025).
BSTFN: A Bayesian Spatio-Temporal Fusion Network for Electric Power Data Asset Circulation Prediction.
(Conference paper).

Adapted for upper-limb motion regression in stroke rehabilitation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import math


class BayesianInferenceModule(nn.Module):
    """
    Bayesian inference for handling uncertainty in predictions.
    Implements Monte Carlo Dropout for approximate Bayesian inference.
    """
    
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Uncertainty parameters (mean and variance)
        self.mean_layer = nn.Linear(hidden_dim, hidden_dim)
        self.var_layer = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, sample_n=5):
        """
        Args:
            x: Input tensor
            sample_n: Number of Monte Carlo samples
        
        Returns:
            mean prediction and uncertainty (variance)
        """
        # Multiple stochastic forward passes
        outputs = []
        
        for _ in range(sample_n):
            h = F.relu(self.fc1(x))
            h = self.dropout(h)
            h = F.relu(self.fc2(h))
            h = self.dropout(h)
            
            mean = self.mean_layer(h)
            var = torch.exp(self.var_layer(h))  # Ensure positivity
            
            # Reparameterization trick
            epsilon = torch.randn_like(mean)
            sample = mean + torch.sqrt(var) * epsilon
            outputs.append(sample)
        
        # Stack outputs
        outputs = torch.stack(outputs)  # (sample_n, batch, dim)
        
        # Compute mean and uncertainty
        mean_pred = outputs.mean(dim=0)
        uncertainty = outputs.var(dim=0)
        
        return mean_pred, uncertainty


class DynamicGCN(nn.Module):
    """
    Dynamic Graph Convolutional Network.
    Captures evolving relationships in the data over time.
    """
    
    def __init__(self, in_features, hidden_dim, num_layers=2):
        super().__init__()
        
        self.gcns = nn.ModuleList()
        self.gcns.append(GCNConv(in_features, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.gcns.append(GCNConv(hidden_dim, hidden_dim))
            
        # Dynamic adjacency adaptation
        self.adj_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, edge_index):
        # x: (batch * joints, features)
        # edge_index: (2, edges)
        
        h = x
        for i, gcn in enumerate(self.gcns):
            h = F.relu(gcn(h, edge_index))
            if i < len(self.gcns) - 1:
                h = F.dropout(h, training=self.training)
        
        return h


class SpatioTemporalInteractionAttention(nn.Module):
    """
    Spatio-Temporal Interaction Attention Mechanism.
    Captures dependencies across both time and space.
    """
    
    def __init__(self, feature_dim, num_heads=4, dropout=0.1):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, time, joints, features)
        
        Returns:
            Attended features (batch, time, joints, features)
        """
        batch, time, joints, feat = x.shape
        
        # Reshape for attention: (batch, time * joints, features)
        x_flat = x.view(batch, -1, feat)
        
        # Compute Q, K, V
        Q = self.q_proj(x_flat)
        K = self.k_proj(x_flat)
        V = self.v_proj(x_flat)
        
        # Multi-head attention
        Q = Q.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = (attn_weights @ V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, -1, feat)
        attn_output = self.out_proj(attn_output)
        
        # Reshape back to (batch, time, joints, features)
        attn_output = attn_output.view(batch, time, joints, -1)
        
        return attn_output


class BSTFN(nn.Module):
    """
    BSTFN: Bayesian Spatio-Temporal Fusion Network
    
    This implementation adapts the BSTFN architecture from power data prediction
    to upper-limb motion regression in stroke rehabilitation.
    
    Core components:
    1. Bayesian Inference Module for uncertainty handling
    2. Dynamic Graph Convolutional Network for spatial modeling
    3. Spatio-Temporal Interaction Attention for dependency capture
    4. Fusion and regression for posture deviation prediction
    """
    
    def __init__(self, num_joints=25, in_features=3, hidden_dim=64, 
                 num_heads=4, dropout_rate=0.2):
        super().__init__()
        
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(in_features, hidden_dim)
        
        # Bayesian Inference Module
        self.bayesian_module = BayesianInferenceModule(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        )
        
        # Dynamic Graph Convolutional Network
        self.dynamic_gcn = DynamicGCN(
            in_features=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=2
        )
        
        # Spatio-Temporal Interaction Attention
        self.st_attention = SpatioTemporalInteractionAttention(
            feature_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate
        )
        
        # Fusion and regression
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x, edge_index):
        """
        Args:
            x: Input tensor (batch, time, joints, features)
            edge_index: Anatomical graph connectivity (2, edges)
        
        Returns:
            Regression output (batch, 1)
        """
        batch, time, joints, feat = x.shape
        
        # === Input Projection ===
        x_proj = self.input_proj(x)  # (batch, time, joints, hidden_dim)
        
        # === Bayesian Inference ===
        # Apply Bayesian module across time dimension
        bayesian_outputs = []
        bayesian_uncertainties = []
        
        for t in range(time):
            x_t = x_proj[:, t, :, :]  # (batch, joints, hidden_dim)
            mean_t, unc_t = self.bayesian_module(x_t)
            bayesian_outputs.append(mean_t)
            bayesian_uncertainties.append(unc_t)
        
        # Stack outputs: (batch, time, joints, hidden_dim)
        h_bayesian = torch.stack(bayesian_outputs, dim=1)
        
        # === Dynamic GCN ===
        # Apply GCN across time dimension
        gcn_outputs = []
        
        for t in range(time):
            x_t = x_proj[:, t, :, :]  # (batch, joints, hidden_dim)
            x_t_flat = x_t.view(-1, self.hidden_dim)  # (batch * joints, hidden_dim)
            
            # Apply dynamic GCN
            h_t = self.dynamic_gcn(x_t_flat, edge_index)
            h_t = h_t.view(batch, joints, -1)  # (batch, joints, hidden_dim)
            gcn_outputs.append(h_t)
        
        h_gcn = torch.stack(gcn_outputs, dim=1)  # (batch, time, joints, hidden_dim)
        
        # === Spatio-Temporal Interaction Attention ===
        h_attention = self.st_attention(x_proj)  # (batch, time, joints, hidden_dim)
        
        # === Feature Fusion ===
        # Pool across time and joints
        h_bayesian_pooled = h_bayesian.mean(dim=[1, 2])  # (batch, hidden_dim)
        h_gcn_pooled = h_gcn.mean(dim=[1, 2])  # (batch, hidden_dim)
        h_attention_pooled = h_attention.mean(dim=[1, 2])  # (batch, hidden_dim)
        
        # Concatenate for fusion
        h_fused = torch.cat([
            h_bayesian_pooled,
            h_gcn_pooled + h_attention_pooled
        ], dim=-1)
        
        h_fused = self.fusion_layer(h_fused)
        
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
    print("BSTFN: Bayesian Spatio-Temporal Fusion Network")
    print("Based on Meng et al., 2025")
    print("=" * 60)
    
    model = BSTFN(num_joints=25, in_features=3, hidden_dim=64, num_heads=4)
    
    batch_size = 4
    time_steps = 10
    x = torch.randn(batch_size, time_steps, 25, 3)
    edge_index = get_upper_limb_edge_index(25)
    
    output = model(x, edge_index)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print("\n✅ BSTFN model created successfully")
