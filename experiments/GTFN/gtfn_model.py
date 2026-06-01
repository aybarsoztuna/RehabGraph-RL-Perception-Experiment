import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class LearnableFusionGate(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, h_spatial, h_temporal):
        concat = torch.cat([h_spatial, h_temporal], dim=-1)
        alpha = self.fusion(concat)
        return alpha * h_spatial + (1 - alpha) * h_temporal, alpha

class MultiScaleTemporalEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=2, dilation=2)
        self.conv4 = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=4, dilation=4)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
    def forward(self, x):
        batch, time, feat = x.shape
        x_conv = x.transpose(1, 2)
        c1 = F.relu(self.conv1(x_conv)).transpose(1, 2)
        c2 = F.relu(self.conv2(x_conv)).transpose(1, 2)
        c4 = F.relu(self.conv4(x_conv)).transpose(1, 2)
        conv_out = (c1 + c2 + c4) / 3
        attn_out, _ = self.attention(conv_out, conv_out, conv_out)
        combined = torch.cat([conv_out, attn_out], dim=-1)
        return self.proj(combined)

class AnatomicalGraphEncoder(nn.Module):
    def __init__(self, num_joints=25, in_features=3, hidden_dim=128):
        super().__init__()
        self.num_joints = num_joints
        self.gcn1 = GCNConv(in_features, 64)
        self.gcn2 = GCNConv(64, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
    def forward(self, x, edge_index):
        batch_size = x.size(0)
        x_flat = x.view(-1, x.size(-1))
        h = F.relu(self.gcn1(x_flat, edge_index))
        h = F.relu(self.gcn2(h, edge_index))
        h = h.view(batch_size, self.num_joints, -1)
        h_attn, _ = self.attention(h, h, h)
        return h_attn.mean(dim=1), h_attn

class GraphTemporalFusionNetwork(nn.Module):
    def __init__(self, num_joints=25, in_features=3, hidden_dim=128):
        super().__init__()
        self.spatial_encoder = AnatomicalGraphEncoder(num_joints, in_features, hidden_dim)
        self.temporal_encoder = MultiScaleTemporalEncoder(hidden_dim, hidden_dim)
        self.fusion_gate = LearnableFusionGate(hidden_dim)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )
    def forward(self, x, edge_index):
        batch, time, joints, feats = x.shape
        spatial_features = []
        for t in range(time):
            h_spat, _ = self.spatial_encoder(x[:, t, :, :], edge_index)
            spatial_features.append(h_spat)
        h_spatial = torch.stack(spatial_features, dim=1)
        h_temporal = self.temporal_encoder(h_spatial)
        h_fused, _ = self.fusion_gate(h_spatial, h_temporal)
        h_pooled = h_fused.mean(dim=1)
        return self.regressor(h_pooled)

def get_upper_limb_edge_index(num_joints=25):
    edges = []
    for i in range(num_joints - 1):
        edges.append([i, i+1])
        edges.append([i+1, i])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

if __name__ == "__main__":
    model = GraphTemporalFusionNetwork()
    print("GTFN model created successfully")
