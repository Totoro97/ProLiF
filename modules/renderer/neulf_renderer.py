import torch
import torch.nn as nn
from modules.fields.mlp import MLP
from modules.embedder.embedder import PointEmbedder


class NeuLFRenderer(nn.Module):
    # Here we reimplement NeuLF only for computing the FPS and model size.
    # We did not perform thorough validations on the rendering quality.
    def __init__(self, field_conf, rgb_hidden_dim, depth_hidden_dim):
        super(NeuLFRenderer, self).__init__()
        self.field = MLP(**field_conf)
        self.rgb_hidden_dim = rgb_hidden_dim
        self.depth_hidden_dim = depth_hidden_dim
        self.rgb_linear = nn.Linear(rgb_hidden_dim, 3)
        self.depth_linear = nn.Linear(depth_hidden_dim, 1)
        self.embedder = PointEmbedder(n_samples=2)  # Not used

    def forward(self, coords, **kwargs):
        feats = self.field(coords)
        rgb = self.rgb_linear(feats[:, :self.rgb_hidden_dim]).sigmoid()
        return rgb

    def full_query(self, coords, **kwargs):
        feats = self.field(coords)
        rgb = self.rgb_linear(feats[:, :self.rgb_hidden_dim]).sigmoid()
        depth = self.depth_linear(feats[:, self.rgb_hidden_dim: self.rgb_hidden_dim + self.depth_hidden_dim]).sigmoid()
        return {
            'colors': rgb,
            'depths': depth
        }
