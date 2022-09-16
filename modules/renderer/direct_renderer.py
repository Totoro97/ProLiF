import torch
import torch.nn as nn
from modules.fields.mlp import MLP
from modules.fields.prolif import ProLiF
from modules.fields.cons_weights import construct_siren_weights
from modules.embedder.embedder import PointEmbedder


class DirectRenderer(nn.Module):
    def __init__(self, **kwargs):
        super(DirectRenderer, self).__init__()
        if 'siren_weight_conf' in kwargs:
            weights, biases = construct_siren_weights(**kwargs['siren_weight_conf'])
            self.field = ProLiF(weights, biases, **kwargs['field_conf'])
        else:
            self.field = MLP(**kwargs['field_conf'])
        self.embedder = PointEmbedder(n_samples=2)  # Not used

    def forward(self, coords, **kwargs):
        rgb = self.field(coords).sigmoid()
        return rgb

    def full_query(self, coords, **kwargs):
        rgb = self.field(coords).sigmoid()
        return {
            'colors': rgb,
            'depths': rgb[:, :1] # not used
        }
