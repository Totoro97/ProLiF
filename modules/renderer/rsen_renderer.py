import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.fields.mlp import MLP
from modules.embedder.embedder import PosEmbedder


class RSENRenderer(nn.Module):
    # Here we reimplement RSEN only for computing the FPS and model size.
    # We did not perform thorough validations on the rendering quality.
    def __init__(self,
                 n_samples,
                 affine_nn_conf,
                 coord_embedder_conf,
                 render_nn_conf,
                 pos_embedder_conf,
                 affine_emb_dim):
        super(RSENRenderer, self).__init__()
        self.n_samples = n_samples
        self.affine_nn = MLP(**affine_nn_conf)
        self.coord_embedder = PosEmbedder(**coord_embedder_conf)
        self.render_nn = MLP(**render_nn_conf)
        self.pos_embedder = PosEmbedder(**pos_embedder_conf)
        self.affine_emb_dim = affine_emb_dim
        self.steps = torch.linspace(0, 1, n_samples + 1)

    def full_query(self, coords, **kwargs):
        B = coords.shape[0]
        pts = coords[:, None, :2] * (1 - self.steps[None, :, None]) + coords[:, None, :2:] * self.steps[None, :, None]
        local_coords = torch.cat([pts[:, :self.n_samples, :], pts[:, 1:, :]], dim=-1).reshape(B * self.n_samples, 4)
        steps = self.steps[None, :self.n_samples, None].expand(B, -1, -1)
        steps_emb = self.coord_embedder(steps.reshape(B * self.n_samples, 1))

        affine_mat_bias = self.affine_nn(torch.cat([local_coords, steps_emb], dim=-1))
        affine_mat  = affine_mat_bias[:, :self.affine_emb_dim * 4].reshape(B * self.n_samples, self.affine_emb_dim, 4)
        affine_bias = affine_mat_bias[:, self.affine_emb_dim * 4: self.affine_emb_dim * 5].reshape(B * self.n_samples, self.affine_emb_dim)

        affine_mat = affine_mat / (affine_mat**2).sum(dim=1, keepdim=True).sum(dim=2, keepdim=True).sqrt() * np.sqrt(128)
        affine_bias = torch.tanh(affine_bias)
        affine_feat = (local_coords[:, None, :] * affine_mat).sum(dim=2) + affine_bias

        affine_feat = self.pos_embedder(affine_feat)
        rgba = self.render_nn(affine_feat).reshape(B, self.n_samples, 4)
        rgb = rgba[:, :, :3].sigmoid()
        alpha = 1 - torch.exp(-F.softplus(rgba[:, :, 3], beta=10))
        trans = torch.cumprod(1 - alpha, dim=-1)
        weights = torch.cat([alpha[:, :1], trans[:, :-1] * alpha[:, 1:]], dim=-1)
        colors = (rgb * weights[..., None]).sum(dim=1) + torch.rand(B, 3) * (1 - weights.sum(dim=1, keepdim=True))
        depths = (steps.reshape(B, self.n_samples) * weights).sum(dim=1, keepdim=True) + (1 - weights.sum(dim=1, keepdim=True))
        return {
            'colors': colors,
            'depths': depths
        }

    def forward(self, coords, **kwargs):
        return self.full_query(coords)['colors']
