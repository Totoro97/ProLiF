import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.fields.mlp import MLP
from modules.fields.upsample_nn import UpsampleNN
from modules.embedder.embedder import PosEmbedder


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class NeRFPatchRenderer(nn.Module):
    def __init__(self,
                 geo_mlp_conf,
                 color_mlp_conf,
                 pts_embedder_conf,
                 dirs_embedder_conf,
                 geo_feat_dim,
                 app_feat_dim,
                 up_sampler_conf,
                 n_samples,
                 n_importance):
        super(NeRFPatchRenderer, self).__init__()
        self.geo_mlp_coarse = MLP(**geo_mlp_conf)
        self.color_mlp_coarse = MLP(**color_mlp_conf)
        self.geo_mlp_fine = MLP(**geo_mlp_conf)
        self.color_mlp_fine = MLP(**color_mlp_conf)
        self.pts_embedder = PosEmbedder(**pts_embedder_conf)
        self.dirs_embedder = PosEmbedder(**dirs_embedder_conf)
        self.density_linear = nn.Linear(geo_feat_dim, 1)
        self.geo_feat_dim = geo_feat_dim
        self.app_feat_dim = app_feat_dim
        self.color_feat_dim = color_mlp_conf['dims'][-1]

        self.upsampler_coarse = UpsampleNN(**up_sampler_conf)
        self.upsampler_fine = UpsampleNN(**up_sampler_conf)
        self.n_samples = n_samples
        self.n_importance = n_importance

    def render_core(self, geo_mlp, color_mlp, coords, steps, steps_bound):
        B, n_samples = steps.shape
        dists = steps_bound[:, 1:] - steps_bound[:, :-1]
        bg_pts = torch.cat([coords[:, :2], torch.zeros([B, 1])], dim=-1)
        ed_pts = torch.cat([coords[:, 2:], torch.ones([B, 1])], dim=-1)

        pts = bg_pts[:, None, :] * (1 - steps)[..., None] + ed_pts[:, None, :] * steps[..., None]
        dirs = (ed_pts - bg_pts)[:, None, :].expand(-1, n_samples, -1)

        feat = geo_mlp(self.pts_embedder(pts.reshape(-1, 3)))
        geo_feat = feat[:, :self.geo_feat_dim]
        app_feat = feat[:, self.geo_feat_dim: self.geo_feat_dim + self.app_feat_dim]
        density = F.relu(self.density_linear(geo_feat) + 0.1).reshape(B, n_samples)   # Add 0.1 here to avoid "Dying ReLU" problem
        dis_density = dists * density
        alpha = 1 - torch.exp(-dis_density)
        trans = torch.exp(-torch.cumsum(dis_density, dim=1))
        weights = torch.cat([alpha[:, :1], trans[:, :-1] * alpha[:, 1:]], dim=1)

        app_feat = torch.cat([app_feat, self.dirs_embedder(dirs.reshape(-1, 3))], dim=1)
        sampled_color_feats = color_mlp(app_feat).reshape(B, n_samples, self.color_feat_dim)

        color_feats = (sampled_color_feats * weights[..., None]).sum(dim=1)
        return weights, color_feats

    def up_sample(self, depths, feats, upsampler, H, W, H_l, W_l, use_upsampler=True):
        depths = depths.reshape(1, 1, H, W)
        feats = feats.reshape(1, H, W, -1).permute(0, 3, 1, 2)
        colors = upsampler(feats, use_upsampler=use_upsampler).sigmoid().permute(0, 2, 3, 1).reshape(H_l * W_l, 3)
        depths = F.interpolate(depths, size=(H_l, W_l), mode='bilinear').reshape(H_l * W_l, 1)

        return depths, colors

    def full_query(self, coords_large, H=-1, W=-1, perturb=True, mode='train', **kwargs):
        B_l = len(coords_large)

        if mode=='train':
            if H < 0:
                H_l = int(np.sqrt(B_l))
                W_l = B_l // H_l
                assert H_l * W_l == B_l
            else:
                H_l = H
                W_l = W
        else:
            H_l = B_l
            W_l = 1

        if mode == 'train':
            factor = 2**self.upsampler_coarse.n_layers
        else:
            factor = 1

        H = H_l // factor
        W = W_l // factor
        coords_large = coords_large.reshape(1, H_l, W_l, 4).permute(0, 3, 1, 2)
        coords = F.interpolate(coords_large, size=(H, W), mode='bilinear')
        coords = coords.permute(0, 2, 3, 1).reshape(H * W, 4)

        B = H * W

        n_samples = self.n_samples
        n_importance = self.n_importance
        steps = torch.linspace(0.5 / n_samples, 1 - 0.5 / n_samples, n_importance)
        if perturb:
            steps = steps[None, :] + (torch.rand(B, n_samples) - 0.5) / n_samples
        else:
            steps = steps[None, :].expand(B, -1)
        steps_bound = torch.cat([torch.zeros([B, 1]), (steps[:, :-1] + steps[:, 1:]) * 0.5, torch.ones([B, 1])], dim=-1)

        weights_coarse, feats_coarse = self.render_core(self.geo_mlp_coarse, self.color_mlp_coarse, coords, steps, steps_bound)
        depths_coarse = (weights_coarse * steps).sum(dim=-1, keepdim=True) + (1 - weights_coarse.sum(dim=-1, keepdim=True))
        depths_coarse, colors_coarse = self.up_sample(depths_coarse, feats_coarse, self.upsampler_coarse, H, W, H_l, W_l, use_upsampler=(mode=='train'))

        # importance sampling
        new_steps = sample_pdf(steps_bound, weights_coarse, n_importance, det=(not perturb)).detach()
        new_steps, _ = torch.sort(torch.cat([steps, new_steps], dim=-1), dim=-1)

        new_steps_bound = torch.cat([torch.zeros([B, 1]), (new_steps[:, :-1] + new_steps[:, 1:]) * 0.5, torch.ones([B, 1])], dim=-1)
        weights_fine, feats_fine = self.render_core(self.geo_mlp_fine, self.color_mlp_fine, coords, new_steps, new_steps_bound)
        depths_fine = (weights_fine * new_steps).sum(dim=-1, keepdim=True) + (1 - weights_fine.sum(dim=-1, keepdim=True))

        depths_fine, colors_fine = self.up_sample(depths_fine, feats_fine, self.upsampler_fine, H, W, H_l, W_l, use_upsampler=(mode=='train'))

        return {
            'colors_coarse': colors_coarse,
            'colors': colors_fine,
            'depths': depths_fine
        }

    def forward(self, coords, H=-1, W=-1, perturb=True, mode='train', **kwargs):
        return self.full_query(coords, H=H, W=W, perturb=perturb, mode=mode)['colors']
