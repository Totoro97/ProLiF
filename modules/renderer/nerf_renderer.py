import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.fields.mlp import MLP
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


class NeRFRenderer(nn.Module):
    def __init__(self,
                 geo_mlp_conf,
                 color_mlp_conf,
                 pts_embedder_conf,
                 dirs_embedder_conf,
                 geo_feat_dim,
                 app_feat_dim,
                 n_samples,
                 n_importance):
        super(NeRFRenderer, self).__init__()
        self.geo_mlp_coarse = MLP(**geo_mlp_conf)
        self.color_mlp_coarse = MLP(**color_mlp_conf)
        self.geo_mlp_fine = MLP(**geo_mlp_conf)
        self.color_mlp_fine = MLP(**color_mlp_conf)
        self.pts_embedder = PosEmbedder(**pts_embedder_conf)
        self.dirs_embedder = PosEmbedder(**dirs_embedder_conf)
        self.density_linear = nn.Linear(geo_feat_dim, 1)
        self.geo_feat_dim = geo_feat_dim
        self.app_feat_dim = app_feat_dim
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
        sampled_colors = color_mlp(app_feat).sigmoid().reshape(B, n_samples, 3)

        colors = (sampled_colors * weights[..., None]).sum(dim=1) + torch.rand(B, 3) * (1 - weights.sum(dim=-1, keepdim=True)) # Add background noise
        return weights, colors

    def full_query(self, coords, perturb=True, **kwargs):
        B = len(coords)
        n_samples = self.n_samples
        n_importance = self.n_importance
        steps = torch.linspace(0.5 / n_samples, 1 - 0.5 / n_samples, n_importance)
        if perturb:
            steps = steps[None, :] + (torch.rand(B, n_samples) - 0.5) / n_samples
        else:
            steps = steps[None, :].expand(B, -1)
        steps_bound = torch.cat([torch.zeros([B, 1]), (steps[:, :-1] + steps[:, 1:]) * 0.5, torch.ones([B, 1])], dim=-1)

        weights_coarse, colors_coarse = self.render_core(self.geo_mlp_coarse, self.color_mlp_coarse, coords, steps, steps_bound)

        # importance sampling
        new_steps = sample_pdf(steps_bound, weights_coarse, n_importance, det=(not perturb)).detach()
        new_steps, _ = torch.sort(torch.cat([steps, new_steps], dim=-1), dim=-1)

        new_steps_bound = torch.cat([torch.zeros([B, 1]), (new_steps[:, :-1] + new_steps[:, 1:]) * 0.5, torch.ones([B, 1])], dim=-1)
        weights_fine, colors_fine = self.render_core(self.geo_mlp_fine, self.color_mlp_fine, coords, new_steps, new_steps_bound)

        return {
            'colors_coarse': colors_coarse,
            'colors': colors_fine,
            'depths': (weights_fine * new_steps).sum(dim=-1, keepdim=True) + (1 - weights_fine.sum(dim=-1, keepdim=True))
        }


    def forward(self, coords, perturb=True, **kwargs):
        return self.full_query(coords, perturb, **kwargs)['colors']
