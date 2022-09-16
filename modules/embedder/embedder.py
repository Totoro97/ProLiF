import torch
import torch.nn as nn
import numpy as np


class PointEmbedder(nn.Module):
    def __init__(self, n_samples):
        self.n_samples = n_samples
        self.steps = torch.linspace(0.5 / self.n_samples, 1 - 0.5 / self.n_samples, self.n_samples)
        super(PointEmbedder, self).__init__()

    def forward(self, coords, perturb=True, steps=None, query_jac=False):
        # bg_pts = torch.cat([coords[:, :2], torch.zeros_like(coords[:, :1])], dim=-1)
        # ed_pts = torch.cat([coords[:, 2:], torch.ones_like(coords[:, :1])], dim=-1)
        bg_pts = coords[:, :2]
        ed_pts = coords[:, 2:]
        if steps is None:
            if perturb:
                steps = self.steps[None, :] + (torch.rand(bg_pts.shape[0], self.n_samples) - 0.5) / self.n_samples
            else:
                steps = self.steps[None, :].expand(coords.shape[0], -1)
        pts = bg_pts[:, None, :] * (1 - steps)[:, :, None] + ed_pts[:, None, :] * steps[:, :, None]
        if not query_jac:
            return pts.reshape(-1, self.n_samples * 2), steps
        else:
            B = coords.shape[0]
            jac = torch.zeros(B, self.n_samples, 2, 4)
            jac[:, :, 0, 0] = 1 - steps
            jac[:, :, 0, 2] = steps
            jac[:, :, 1, 1] = 1 - steps
            jac[:, :, 1, 3] = steps

            return pts.reshape(-1, self.n_samples * 2), steps, jac.reshape(-1, self.n_samples * 2, 4)

    def update_n_samples(self, n_samples):
        self.n_samples = n_samples
        self.steps = torch.linspace(0.5 / self.n_samples, 1 - 0.5 / self.n_samples, self.n_samples)

    def update_progress(self, iter_step):
        pass


class RandPosEmbedder(nn.Module):
    def __init__(self, d_in, layers, freq, channels, window_begin=0, window_end=50000):
        super(RandPosEmbedder, self).__init__()
        emb_w = torch.randn(d_in, layers, freq, channels)
        emb_w = emb_w / torch.linalg.norm(emb_w, ord=2, dim=0, keepdim=True)
        emb_w = emb_w * (2**torch.arange(freq))[None, None, :, None]
        # emb_w = emb_w.reshape(d_in, freq * channels)
        self.mask = torch.zeros([freq])
        self.register_parameter('emb', nn.Parameter(emb_w))
        self.emb.requires_grad_(False)
        self.freq = freq
        self.channels = channels
        self.window_begin = window_begin
        self.window_end = window_end
        self.layers = layers

    def update_progress(self, iter_step):
        progress = (iter_step - self.window_begin) / (self.window_end - self.window_begin)
        progress = np.clip(progress, 0, 1)
        begin_idx = int(progress * self.freq)
        mid_ratio = progress * self.freq - begin_idx
        if begin_idx == self.freq:
            begin_idx = self.freq - 1
            mid_ratio = 1.0

        self.mask = torch.cat(
            [torch.ones([begin_idx]), torch.ones([1]) * mid_ratio, torch.zeros([self.freq - begin_idx - 1])])

    def forward(self, x):
        y = (x[:, :, None, None, None] * self.emb[None, :, :, :, :]).sum(dim=1)
        cos_val = (torch.cos(y) * self.mask[None, None, :, None]).reshape(-1, self.layers, self.freq * self.channels)
        sin_val = (torch.sin(y) * self.mask[None, None, :, None]).reshape(-1, self.layers, self.freq * self.channels)
        y = torch.cat([x[:, None, :].expand(-1, self.layers, -1), cos_val, sin_val], dim=-1)
        return y


class PosEmbedder(nn.Module):
    def __init__(self, n_freq):
        super(PosEmbedder, self).__init__()
        self.mul = 2.**torch.arange(n_freq)

    def forward(self, x):
        B = x.shape[0]
        cos_val = torch.cos(x[:, :, None] * self.mul[None, None, :])
        sin_val = torch.sin(x[:, :, None] * self.mul[None, None, :])

        return torch.cat([x, cos_val.reshape(B, -1), sin_val.reshape(B, -1)], dim=-1)
