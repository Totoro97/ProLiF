import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Regularizer:
    def __init__(self,
                 dataset,
                 renderer,
                 batch_size,
                 reg_weight,
                 reg_window_begin,
                 reg_window_end,
                 color_weight,
                 density_weight,
                 r_alpha,
                 sml1_beta,
                 n_apps=1):
        self.name = 'regularizer'
        self.renderer = renderer
        self.dataset = dataset
        self.batch_size = batch_size

        self.reg_weight = reg_weight
        self.reg_window_begin = reg_window_begin
        self.reg_window_end = reg_window_end

        self.color_weight = color_weight
        self.density_weight = density_weight
        self.r_alpha = r_alpha

        self.iter_step = 0
        self.sml1_beta = sml1_beta

        self.n_apps = n_apps

    def update_progress(self, iter_step):
        self.iter_step = iter_step

    def get_density_loss(self, coords, steps_bound, steps, render_result):
        B = steps.shape[0]

        bias_0 = torch.randn(B, 2)
        bias_1 = torch.cat([-bias_0[:, 1:], bias_0[:, :1]], dim=-1)
        bias = torch.stack([bias_0, bias_1], dim=1)  # B, 2, 2
        bias = bias / torch.linalg.norm(bias, ord=2, dim=-1, keepdim=True)

        bias = torch.cat([bias[:, None, :, :] * steps[:, :, None, None],
                          bias[:, None, :, :] * -(1 - steps)[:, :, None, None]], dim=-1)  # B, N, 2, 4

        density_jac = render_result['density_jac']
        density_loss = ((bias[:, :, :, :] * density_jac[:, :, None, :]).sum(dim=-1)**2).sum(dim=-1).mean()

        return density_loss

    def get_color_loss(self, coords, steps_bound, steps, render_result):
        B = steps.shape[0]

        depths = render_result['depths']
        bias_0 = torch.randn(B, 2)
        bias_1 = torch.cat([-bias_0[:, 1:], bias_0[:, :1]], dim=-1)
        bias = torch.stack([bias_0, bias_1], dim=1)  # B, 2, 2
        bias = bias / torch.linalg.norm(bias, ord=2, dim=-1, keepdim=True)
        bias = torch.cat(
            [depths[:, :, None, None] * bias[:, None, :, :], -(1 - depths)[:, :, None, None] * bias[:, None, :, :]],
            dim=-1)  # B, 1, 2, 4

        color_grad = render_result['colors_jac']
        color_error = torch.linalg.norm((color_grad[:, :, None, :] * bias).sum(dim=-1), ord=2, dim=-1)
        color_loss = F.smooth_l1_loss(color_error, torch.zeros_like(color_error), beta=self.sml1_beta)

        return color_loss

    def get_loss(self):
        if self.iter_step < self.reg_window_begin:
            return 0., dict()

        B = self.batch_size
        coords = self.dataset.rand_coords_from_rand_pose(B)
        n_samples = self.renderer.embedder.n_samples
        steps_bound = torch.linspace(0, 1, n_samples + 1)[None].expand(B, -1)
        steps_ratio = torch.rand(B, n_samples)
        steps = steps_bound[:, :-1] * steps_ratio + steps_bound[:, 1:] * (1 - steps_ratio)
        render_result = self.renderer.full_query(coords,
                                                 steps=steps,
                                                 query_jac=True,
                                                 app_idx=np.random.randint(self.n_apps))

        color_loss = 0.
        if self.color_weight > 1e-5:
            color_loss = self.get_color_loss(coords, steps_bound, steps, render_result)

        density_loss = 0.
        if self.density_weight > 1e-5:
            density_loss = self.get_density_loss(coords, steps_bound, steps, render_result)

        # gradually increase weights of color loss
        progress = (self.iter_step - self.reg_window_begin) / (self.reg_window_end - self.reg_window_begin)
        progress = np.max([0, np.min([1., progress])])
        factor = (np.cos(np.pi * (1 - progress)) + 1.0) * 0.5 * (1 - self.r_alpha) + self.r_alpha

        r_loss = color_loss * self.color_weight * factor + density_loss * self.density_weight
        loss = r_loss * self.reg_weight

        loss_dict = {
            'color_loss': color_loss,
            'density_loss': density_loss,
        }

        return loss, loss_dict
