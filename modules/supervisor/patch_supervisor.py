import torch
import torch.nn as nn
import lpips
import numpy as np


class PatchSupervisor:
    def __init__(self,
                 dataset,
                 renderer,
                 lp_weight,
                 batch_sizes,
                 mile_stones,
                 lp_patch_h,
                 lp_patch_w,
                 image_down_levels,
                 size_align=1,
                 color_keys=('colors',)):
        self.name = 'patch_supervisor'
        self.dataset = dataset
        self.renderer = renderer
        self.iter_step = 0
        self.ms_idx = 0
        self.lp_weight = lp_weight
        self.batch_sizes = batch_sizes
        self.mile_stones = mile_stones
        self.lp_patch_h = lp_patch_h
        self.lp_patch_w = lp_patch_w
        self.image_down_levels = image_down_levels
        self.image_perm = self.get_image_perm()
        self.lp_loss_func = lpips.LPIPS(net='vgg')
        self.size_align = size_align
        self.color_keys = color_keys

    def get_loss(self):
        img_idx = self.image_perm[self.iter_step % len(self.image_perm)]
        ms_idx = self.ms_idx
        batch_size = self.batch_sizes[ms_idx]
        if self.lp_patch_h[ms_idx] < 0:   # Full image
            assert self.lp_patch_w[ms_idx] < 0
            assert batch_size == 1
            patch_h = self.dataset.H // self.image_down_levels[ms_idx]
            patch_w = self.dataset.W // self.image_down_levels[ms_idx]
            patch_h = patch_h // self.size_align * self.size_align
            patch_w = patch_w // self.size_align * self.size_align
            coords, gt_colors = self.dataset.coords_data_of_camera(img_idx,
                                                                   down_level=self.image_down_levels[ms_idx])
        else:
            assert self.lp_patch_w[ms_idx] >= 0
            patch_h, patch_w = self.lp_patch_h[ms_idx], self.lp_patch_w[ms_idx]
            patch_h = np.min([patch_h, self.dataset.H // self.image_down_levels[ms_idx]])
            patch_w = np.min([patch_w, self.dataset.W // self.image_down_levels[ms_idx]])
            patch_h = patch_h // self.size_align * self.size_align
            patch_w = patch_w // self.size_align * self.size_align
            coords, gt_colors = self.dataset.rand_coords_data_patch_of_camera(img_idx,
                                                                              patch_h=patch_h,
                                                                              patch_w=patch_w,
                                                                              batch_size=batch_size,
                                                                              stride=self.image_down_levels[ms_idx])

        # BGR -> RGB
        gt_colors = torch.stack([gt_colors[:, 2], gt_colors[:, 1], gt_colors[:, 0]], dim=1)
        gt_img = gt_colors.reshape(batch_size, patch_h, patch_w, 3).permute(0, 3, 1, 2) * 2.0 - 1.0

        render_result = self.renderer.full_query(coords, H=patch_h, W=patch_w)
        loss = 0.
        loss_dict = dict()
        for color_key in self.color_keys:
            colors = render_result[color_key]
            # BGR -> RGB
            colors = torch.stack([colors[:, 2], colors[:, 1], colors[:, 0]], dim=1)
            pd_img = colors.reshape(batch_size, patch_h, patch_w, 3).permute(0, 3, 1, 2) * 2.0 - 1.0
            lp_loss = self.lp_loss_func(gt_img, pd_img).mean()
            loss_dict['{}_loss'.format(color_key)] = lp_loss
            loss = loss + lp_loss * self.lp_weight

        return loss, loss_dict

    def get_image_perm(self):
        image_list = [i for i in range(self.dataset.n_images) if i % 8 != 0]
        image_list = torch.tensor(image_list).cuda()
        image_list = image_list[torch.randperm(len(image_list))]
        return image_list

    def update_progress(self, iter_step):
        self.iter_step = iter_step
        if iter_step % len(self.image_perm) == 0:
            self.image_perm = self.get_image_perm()
        while self.ms_idx > 0 and self.mile_stones[self.ms_idx] > self.iter_step:
            self.ms_idx -= 1
        while self.ms_idx + 1 < len(self.mile_stones) and self.mile_stones[self.ms_idx + 1] <= self.iter_step:
            self.ms_idx += 1
