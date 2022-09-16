import torch
import torch.nn as nn


class PixelSupervisor:
    def __init__(self,
                 dataset,
                 renderer,
                 color_weight,
                 gray_weight,
                 batch_size,
                 use_all_images=False,
                 extra_query_conf=None):
        self.dataset = dataset
        self.renderer = renderer
        self.color_weight = color_weight
        self.gray_weight = gray_weight
        self.batch_size = batch_size
        self.iter_step = 0
        self.use_all_images = use_all_images
        self.image_perm = self.get_image_perm()

        if extra_query_conf is None:
            self.extra_query_conf = dict()
        else:
            self.extra_query_conf = extra_query_conf

    def get_loss(self):
        img_idx = self.image_perm[self.iter_step % len(self.image_perm)]
        coords, gt_colors = self.dataset.rand_coords_data_of_camera(img_idx, self.batch_size)
        gt_colors = gt_colors.cuda()
        colors = self.renderer(coords, **self.extra_query_conf)

        color_loss = 0.
        psnr = 100.
        if self.color_weight > 1e-5:
            color_loss = ((gt_colors - colors) ** 2).mean()
            psnr = 20.0 * torch.log10(1.0 / (((gt_colors - colors) ** 2).mean()).sqrt())

        gray_loss = 0.
        if self.gray_weight > 1e-5:
            gt_gray = 0.114 * gt_colors[:, 0] + 0.587 * gt_colors[:, 1] + 0.299 * gt_colors[:, 2]
            pred_gray = 0.114 * colors[:, 0] + 0.587 * colors[:, 1] + 0.299 * colors[:, 2]
            gray_loss = ((gt_gray - pred_gray) ** 2).mean()

        loss = color_loss * self.color_weight + gray_loss * self.gray_weight


        loss_dict = {
            'color_loss': color_loss,
            'gray_loss': gray_loss,
            'psnr': psnr
        }

        return loss, loss_dict

    def get_image_perm(self):
        if self.use_all_images:
            image_list = [ i for i in range(self.dataset.n_images) ]
        else:
            # train/test split of LLFF dataset
            image_list = [i for i in range(self.dataset.n_images) if i % 8 != 0]
        image_list = torch.tensor(image_list).cuda()
        image_list = image_list[torch.randperm(len(image_list))]
        return image_list

    def update_progress(self, iter_step):
        self.iter_step = iter_step
        if iter_step % len(self.image_perm) == 0:
            self.image_perm = self.get_image_perm()
