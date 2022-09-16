import torch
import torch.nn as nn
import numpy as np
import clip


class CLIPLoss(nn.Module):
    def __init__(self, device='cuda', text='yellow flowers', patch_size=224):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load('ViT-B/32', device=device)
        self.text = clip.tokenize([text]).to(device)
        self.pooling = torch.nn.AdaptiveAvgPool2d((patch_size, patch_size))
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    def forward(self, image):
        # Input image is RGB of [0, 1]
        image = image.sub(self.mean).div(self.std)
        image = self.pooling(image)
        logits_per_image, _ = self.model(image, self.text)

        return 1 - logits_per_image / 100.


class ClipSupervisor:
    def __init__(self,
                 dataset,
                 renderer,
                 clip_weight,
                 psize,
                 clip_psize,
                 text,
                 extra_query_conf=None):
        self.dataset = dataset
        self.renderer = renderer
        self.psize = psize
        self.clip_weight = clip_weight

        if clip_psize < 0:
            self.clip_psize = self.psize
        else:
            self.clip_psize  = clip_psize
        # self.clip_psize = 2 ** int(np.log2(psize + 1e-5))
        self.clip_loss = CLIPLoss(text=text, patch_size=self.clip_psize)
        self.iter_step = 0

        if extra_query_conf is None:
            self.extra_query_conf = dict()
        else:
            self.extra_query_conf = extra_query_conf

    def get_loss(self):
        coords = self.dataset.rand_coords_patch_from_rand_pose(self.clip_psize,
                                                               self.clip_psize,
                                                               stride=self.psize / self.clip_psize)
        render_result = self.renderer.full_query(coords, **self.extra_query_conf)
        patch = render_result['colors'].reshape(1, self.clip_psize, self.clip_psize, 3).permute(0, 3, 1, 2)
        patch = torch.stack([patch[:, 2], patch[:, 1], patch[:, 0]], dim=1)   # BGR -> RGB
        clip_loss = self.clip_loss(patch)

        loss = clip_loss * self.clip_weight
        loss_dict = {
            'clip_loss': clip_loss,
        }

        return loss, loss_dict

    def update_progress(self, iter_step):
        self.iter_step = iter_step
