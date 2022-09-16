import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from modules.embedder.embedder import PointEmbedder
from modules.fields.prolif import ProLiF, construct_cp_weights, construct_cp_weights_wn
from modules.fields.cons_weights import construct_siren_weights


# This function is for debugging
def query_gradient(x, y):
    d_output = torch.ones_like(y, requires_grad=False, device=y.device)
    gradients = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=d_output,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return gradients


class ProLiFRenderer(nn.Module):
    def __init__(self,
                 embedder_conf,
                 weight_conf,
                 field_conf):
        super(ProLiFRenderer, self).__init__()
        self.embedder = PointEmbedder(**embedder_conf)
        weights, biases = construct_siren_weights(**weight_conf)
        self.field_conf = field_conf
        self.field = ProLiF(weights, biases, **field_conf)

        self.scale = 1.0
        self.stage = 0

    def full_query(self, coords, perturb=True, steps=None, query_jac=False, idx=0, **kwargs):
        beta = 10.   # WARN: Hard code
        B = coords.shape[0]
        embedding = None

        if query_jac:
            emb, steps, jac = self.embedder(coords, perturb=perturb, steps=steps, query_jac=query_jac)
            rgba, jac = self.field.forward_with_jacobian(emb, jac, embedding=embedding)
            rgba = rearrange(rgba, 'b (n c) -> b n c', c=4)
            jac = rearrange(jac, 'b (n c) d -> b n c d', c=4, d=4)

            rgb, density = rgba[:, :, :3], rgba[:, :, 3]
            rgb = rearrange(rgb, 'b n c -> b (n c)', c=3)
            rgb_jac = rearrange(jac[:, :, :3, :], 'b n c d -> b (n c) d')
            density_jac = rearrange(jac[:, :, 3:4, :], 'b n c d -> b (n c) d')

            rgb_jac = (torch.sigmoid(rgb) * torch.sigmoid(-rgb))[..., None] * rgb_jac
            rgb = torch.sigmoid(rgb)

            density_jac = torch.sigmoid(density * beta)[..., None] * density_jac
            density = F.softplus(density, beta=beta)

            cum_density_jac = torch.cumsum(density_jac, dim=1)
            cum_density = torch.cumsum(density, dim=1)

            cum_density_jac = torch.cat([torch.zeros([B, 1, 4]), cum_density_jac], dim=1)[:, :-1, :]
            cum_density = torch.cat([torch.zeros([B, 1]), cum_density], dim=1)[:, :-1]

            trans_jac = (-self.scale * torch.exp(-self.scale * cum_density))[..., None] * cum_density_jac
            trans = torch.exp(-self.scale * cum_density)

            alpha_jac = (self.scale * torch.exp(-density * self.scale))[..., None] * density_jac
            alpha = 1 - torch.exp(-self.scale * density)

            weights_jac = alpha_jac * trans[..., None] + alpha[..., None] * trans_jac
            weights = alpha * trans

            depths_jac = (steps[..., None] * weights_jac).sum(dim=1) + -weights_jac.sum(dim=1)

            rgb = rearrange(rgb, 'b (n c) -> b n c', c=3)
            rgb_jac = rearrange(rgb_jac, 'b (n c) d -> b n c d', c=3)

            bg_color = torch.rand(B, 3)
            w_rgb_jac = rgb_jac * weights[:, :, None, None] + rgb[..., None] * weights_jac[:, :, None, :]
            colors_jac = w_rgb_jac.sum(dim=1) + bg_color[..., None] * -weights_jac.sum(dim=1)[:, None, :]
            colors = (rgb * weights[..., None]).sum(dim=1) + bg_color * (1 - weights.sum(dim=1))[:, None]

            rgb_density_jac = rgb_jac * density[:, :, None, None] + rgb[:, :, :, None] * density_jac[:, :, None, :]

            alpha = 1 - torch.exp(-density * self.scale)
        else:
            emb, steps = self.embedder(coords, perturb=perturb, steps=steps)
            rgba = rearrange(self.field(emb, embedding=embedding), 'b (n c) -> b n c', c=4)
            rgb = torch.sigmoid(rgba[:, :, :3])
            density = F.softplus(rgba[:, :, 3], beta=beta)
            trans = torch.exp(-torch.cumsum(density * self.scale, dim=-1))
            trans = torch.cat([torch.ones([B, 1]), trans], dim=-1)
            alpha = 1 - torch.exp(-density * self.scale)

            weights = alpha * trans[:, :-1]
            colors = (rgb * weights[:, :, None]).sum(dim=1, keepdim=False) + torch.rand(B, 3) * trans[:, -1:]
            trans = trans[:, :-1]

            rgb_jac = None
            colors_jac = None
            depths_jac = None
            density_jac = None
            rgb_density_jac = None
            alpha_jac = None

        depths = (steps * weights).sum(dim=1, keepdim=True) + (1 - weights.sum(dim=-1, keepdim=True))
        return {
            'colors': colors,
            'depths': depths,
            'weights': weights,
            'trans': trans,
            'alpha': alpha,
            'density': density,
            'sampled_color': rgb,
            'rgb_jac': rgb_jac,
            'alpha_jac': alpha_jac,
            'colors_jac': colors_jac,
            'depths_jac': depths_jac,
            'density_jac': density_jac,
            'rgb_density_jac': rgb_density_jac,
            'steps': steps
        }

    def forward(self, coords, perturb=True, steps=None, idx=0, **kwargs):
        return self.full_query(coords, perturb=perturb, steps=steps, idx=idx)['colors']

    def merge(self, sub_div_inputs=True, sub_div_outputs=True):
        if not self.field.weight_norm:
            weights, biases = self.field.weights_biases()
            new_weights, new_biases = construct_cp_weights(weights,
                                                           biases,
                                                           skips=self.field_conf['skips'],
                                                           sub_div_inputs=sub_div_inputs,
                                                           sub_div_outputs=sub_div_outputs)
            self.field = ProLiF(new_weights, new_biases, **self.field_conf)
        else:
            weights, biases, weights_g = self.field.weights_biases()
            new_weights, new_biases, new_weights_g = construct_cp_weights_wn(weights,
                                                                             biases,
                                                                             weights_g,
                                                                             skips=self.field_conf['skips'],
                                                                             sub_div_inputs=sub_div_inputs,
                                                                             sub_div_outputs=sub_div_outputs)
            self.field = ProLiF(new_weights, new_biases, weights_g=new_weights_g, **self.field_conf)

        self.stage += 1

        if sub_div_outputs:
            self.embedder.update_n_samples(self.embedder.n_samples * 2)
            self.scale = self.scale * 0.5
