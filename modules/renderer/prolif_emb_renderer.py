import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from modules.embedder.embedder import PointEmbedder
from modules.fields.prolif import ProLiF, construct_cp_weights, construct_cp_weights_wn
from modules.fields.cons_weights import construct_field_weights


class ProLiFEmbRenderer(nn.Module):
    def __init__(self,
                 embedder_conf,
                 geo_weights_conf,
                 geo_field_conf,
                 app_weights_conf,
                 app_field_conf,
                 rgb_weights_conf,
                 rgb_field_conf,
                 n_embeddings,
                 embedding_dim):
        super(ProLiFEmbRenderer, self).__init__()
        self.embedder = PointEmbedder(**embedder_conf)
        # weights, biases = construct_siren_weights(**siren_conf)
        self.geo_weights_conf = geo_weights_conf
        self.geo_field_conf = geo_field_conf
        geo_weights, geo_bias = construct_field_weights(geo_field_conf['activation'], geo_weights_conf)
        self.geo_field = ProLiF(geo_weights, geo_bias, **geo_field_conf)

        self.app_weights_conf = app_weights_conf
        self.app_field_conf = app_field_conf
        app_weights, app_bias = construct_field_weights(app_field_conf['activation'], app_weights_conf)
        self.app_field = ProLiF(app_weights, app_bias, **app_field_conf)

        self.rgb_weights_conf = rgb_weights_conf
        self.rgb_field_conf = rgb_field_conf
        rgb_weights, rgb_bias = construct_field_weights(rgb_field_conf['activation'], rgb_weights_conf)
        self.rgb_field = ProLiF(rgb_weights, rgb_bias, **rgb_field_conf)

        self.n_embeddings = n_embeddings
        self.embeddings = nn.Embedding(n_embeddings, embedding_dim)

        assert geo_weights_conf['d_subfields'] == app_weights_conf['d_subfields'] == rgb_weights_conf['d_subfields']
        self.d_subfields = geo_weights_conf['d_subfields']
        self.init_n_samples = self.d_subfields
        self.scale = 1.0
        self.stage = 0

    def full_query(self, coords, perturb=True, steps=None, query_jac=False, app_idx=0, detach_geo=False, detach_app=False, **kwargs):
        beta = 10.   # WARN: Hard code
        B = coords.shape[0]
        embedding = None
        if self.embeddings is not None:
            embedding = self.embeddings(torch.LongTensor([app_idx]).cuda()).expand(B, -1)

        if query_jac:
            emb, steps, jac = self.embedder(coords, perturb=perturb, steps=steps, query_jac=query_jac)

            # rgba, jac = self.field.forward_with_jacobian(emb, jac, embedding=embedding)
            # rgba = rearrange(rgba, 'b (n c) -> b n c', c=4)
            # jac = rearrange(jac, 'b (n c) d -> b n c d', c=4, d=4)

            # rgb, density = rgba[:, :, :3], rgba[:, :, 3]
            # rgb = rearrange(rgb, 'b n c -> b (n c)', c=3)
            # rgb_jac = rearrange(jac[:, :, :3, :], 'b n c d -> b (n c) d')
            # density_jac = rearrange(jac[:, :, 3:4, :], 'b n c d -> b (n c) d')

            density, density_jac = self.geo_field.forward_with_jacobian(emb, jac, embedding=None)
            app_feats, app_feats_jac = self.app_field.forward_with_jacobian(emb, jac, embedding=None)
            if detach_geo:
                density = density.detach()
                density_jac = density_jac.detach()

            if detach_app:
                app_feats = app_feats.detach()
                app_feats_jac = app_feats_jac.detach()

            app_feats = rearrange(app_feats, 'b (n c) -> b n c', n=self.init_n_samples)
            app_feats_jac = rearrange(app_feats_jac, 'b (n c) d -> b n c d', n=self.init_n_samples)
            embedding = rearrange(embedding, 'b (n c) -> b n c', n=self.init_n_samples)
            app_feats = torch.cat([app_feats, embedding], dim=2)
            app_feats_jac = torch.cat([app_feats_jac, torch.zeros(list(embedding.shape) + [4])], dim=2)

            app_feats = rearrange(app_feats, 'b n c -> b (n c)')
            app_feats_jac = rearrange(app_feats_jac, 'b n c d -> b (n c) d')

            rgb, rgb_jac = self.rgb_field.forward_with_jacobian(app_feats, app_feats_jac, embedding=None)

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

            fake_alpha_jac = torch.exp(-density)[..., None] * density_jac
            fake_alpha = 1 - torch.exp(-density)

            weights_jac = alpha_jac * trans[..., None] + alpha[..., None] * trans_jac
            weights = alpha * trans

            depths_jac = (steps[..., None] * weights_jac).sum(dim=1) + -weights_jac.sum(dim=1)

            rgb = rearrange(rgb, 'b (n c) -> b n c', c=3)
            rgb_jac = rearrange(rgb_jac, 'b (n c) d -> b n c d', c=3)

            bg_color = torch.rand(B, 3)
            w_rgb_jac = rgb_jac * weights[:, :, None, None] + rgb[..., None] * weights_jac[:, :, None, :]
            colors_jac = w_rgb_jac.sum(dim=1) + bg_color[..., None] * -weights_jac.sum(dim=1)[:, None, :]
            colors = (rgb * weights[..., None]).sum(dim=1) + bg_color * (1 - weights.sum(dim=1))[:, None]

            rgb_fake_alpha_jac = rgb_jac * fake_alpha[:, :, None, None] + rgb[:, :, :, None] * fake_alpha_jac[:, :, None, :]
            rgb_density_jac = rgb_jac * density[:, :, None, None] + rgb[:, :, :, None] * density_jac[:, :, None, :]

            alpha = 1 - torch.exp(-density * self.scale)
        else:
            emb, steps = self.embedder(coords, perturb=perturb, steps=steps)
            app_feats = rearrange(self.app_field(emb), 'b (n c) -> b n c', n=self.init_n_samples)
            if detach_app:
                app_feats = app_feats.detach()
            embedding = rearrange(embedding, 'b (n c) -> b n c', n=self.init_n_samples)
            app_feats = torch.cat([app_feats, embedding], dim=2)
            app_feats = rearrange(app_feats, 'b n c -> b (n c)')

            rgb = rearrange(self.rgb_field(app_feats, embedding=None), 'b (n c) -> b n c', c=3)
            rgb = torch.sigmoid(rgb)
            density = self.geo_field(emb, embedding=None)
            if detach_geo:
                density = density.detach()

            density = F.softplus(density, beta=beta)

            trans = torch.exp(-torch.cumsum(density * self.scale, dim=-1))
            trans = torch.cat([torch.ones([B, 1]), trans], dim=-1)
            alpha = 1 - torch.exp(-density * self.scale)

            weights = alpha * trans[:, :-1]
            colors = (rgb * weights[:, :, None]).sum(dim=1, keepdim=False) + torch.rand(B, 3) * trans[:, -1:]
            trans = trans[:, :-1]

            rgb_jac = None
            fake_alpha = None
            fake_alpha_jac = None
            rgb_fake_alpha_jac = None
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
            'fake_alpha': fake_alpha,
            'density': density,
            'sampled_color': rgb,
            'rgb_jac': rgb_jac,
            'alpha_jac': alpha_jac,
            'fake_alpha_jac': fake_alpha_jac,
            'rgb_fake_alpha_jac': rgb_fake_alpha_jac,
            'colors_jac': colors_jac,
            'depths_jac': depths_jac,
            'density_jac': density_jac,
            'rgb_density_jac': rgb_density_jac,
            'steps': steps
        }

    def forward(self, coords, perturb=True, steps=None, app_idx=0, detach_geo=False, detach_app=False, **kwargs):  # idx is the app idx
        return self.full_query(coords, perturb=perturb, steps=steps, app_idx=app_idx, detach_geo=detach_geo, detach_app=detach_app)['colors']

    def merge(self, sub_div_inputs=True, sub_div_outputs=True):
        assert self.geo_field.weight_norm and self.app_field.weight_norm and self.rgb_field.weight_norm
        # geo field
        weights, biases, weights_g = self.geo_field.weights_biases()
        new_weights, new_biases, new_weights_g = construct_cp_weights_wn(weights,
                                                                         biases,
                                                                         weights_g,
                                                                         skips=self.geo_field_conf['skips'],
                                                                         sub_div_inputs=sub_div_inputs,
                                                                         sub_div_outputs=sub_div_outputs,
                                                                         input_coord_dim=self.geo_weights_conf['d_in'],
                                                                         output_coord_dim=self.geo_weights_conf['d_out']
                                                                         )
        self.geo_field = ProLiF(new_weights, new_biases, weights_g=new_weights_g, **self.geo_field_conf)

        # app field
        weights, biases, weights_g = self.app_field.weights_biases()
        new_weights, new_biases, new_weights_g = construct_cp_weights_wn(weights,
                                                                         biases,
                                                                         weights_g,
                                                                         skips=self.app_field_conf['skips'],
                                                                         sub_div_inputs=sub_div_inputs,
                                                                         sub_div_outputs=False,
                                                                         input_coord_dim=self.app_weights_conf['d_in'],
                                                                         output_coord_dim=self.app_weights_conf['d_out'])
        self.app_field = ProLiF(new_weights, new_biases, weights_g=new_weights_g, **self.app_field_conf)

        # rgb field
        weights, biases, weights_g = self.rgb_field.weights_biases()
        new_weights, new_biases, new_weights_g = construct_cp_weights_wn(weights,
                                                                         biases,
                                                                         weights_g,
                                                                         skips=self.rgb_field_conf['skips'],
                                                                         sub_div_inputs=False,
                                                                         sub_div_outputs=sub_div_outputs,
                                                                         input_coord_dim=self.rgb_weights_conf['d_in'],
                                                                         output_coord_dim=self.rgb_weights_conf['d_out'])
        self.rgb_field = ProLiF(new_weights, new_biases, weights_g=new_weights_g, **self.rgb_field_conf)
        self.stage += 1

        if sub_div_outputs:
            self.embedder.update_n_samples(self.embedder.n_samples * 2)
            self.scale = self.scale * 0.5
