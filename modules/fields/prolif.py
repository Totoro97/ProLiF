import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.fields.cons_weights import construct_siren_weights, construct_relu_weights
from einops import rearrange
from copy import deepcopy


def siren_acts(first_omega_0, omega_0):
    assert first_omega_0 == omega_0
    act = lambda x: torch.sin(x * omega_0)
    d_act = lambda x: omega_0 * torch.cos(x * omega_0)
    return act, d_act


def softplus_acts(beta):
    act = lambda x: F.softplus(x, beta=beta)
    d_act = lambda x: torch.sigmoid(x * beta)
    return act, d_act


class SubProLiF(nn.Module):
    def __init__(self,
                 weights,  # n_layers x [d_out, d_in]
                 biases,   # n_layers x [d_out]
                 weight_norm,
                 activation,
                 act_conf,
                 weights_g=None,  # n_layers x [d_out, 1]
                 skips=()):
        super(SubProLiF, self).__init__()
        # Parameters
        self.n_layers = len(weights)
        self.layers = nn.ModuleList()
        self.skips = skips

        # Build layers
        for i in range(self.n_layers):
            weight = weights[i]
            bias = biases[i]
            layer = nn.Linear(weight.shape[1], weight.shape[0])

            with torch.no_grad():
                layer.weight.data = weight.clone()
                layer.bias.data = bias.clone()

            if weight_norm:
                layer = nn.utils.weight_norm(layer)
                if weights_g is not None:
                    with torch.no_grad():
                        layer.weight_v.data = weight.clone()
                        layer.bias.data = bias.clone()
                        layer.weight_g.data = weights_g[i].clone()

            self.layers.append(layer)

        # Activation
        if activation == 'sine':
            self.act, self.d_act = siren_acts(**act_conf)
        else:
            assert activation == 'softplus'
            self.act, self.d_act = softplus_acts(**act_conf)

    def forward(self, emb, embedding=None):
        after_fst = self.layers[0](emb)
        after_fst = self.act(after_fst)
        x = after_fst
        for i in range(1, self.n_layers - 1):
            if i in self.skips:
                x = torch.cat([after_fst, x], dim=-1)
            x = self.layers[i](x)
            x = self.act(x)

        x = self.layers[-1](x)
        return x

    def forward_with_jacobian(self, emb, jac, embedding=None):
        after_fst = self.layers[0](emb)
        after_fst_jac = torch.matmul(self.layers[0].weight, jac)
        after_act = self.act(after_fst)
        after_act_jac = self.d_act(after_fst)[..., None] * after_fst_jac

        after_fst_act, after_fst_act_jac = after_act, after_act_jac

        for i in range(1, self.n_layers - 1):
            if i in self.skips:
                after_act = torch.cat([after_fst_act, after_act], dim=1)
                after_act_jac = torch.cat([after_fst_act_jac, after_act_jac], dim=1)
            after_lin = self.layers[i](after_act)
            after_lin_jac = torch.matmul(self.layers[i].weight, after_act_jac)
            after_act = self.act(after_lin)
            after_act_jac = self.d_act(after_lin)[..., None] * after_lin_jac

        after_lin = self.layers[-1](after_act)
        after_lin_jac = torch.matmul(self.layers[-1].weight, after_act_jac)

        return after_lin, after_lin_jac


class ProLiF(nn.Module):
    def __init__(self,
                 weights,  # list: n_layers x [n_subfields, d_out, d_in]
                 biases,   # list: n_layers x [n_subfields, d_out]
                 weight_norm,
                 d_hidden,
                 activation,
                 act_conf,
                 weights_g = None,   # list: n_layers x [n_subfields, d_out, 1]
                 skips=(),
                 ):
        super(ProLiF, self).__init__()
        self.n_layers = len(weights)
        self.n_sub_fields = weights[0].shape[0]
        self.sub_fields = nn.ModuleList()
        self.weight_norm = weight_norm
        for i in range(self.n_sub_fields):
            cur_weights = [ weight[i] for weight in weights ]
            cur_biases =  [ bias[i] for bias in biases ]
            if weights_g is not None:
                cur_weights_g = [ weight_g[i] for weight_g in weights_g ]
            else:
                cur_weights_g = None
            self.sub_fields.append(SubProLiF(cur_weights,
                                             cur_biases,
                                             weights_g=cur_weights_g,
                                             weight_norm=weight_norm,
                                             activation=activation,
                                             act_conf=act_conf,
                                             skips=skips))

    def forward(self, emb, embedding=None):
        x = rearrange(emb, 'b (n c) -> b n c', n=self.n_sub_fields)
        out = []
        if embedding is not None:
            embedding = rearrange(embedding, 'b (n c) -> b n c', n=self.n_sub_fields)
            for i in range(self.n_sub_fields):
                out.append(self.sub_fields[i](x[:, i, :], embedding=embedding[:, i, :]))
        else:
            for i in range(self.n_sub_fields):
                out.append(self.sub_fields[i](x[:, i, :], embedding=None))

        return torch.cat(out, dim=-1)

    def forward_with_jacobian(self, emb, jac, embedding=None):
        x = rearrange(emb, 'b (n c) -> b n c', n=self.n_sub_fields)
        jac = rearrange(jac, 'b (n c) d -> b n c d', n=self.n_sub_fields, d=4)
        outs = []
        outs_jac = []

        if embedding is not None:
            embedding = rearrange(embedding, 'b (n c) -> b n c', n=self.n_sub_fields)
            for i in range(self.n_sub_fields):
                out, out_jac =\
                    self.sub_fields[i].forward_with_jacobian(x[:, i, :], jac[:, i, :, :], embedding=embedding[:, i, :])
                outs.append(out)
                outs_jac.append(out_jac)
        else:
            for i in range(self.n_sub_fields):
                out, out_jac = self.sub_fields[i].forward_with_jacobian(x[:, i, :], jac[:, i, :, :], embedding=None)
                outs.append(out)
                outs_jac.append(out_jac)

        return torch.cat(outs, dim=1), torch.cat(outs_jac, dim=1)

    def weights_biases(self):
        if not self.weight_norm:
            weights = []
            biases = []
            with torch.no_grad():
                for i in range(self.n_layers):
                    cur_weights = [self.sub_fields[j].layers[i].weight.data.clone() for j in range(self.n_sub_fields)]
                    cur_biases =  [self.sub_fields[j].layers[i].bias.data.clone() for j in range(self.n_sub_fields)]

                    weights.append(torch.stack(cur_weights, dim=0))
                    biases.append(torch.stack(cur_biases, dim=0))

            return weights, biases
        else:
            weights = []
            biases = []
            weights_g = []
            with torch.no_grad():
                for i in range(self.n_layers):
                    cur_weights = [self.sub_fields[j].layers[i].weight_v.data.clone() for j in range(self.n_sub_fields)]
                    cur_biases =  [self.sub_fields[j].layers[i].bias.data.clone() for j in range(self.n_sub_fields)]
                    cur_weights_g = [self.sub_fields[j].layers[i].weight_g.data.clone() for j in range(self.n_sub_fields)]

                    weights.append(torch.stack(cur_weights, dim=0))
                    biases.append(torch.stack(cur_biases, dim=0))
                    weights_g.append(torch.stack(cur_weights_g, dim=0))

            return weights, biases, weights_g


# ----------------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def construct_sub_div_input_weights(org_weight,    # n_subfields, d_out, d_in
                                    org_bias,      # n_subfields, d_out
                                    org_weight_g=None,  # n_subfields, d_out, 1  (if use weight_norm)
                                    coord_dim=2    # X, Y
                                    ):
    bias = org_bias.clone()
    if org_weight_g is None:
        weight = rearrange(org_weight, 'n o (s c) -> n o s c', c=coord_dim).clone()
        weight = torch.cat([weight, weight], dim=-1) * 0.5
        weight = rearrange(weight, 'n o s c -> n o (s c)')
        return weight, bias
    else:
        weight = rearrange(org_weight, 'n o (s c) -> n o s c', c=coord_dim).clone()
        weight = torch.cat([weight, weight], dim=-1)
        weight = rearrange(weight, 'n o s c -> n o (s c)')
        weight_g = org_weight_g.clone() / np.sqrt(2)
        return weight, bias, weight_g


@torch.no_grad()
def construct_sub_div_output_weights(org_weight,          # n_subfields, d_out, d_in
                                     org_bias,            # n_subfields, d_out
                                     org_weight_g=None,   # n_subfields, d_out, 1
                                     coord_dim=4  # RGBA
                                     ):
    bias = rearrange(org_bias, 'n (s c) -> n s c', c=coord_dim).clone()
    bias = torch.cat([bias, bias], dim=2)
    bias = rearrange(bias, 'n s c -> n (s c)')

    weight = rearrange(org_weight, 'n (s c) i -> n s c i', c=coord_dim).clone()
    weight = torch.cat([weight, weight], dim=2)
    weight = rearrange(weight, 'n s c i -> n (s c) i')

    if org_weight_g is None:
        return weight, bias
    else:
        weight_g = rearrange(org_weight_g, 'n (s c) 1 -> n s c 1', c=coord_dim).clone()
        weight_g = torch.cat([weight_g, weight_g], dim=2)
        weight_g = rearrange(weight_g, 'n s c 1 -> n (s c) 1')
        return weight, bias, weight_g


@torch.no_grad()
def construct_merge_weights(org_weight,
                            org_bias,
                            org_weight_g=None,
                            fill_val=0.):
    n_subfields, d_out, d_in = org_weight.shape
    weight = torch.ones([n_subfields // 2, d_out * 2, d_in * 2]) * fill_val
    weight[:, :d_out, :d_in] = org_weight[0::2, :, :].clone()
    weight[:, d_out:, d_in:] = org_weight[1::2, :, :].clone()

    bias = rearrange(org_bias, '(n c) o -> n (c o)', c=2).clone()

    if org_weight_g is None:
        return weight, bias
    else:
        weight_g = rearrange(org_weight_g, '(n c) o 1 -> n (c o) 1', c=2).clone()
        return weight, bias, weight_g


@torch.no_grad()
def construct_merge_weights_skip(org_weight,
                                 org_bias,
                                 skip_dim,
                                 org_weight_g = None,
                                 fill_val = 0.):
    n_subfields, d_out, d_in = org_weight.shape
    weight = torch.ones([n_subfields // 2, d_out * 2, d_in * 2]) * fill_val
    weight[:, :d_out, :skip_dim]              = org_weight[0::2, :, :skip_dim].clone()
    weight[:, d_out:, skip_dim: skip_dim * 2] = org_weight[1::2, :, :skip_dim].clone()

    weight[:, :d_out, skip_dim * 2: skip_dim + d_in] = org_weight[0::2, :, skip_dim:].clone()
    weight[:, d_out:, skip_dim + d_in: d_in * 2]     = org_weight[1::2, :, skip_dim:].clone()

    bias = rearrange(org_bias, '(n c) o -> n (c o)', c=2).clone()

    if org_weight_g is None:
        return weight, bias
    else:
        weight_g = rearrange(org_weight_g, '(n c) o 1 -> n (c o) 1', c=2).clone()
        return weight, bias, weight_g


@torch.no_grad()
def construct_cp_weights(org_weights,
                         org_biases,
                         skips=(),
                         sub_div_inputs=True,
                         sub_div_outputs=True):
    # Not thoroughly testet, may be buggy
    weights = []
    biases = []
    n_layers = len(org_weights)
    # first layer:
    fst_weight, fst_bias = construct_merge_weights(org_weights[0], org_biases[0])
    if sub_div_inputs:
        fst_weight, fst_bias = construct_sub_div_input_weights(fst_weight, fst_bias)

    weights.append(fst_weight)
    biases.append(fst_bias)

    # hidden layers:
    for i in range(1, n_layers - 1):
        if i in skips:
            weight, bias = construct_merge_weights_skip(org_weights[i], org_biases[i], skip_dim=org_weights[0].shape[1])
        else:
            weight, bias = construct_merge_weights(org_weights[i], org_biases[i])
        weights.append(weight)
        biases.append(bias)

    lst_weight, lst_bias = construct_merge_weights(org_weights[-1], org_biases[-1])
    if sub_div_outputs:
        lst_weight, lst_bias = construct_sub_div_output_weights(lst_weight, lst_bias)
    weights.append(lst_weight)
    biases.append(lst_bias)

    return weights, biases


@torch.no_grad()
def construct_cp_weights_wn(org_weights,
                            org_biases,
                            org_weights_g,
                            skips=(),
                            sub_div_inputs=True,
                            sub_div_outputs=True,
                            input_coord_dim=2,
                            output_coord_dim=4):
    weights = []
    biases = []
    weights_g = []
    n_layers = len(org_weights)

    # first layer:
    fst_weight, fst_bias, fst_weight_g =\
        construct_merge_weights(org_weights[0], org_biases[0], org_weight_g=org_weights_g[0])
    if sub_div_inputs:
        fst_weight, fst_bias, fst_weight_g =\
            construct_sub_div_input_weights(fst_weight, fst_bias, fst_weight_g, coord_dim=input_coord_dim)

    weights.append(fst_weight)
    biases.append(fst_bias)
    weights_g.append(fst_weight_g)

    # hidden layers:
    for i in range(1, n_layers - 1):
        if i in skips:
            weight, bias, weight_g = construct_merge_weights_skip(org_weights[i],
                                                                  org_biases[i],
                                                                  org_weight_g=org_weights_g[i],
                                                                  skip_dim=org_weights[0].shape[1])
        else:
            weight, bias, weight_g = construct_merge_weights(org_weights[i],
                                                             org_biases[i],
                                                             org_weight_g=org_weights_g[i])
        weights.append(weight)
        biases.append(bias)
        weights_g.append(weight_g)

    # last layer
    lst_weight, lst_bias, lst_weight_g =\
        construct_merge_weights(org_weights[-1], org_biases[-1], org_weight_g=org_weights_g[-1])
    if sub_div_outputs:
        lst_weight, lst_bias, lst_weight_g =\
            construct_sub_div_output_weights(lst_weight, lst_bias, org_weight_g=lst_weight_g, coord_dim=output_coord_dim)
    weights.append(lst_weight)
    biases.append(lst_bias)
    weights_g.append(lst_weight_g)

    return weights, biases, weights_g


@torch.no_grad()
def construct_adam_state_dict_wn_single(org_state_dict,      # Consider weight normalization
                                        org_n_sub_fields,
                                        n_layers,
                                        skips,
                                        skip_dim,
                                        sub_div_inputs=True,
                                        sub_div_outputs=True,
                                        input_coord_dim=2,
                                        output_coord_dim=4,
                                        idx=0,
                                        new_idx=0):
    state = {}

    for i in range(0, org_n_sub_fields, 2):
        for j in range(0, n_layers):
            state[new_idx] = dict()      # bias
            state[new_idx + 1] = dict()  # weight_g
            state[new_idx + 2] = dict()  # weight_v
            for key in ['exp_avg', 'exp_avg_sq']:
                org_bias = torch.stack(
                    [org_state_dict['state'][idx][key], org_state_dict['state'][idx + 3 * n_layers][key]], dim=0)
                org_weight_g = torch.stack(
                    [org_state_dict['state'][idx + 1][key], org_state_dict['state'][idx + 1 + 3 * n_layers][key]],
                    dim=0)
                org_weight_v = torch.stack(
                    [org_state_dict['state'][idx + 2][key], org_state_dict['state'][idx + 2 + 3 * n_layers][key]],
                    dim=0)
                if key == 'exp_avg':
                    fill_val = 0.
                else:
                    # A trick to suppress the learning step of new weights in the beggining training iterations
                    fill_val = 1.0 + org_weight_v.mean() * float(org_weight_v.shape[1])**2

                if j in skips:
                    weight, bias, weight_g = construct_merge_weights_skip(org_weight_v,
                                                                          org_bias,
                                                                          org_weight_g=org_weight_g,
                                                                          skip_dim=skip_dim,
                                                                          fill_val=fill_val)
                else:
                    weight, bias, weight_g = construct_merge_weights(org_weight_v,
                                                                     org_bias,
                                                                     org_weight_g=org_weight_g,
                                                                     fill_val=fill_val)

                if j == 0:
                    # first layer
                    if sub_div_inputs:
                        weight, bias, weight_g = construct_sub_div_input_weights(weight,
                                                                                 bias,
                                                                                 org_weight_g=weight_g,
                                                                                 coord_dim=input_coord_dim)
                        scale = 1 / np.sqrt(2) if key == 'exp_avg' else 0.5
                    else:
                        scale = 1
                    state[new_idx][key] = bias[0]
                    state[new_idx + 1][key] = weight_g[0] * scale
                    state[new_idx + 2][key] = weight[0] * scale
                elif j + 1 == n_layers:
                    # last layer
                    if sub_div_outputs:
                        weight, bias, weight_g = construct_sub_div_output_weights(weight,
                                                                                  bias,
                                                                                  org_weight_g=weight_g,
                                                                                  coord_dim=output_coord_dim)

                    state[new_idx][key] = bias[0]
                    state[new_idx + 1][key] = weight_g[0]
                    state[new_idx + 2][key] = weight[0]
                else:
                    # hidden layers
                    state[new_idx][key] = bias[0]
                    state[new_idx + 1][key] = weight_g[0]
                    state[new_idx + 2][key] = weight[0]

            state[new_idx]['step'] =\
                state[new_idx + 1]['step'] = state[new_idx + 2]['step'] = org_state_dict['state'][idx]['step']
            idx += 3
            new_idx += 3

        idx += 3 * n_layers

    return state, idx, new_idx


def construct_adam_state_dict_wn(org_state_dict, field_confs):
    state = dict()
    idx, new_idx = 0, 0
    for field_conf in field_confs:
        cur_state, idx, new_idx =\
            construct_adam_state_dict_wn_single(org_state_dict, **field_conf, idx=idx, new_idx=new_idx)
        state.update(cur_state)

    while idx != len(org_state_dict['state']):   # Means there is extra embedding parameters here
        state[new_idx] = deepcopy(org_state_dict['state'][idx])
        new_idx += 1
        idx += 1

    param_groups = deepcopy(org_state_dict['param_groups'])
    param_groups[0]['params'] = [ k for k in range(new_idx) ]
    ret = { 'state': state, 'param_groups': param_groups }

    for key in org_state_dict:
        if key not in ret:
            ret[key] = deepcopy(org_state_dict[key])

    return ret

