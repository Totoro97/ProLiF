import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@torch.no_grad()
def construct_uniform_tensor(shape, perturb=True):
    t_size = 1
    for i in shape:
        t_size *= i
    t = torch.linspace(0.5 / t_size, 1 - 0.5 / t_size, t_size)
    if perturb:
        t = t + (torch.rand(t_size) - 0.5) / t_size

    t = t[torch.randperm(t_size)]

    return t.reshape(*shape)


@torch.no_grad()
def construct_siren_weights(d_subfields,
                            d_in,
                            d_out,
                            d_hidden,
                            n_layers,
                            skips,
                            first_omega_0=30.,
                            omega_0=30.,
                            bias_mul=1.):
    weights = []
    biases = []
    # first layer
    fst_weight = (construct_uniform_tensor((d_subfields, d_hidden, d_in)) - 0.5) * 2.0 / d_in
    fst_bias = (construct_uniform_tensor((d_subfields, d_hidden)) - 0.5) * 2.0 / np.sqrt(d_in)
    weights.append(fst_weight)
    biases.append(fst_bias)

    # other layers
    for i in range(1, n_layers):
        if i in skips:
            cur_d_in = d_hidden + d_hidden
        else:
            cur_d_in = d_hidden
        if i + 1 == n_layers:
            cur_d_out = d_out
        else:
            cur_d_out = d_hidden

        weight = (construct_uniform_tensor((d_subfields, cur_d_out, cur_d_in)) - 0.5) *\
                 2.0 * -np.sqrt(6 / cur_d_in) / omega_0
        bias = (construct_uniform_tensor((d_subfields, cur_d_out)) - 0.5) * 2.0 / np.sqrt(cur_d_in) * bias_mul

        weights.append(weight)
        biases.append(bias)

    return weights, biases


@torch.no_grad()
def construct_relu_weights(d_subfields,
                           d_in,
                           d_out,
                           d_hidden,
                           n_layers,
                           skips):
    weights = []
    biases = []
    # first layer
    fst_weight = (construct_uniform_tensor((d_subfields, d_hidden, d_in)) - 0.5) * 2.0 / np.sqrt(d_in)
    fst_bias = (construct_uniform_tensor((d_subfields, d_hidden)) - 0.5) * 2.0 / np.sqrt(d_in)
    weights.append(fst_weight)
    biases.append(fst_bias)

    # other layers
    for i in range(1, n_layers):
        if i in skips:
            cur_d_in = d_hidden + d_hidden
        else:
            cur_d_in = d_hidden
        if i + 1 == n_layers:
            cur_d_out = d_out
        else:
            cur_d_out = d_hidden

        weight = (construct_uniform_tensor((d_subfields, cur_d_out, cur_d_in)) - 0.5) * 2.0 / np.sqrt(cur_d_in)
        bias = (construct_uniform_tensor((d_subfields, cur_d_out)) - 0.5) * 2.0 / np.sqrt(cur_d_in)

        weights.append(weight)
        biases.append(bias)

    return weights, biases


def construct_field_weights(act_type, weights_conf):
    if act_type == 'sine':
        return construct_siren_weights(**weights_conf)
    else:
        assert act_type in ['relu', 'softplus']
        return construct_relu_weights(**weights_conf)
