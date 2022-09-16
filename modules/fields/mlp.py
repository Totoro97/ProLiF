import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self,
                 dims,
                 skips=(),
                 activation='relu',
                 weight_norm=True,
                 last_act=False):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList()
        self.n_dims = len(dims)
        self.skips = skips
        self.last_act = last_act
        for i in range(self.n_dims - 1):
            if i in self.skips:
                d_in = dims[i] + dims[0]
            else:
                d_in = dims[i]
            d_out = dims[i + 1]
            lin = nn.Linear(d_in, d_out)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            self.linears.append(lin)
        assert activation == 'relu'
        self.act = nn.ReLU()

    def forward(self, x):
        y = x
        for i in range(self.n_dims - 2):
            if i in self.skips:
                y = torch.cat([y, x], dim=-1)
            y = self.linears[i](y)
            y = self.act(y)

        if (self.n_dims - 2) in self.skips:
            y = torch.cat([y, x], dim=-1)

        y = self.linears[-1](y)
        if self.last_act:
            return self.act(y)
        else:
            return y

