import torch
import torch.nn as nn
import torch.nn.functional as F

# A simple up-sampler.
class UpsampleNN(nn.Module):
    def __init__(self,
                 dims,
                 factors):
        super(UpsampleNN, self).__init__()
        self.convs = nn.ModuleList()
        self.samplers = nn.ModuleList()
        self.n_layers = len(dims) - 1
        for i in range(self.n_layers):
            self.convs.append(nn.Conv2d(dims[i], dims[i+1], kernel_size=(1, 1)))
        for i in range(self.n_layers):
            self.samplers.append(nn.Upsample(scale_factor=factors[i], mode='bilinear'))

    def forward(self, x, use_upsampler=True):
        for i in range(self.n_layers):
            if use_upsampler:
                x = self.samplers[i](x)
            x = self.convs[i](x)
            if i + 1 < self.n_layers:
                x = F.relu(x)

        return x
