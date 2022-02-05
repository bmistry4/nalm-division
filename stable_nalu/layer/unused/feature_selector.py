import torch.nn as nn
import torch
import math
import numpy as np

"""
Ack:
Paper: https://arxiv.org/pdf/1810.04247.pdf (Feature Selection using Stochastic Gates)
Code: https://github.com/runopti/stg/blob/master/python/stg/layers.py
"""
class FeatureSelector(nn.Module):
    def __init__(self, input_dim, output_dim, sigma, device):
        super(FeatureSelector, self).__init__()
        self.mu = torch.nn.Parameter(0.01 * torch.randn(input_dim, output_dim), requires_grad=True) # TODO: shouldn't mu have a 0.5 init?
        self.noise = torch.randn(self.mu.size())
        self.sigma = sigma
        self.device = device

    def forward(self, prev_x):
        z = self.mu + self.sigma * self.noise.normal_() * self.training
        stochastic_gate = self.hard_sigmoid(z)
        new_x = prev_x * stochastic_gate
        return new_x

    def hard_sigmoid(self, x):
        return torch.clamp(x + 0.5, 0.0, 1.0)

    def regularizer(self, x):
        ''' Gaussian CDF. '''
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))  # TODO: How?

    def _apply(self, fn):
        super(FeatureSelector, self)._apply(fn)
        self.noise = fn(self.noise)
        return self

    def get_gates(self, mode):
        if mode == 'raw':
            return self.mu.detach().cpu().numpy()
        elif mode == 'prob':
            return np.minimum(1.0, np.maximum(0.0, self.mu.detach().cpu().numpy() + 0.5))
        else:
            raise NotImplementedError()
