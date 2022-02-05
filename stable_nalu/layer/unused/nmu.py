import scipy.optimize
import numpy as np
import torch
import math

from ..abstract import ExtendedTorchModule
from ..functional import mnac, Regualizer, RegualizerNMUZ, sparsity_error
from ._abstract_recurrent_cell import AbstractRecurrentCell

"""
NOT USED re_regualized_linear_mnac.py copy (containing additional code which didn't work out in the end) 
Set up for sNMU (if flag is set). Needs to be in folder 1 level above to work.
Got commented out for the pass-through versions of the nmu at different levels
- W, and out before noise
"""
class ReRegualizedLinearMNACLayer(ExtendedTorchModule):
    """Implements the NAC (Neural Accumulator)

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """

    def __init__(self, in_features, out_features,
                 nac_oob='regualized', regualizer_shape='squared',
                 mnac_epsilon=0, mnac_normalized=False, regualizer_z=0,
                 **kwargs):
        super().__init__('nac', **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.mnac_normalized = mnac_normalized
        self.mnac_epsilon = mnac_epsilon
        self.nac_oob = nac_oob

        self._regualizer_bias = Regualizer(
            support='mnac', type='bias',
            shape=regualizer_shape, zero_epsilon=mnac_epsilon
        )
        self._regualizer_oob = Regualizer(
            support='mnac', type='oob',
            shape=regualizer_shape, zero_epsilon=mnac_epsilon,
            zero=self.nac_oob == 'clip'
        )
        self._regualizer_nmu_z = RegualizerNMUZ(
            zero=regualizer_z == 0
        )

        self.W = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        # self.W = torch.nn.Parameter(torch.ones(out_features, in_features))      # TODO - use when fixed NMU
        # self.register_buffer('W', torch.ones(out_features, in_features))  # have fixed weight of 1's (i.e. add)
        self.register_parameter('bias', None)
        self.use_noise = kwargs['nmu_noise']

    def reset_parameters(self):
        # self.W.requires_grad = False  # TODO - use when fixed NMU
        std = math.sqrt(0.25)
        r = min(0.25, math.sqrt(3.0) * std)
        torch.nn.init.uniform_(self.W, 0.5 - r, 0.5 + r)
        # self.W = torch.nn.Parameter(torch.Tensor([[0.5848, 0.3222]]))  # TODO - used fixed NMU

        self._regualizer_nmu_z.reset()

    def optimize(self, loss):
        self._regualizer_nmu_z.reset()

        if self.nac_oob == 'clip':
            self.W.data.clamp_(0.0 + self.mnac_epsilon, 1.0)

    def regualizer(self):
        return super().regualizer({
            'W': self._regualizer_bias(self.W),
            'z': self._regualizer_nmu_z(self.W),
            'W-OOB': self._regualizer_oob(self.W)
        })

    def forward(self, x, reuse=False):
        if self.use_noise and self.training:
            noise = torch.Tensor(x.shape).uniform_(1, 5)  # [B,I]
            x *= noise

        if self.allow_random:
            self._regualizer_nmu_z.append_input(x)

        W = torch.clamp(self.W, 0.0 + self.mnac_epsilon, 1.0) \
            if self.nac_oob == 'regualized' \
            else self.W

        # discrete_W = torch.where(W >= 0.5, torch.ones(W.shape), torch.zeros_like(W))
        # W = W + (discrete_W - W).detach()

        self.writer.add_histogram('W', W)
        self.writer.add_tensor('W', W)
        self.writer.add_scalar('W/sparsity_error', sparsity_error(W), verbose_only=False)

        if self.mnac_normalized:
            c = torch.std(x)
            x_normalized = x / c
            z_normalized = mnac(x_normalized, W, mode='prod')
            out = z_normalized * (c ** torch.sum(W, 1))
        else:
            out = mnac(x, W, mode='prod')
            # out = out + (mnac(x, discrete_W, mode='prod') - out).detach()
            # out_discrete = mnac(x, discrete_W, mode='prod')
            # out = x.prod(1).unsqueeze(1)

        if self.use_noise and self.training:
            # denominator shape: ([O, I] * [B, I] + 1 - [B,O]) --> [B,O].prod(1).view(-1,1) --> [B].view(B,-1) --> [B,O]
            out = out / (W * noise + 1 - W).prod(axis=1).view(x.shape[0], -1)  # * = elemwise mul
            # out_discrete = out_discrete / (discrete_W * noise + 1 - discrete_W).prod(axis=1).view(x.shape[0], -1)  # * = elemwise mul
            # out = out + (out_discrete - out).detach()
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


class ReRegualizedLinearMNACCell(AbstractRecurrentCell):
    """Implements the NAC (Neural Accumulator) as a recurrent cell

    Arguments:
        input_size: number of ingoing features
        hidden_size: number of outgoing features
    """

    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__(ReRegualizedLinearMNACLayer, input_size, hidden_size, **kwargs)
