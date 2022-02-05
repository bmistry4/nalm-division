import scipy.optimize
import numpy as np
import torch
import math

from ..abstract import ExtendedTorchModule
from ..functional import Regualizer, RegualizerNAUZ, sparsity_error
from ._abstract_recurrent_cell import AbstractRecurrentCell
import itertools
from torch.distributions.beta import Beta
from torch.distributions.normal import Normal

"""
NOT USED re_regualized_linear_nac.py copy (containing additional code which didn't work out in the end) 
Set up for beta-NAU. Needs to be in folder 1 level above to work.
Got commented out for the pass-through versions of the nau at different levels
- W, out before noise, out after noise
"""
def generate_intial_W(id):
    # generate permutations
    W_permutations = ([p for p in itertools.product([-0.5, 0, 0.5], repeat=8)])
    return torch.nn.Parameter(torch.Tensor(list(W_permutations[id])).reshape(2, -1))


class ReRegualizedLinearNACLayer(ExtendedTorchModule):
    """Implements the RegualizedLinearNAC

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """

    def __init__(self, in_features, out_features,
                 nac_oob='regualized', regualizer_shape='squared', regualizer_z=0,
                 **kwargs):
        super().__init__('nac', **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.nac_oob = nac_oob
        self.use_noise = kwargs['nau_noise']
        # self.p = 0.25
        self._regualizer_bias = Regualizer(
            support='nac', type='bias',
            shape=regualizer_shape,
        )
        self._regualizer_oob = Regualizer(
            support='nac', type='oob',
            shape=regualizer_shape,
            zero=self.nac_oob == 'clip'
        )
        self._regualizer_nau_z = RegualizerNAUZ(
            zero=regualizer_z == 0
        )

        self.W = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_parameter('bias', None)

    def reset_parameters(self):
        # std = math.sqrt(2.0 / (self.in_features + self.out_features))
        # r = min(0.5, math.sqrt(3.0) * std)
        # torch.nn.init.uniform_(self.W, -r, r)
        self.W.data = (Beta(torch.tensor([7.]), torch.tensor([7.])).sample(self.W.shape).squeeze(
            -1) * 2) - 1  # sample in range [-1,1]
        # self.W.data = (Normal(torch.tensor([0.5]), torch.tensor([math.sqrt(1/60)])).sample(self.W.shape).squeeze(-1)*2)-1    # sample in range [-1,1]

        # self.W = torch.nn.Parameter(torch.Tensor([[0.5, 0, 0, 0], [0, 0.5, 0, -0]]))  # TODO - used fixed NAU

    def optimize(self, loss):
        self._regualizer_nau_z.reset()

        if self.nac_oob == 'clip':
            self.W.data.clamp_(-1.0, 1.0)

    def regualizer(self):
        return super().regualizer({
            'W': self._regualizer_bias(self.W),
            'z': self._regualizer_nau_z(self.W),
            'W-OOB': self._regualizer_oob(self.W)
        })

    def forward(self, x, reuse=False):
        # self.p = self.p - 0.00001
        # if self.p < 0:
        #  self.p = 0.
        # x = torch.nn.functional.dropout(x, self.p, self.training)

        if self.use_noise and self.training:
            a = 1 / x.var()

            # additive noise - unique f.e. element
            noise = torch.Tensor(x.shape).uniform_(-a, a)  # [B,I]
            x += noise

            # multiplicative noise - unique f.e. b.item
            # noise = torch.Tensor(x.shape[0],1).uniform_(-a, a)  # [B,1]
            # x *= noise

        if self.allow_random:
            self._regualizer_nau_z.append_input(x)

        W = torch.clamp(self.W, -1.0, 1.0)

        # set values to -1 if <=-0.5, 1 if >=0.5, or 0 otherwise
        # discrete_W = torch.where(W <= -0.5, torch.empty(W.shape).fill_(-1.),
        #             torch.where(W >= 0.5, torch.ones(W.shape), torch.zeros_like(W)))
        # W = W + (discrete_W - W).detach()

        self.writer.add_histogram('W', W)
        self.writer.add_tensor('W', W)
        self.writer.add_scalar('W/sparsity_error', sparsity_error(W), verbose_only=False)

        out = torch.nn.functional.linear(x, W, self.bias)
        # out = out + (torch.nn.functional.linear(x, discrete_W, self.bias) - out).detach()

        if self.use_noise and self.training:
            # denoise additive noise
            # out = [B,O] - [B,I][O,I]^T
            out = out - torch.nn.functional.linear(noise, W, bias=None)

            # denoise multiplicative noise
            # out /= noise
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


class ReRegualizedLinearNACCell(AbstractRecurrentCell):
    """Implements the RegualizedLinearNAC as a recurrent cell

    Arguments:
        input_size: number of ingoing features
        hidden_size: number of outgoing features
    """

    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__(ReRegualizedLinearNACLayer, input_size, hidden_size, **kwargs)
