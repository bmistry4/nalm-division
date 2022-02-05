import torch
import math

from ..abstract import ExtendedTorchModule
from ..functional import mnac, Regualizer, RegualizerNMUZ, sparsity_error
from ._abstract_recurrent_cell import AbstractRecurrentCell


class ConcatReciprocalNMULayer(ExtendedTorchModule):
    """Implements the NMRU

    Arguments:
        in_features: number of ingoing features
        out_features: number of outgoing features
    """

    def __init__(self, in_features, out_features,
                 nac_oob='regualized', regualizer_shape='squared',
                 mnac_epsilon=0, mnac_normalized=False, regualizer_z=0,
                 **kwargs):
        super().__init__('nmru', **kwargs)
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

        self.W = torch.nn.Parameter(torch.Tensor(out_features, in_features))  # [O, I]
        self.W_reciprocal = torch.nn.Parameter(torch.Tensor(out_features, in_features))  # [O, I]
        self.register_parameter('bias', None)
        self.use_noise = kwargs['nmu_noise']

    def reset_parameters(self):
        std = math.sqrt(0.25)
        r = min(0.25, math.sqrt(3.0) * std)
        torch.nn.init.uniform_(self.W, 0.5 - r, 0.5 + r)
        torch.nn.init.uniform_(self.W_reciprocal, 0.5 - r, 0.5 + r)

        # self.W = torch.nn.Parameter(torch.Tensor([[1, 0., 0, 1]]))  # TODO - used fixed NMU
        # self.W.requires_grad = False

        self._regualizer_nmu_z.reset()

    def optimize(self, loss):
        self._regualizer_nmu_z.reset()

        if self.nac_oob == 'clip':
            self.W.data.clamp_(0.0 + self.mnac_epsilon, 1.0)
            self.W_reciprocal.data.clamp_(0.0 + self.mnac_epsilon, 1.0)

    def regualizer(self):
        return super().regualizer({
            'W': self._regualizer_bias(self.W),
            'z': self._regualizer_nmu_z(self.W),
            'W-OOB': self._regualizer_oob(self.W)
        })

    def forward(self, x):
        if self.use_noise and self.training:
            noise = torch.Tensor(x.shape).uniform_(1, 5).to(self.W.device)  # [B,I]
            x *= noise

        if self.allow_random:
            self._regualizer_nmu_z.append_input(x)

        self.writer.add_histogram('W', self.W)
        self.writer.add_tensor('W', self.W)
        self.writer.add_scalar('W/sparsity_error', sparsity_error(self.W), verbose_only=False)

        out = mnac(x, self.W, mode='prod')
        out_reciprocal = mnac(x.reciprocal(), self.W_reciprocal, mode='prod')

        # apply denoising if sNMU is used
        if self.use_noise and self.training:
            # [B,O] / mnac([B,I], [O,I] 'prod') --> [B,O] / [B,O] --> [B,O]
            out = out / mnac(noise, self.W, mode='prod')
            out_reciprocal = out_reciprocal / mnac(noise, self.W_reciprocal, mode='prod')

        return out * out_reciprocal

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


class ConcatReciprocalNMUCell(AbstractRecurrentCell):
    """Implements the NAC (Neural Accumulator) as a recurrent cell

    Arguments:
        input_size: number of ingoing features
        hidden_size: number of outgoing features
    """

    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__(ConcatReciprocalNMULayer, input_size, hidden_size, **kwargs)
