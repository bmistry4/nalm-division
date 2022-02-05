import torch

from stable_nalu.abstract import ExtendedTorchModule
from stable_nalu.functional import Regualizer, RegualizerNAUZ, sparsity_error
from stable_nalu.layer._abstract_recurrent_cell import AbstractRecurrentCell
import itertools
from stable_nalu.layer.unused.feature_selector import FeatureSelector

'''
Paper: https://arxiv.org/pdf/1810.04247.pdf (Feature Selection using Stochastic Gates)

STG for addition
In experiment file still need to add reg penalty: 
 reg_stg_add = torch.mean(model.layer_1.layer.stg((model.layer_1.layer.mu + 0.5) / model.layer_1.layer.sigma))
 loss_train = ... + (reg_stg_add * model.layer_1.layer.lam)

For printing gates use: 
 print(model.layer_1.layer.stg.get_gates('prob'))
 
'''


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

        # self.W = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        # self.register_parameter('bias', None)

        self.stg = FeatureSelector(in_features, out_features, sigma=0.5, device='cpu')
        self.reg = self.stg.regularizer
        self.lam = 0.7  # TODO: tune!!! using 0.5 will work for [1.1,1.3] (which doesn't work for baseline). 0.7 works for [1.1,1.2] s0
        self.mu = self.stg.mu
        self.sigma = self.stg.sigma

    def reset_parameters(self):
        pass
        # std = math.sqrt(2.0 / (self.in_features + self.out_features))
        # r = min(0.5, math.sqrt(3.0) * std)
        # torch.nn.init.uniform_(self.W, -r, r)

        # self.W = generate_intial_W(self.seed)
        # self.W = torch.nn.Parameter(torch.Tensor([[0.5, 0, 0, 0], [0, 0.5, 0, -0]]))  # TODO - used fixed NAU

    def optimize(self, loss):
        self._regualizer_nau_z.reset()

        # if self.nac_oob == 'clip':
        #     self.W.data.clamp_(-1.0, 1.0)

    def regualizer(self):
        return super().regualizer({
            # 'W': self._regualizer_bias(self.W),
            # 'z': self._regualizer_nau_z(self.W),
            # 'W-OOB': self._regualizer_oob(self.W)
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

        # TODO - STG with manual addition
        x = x.unsqueeze(2).repeat(1, 1, self.out_features)  # [B,I] -> [B,I,O]
        x = self.stg(x)
        out = x.sum(axis=1).squeeze(1)  # [B,I,O] -> [B,1,O] -> [B,O]
        return out  # [B,O]

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
