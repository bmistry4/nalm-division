import torch

from stable_nalu.abstract import ExtendedTorchModule
from stable_nalu.functional import mnac, Regualizer, RegualizerNMUZ, sparsity_error
from stable_nalu.layer._abstract_recurrent_cell import AbstractRecurrentCell
from stable_nalu.layer.unused.feature_selector import FeatureSelector

'''
Paper: https://arxiv.org/pdf/1810.04247.pdf (Feature Selection using Stochastic Gates)

STG for multiplication
In experiment file still need to add reg penalty: 
 reg_stg_mul = torch.mean(model.layer_2.layer.stg((model.layer_2.layer.mu + 0.5) / model.layer_2.layer.sigma))
 loss_train = ... + (reg_stg_mul * model.layer_2.layer.lam)

For printing gates use: 
 print(model.layer_2.layer.stg.get_gates('prob'))

'''


def _standard_truncnorm_sample(lower_bound, upper_bound, sample_shape=torch.Size()):
    r"""
    Implements accept-reject algorithm for doubly truncated standard normal distribution.
    (Section 2.2. Two-sided truncated normal distribution in [1])
    [1] Robert, Christian P. "Simulation of truncated normal variables." Statistics and computing 5.2 (1995): 121-125.
    Available online: https://arxiv.org/abs/0907.4010
    Args:
        lower_bound (Tensor): lower bound for standard normal distribution. Best to keep it greater than -4.0 for
        stable results
        upper_bound (Tensor): upper bound for standard normal distribution. Best to keep it smaller than 4.0 for
        stable results
    """
    x = torch.randn(sample_shape)
    done = torch.zeros(sample_shape).byte()
    while not done.all():
        proposed_x = lower_bound + torch.rand(sample_shape) * (upper_bound - lower_bound)
        if (upper_bound * lower_bound).lt(0.0):  # of opposite sign
            log_prob_accept = -0.5 * proposed_x ** 2
        elif upper_bound < 0.0:  # both negative
            log_prob_accept = 0.5 * (upper_bound ** 2 - proposed_x ** 2)
        else:  # both positive
            assert (lower_bound.gt(0.0))
            log_prob_accept = 0.5 * (lower_bound ** 2 - proposed_x ** 2)
        prob_accept = torch.exp(log_prob_accept).clamp_(0.0, 1.0)
        accept = torch.bernoulli(prob_accept).byte() & ~done
        if accept.any():
            accept = accept.bool()
            x[accept] = proposed_x[accept]
            accept = accept.byte()
            done |= accept
    return x


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

        # self.W = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        # self.W = torch.nn.Parameter(torch.ones(out_features, in_features))      # TODO - use when fixed NMU
        # self.register_buffer('W', torch.ones(out_features, in_features))  # have fixed weight of 1's (i.e. add)
        self.register_parameter('bias', None)
        self.use_noise = kwargs['nmu_noise']

        self.stg = FeatureSelector(in_features, out_features, sigma=0.5, device='cpu')
        self.reg = self.stg.regularizer
        self.lam = 0.05  # TODO: tune!!! using 0.5 will work for [1.1,1.3] (which doesn't work for baseline). 0.7 works for [1.1,1.2] s0
        self.mu = self.stg.mu
        self.sigma = self.stg.sigma

    def reset_parameters(self):
        # self.W.requires_grad = False  # TODO - use when fixed NMU
        # std = math.sqrt(0.25)
        # r = min(0.25, math.sqrt(3.0) * std)
        # torch.nn.init.uniform_(self.W, 0.5 - r, 0.5 + r)

        # TODO - STG Linear weight init (with modifications for std and bounds)
        # stddev = torch.tensor(std)  # torch.tensor(0.1)
        # shape = self.W.shape
        # self.W = torch.nn.Parameter(_standard_truncnorm_sample(lower_bound=1 * stddev, upper_bound=1.5 * stddev,
        #                                                        sample_shape=shape))

        self._regualizer_nmu_z.reset()

    def optimize(self, loss):
        self._regualizer_nmu_z.reset()

        # if self.nac_oob == 'clip':
        #     self.W.data.clamp_(0.0 + self.mnac_epsilon, 1.0)

    def regualizer(self):
        return super().regualizer({
            # 'W': self._regualizer_bias(self.W),
            # 'z': self._regualizer_nmu_z(self.W),
            # 'W-OOB': self._regualizer_oob(self.W)
        })

    def forward(self, x, reuse=False):
        # TODO: when using STG on NMU
        x = x.unsqueeze(2).repeat(1, 1, self.out_features)  # [B,I] -> [B,I,O]
        x = self.stg(x)
        out = x.prod(axis=1)  # [B,I, O] -> [B,1,O] -> [B,O]
        return out

        # TODO: when using STG on NAU and just want to multiply manually
        # return x.prod(axis=1).unsqueeze(1)

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
