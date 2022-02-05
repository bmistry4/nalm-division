import math

import torch
from torch.nn import functional as F

from stable_nalu.abstract import ExtendedTorchModule
from stable_nalu.functional import Regualizer, RegualizerNAUZ, sparsity_error


class EnsembleGumbelSoftmaxNAU(ExtendedTorchModule):
    """Implements the RegualizedLinearNAC using ESG

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
        # self._regualizer_bias = Regualizer(
        #     support='nac', type='bias',
        #     shape=regualizer_shape  # , zero=True
        # )
        # self._regualizer_oob = Regualizer(
        #     support='nac', type='oob',
        #     shape=regualizer_shape,
        #     zero=self.nac_oob == 'clip'
        # )
        self._regualizer_nau_z = RegualizerNAUZ(
            zero=regualizer_z == 0
        )

        self.register_buffer('W', torch.ones(1, in_features))  # have fixed weight of 1's (i.e. add)
        self.W_logits = torch.nn.Parameter(torch.Tensor(in_features, out_features))
        self.register_parameter('bias', None)
        self.mask = None
        self.tau = torch.tensor(1, dtype=torch.float32)
        # self.mask = torch.tensor([[1,0.], [0,1]], requires_grad=False)

    def reset_parameters(self):
        std = math.sqrt(0.25)
        r = min(0.25, math.sqrt(3.0) * std)
        torch.nn.init.uniform_(self.W_logits, 0.5 - r, 0.5 + r)

    def optimize(self, loss):
        self._regualizer_nau_z.reset()

        # if self.nac_oob == 'clip':
        #     self.W.data.clamp_(-1.0, 1.0)

    def regualizer(self):
        return super().regualizer({
            # 'W': self._regualizer_bias(self.W),
            'z': self._regualizer_nau_z(self.W),
            # 'W-OOB': self._regualizer_oob(self.W)
        })

    def forward(self, x, reuse=False):
        self.tau = min(1, math.exp(-1e-5 * self.writer.get_iteration()))
        if self.allow_random:
            self._regualizer_nau_z.append_input(x)

        x = x.unsqueeze(2).repeat(1, 1, self.W_logits.shape[-1])  # [B,I] -> [B,I,O]
        # is a must to be able to be ble to learn selection
        logits = x * self.W_logits  # [B,I,O] * [I,O] -> [B,I,O]
        z = []
        # number of OH masks - min. = subset size on input
        for i in range(math.ceil(x.shape[1]*0.25)):
            # do softmax on input axis (so can't have 2 1's on input dim)
            z.append(F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=1))  # z element = [B,I,O]
        z = torch.stack(z, dim=0)  # [M, B, I, O]
        # find max on element level, reducing the different OH masks to a single binary mask
        # TODO: Try mean mask (which doesn't consider 0's -https://discuss.pytorch.org/t/use-tensor-mean-method-but-ignore-0-values/60170/4
        mask = torch.max(z, axis=0).values  # [B,I,O]
        # mask = torch.mean(torch.max(z, axis=0).values, 0)  # [B,I,O] -> [I,O]
        # mask = F.gumbel_softmax(mask, tau=1, hard=True, dim=0)    # too stochastic so don't use
        # mask = self.mask.unsqueeze(0).repeat(x.shape[0],1,1)  #hardcoded mask
        self.mask = torch.mean(mask, 0)
        # self.mask = mask

        x = x * mask  # [B,I,O]
        # TODO: not using logit x for multiplication with W, just just logit_W to do gumble. THerefore they are separate paths - results were the same?
        #x = x / self.W_logits
        out = self.W.unsqueeze(0).repeat(x.shape[0], 1, 1) @ x  # [B,1,I] @ [B,I,O] = [B,1,O]
        out = out.squeeze(1)  # [B,1,O] -> [B,O]

        # self.writer.add_histogram('W', W)
        # self.writer.add_tensor('W', W)
        # self.writer.add_scalar('W/sparsity_error', sparsity_error(W), verbose_only=False)

        # out = torch.nn.functional.linear(x, self.W, self.bias)
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )
