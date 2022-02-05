import math

import torch

from stable_nalu.abstract import ExtendedTorchModule
from stable_nalu.functional import Regualizer, RegualizerNAUZ, sparsity_error

"""
CONFIG THAT WORKS FOR INPUT SIZE 10:
--operation mul --first-layer SigmoidNAU --layer-type ReRegualizedLinearMNAC --interpolation-range [1.1,1.2] 
--extrapolation-range [1.2,6] --seed 0 --max-iterations 50000 --no-cuda --remove-existing-data --name-prefix 2-layer
--input-size 10 --subset-ratio 0.5 --overlap-ratio 0 --regualizer-scaling-start 35000 --regualizer-scaling-end 45000
--nmu-noise

CONFIG THAT WORKS FOR INPUT SIZE 4 with overlap but needs early stopping:
--operation mul --first-layer SigmoidNAU --layer-type ReRegualizedLinearMNAC --interpolation-range [1.1,1.2] 
--extrapolation-range [1.2,6] --seed 0 --max-iterations 30000 --no-cuda --remove-existing-data --name-prefix 2-layer 
--input-size 4 --subset-ratio 0.5 --overlap-ratio 0.5 --regualizer-scaling-start 35000 --regualizer-scaling-end 45000 
--nmu-noise

CURRENTLY TRYING TO GET WORKING
--operation mul --first-layer SigmoidNAU --layer-type ReRegualizedLinearMNAC 
--interpolation-range [1.1,1.2] --extrapolation-range [1.2,6] --seed 0 --max-iterations 100000 
--no-cuda --remove-existing-data --name-prefix 2-layer --input-size 10 --subset-ratio 0.25
--overlap-ratio 0.5 --regualizer-scaling-start 65000 --regualizer-scaling-end 85000

"""
class SigmoidNAU(ExtendedTorchModule):
    """Implements the RegualizedLinearNAC using Sigmoid selection NAU

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
        #     support='mnac', type='bias',
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
        self.bn = torch.nn.BatchNorm1d(out_features)
        self.l1 = torch.nn.Linear(out_features,out_features, bias=True)

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
            # 'W': self._regualizer_bias(self.W_logits),  # FIXME: Helps once slection thr is correct
            'z': self._regualizer_nau_z(self.W),
            # 'W-OOB': self._regualizer_oob(self.W)
        })

    def forward(self, x, reuse=False):
        if self.allow_random:
            self._regualizer_nau_z.append_input(x)

        x = x.unsqueeze(2).repeat(1, 1, self.W_logits.shape[-1])  # [B,I] -> [B,I,O]
        # FIXME: Maybe don't need x *
        logits = self.W_logits  # [I,O]
        mask = torch.sigmoid(self.bn(self.l1(logits)))  # [I,O] between 0-1
        self.mask = mask
        if not self.training:
            # binarise mask for val and test set -> >0.5 = 1 and <0.5 = 0
            mask = torch.round(mask)
        x = x * mask  # [B,I,O]
        out = self.W.unsqueeze(0).repeat(x.shape[0], 1, 1) @ x  # [B,1,I] @ [B,I,O] = [B,1,O]
        out = out.squeeze(1)  # [B,1,O] -> [B,O]
        # self.mask = torch.mean(mask, 0)
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )
