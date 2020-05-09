# PixelCNN XY model

import torch
import math
from numpy import log
from torch import nn

from utils import default_dtype_torch


class ResBlock(nn.Module):
    def __init__(self, block):
        super(ResBlock, self).__init__()
        self.block = block

    def forward(self, x):
        return x + self.block(x)


class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self.exclusive = kwargs.pop('exclusive')
        super(MaskedConv2d, self).__init__(*args, **kwargs)

        _, _, kh, kw = self.weight.shape
        self.register_buffer('mask', torch.ones([kh, kw]))
        self.mask[kh // 2, kw // 2 + (not self.exclusive):] = 0
        self.mask[kh // 2 + 1:] = 0
        self.weight.data *= self.mask

        # Correction to Xavier initialization
        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())

    def forward(self, x):
        return nn.functional.conv2d(x, self.mask * self.weight, self.bias,
                                    self.stride, self.padding, self.dilation,
                                    self.groups)

    def extra_repr(self):
        return (super(MaskedConv2d, self).extra_repr() +
                ', exclusive={exclusive}'.format(**self.__dict__))


class PixelCNN(nn.Module):
    def __init__(self, **kwargs):
        super(PixelCNN, self).__init__()
        self.L = kwargs['L']
        self.net_depth = kwargs['net_depth']
        self.net_width = kwargs['net_width']
        self.half_kernel_size = kwargs['half_kernel_size']
        self.bias = kwargs['bias']
        self.o2 = kwargs['o2']
        self.res_block = kwargs['res_block']
        self.x_hat_clip = kwargs['x_hat_clip']
        self.final_conv = kwargs['final_conv']
        self.epsilon = kwargs['epsilon']
        self.device = kwargs['device']

        # Force the first x_hat to be 1.0
        if self.bias and not self.o2:
            self.register_buffer('x_hat_mask', torch.ones([self.L] * 2))
            self.x_hat_mask[0, 0] = 0
            self.register_buffer('x_hat_bias', torch.zeros([self.L] * 2))
            self.x_hat_bias[0, 0] = 1.0

        layers = []
        layers.append(
            MaskedConv2d(
                1,
                2 if self.net_depth == 1 else self.net_width,
                self.half_kernel_size * 2 + 1,
                padding=self.half_kernel_size,
                bias=self.bias,
                exclusive=True)) #out channel = 2
        for count in range(self.net_depth - 2):
            if self.res_block:
                layers.append(
                    self._build_res_block(self.net_width, self.net_width))
            else:
                layers.append(
                    self._build_simple_block(self.net_width, self.net_width))
        if self.net_depth >= 2:
            layers.append(
                self._build_simple_block(
                    self.net_width, self.net_width if self.final_conv else 2)) #out channel = 2
        if self.final_conv:
            layers.append(nn.PReLU(self.net_width, init=0.5))
            layers.append(nn.Conv2d(self.net_width, 2, 1)) #out channel = 2
        layers.append(nn.Softplus())
        # layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def _build_simple_block(self, in_channels, out_channels):
        layers = []
        layers.append(nn.PReLU(in_channels, init=0.5))
        layers.append(
            MaskedConv2d(
                in_channels,
                out_channels,
                self.half_kernel_size * 2 + 1,
                padding=self.half_kernel_size,
                bias=self.bias,
                exclusive=False))
        block = nn.Sequential(*layers)
        return block

    def _build_res_block(self, in_channels, out_channels):
        layers = []
        layers.append(nn.Conv2d(in_channels, in_channels, 1, bias=self.bias))
        layers.append(nn.PReLU(in_channels, init=0.5))
        layers.append(
            MaskedConv2d(
                in_channels,
                out_channels,
                self.half_kernel_size * 2 + 1,
                padding=self.half_kernel_size,
                bias=self.bias,
                exclusive=False))
        block = ResBlock(nn.Sequential(*layers))
        return block

    def forward(self, x):
        x_hat = self.net(x)

        if self.x_hat_clip:
            # Clip value and preserve gradient
            with torch.no_grad():
                delta_x_hat = torch.clamp(x_hat, self.x_hat_clip,
                                          1 - self.x_hat_clip) - x_hat
            assert not delta_x_hat.requires_grad
            x_hat = x_hat + delta_x_hat

        # Force the first x_hat to be 1.0
        if self.bias and not self.o2:
            x_hat = x_hat * self.x_hat_mask + self.x_hat_bias

        return x_hat


    def sample(self, batch_size):
        sample = torch.zeros(
            [batch_size, 1, self.L, self.L],
            dtype=default_dtype_torch,
            device=self.device)
        sample0 = torch.ones(
            [batch_size, 1, self.L, self.L],
            dtype=default_dtype_torch,
            device=self.device)*0.5
        for i in range(self.L):
            for j in range(self.L):
                x_hat = self.forward(sample) # q_theta
                alpha=x_hat[:,0,i,j]
                beta=x_hat[:,1,i,j]
                # sample[:,0, i, j]= (torch.distributions.Normal(alpha,beta).sample().to(default_dtype_torch) % 1 + 1)/2  # normalization dis。
                # sample[:,0, i, j]= torch.distributions.Uniform(alpha,beta).sample().to(default_dtype_torch) # uniform dis.
                sample[:,0, i, j]= torch.distributions.Beta(alpha,beta).sample().to(default_dtype_torch) # beta dis.

        # Chekc the distribution
        # import matplotlib.pyplot as plt
        # plt.hist(x_hat[:,0, 1, 1].cpu().numpy(), bins=100, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
        # plt.show()

        if self.o2:
            # random angular change
            rotate = torch.randn(
                [batch_size, 1, 1, 1],
                dtype=sample.dtype,
                device=sample.device)
            sample += rotate
            sample = sample % (2 * math.pi)

        return sample, x_hat

    def _log_prob(self,sample,x_hat):
        alpha=x_hat[:,0,:,:]
        beta=x_hat[:,1,:,:]
        log_prob = torch.distributions.Beta(alpha,beta).log_prob(sample[:,0,:,:])
        log_prob = log_prob.view(log_prob.shape[0], -1).sum(dim=1)
        return log_prob

    def log_prob(self, sample, batch_size):
        x_hat = self.forward(sample)
        log_prob = self._log_prob(sample, x_hat)

        if self.o2:
            # Density estimation on rotated sample
            rotate = torch.randn(
                [batch_size, 1, 1, 1],
                dtype=sample.dtype,
                device=sample.device)
            sample_rot = (sample + rotate)% (2 * math.pi)
            x_hat_rot = self.forward(sample_rot)
            log_prob_rot = self._log_prob(sample_rot, x_hat_rot)
            log_prob = torch.logsumexp(
                torch.stack([log_prob, log_prob_rot]), dim=0)
            log_prob = log_prob - log(2)

        return log_prob
