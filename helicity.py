# calculate the correlation parts of the helicity modulus

import torch
import math
from args import args
from utils import default_dtype_torch


def helicity(sample):
    sample=sample*2*math.pi # angular
    # output = torch.zeros(args.batch_size,
    #         dtype=default_dtype_torch,
    #         device=sample.device)

    # calculate the derivation of energy
    term = torch.sin(sample[:, :, 1:, :] - sample[:, :, :-1, :])
    term = term.sum(dim=(1, 2, 3))
    output = term
    term = torch.sin(sample[:, :, :, 1:] - sample[:, :, :, :-1])
    term = term.sum(dim=(1, 2, 3))
    output += term
    term = torch.sin(sample[:, :, 0, :] - sample[:, :, -1, :])
    term = term.sum(dim=(1, 2))
    output += term
    term = torch.sin(sample[:, :, :, 0] - sample[:, :, :, -1])
    term = term.sum(dim=(1, 2))
    output += term

    return output

