# 2D binary layers XY model
# default option is 'square, periodic boundary'

import torch
import math
from args import args
import numpy as np
from scipy.optimize import curve_fit

def correlator(sample):
    sample=sample*2*math.pi # angular
    kdata = np.linspace(1, args.L, args.L)
    # construct fitting function
    corfun = correlation(sample, args.L)

    popt, pcov = curve_fit(func, kdata, np.log(corfun))
    # print(pcov)
    
    r_alg = np.exp(popt[0])
    r_exp = popt[1]

    return r_alg,r_exp,pcov

def correlation(sample, k):
    sample=sample*2*math.pi # angular


    output = np.linspace(1, args.L, args.L)
    for i in range(1, k):
        sample_t = sample.cpu().numpy()
        termx = np.cos(sample_t[:, :, i:, :] - sample_t[:, :, :-i, :]) # k sites in x direction
        termy = np.cos(sample_t[:, :, :, i:] - sample_t[:, :, :, :-i]) # y direction
        output[i] += (termx.sum(axis=(1, 2, 3))+ termy.sum(axis=(1, 2, 3))).mean()

        # periodic boudanry condition
        termx = np.cos(sample_t[:, :, -i:-1, :] - sample_t[:, :, 0:i-1, :])
        termy = np.cos(sample_t[:, :, :, -i:-1] - sample_t[:, :, :, 0:i-1])
        output[i] += (termx.sum(axis=(1, 2, 3))+ termy.sum(axis=(1, 2, 3))).mean()
    return output


# def func(x, a, b, c): # log fitting
#     return c*(np.log(x) - a) - x / b

def func(x, a, b): # log fitting
    return (np.log(x) - a) - x / b
