#!/usr/bin/env python3
#
# Sample from a trained network

import re
import time
import os
from collections import namedtuple
from math import sqrt
from uncertainties import ufloat

import numpy as np
import torch

import xy
import correlator
from helicity import helicity
from args import args
# from pixelcnn_xy import PixelCNN
from pixelcnn_xy_multi import PixelCNN
from utils import get_ham_args_features, ignore_param, print_args

default_dtype = np.float32

# Set args here or through CLI to match the state
args.ham = 'fm'
args.lattice = 'sqr'
args.boundary = 'periodic'
# args.L = 16
# args.beta = 0.6

args.net = 'pixelcnn_xy'
args.net_depth = 3
args.net_width = 32
# args.channel = 6
args.half_kernel_size = 1
args.bias = True
args.O2 = False
args.res_block = False
args.lr_schedule = True
args.beta_anneal = 0.998
args.clip_grad = 1.0


args.batch_size = 10**3
args.max_step = 1
args.print_step = 1
state_dir = 'out'

sample_con = 0

ham_args, features = get_ham_args_features()
if sample_con:
    out_filename = '{state_dir}/{ham_args}/{features}/out{args.out_infix}_sample/'.format(**locals())
    os.makedirs(out_filename, exist_ok = True)

state_filename = '{state_dir}/{ham_args}/{features}/out{args.out_infix}_save/{{}}.state'.format(
    **locals())
log_filename = '{state_dir}/{ham_args}/{features}/out{args.out_infix}.log'.format(
    **locals())
final_window = 2000

Run = namedtuple('Run', [
    'step', 'F_arr', 'F_std_arr', 'E_arr', 'param_count', 'used_time'
])


def read_log(filename):
    step_arr = []
    F_arr = []
    F_std_arr = []
    S_arr = []
    E_arr = []
    param_count = 0
    used_time = 0
    with open(filename, 'r') as f:
        for line in f:
            match = re.compile(r'parameters: (.*)').search(line)
            if match:
                param_count = int(match.group(1))
                continue

            match = re.compile(
                r'step = (.*?),'
            ).search(line)
            if match:
                step_arr.append(int(match.group(1)))

            match = re.compile(
                r'F = (.*?),.*F_std = (.*?),.*E = (.*?),'
                ).search(line)
            if match:
                F_arr.append(float(match.group(1)))
                F_std_arr.append(float(match.group(2)))
                E_arr.append(float(match.group(3)))

    step_arr = np.array(step_arr, dtype=int)
    F_arr = np.array(F_arr, dtype=default_dtype)
    F_std_arr = np.array(F_std_arr, dtype=default_dtype)
    E_arr = np.array(E_arr, dtype=default_dtype)

    idx = np.argsort(step_arr)
    step_arr = step_arr[idx]
    F_arr = F_arr[idx]
    F_std_arr = F_std_arr[idx]
    E_arr = E_arr[idx]

    return Run(step_arr, F_arr, F_std_arr, E_arr, param_count,
               used_time)


def get_state_step():
    run = read_log(log_filename)
    F_cumsum = np.cumsum(run.F_arr[1:])
    F_avg = (F_cumsum[final_window:] - F_cumsum[:-final_window]) / final_window
    step = np.argmin(F_avg)
    step = int(round((step + final_window) / args.save_step)) * args.save_step
    step = 10000
    # if save all states in each step, we can find the accurate free energy
    return step


def get_mean_err(count, x_sum, x_sqr_sum):
    x_mean = x_sum / count
    x_sqr_mean = x_sqr_sum / count
    x_std = sqrt(abs(x_sqr_mean - x_mean**2))
    x_err = x_std / sqrt(count)
    x_ufloat = ufloat(x_mean, x_err)
    return x_ufloat, x_mean, x_err


if __name__ == '__main__':
    print_args(print_fn=print)

    if args.net == 'pixelcnn_xy':
        net = PixelCNN(**vars(args))
    else:
        raise ValueError('Unknown net: {}'.format(args.net))
    net.to(args.device)
    print('{}\n'.format(net))

    state_filename = state_filename.format(get_state_step())
    print(state_filename)
    state = torch.load(state_filename, map_location=args.device)
    ignore_param(state['net'], net)
    net.load_state_dict(state['net'])

    F_sum = 0
    F_sqr_sum = 0
    v_sum = 0
    v_sqr_sum = 0
    # S_sum = 0
    # S_sqr_sum = 0
    E_sum = 0
    E_sqr_sum = 0
    helicity_modulus_sum = 0
    helicity_modulus_sqr_sum = 0

    start_time = time.time()
    for step in range(args.max_step):
        with torch.no_grad():
            sample, x_hat = net.sample(args.batch_size)
            log_prob = net.log_prob(sample, args.batch_size)
            energy, vortices= xy.energy(sample, args.ham, args.lattice,
                                  args.boundary)
            energy = energy / args.L**2
            vortices = vortices /args.L**2
            free_energy = energy + 1 / args.beta * log_prob / args.L**2
            # entropy = -log_prob / args.L**2

            F_sum += free_energy.sum().item()
            F_sqr_sum += (free_energy**2).sum().item()
            # S_sum += entropy.sum().item()
            # S_sqr_sum += (entropy**2).sum().item()
            E_sum += energy.sum().item()
            E_sqr_sum += (energy**2).sum().item()
            v_sum += vortices.sum().item()
            v_sqr_sum += (vortices**2).sum().item()
 
            # helicity
            correlations = helicity(sample)
            helicity_modulus = - (energy/2 ) - (args.beta/ args.L**2)*(correlations**2 )/2
            helicity_modulus_sum += helicity_modulus.sum().item()
            helicity_modulus_sqr_sum += (helicity_modulus**2).sum().item()


        if args.print_step and (step + 1) % args.print_step == 0:
            count = args.batch_size * (step + 1)
            print(count)
            F_ufloat, F_mean, F_err = get_mean_err(count, F_sum, F_sqr_sum)
            # S_ufloat, S_mean, F_err = get_mean_err(count, S_sum, S_sqr_sum)
            E_ufloat, E_mean, E_err = get_mean_err(count, E_sum, E_sqr_sum)
            v_ufloat, v_mean, v_err = get_mean_err(count, v_sum, v_sqr_sum)
            Cv = args.beta**2 * (E_sqr_sum / count -
                                 (E_sum / count)**2) * args.L**2
            rho_ufloat, rho_mean, rho_err = get_mean_err(count, helicity_modulus_sum, helicity_modulus_sqr_sum)
            used_time = time.time() - start_time
            print(
                'count = {}, F = {:.2u}, v = {:.2u}, E = {:.2u}, Cv = {:.8g}, Rho = {:.8g}, used_time = {:.3f}'
                .format(count, F_ufloat, v_ufloat, E_ufloat, Cv,rho_ufloat, used_time))

        if sample_con:
            # save configurations
            sample_array=sample.cpu().numpy()
            np.savetxt('{}sample{}.txt'.format(out_filename, step),sample_array.reshape(args.batch_size,-1))

            # save energy distribution
            # energy_array=energy.cpu().numpy()
            # np.savetxt('{}energy{}.txt'.format(out_filename, step),energy_array.reshape(args.batch_size,-1))
         
        # correlation length
        with torch.no_grad():
           r_alg, r_exp, pcov= correlator.correlator(sample)
        
        print('r_alg={:.8g}, r_exp={:.8g}, r_exp_std={:.8g}'
            .format(
                r_alg,
                r_exp,
                pcov[1,1]
            ))

        # save observables
        os.makedirs('results', exist_ok = True)
        f = open('results/resultsL{}_ch{}.csv'.format(args.L, args.channel),'a+')
        np.savetxt(f,[args.beta,args.L,
            F_mean,
            F_err,
            E_mean,
            E_err,
            v_mean,
            Cv,
            rho_mean,
            r_alg,
            r_exp,
            pcov[1,1],
            used_time
            ],newline='\t')
        f.write("\n")   
        f.close

