#!/usr/bin/env python3
#
# Recognizing the topological phase transition by Variational Autoregressive Networks
# 2d classical XY model

import time

# import pytorch
import numpy as np
from numpy import sqrt
from torch import nn
import torch
from scipy import signal
import math as m

# import xy model
import xy
import correlator
from helicity import helicity

# parameters setup
from args import args

# import PixelCNN for xy model
from pixelcnn_xy_multi import PixelCNN
# utilities
from utils import (
    clear_checkpoint,
    clear_log,
    get_last_checkpoint_step,
    ignore_param,
    init_out_dir,
    my_log,
    print_args,
)


def main():

    start_time = time.time()
    # initialize output dir
    init_out_dir()
    # check point
    if args.clear_checkpoint:
        clear_checkpoint()
    last_step = get_last_checkpoint_step()
    if last_step >= 0:
        my_log('\nCheckpoint found: {}\n'.format(last_step))
    else:
        clear_log()
    print_args()

    if args.net == 'pixelcnn_xy':
        net = PixelCNN(**vars(args))
    else:
        raise ValueError('Unknown net: {}'.format(args.net))
    net.to(args.device)
    my_log('{}\n'.format(net))

    # parameters of networks
    params = list(net.parameters())
    params = list(filter(lambda p: p.requires_grad, params)) # parameters with gradients
    nparams = int(sum([np.prod(p.shape) for p in params]))
    my_log('Total number of trainable parameters: {}'.format(nparams))
    named_params = list(net.named_parameters())
    # optimizers
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr)
    elif args.optimizer == 'sgdm':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=args.lr, alpha=0.99)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))
    elif args.optimizer == 'adam0.5':
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.5, 0.999))
    else:
        raise ValueError('Unknown optimizer: {}'.format(args.optimizer))
    # learning rates
    if args.lr_schedule:
        # 0.92**80 ~ 1e-3
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.92, patience=100, threshold=1e-4, min_lr=1e-6)
    # read last step
    if last_step >= 0:
        state = torch.load('{}_save/{}.state'.format(args.out_filename,
                                                     last_step))
        ignore_param(state['net'], net)
        net.load_state_dict(state['net'])
        if state.get('optimizer'):
            optimizer.load_state_dict(state['optimizer'])
        if args.lr_schedule and state.get('scheduler'):
            scheduler.load_state_dict(state['scheduler'])

    init_time = time.time() - start_time
    my_log('init_time = {:.3f}'.format(init_time))

    # start training
    my_log('Training...')
    sample_time = 0
    train_time = 0
    start_time = time.time()
    for step in range(last_step + 1, args.max_step + 1):
        optimizer.zero_grad() # clear last step

        sample_start_time = time.time()
        with torch.no_grad():
            sample, x_hat= net.sample(args.batch_size) # sample from networks with batch_size = 10**3 (default)
        assert not sample.requires_grad
        assert not x_hat.requires_grad
        sample_time += time.time() - sample_start_time

        train_start_time = time.time()

        # log probabilities
        log_prob = net.log_prob(sample, args.batch_size) 

        # 0.998**9000 ~ 1e-8
        beta = args.beta * (1 - args.beta_anneal**step) # anneal process to avoid mode collapse
        with torch.no_grad():
            energy, vortices= xy.energy(sample, args.ham, args.lattice,
                                  args.boundary)
            loss = log_prob + beta * energy # construct loss function(free energy）from configurations

        assert not energy.requires_grad
        assert not loss.requires_grad
        loss_reinforce = torch.mean((loss - loss.mean()) * log_prob)
        loss_reinforce.backward() # back propagation

        if args.clip_grad:
            nn.utils.clip_grad_norm_(params, args.clip_grad)

        optimizer.step()

        if args.lr_schedule:
            scheduler.step(loss.mean())

        train_time += time.time() - train_start_time

        # export physical observables
        if args.print_step and step % args.print_step == 0:
            free_energy_mean = loss.mean() / beta / (args.L**2) # free energy density
            free_energy_std = loss.std() / beta / (args.L**2) 
            entropy_mean = -log_prob.mean() / (args.L**2) # entropy density
            energy_mean = (energy/ (args.L**2)).mean() # energy density
            energy_std= (energy/ (args.L**2)).std()
            vortices = vortices.mean()/args.L **2 # vortices density

            # heat_capacity=(((energy/ (args.L**2))**2).mean()- ((energy/ (args.L**2)).mean())**2)  *(beta**2)
            
            # magnetization
            # mag = torch.cos(sample).sum(dim=(2,3)).mean(dim=0) # M_x (M_x,M_y)=(cos(theta), sin(theta))
            # mag_mean = mag.mean() 
            # mag_sqr_mean = (mag**2).mean()
            # sus_mean = mag_sqr_mean/args.L**2


            # log
            if step > 0:
                sample_time /= args.print_step
                train_time /= args.print_step
            used_time = time.time() - start_time
            # hyperparameters in training
            my_log(
                'step = {}, lr = {:.3g}, loss={:.8g}, beta = {:.8g}, sample_time = {:.3f}, train_time = {:.3f}, used_time = {:.3f}'
                .format(
                    step,
                    optimizer.param_groups[0]['lr'],
                    loss.mean(),
                    beta,
                    sample_time,
                    train_time,
                    used_time,
                ))
            # observables
            my_log(
                'F = {:.8g}, F_std = {:.8g}, E = {:.8g}, E_std={:.8g}, v={:.8g}'
                .format(
                    free_energy_mean.item(),
                    free_energy_std.item(),
                    energy_mean.item(),
                    energy_std.item(),
                    vortices.item(),
                ))

            sample_time = 0
            train_time = 0
            # save sample
            if args.save_sample and step % args.save_step == 0:
                # save traning state
                # state = {
                #     'sample': sample,
                #     'x_hat': x_hat,
                #     'log_prob': log_prob,
                #     'energy': energy,
                #     'loss': loss,
                # }
                # torch.save(state, '{}_save/{}.sample'.format(
                #     args.out_filename, step))
                
                # Recognize the Phase Transition  
                # helicity
                with torch.no_grad():
                    correlations = helicity(sample)
                helicity_modulus = - ((energy/ args.L**2).mean() ) - (args.beta * correlations**2 / args.L**2).mean()
                my_log('Rho={:.8g}'.format(helicity_modulus.item()))
                # correlation length
                # with torch.no_grad():
                #    r_alg, r_exp, pcov= correlator.correlator(sample)
                # my_log('Rho={:.8g},r_alg={:.8g}, r_exp={:.8g}, r_exp_std={:.8g}'
                #     .format(
                #         helicity_modulus.item(),
                #         r_alg,
                #         r_exp,
                #         pcov[1,1]
                #     ))

                # save configurations
                sample_array=sample.cpu().numpy()
                np.savetxt('{}_save/sample{}.txt'.format(args.out_filename, step),sample_array.reshape(args.batch_size,-1))
                # save observables
                np.savetxt('{}_save/results{}.csv'.format(args.out_filename, step),[beta,step,
                    free_energy_mean,
                    free_energy_std,
                    energy_mean,
                    energy_std,
                    vortices,                   
                    helicity_modulus,
                    ])

        # save net
        if (args.out_filename and args.save_step
                and step % args.save_step == 0):
            state = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            if args.lr_schedule:
                state['scheduler'] = scheduler.state_dict()
            torch.save(state, '{}_save/{}.state'.format(
                args.out_filename, step))

        # visualization in each visual_step
        if (args.out_filename and args.visual_step
                and step % args.visual_step == 0):
            # torchvision.utils.save_image(
            #     sample,
            #     '{}_img/{}.png'.format(args.out_filename, step),
            #     nrow=int(sqrt(sample.shape[0])),
            #     padding=0,
            #     normalize=True)
            # print sample
            if args.print_sample:
                x_hat_alpha = x_hat[:,0,:,:].view(x_hat.shape[0], -1).cpu().numpy() # alpha
                x_hat_std1 = np.std(x_hat_alpha, axis=0).reshape([args.L] * 2) 
                x_hat_beta= x_hat[:,1,:,:].view(x_hat.shape[0], -1).cpu().numpy() # beta
                x_hat_std2 = np.std(x_hat_beta, axis=0).reshape([args.L] * 2)

                energy_np = energy.cpu().numpy()
                energy_count = np.stack(
                    np.unique(energy_np, return_counts=True)).T

                my_log(
                    '\nsample\n{}\nalpha\n{}\nbeta\n{}\nlog_prob\n{}\nenergy\n{}\nloss\n{}\nalpha_std\n{}\nbeta_std\n{}\nenergy_count\n{}\n'
                    .format(
                        sample[:args.print_sample, 0],
                        x_hat[:args.print_sample, 0],
                        x_hat[:args.print_sample, 1],
                        log_prob[:args.print_sample],
                        energy[:args.print_sample],
                        loss[:args.print_sample],
                        x_hat_std1,
                        x_hat_std2,
                        energy_count,
                    ))
            # print gradient
            if args.print_grad:
                my_log('grad max_abs min_abs mean std')
                for name, param in named_params:
                    if param.grad is not None:
                        grad = param.grad
                        grad_abs = torch.abs(grad)
                        my_log('{} {:.3g} {:.3g} {:.3g} {:.3g}'.format(
                            name,
                            torch.max(grad_abs).item(),
                            torch.min(grad_abs).item(),
                            torch.mean(grad).item(),
                            torch.std(grad).item(),
                        ))
                    else:
                        my_log('{} None'.format(name))
                my_log('')


if __name__ == '__main__':
    main()
