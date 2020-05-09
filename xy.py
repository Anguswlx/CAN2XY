# 2D classical XY model

import torch
import math

def energy(sample, ham, lattice, boundary):
    sample=sample*2*math.pi # angular

    energy = xyenergy(sample, ham, lattice, boundary)

    vortices = vortex(sample, boundary)

    return energy, vortices

def xyenergy(sample, ham, lattice, boundary):
    # calculate the energy
    term = torch.cos(sample[:, :, 1:, :] - sample[:, :, :-1, :])
    term = term.sum(dim=(1, 2, 3))
    output = term
    term = torch.cos(sample[:, :, :, 1:] - sample[:, :, :, :-1])
    term = term.sum(dim=(1, 2, 3))
    output += term
    if lattice == 'tri':
        term = torch.cos(sample[:, :, 1:, 1:] - sample[:, :, :-1, :-1])
        term = term.sum(dim=(1, 2, 3))
        output += term

    if boundary == 'periodic':
        term = torch.cos(sample[:, :, 0, :] - sample[:, :, -1, :])
        term = term.sum(dim=(1, 2))
        output += term
        term = torch.cos(sample[:, :, :, 0] - sample[:, :, :, -1])
        term = term.sum(dim=(1, 2))
        output += term
        if lattice == 'tri':
            term = torch.cos(sample[:, :, 0, 1:] - sample[:, :, -1, :-1])
            term = term.sum(dim=(1, 2))
            output += term
            term = torch.cos(sample[:, :, 1:, 0] - sample[:, :, :-1, -1])
            term = term.sum(dim=(1, 2))
            output += term
            term = torch.cos(sample[:, :, 0, 0] - sample[:, :, -1, -1])
            term = term.sum(dim=1)
            output += term
            
    if ham == 'fm':
        output *= -1
    
    return output

def vortex(sample, boundary):
# count the number of vortices
    theta=torch.fmod(sample[:, :, 1:, :-1] - sample[:, :, :-1, :-1],math.pi)-(sample[:, :, 1:, :-1] - sample[:, :, :-1, :-1])//math.pi*math.pi
    theta+=torch.fmod(sample[:, :, :-1, :-1]-sample[:, :, :-1, 1:],math.pi)-(sample[:, :, :-1, :-1]-sample[:, :, :-1, 1:])//math.pi*math.pi
    theta+=torch.fmod(sample[:, :, :-1, 1:] - sample[:, :, 1:, 1:],math.pi)-(sample[:, :, :-1, 1:] - sample[:, :, 1:, 1:])//math.pi*math.pi
    theta+=torch.fmod(sample[:, :, 1:, 1:]-sample[:, :, 1:, :-1],math.pi)-(sample[:, :, 1:, 1:]-sample[:, :, 1:, :-1])//math.pi*math.pi
    vortices=(theta.abs()/2/math.pi).sum(dim=(1, 2, 3))

    if boundary == 'periodic':
        theta1=torch.fmod(sample[:, :, 1:, -1]-sample[:, :, :-1, -1],math.pi)-(sample[:, :, 1:, -1]-sample[:, :, :-1, -1])//math.pi*math.pi
        theta1+=torch.fmod(sample[:, :, :-1, -1]-sample[:, :, :-1, 0],math.pi)-(sample[:, :, :-1, -1]-sample[:, :, :-1, 0])//math.pi*math.pi
        theta1+=torch.fmod(sample[:, :, :-1, 0]-sample[:, :, 1:, 0],math.pi)-(sample[:, :, :-1, 0]-sample[:, :, 1:, 0])//math.pi*math.pi
        theta1+=torch.fmod(sample[:, :, 1:, 0]-sample[:, :, 1:, -1],math.pi)-(sample[:, :, 1:, 0]-sample[:, :, 1:, -1])//math.pi*math.pi
        vortices+=(theta1.abs()/2/math.pi).sum(dim=(1, 2))
        # calculate the transverse

        theta2=torch.fmod(sample[:, :, 0, :-1]-sample[:, :, -1, :-1],math.pi)-(sample[:, :, 0, :-1]-sample[:, :, -1, :-1])//math.pi*math.pi
        theta2+=torch.fmod(sample[:, :, -1, :-1]-sample[:, :, -1, 1:],math.pi)-(sample[:, :, -1, :-1]-sample[:, :, -1, 1:])//math.pi*math.pi
        theta2+=torch.fmod(sample[:, :, -1, 1:]-sample[:, :, 0, 1:],math.pi)-(sample[:, :, -1, 1:]-sample[:, :, 0, 1:])//math.pi*math.pi
        theta2+=torch.fmod(sample[:, :, 0, 1:]-sample[:, :, 0, :-1],math.pi)-(sample[:, :, 0, 1:]-sample[:, :, 0, :-1])//math.pi*math.pi
        vortices+=(theta2.abs()/2/math.pi).sum(dim=(1, 2))
        # calculate the longitudinal

        theta3=torch.fmod(sample[:, :, 0, -1]-sample[:, :, -1, -1],math.pi)-(sample[:, :, 0, -1]-sample[:, :, -1, -1])//math.pi*math.pi
        theta3+=torch.fmod(sample[:, :, -1, -1]-sample[:, :, -1, 0],math.pi)-(sample[:, :, -1, -1]-sample[:, :, -1, 0])//math.pi*math.pi
        theta3+=torch.fmod(sample[:, :, -1, 0]-sample[:, :, 0, 0],math.pi)-(sample[:, :, -1, 0]-sample[:, :, 0, 0])//math.pi*math.pi
        theta3+=torch.fmod(sample[:, :, 0, 0]-sample[:, :, 0, -1],math.pi)-(sample[:, :, 0, 0]-sample[:, :, 0, -1])//math.pi*math.pi
        vortices+=(theta3.abs()/2/math.pi).sum(dim=1)
        # calculate the corner

    vortices=vortices/2
    return vortices
