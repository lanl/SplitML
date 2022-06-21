"""
Custom Pytorch loss functions for complex data.

Author: Natalie Klein
"""


import torch

def logmse(input, target):
    """
    log MSE loss function for regression. (similar to SDR)
    """
    sq_resid = torch.square(input - target).sum(-1)
    return 10 * torch.mean(torch.log(sq_resid))

def complex_mse(input, target):
    """
    Complex MSE loss function for regression.
    """
    resid = input - target
    return torch.mean(resid * torch.conj(resid)).real

def complex_logmse(input, target):
    """
    Complex logMSE loss function for regression.
    """
    resid = input - target
    r_sq_resid = torch.square(resid.real).sum(-1)
    i_sq_resid = torch.square(resid.imag).sum(-1)
    return 10 * torch.mean(torch.log(r_sq_resid + 1e-6)) + 10 * torch.mean(torch.log(i_sq_resid + 1e-6))

def complex_norm(x):
    return torch.sum(x * torch.conj(x), axis=-1)

def complex_innerprod(input, target):
    num = torch.sum(input * torch.conj(target), axis=-1)
    denom = torch.sqrt(complex_norm(input).real) * torch.sqrt(complex_norm(target).real)
    return -torch.mean(torch.abs(num/denom))

def complex_innerprod_nonorm(input, target):
    num = torch.sum(input * torch.conj(target), axis=-1)
    return -torch.mean(torch.abs(num))

def complex_sdr(input, target):
    """
    Complex SDR loss function for regression.
    """
    resid = input - target
    num = target.abs()**2
    denom = resid.abs()**2
    return -torch.mean(torch.log10(num) - torch.log10(denom))