"""
Custom Pytorch activation functions and Modules for complex-valued tensors.

Author: Natalie Klein
"""

from torch.nn import Module
from torch.nn.functional import prelu, relu, tanh
from torch import sigmoid
import torch

def complex_dual_activ(input, activ, kwarg_dict):
    """
    Activations that apply separately to real and imaginary parts with activation activ.
    """
    output = torch.zeros_like(input)
    output.real = activ(input.real, **kwarg_dict)
    output.imag = activ(input.imag, **kwarg_dict)
    return output

def complex_phase_tanh(input):
    """
    Hirose paper; amp-phase split type activation.
    """
    output = torch.zeros_like(input)
    r = torch.tanh(input.abs())
    output.real = r*torch.cos(input.angle())
    output.imag = r*torch.sin(input.angle())
    return output

def complex_cardioid(input):
    """
    Stella Xu paper; amplitude is modulated by phase, extension of ReLU.
    """
    return 0.5 * (1 + torch.cos(input.angle())) * input

class ComplexReLU(Module):

    def forward(self, input):
        return complex_dual_activ(input, relu, {})

class ComplexSigmoid(Module):

    def forward(self, input):
        return complex_dual_activ(input, sigmoid, {})

class ComplexTanh(Module):

    def forward(self, input):
        return complex_dual_activ(input, tanh, {})

class ComplexPReLU(Module):

    def __init__(self, num_parameters=1, init=0.25, device=None, dtype=None):
        self.num_parameters = num_parameters
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ComplexPReLU, self).__init__()
        self.weight = torch.nn.Parameter(torch.empty(num_parameters, **factory_kwargs).fill_(init))

    def forward(self, input):
        return complex_dual_activ(input, prelu, {'weight': self.weight})
    
class ComplexPhaseTanh(Module):

    def forward(self, input):
        return complex_phase_tanh(input)

class ComplexCardioid(Module):

    def forward(self, input):
        return complex_cardioid(input)