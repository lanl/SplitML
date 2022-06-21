"""
Custom Pytorch layers for complex-valued tensors.


ComplexiSTFT based on code from torchaudio.

BSD 2-Clause License

Copyright (c) 2017 Facebook Inc. (Soumith Chintala), 
Copyright (c) 2022, Triad National Security, LLC
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

import torch
from torch.nn import Module, Conv1d, ConvTranspose1d

class ComplexConv1d(Module):
    """ Pytorch Conv1d adapted to complex-valued
    """

    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding = 0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv1d, self).__init__()
        self.conv_r = Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
    def forward(self, input):    
        fwd_rr = self.conv_r(input.real)
        fwd_ri = self.conv_r(input.imag)
        fwd_ir = self.conv_i(input.real)
        fwd_ii = self.conv_i(input.imag)
        output = torch.zeros_like(fwd_rr.type(input.dtype))
        output.real = fwd_rr - fwd_ii
        output.imag = fwd_ri + fwd_ir
        return output

class ComplexConvTranspose1d(Module):
    """ Pytorch ConvTranspose1d adapted to complex-valued
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'):

        super(ComplexConvTranspose1d, self).__init__()

        self.conv_tran_r = ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding,
                                       output_padding, groups, bias, dilation, padding_mode)
        self.conv_tran_i = ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding,
                                       output_padding, groups, bias, dilation, padding_mode)

    def forward(self,input):
        fwd_rr = self.conv_tran_r(input.real)
        fwd_ri = self.conv_tran_r(input.imag)
        fwd_ir = self.conv_tran_i(input.real)
        fwd_ii = self.conv_tran_i(input.imag)
        output = torch.zeros_like(fwd_rr.type(input.dtype))
        output.real = fwd_rr - fwd_ii
        output.imag = fwd_ri + fwd_ir
        return output

class ComplexiSTFT(Module):
    """ torchaudio.functional.inverse_spectrogram, adapted to give complex-valued output
    """

    def __init__(self, n_fft, hop_length, win_length, pad=0):
        super(ComplexiSTFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.pad = pad
        self.window = torch.hann_window(self.win_length)
        self.center = True
        self.length = None

    def forward(self, input):
        shape = input.size()
        spectrogram = input.reshape(-1, shape[-2], shape[-1])
        waveform = torch.istft(
            input=spectrogram,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(input.get_device()),
            center=self.center,
            normalized=False,
            onesided=False,
            length=self.length + 2 * self.pad if self.length is not None else None,
            return_complex=True
        )
        if self.length is not None and self.pad > 0:
            # remove padding from front and back
            waveform = waveform[:, self.pad:-self.pad]

        # unpack batch
        waveform = waveform.reshape(shape[:-2] + waveform.shape[-1:])
        return waveform