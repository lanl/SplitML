"""
Dual-real complex ConvTasNet, based on torchaudio implementation of ConvTasNet.
Treats real/imag as two channels rather than separate inputs.

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
import torch.nn as nn
from torch.nn import Module, ModuleList, Sequential
from torchaudio.transforms import Spectrogram
from torchaudio.functional import inverse_spectrogram

class DualRealChannelsConvTasNet(Module):
    """
    Extension of ConvTasNet to handle complex-valued as [real, imaginary] in two channels; similar to torchaudio version but with some
    added options and capabilities.
    """

    def __init__(self, 
                # encoder/decoder settings
                enc_kernel_size=16, enc_num_feats=512, enc_depth=3, 
                enc_activate=nn.PReLU, stft_layer=False,
                # mask generator network parameters
                msk_kernel_size=3, msk_num_feats=128, msk_num_hidden_feats=32, # was 512
                msk_num_layers=8, msk_num_stacks=3, dc_activ=nn.PReLU, msk_activate=nn.Sigmoid,
                # weight sharing flags - no effect
                share_encoder=True, share_decoder=True, share_mask=True):

        super(DualRealChannelsConvTasNet, self).__init__()

        self.enc_kernel_size = enc_kernel_size
        self.enc_num_feats = enc_num_feats
        enc_stride = enc_kernel_size // 2
        self.enc_stride = enc_stride

        # Encoder
        self.encoder = ModuleList([])
        # first encoder layer
        self.encoder.append(nn.Conv1d(2, enc_num_feats, enc_kernel_size, 
                                               bias=False, stride=enc_stride))

        # remaining encoder layers
        if enc_depth > 1:
            for _ in range(enc_depth - 1):
                self.encoder.append(nn.Conv1d(enc_num_feats, enc_num_feats, 3, bias=False)) # TODO use enc_kernel_size or other?
                self.encoder.append(enc_activate())

        # Mask generator
        self.mask_generator = MaskGenerator(input_dim=enc_num_feats, kernel_size=msk_kernel_size, num_feats=msk_num_feats,
                                                    num_hidden=msk_num_hidden_feats, num_layers=msk_num_layers,
                                                    num_stacks=msk_num_stacks, activ=dc_activ, msk_activate=msk_activate)
        
        # Decoder
        self.decoder = ModuleList([])
            # decoder layers
        if enc_depth > 1:
            for _ in range(enc_depth - 1):
                self.decoder.append(nn.ConvTranspose1d(enc_num_feats, enc_num_feats, 3, bias=False))
                self.decoder.append(enc_activate())
        # final decoder layer 
        self.decoder.append(nn.ConvTranspose1d(enc_num_feats, 2, enc_kernel_size, bias=False, stride=enc_stride))

    def _align_num_frames_with_strides(self, input):
        """
        Padding
        """
        batch_size, num_channels, num_frames = input.shape
        is_odd = self.enc_kernel_size % 2
        num_strides = (num_frames - is_odd) // self.enc_stride
        num_remainings = num_frames - (is_odd + num_strides * self.enc_stride)
        if num_remainings == 0:
            return input, 0

        num_paddings = self.enc_stride - num_remainings
        pad = torch.zeros(
            batch_size,
            num_channels,
            num_paddings,
            dtype=input.dtype,
            device=input.device,
        )
        return torch.cat([input, pad], 2), num_paddings

    def forward(self, input):
        """Perform source separation. Generate audio source waveforms.

        Args:
            input (list of torch.Tensor): length 2 for real, imag: 3D Tensor with shape [batch, channel==1, frames]

        Returns:
            Tensor: list 2 for real imag: 3D Tensor with shape [batch, channel==num_sources, frames]
        """

        # reshape complex input
        input = torch.cat([input.real, input.imag], dim=1)
        #if input.ndim != 3 or input.shape[1] != 1:
        #    raise ValueError(f"Expected 3D tensor (batch, channel==2, frames). Found: {input.shape}")

        # B: batch size
        # L: input frame length
        # L': padded input frame length
        # F: feature dimension
        # M: feature frame length
        # S: number of sources
        feats, num_pads = self._align_num_frames_with_strides(input)  # B, 2, L'
        batch_size, num_padded_frames = feats.shape[0], feats.shape[2]
        for e in self.get_submodule('encoder'):
            feats = e(feats)  # B, F, M
        masked = self.get_submodule('mask_generator')(feats) * feats.unsqueeze(1)  # B, S, F, M
        masked = masked.view(batch_size * 2, self.enc_num_feats, -1)  # B*S, F, M
        for d in self.get_submodule('decoder'):
            masked = d(masked)
        output = masked.view(batch_size, 2, 2, num_padded_frames)  # B, S, L'
        if num_pads > 0:
            output = output[..., :-num_pads]  # B, S, L
        output = torch.complex(output[:, :, 0, :], output[:, :, 1, :])
        return output

class MaskGenerator(Module):
    """
    Temporal convolution network (TCN) separation module; adapted from torchaudio to potentially return complex-valued mask.
    """

    def __init__(self, input_dim, kernel_size, num_feats, num_hidden, num_layers, num_stacks, activ=nn.PReLU, msk_activate=nn.Sigmoid):
        super(MaskGenerator, self).__init__()

        self.input_dim = input_dim
        self.input_conv = nn.Conv1d(input_dim, num_feats, 1)
        self.receptive_field = 0
        self.conv_layers = ModuleList([])
        for s in range(num_stacks):
            for l in range(num_layers):
                multi = 2 ** l
                self.conv_layers.append(ConvBlock(num_feats, num_hidden, kernel_size, dilation=multi, activ=activ,
                                                        padding=multi, no_residual=(l == (num_layers - 1) and s == (num_stacks - 1))))
                self.receptive_field += kernel_size if s == 0 and l == 0 else (kernel_size - 1) * multi        

        self.output_prelu = nn.PReLU()
        self.output_conv = nn.Conv1d(num_feats, input_dim * 2, 1)
        self.mask_activate = msk_activate()

    def forward(self, input):   
        batch_size = input.shape[0]
        output = self.input_conv(input)
        for layer in self.conv_layers:
            residual, skip = layer(output)
            if residual is not None:
                output = output + residual
            output = output + skip
        output = self.output_prelu(output)
        output = self.output_conv(output)
        output = self.mask_activate(output)
        return output.view(batch_size, 2, self.input_dim, -1)
        
class ConvBlock(Module):
    """
    1D convolutional block; no GroupNorm.
    """

    def __init__(self, io_channels, hidden_channels, kernel_size, padding, activ=nn.PReLU, dilation=1, no_residual=False):
        super(ConvBlock, self).__init__()

        self.conv_layers = Sequential(nn.Conv1d(io_channels, hidden_channels, 1), 
                                      activ(),
                                      nn.Conv1d(hidden_channels, hidden_channels, kernel_size, 
                                                    padding=padding, dilation=dilation, groups=hidden_channels),
                                      activ()
                                     )
        if no_residual:
            self.res_out = None
        else:
            self.res_out = nn.Conv1d(hidden_channels, io_channels, 1)
        self.skip_out = nn.Conv1d(hidden_channels, io_channels, 1)

    def forward(self, input):
        feature = self.conv_layers(input)
        if self.res_out is None:
            residual = None
        else:
            residual = self.res_out(feature)
        skip_out = self.skip_out(feature)
        return residual, skip_out
