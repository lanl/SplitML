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
import math
from torch.nn import Module, Conv1d, ConvTranspose1d, Linear
from splitml.activations import ComplexDropout, ComplexReLU

class ComplexLinear(Module):
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = Linear(in_features, out_features)
        self.fc_i = Linear(in_features, out_features)

    def forward(self, input):
        fwd_rr = self.fc_r(input.real)
        fwd_ri = self.fc_r(input.imag)
        fwd_ir = self.fc_i(input.real)
        fwd_ii = self.fc_i(input.imag)
        output = torch.zeros_like(fwd_rr.type(input.dtype))
        output.real = fwd_rr - fwd_ii
        output.imag = fwd_ri + fwd_ir
        return output

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
            window=self.window.to(input.device),
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
    

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.W_o = torch.nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output
     
class PositionWiseFeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = torch.nn.Linear(d_model, d_ff)
        self.fc2 = torch.nn.Linear(d_ff, d_model)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
        
class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = torch.nn.Identity(d_model)
        self.norm2 = torch.nn.Identity(d_model)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
   

class ComplexAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super(ComplexAttention, self).__init__()
        self.MH = MultiHeadAttention(d_model, num_heads)

    def forward(self, x, mask=None):
        A = x.real
        B = x.imag
        r = self.MH(A,A,A, mask=None) - self.MH(A,B,B,mask=None) - self.MH(B,A,B,mask=None) - self.MH(B,B,A,mask=None)
        i = self.MH(A,A,B, mask=None) + self.MH(A,B,A, mask=None) + self.MH(B,A,A,mask=None) - self.MH(B,B,B,mask=None) 
        attn_output = r + 1j*i
        return attn_output

class ComplexPositionWiseFeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super(ComplexPositionWiseFeedForward, self).__init__()
        self.fc1 = ComplexLinear(d_model, d_ff)
        self.fc2 = ComplexLinear(d_ff, d_model)
        self.relu = ComplexReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class ComplexEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(ComplexEncoderLayer, self).__init__()
        self.self_attn = ComplexAttention(d_model, num_heads) 
        self.feed_forward = ComplexPositionWiseFeedForward(d_model, d_ff)
        self.norm1 = torch.nn.Identity(d_model)   
        self.norm2 = torch.nn.Identity(d_model) 
        self.dropout = ComplexDropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, mask=None) 
        x = self.norm1(x + self.dropout(attn_output.real, attn_output.imag))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output.real, ff_output.imag)) 
        return x.cfloat()
