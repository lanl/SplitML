"""
Transformers for signal denoising

Author: Amber Day
"""

import torch
import torch.nn as nn
from splitml.layers import ComplexLinear, PositionalEncoding, ComplexEncoderLayer, ComplexDropout, EncoderLayer


class ComplexTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(ComplexTransformer, self).__init__()
        self.encoder_embedding = ComplexLinear(1, d_model)
        self.decoder_embedding = ComplexLinear(1, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([ComplexEncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = ComplexLinear(d_model, 1)
        self.dropout = ComplexDropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src):
        src_mask, tgt_mask = None, None

        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src).real),self.positional_encoding(self.encoder_embedding(src).imag))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        output = self.fc(enc_output)
        return output

class DRTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(DRTransformer, self).__init__()
        self.encoder_embedding = nn.Linear(1, d_model)
        self.decoder_embedding = nn.Linear(1, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src):
        src_mask, tgt_mask = None, None

        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)


        output = self.fc(enc_output)
        return output

class DRCTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(DRCTransformer, self).__init__()
        self.encoder_embedding = nn.Linear(2, d_model)
        self.decoder_embedding = nn.Linear(2, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, 2)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src):
        src_mask, tgt_mask = None, None

        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        output = self.fc(enc_output)
        return output