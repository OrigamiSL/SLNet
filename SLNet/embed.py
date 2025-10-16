import torch
import torch.nn as nn

import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, start_pos=0, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.start_pos = start_pos

    def forward(self, x):
        pos = self.pe[:, self.start_pos:x.size(1) + self.start_pos].unsqueeze(2)
        return pos


class TokenEmbedding(nn.Module):
    def __init__(self, channel, d_model, start_pos=0):
        super(TokenEmbedding, self).__init__()
        if start_pos == 0:
            self.tokenConv = nn.Conv2d(in_channels=channel, out_channels=d_model, kernel_size=1, padding=0)
        else:
            self.tokenConv = nn.Conv2d(in_channels=channel, out_channels=d_model, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, channel, d_model, start_pos=0):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(channel=channel, d_model=d_model, start_pos=start_pos)
        self.position_embedding = PositionalEmbedding(d_model=d_model, start_pos=start_pos)

    def forward(self, x):
        pos = self.position_embedding(x)
        x = self.value_embedding(x) + pos
        return x


class DataEmbeddingD(nn.Module):
    def __init__(self, bins, d_model, start_pos=0):
        super(DataEmbeddingD, self).__init__()

        self.value_embedding = nn.Embedding(bins + 1, d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model, start_pos=start_pos)

    def forward(self, x):  # B, L, V
        pos = self.position_embedding(x)
        x = self.value_embedding(x) + pos
        return x


class DataEmbeddingL(nn.Module):
    def __init__(self, patch, d_model):
        super(DataEmbeddingL, self).__init__()
        self.patch = patch
        self.value_embedding = nn.Linear(patch, d_model)
        self.pre_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, L, V = x.shape
        x = x.transpose(1, 2).contiguous().view(B, V, -1, self.patch)
        x = self.value_embedding(x)
        return self.pre_norm(x)
