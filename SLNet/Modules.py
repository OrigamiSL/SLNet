import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class Attn_PatchLevel(nn.Module):
    def __init__(self, d_model, dropout=0.1, head=4):
        super(Attn_PatchLevel, self).__init__()
        self.head = head
        self.query_projection = nn.Linear(d_model, d_model)
        self.kv_projection = nn.Linear(d_model, d_model)

        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        B, V, P, D = queries.shape
        _, _, S, D = keys.shape

        if self.head == 1:
            queries = self.query_projection(queries)
            keys = self.kv_projection(keys)
            values = self.kv_projection(values)
            scale = 1. / math.sqrt(D)

            scores = torch.einsum("bvpd,bvsd->bvps", queries, keys)  # [B V P P] / [B P V V]

            attn = self.dropout(torch.softmax(scale * scores, dim=-1))  # [B V P P] / [B P V V]
            out = torch.einsum("bvps,bvsd->bvpd", attn, values)  # [B V P D] / [B V P D]
        else:
            H = self.head
            d = D // H
            queries = self.query_projection(queries).contiguous().view(B, V, P, H, d)
            keys = self.kv_projection(keys).contiguous().view(B, V, S, H, d)
            values = self.kv_projection(values).contiguous().view(B, V, S, H, d)
            scale = 1. / math.sqrt(d)

            scores = torch.einsum("bvphd,bvshd->bvhps", queries, keys)  # [B V H P P] / [B P H V V]
            attn = self.dropout(torch.softmax(scale * scores, dim=-1))  # [B V H P P] / [B P H V V]
            out = torch.einsum("bvhps,bvshd->bvphd", attn, values)  # [B V P H d] / [B P V H D]
            out = out.contiguous().view(B, V, P, -1)

        return self.out_projection(out)  # [B V P D] / [B P V D]


class Encoder_Cross(nn.Module):
    def __init__(self, d_model, dropout=0.1, split=False):
        super(Encoder_Cross, self).__init__()
        self.patch_dim = d_model
        self.attn1 = Attn_PatchLevel(self.patch_dim, dropout)
        self.attn2 = Attn_PatchLevel(self.patch_dim, dropout)

        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm(self.patch_dim)
        self.norm2 = nn.LayerNorm(self.patch_dim)
        self.norm3 = nn.LayerNorm(self.patch_dim)
        self.norm4 = nn.LayerNorm(self.patch_dim)

        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(self.patch_dim, 2 * self.patch_dim)
        self.linear2 = nn.Linear(2 * self.patch_dim, self.patch_dim)
        self.linear3 = nn.Linear(self.patch_dim, 2 * self.patch_dim)
        self.linear4 = nn.Linear(2 * self.patch_dim, self.patch_dim)
        self.transfer_linear = nn.Linear(self.patch_dim * 2, self.patch_dim)
        self.split = split

    def forward(self, x):
        B, V, P, D = x.shape
        m = self.dropout(self.attn1(x, x, x))
        x = self.norm1(x + m)
        y = x.clone()
        y = self.activation(self.linear1(y))
        x = x + self.dropout(self.linear2(y))

        x = self.norm2(x).permute(0, 2, 1, 3)  # B, P, V, D
        n = self.dropout(self.attn2(x, x, x))
        x = self.norm3(x + n)
        z = x.clone()
        z = self.activation(self.linear3(z))
        x = x + self.dropout(self.linear4(z))
        x = self.norm4(x)
        x_out = x.permute(0, 2, 1, 3)  # B, V, P, D

        if self.split:
            x_next = x_out.contiguous().view(B, V, P // 2, 2 * D)
            x_next = self.transfer_linear(x_next)
        else:
            x_next = x_out

        return x_out, x_next


class Long_Encoder(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(Long_Encoder, self).__init__()
        self.patch_dim = d_model
        self.attn1 = Attn_PatchLevel(self.patch_dim, dropout)
        self.attn2 = Attn_PatchLevel(self.patch_dim, dropout)

        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm(self.patch_dim)
        self.norm2 = nn.LayerNorm(self.patch_dim)
        self.norm3 = nn.LayerNorm(self.patch_dim)
        self.norm4 = nn.LayerNorm(self.patch_dim)

        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(self.patch_dim, 2 * self.patch_dim)
        self.linear2 = nn.Linear(2 * self.patch_dim, self.patch_dim)
        self.linear3 = nn.Linear(self.patch_dim, 2 * self.patch_dim)
        self.linear4 = nn.Linear(2 * self.patch_dim, self.patch_dim)

    def forward(self, x, y):
        y = y.permute(0, 2, 1, 3)  # B, P, V, D
        y = self.norm1(y + self.dropout(self.attn1(y, y, y)))
        n = y.clone()
        n = self.activation(self.linear1(n))
        y = y + self.dropout(self.linear2(n))
        y = self.norm2(y).permute(0, 2, 1, 3)  # B, V, P, D

        m = self.dropout(self.attn2(x, y, y))
        x = self.norm3(x + m)
        z = x.clone()
        z = self.activation(self.linear3(z))
        x = x + self.dropout(self.linear4(z))

        x_out = self.norm4(x)  # B, V, P, D

        return x_out


class Decoder(nn.Module):
    def __init__(self, d_model, dropout=0.1, split=False):
        super(Decoder, self).__init__()
        self.patch_dim = d_model
        self.attn = Attn_PatchLevel(self.patch_dim, dropout)

        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm(self.patch_dim)
        self.norm2 = nn.LayerNorm(self.patch_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(self.patch_dim, 2 * self.patch_dim)
        self.linear2 = nn.Linear(2 * self.patch_dim, self.patch_dim)
        self.transfer_linear = nn.Linear(self.patch_dim // 2, self.patch_dim)
        self.split = split

    def forward(self, x, y):
        B, V, P, D = x.shape
        attn_x = self.attn(x, y, y)
        x = self.norm1(x + self.dropout(attn_x))
        z = x.clone()
        z = self.activation(self.linear1(z))
        x = x + self.dropout(self.linear2(z))
        x = self.norm2(x)

        if self.split:
            x = x.contiguous().view(B, V, P * 2, D // 2)
            x = self.transfer_linear(x)

        return x
