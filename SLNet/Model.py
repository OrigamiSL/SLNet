# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from SLNet.Modules import *
from SLNet.embed import *
from utils.RevIN import RevIN
from loguru import logger


class Encoder_process(nn.Module):
    def __init__(self, input_len, layer_num, patch_size, d_model=4, dropout=0.05):
        super(Encoder_process, self).__init__()
        self.input_len = input_len
        self.patch_size = patch_size
        self.layer_num = layer_num
        self.d_model = d_model

        self.Embed1 = DataEmbedding(1, d_model)

        self.encoders1 = ([Encoder_Cross(patch_size[0] * d_model, dropout, split=True)
                           for _ in range(layer_num - 1)] +
                          [Encoder_Cross(patch_size[0] * d_model, dropout, split=False)])
        self.encoders1 = nn.ModuleList(self.encoders1)
        # self.encoders2 = [Encoder_Cross(patch_size[0] * d_model, dropout, split=False)
        #                   for _ in range(layer_num)]
        # self.encoders2 = nn.ModuleList(self.encoders2)

        self.Embedl1 = DataEmbeddingL(patch_size[1], patch_size[0] * d_model)
        self.Embedl2 = DataEmbeddingL(patch_size[2], patch_size[0] * d_model)
        if len(self.patch_size) == 4:
            self.Embedl3 = DataEmbeddingL(patch_size[3], patch_size[0] * d_model)

        self.Lencoders = [Long_Encoder(patch_size[0] * d_model, dropout)
                          for _ in range(2 + layer_num)]
        self.Lencoders = nn.ModuleList(self.Lencoders)

    def forward(self, x_enc, x_long_1, x_long_2, x_long_3):
        x_enc = self.Embed1(x_enc.unsqueeze(-1)).transpose(1, 2)

        B, V, L, D = x_enc.shape
        x_patch_attn = x_enc.contiguous().view(B, V, -1, self.patch_size[0] * D)

        x_lenc_1 = self.Embedl1(x_long_1)
        x_lenc_2 = self.Embedl2(x_long_2)
        if x_long_3 is not None:
            x_lenc_3 = self.Embedl3(x_long_3)
            x_lenc_2 = self.Lencoders[0](x_lenc_2, x_lenc_3)
        x_lenc_1 = self.Lencoders[1](x_lenc_1, x_lenc_2)

        encoder_out_list = []
        for i in range(self.layer_num):
            # _, x_patch_attn = self.encoders1[i](x_patch_attn)
            # x_out, x_patch_attn = self.encoders2[i](x_patch_attn)
            x_out, x_patch_attn = self.encoders1[i](x_patch_attn)
            x_out_l = self.Lencoders[i + 2](x_out, x_lenc_1)
            x_out = torch.cat([x_out.clone(), x_out_l], dim=-2)  # B, V, 2 * L', D
            encoder_out_list.append(x_out)

        return encoder_out_list


class Decoder_process(nn.Module):
    def __init__(self, input_len, layer_num=3, patch_size=12, d_model=4, dropout=0.05):
        super(Decoder_process, self).__init__()
        self.patch_size = patch_size
        self.layer_num = layer_num
        self.d_model = d_model // (2 ** (self.layer_num - 1))
        self.b_patch_size = self.patch_size * 2 ** (self.layer_num - 1)
        self.decoders1 = ([Decoder(self.patch_size * d_model, dropout, split=True)
                           for i in range(layer_num - 1)] +
                          [Decoder(self.patch_size * d_model, dropout, split=False)])
        self.decoders1 = nn.ModuleList(self.decoders1)
        # self.decoders2 = [Decoder(self.patch_size * d_model, dropout, split=False)
        #                   for i in range(layer_num)]
        # self.decoders2 = nn.ModuleList(self.decoders2)
        self.projection_out = nn.Linear(d_model, 1)
        self.projection_repr = nn.Linear(d_model, self.d_model)
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, enc_list, y_dec):
        B, V, L, D = y_dec.shape
        y_dec = y_dec.contiguous().view(B, V, -1, self.b_patch_size * self.d_model)

        for i in range(self.layer_num):
            y_dec = self.decoders1[i](y_dec, enc_list[-1 - i])
            # y_dec = self.decoders2[i](y_dec, enc_list[-1 - i])
        y_dec = y_dec.contiguous().view(B, V, L, -1)
        x_out = self.projection_out(y_dec)
        y_repr = self.norm(self.projection_repr(y_dec))
        return x_out, y_repr


class Model(nn.Module):
    def __init__(self, pred_len, layer_num, basic_input,
                 patch_size, boundary, bins, d_model, dropout):
        super(Model, self).__init__()
        self.basic_input = basic_input
        self.pred_len = pred_len
        self.patch_size = patch_size
        self.boundary = boundary
        self.layer_num = layer_num
        self.bins = bins
        self.d_model = d_model

        self.b_patch_size = self.patch_size * 2 ** (self.layer_num - 1)
        self.revin = RevIN()
        self.Embed = nn.Linear(1, d_model // (2 ** (self.layer_num - 1)))
        self.Embed_norm = nn.LayerNorm(d_model // (2 ** (self.layer_num - 1)))
        self.Dec_pos = PositionalEmbedding(d_model // (2 ** (self.layer_num - 1)))
        self.dropout = nn.Dropout(dropout)

        self.Encoder_process = (
            Encoder_process(basic_input, layer_num, patch_size, d_model, dropout))
        self.Decoder_process = (
            Decoder_process(basic_input, layer_num, patch_size[0], d_model, dropout))

    def forward(self, x, GT, flag='train'):
        if x.ndim == 2:
            x = x.unsqueeze(0)
            GT = GT.unsqueeze(0)
        # IN
        B, _, V = x.shape

        self.revin(x[:, -self.boundary[0]:, :], 'stats')
        x_enc = self.revin(x[:, -self.boundary[0]:, :], 'norm')  # [B Lin V]
        # x_enc = self.revin(x[:, -self.basic_input:, :], 'norm')
        if len(self.boundary) == 2:
            x_long_1 = self.revin(x[:, -self.boundary[0]:, :], 'norm')
            x_long_2 = self.revin(x[:, -self.boundary[1]:, :], 'norm')
            x_long_3 = self.revin(x[:,
                                  -self.patch_size[-1] * (x.shape[1] // self.patch_size[-1]):,
                                  :], 'norm')
        else:
            x_long_1 = self.revin(x[:, -self.boundary[0]:, :], 'norm')
            x_long_2 = self.revin(x[:,
                                  -self.patch_size[-1] * (x.shape[1] // self.patch_size[-1]):,
                                  :], 'norm')
            x_long_3 = None
        GT = self.revin(GT, 'norm')

        # Tokenization
        data_max = torch.max(x_long_1, dim=1, keepdim=True)[0]  # B, 1, V
        data_min = torch.min(x_long_1, dim=1, keepdim=True)[0]  # B, 1, V
        data_interval = ((data_max - data_min) / self.bins).expand(B, self.bins, V)
        Word = torch.cumsum(data_interval, dim=1) + data_min
        Word = torch.cat([data_min, Word], dim=1).unsqueeze(-1)  # B, bins + 1, V, 1
        Word_Embed = self.dropout(self.Embed_norm(self.Embed(Word)))  # B, bins + 1, V, D
        if flag == 'train':
            GT_max = data_max.expand(B, self.pred_len, V)
            GT_min = data_min.expand(B, self.pred_len, V)
            GT = torch.where(GT > GT_max, GT_max, GT)
            GT = torch.where(GT < GT_min, GT_min, GT)
            GT = torch.round(self.bins * (GT - GT_min) / (GT_max - GT_min + 1e-6))
            GT = GT.transpose(1, 2)  # B, V, L

        # Encoder
        enc_list = self.Encoder_process(x_enc, x_long_1, x_long_2, x_long_3)

        # Y initialization
        y = torch.zeros([B, self.pred_len, V]).to(x.device)
        y_dec = y
        y_dec = self.Embed(y_dec.unsqueeze(-1)) + self.Dec_pos(y_dec)

        # Decoder
        x_out, y_repr = self.Decoder_process(enc_list, y_dec.transpose(1, 2))  # B, V, L_pred, D

        # Detokenization and RevIN
        Match_score = torch.einsum("bvld,bnvd->bvln", y_repr, Word_Embed)
        Pred = self.revin(x_out.squeeze(-1).transpose(1, 2), 'denorm')
        if flag == 'train':
            Match_score = Match_score.contiguous().view(-1, self.bins + 1)
            GT = GT.contiguous().view(-1)
            return Pred, Match_score, GT.long()
        else:
            Match_score = torch.softmax(Match_score.transpose(1, 2), dim=-1)
            Best_value, Best_index = Match_score.topk(dim=-1, k=1)[0], Match_score.topk(dim=-1, k=1)[1]
            return Pred, Best_value.squeeze(-1), Best_index.squeeze(-1)  # B, L, V; B, L, V
