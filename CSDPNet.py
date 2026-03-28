#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   decoder.py
@Time    :   2021/07/28
@Author  :   deyiwang
@Mail    :   deyiwang@qq.com
@Version :   1.0
@Description:the whole structure of traj-gru
实现步骤:
1, 复现trajgru-----------done
2, 添加attention
3, 添加3d conv(实验)
4, 增加DLA结构(实验)
5, 将gru更改为srnn
'''

import torch
from torch import nn
# from ConvRNN import CGRU_cell, CLSTM_cell
from CBSConv import CBS_Conv
from CTSFF import CTSFF

import einops


class EfficientAdditiveAttnetion(nn.Module):
    """
    Efficient Additive Attention module for SwiftFormer.
    Input: tensor in shape [B, N, D]
    Output: tensor in shape [B, N, D]
    """

    def __init__(self, in_dims=512, token_dim=256, num_heads=2):
        super().__init__()

        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)

        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x):
        query = self.to_query(x)
        key = self.to_key(x)

        query = torch.nn.functional.normalize(query, dim=-1)  # BxNxD
        key = torch.nn.functional.normalize(key, dim=-1)  # BxNxD

        query_weight = query @ self.w_g  # BxNx1 (BxNxD @ Dx1)
        A = query_weight * self.scale_factor  # BxNx1

        A = torch.nn.functional.normalize(A, dim=1)  # BxNx1

        G = torch.sum(A * query, dim=1)  # BxD

        G = einops.repeat(
            G, "b d -> b repeat d", repeat=key.shape[1]
        )  # BxNxD

        out = self.Proj(G * key) + query  # BxNxD

        out = self.final(out)  # BxNxD

        return out



class CGRU_cell(nn.Module):
    """
    ConvGRU Cell
    """
    def __init__(self, shape, input_channels, filter_size, num_features):
        super(CGRU_cell, self).__init__()
        self.shape = shape
        self.input_channels = input_channels
        # kernel_size of input_to_state equals state_to_state
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      2 * self.num_features, self.filter_size, 1,
                      self.padding),
            nn.GroupNorm(2 * self.num_features // 32, 2 * self.num_features))
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features,
                      self.num_features, self.filter_size, 1, self.padding),
            nn.GroupNorm(self.num_features // 32, self.num_features))

    def forward(self, inputs=None, hidden_state=None, seq_len=10):
        # print(inputs.shape)
        # seq_len=10 for moving_mnist
        if hidden_state is not None:
            device = hidden_state.device
        elif inputs is not None:
            device = inputs.device
        else:
            device = self.conv1[0].weight.device
        if hidden_state is None:
            htprev = torch.zeros(inputs.size(1), self.num_features,
                                 self.shape[0], self.shape[1], device=device)
        else:
            htprev = hidden_state
        output_inner = []
        for index in range(seq_len):
            if inputs is None:
                x = torch.zeros(htprev.size(0), self.input_channels,
                                self.shape[0], self.shape[1], device=device)
            else:
                x = inputs[index, ...]

            combined_1 = torch.cat((x, htprev), 1)  # X_t + H_t-1
            gates = self.conv1(combined_1)  # W * (X_t + H_t-1)

            zgate, rgate = torch.split(gates, self.num_features, dim=1)
            # zgate, rgate = gates.chunk(2, 1)
            z = torch.sigmoid(zgate)
            r = torch.sigmoid(rgate)

            combined_2 = torch.cat((x, r * htprev),
                                   1)  # h' = tanh(W*(x+r*H_t-1))
            ht = self.conv2(combined_2)
            ht = torch.tanh(ht)
            htnext = (1 - z) * htprev + z * ht
            output_inner.append(htnext)
            htprev = htnext
            # print(torch.stack(output_inner).shape, htnext.shape)
            # torch.Size([10, 8, 64, 64, 64]) torch.Size([8, 64, 64, 64])
        return torch.stack(output_inner), htnext


class CSDPNet(nn.Module):
    def __init__(self, sizevalue = 2, batchsize = 8, seqNum = 4, filter_size=5, attention_type='ctsff'):
        super().__init__()
        self.seq_len = seqNum
        self.attention_type = attention_type
        #-----encoder
        # stage-1 (self, inc, outc, num_param, stride=1, bias=None):
        # self.conv_1_leaky_1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
        #                                      nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_1_leaky_1 = nn.Sequential(CBS_Conv(inc=1, outc=16, num_param=3),
                                             nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.rnn1_0 = CGRU_cell(shape=(64,64), input_channels=16, filter_size=filter_size, num_features=64)
        # stage-2
        # self.conv_2_leaky_2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
        #                                     nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_2_leaky_2 = nn.Sequential(CBS_Conv(inc=64, outc=64, num_param=3, stride=2),
                                            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.rnn2_0 = CGRU_cell(shape=(32,32), input_channels=64, filter_size=filter_size, num_features=96)
        # stage-3
        # self.conv_3_leaky_3 = nn.Sequential(nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=2, padding=1),
        #                                      nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_3_leaky_3 = nn.Sequential(CBS_Conv(inc=96, outc=96, num_param=3, stride=2),
                                             nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.rnn3_0 = CGRU_cell(shape=(16,16), input_channels=96, filter_size=filter_size, num_features=96)
        #-----decoder
        self.rnn3_1 = CGRU_cell(shape=(16,16), input_channels=96, filter_size=filter_size, num_features=96)
        self.deconv_1_leaky_deconv_1 = nn.Sequential(nn.ConvTranspose2d(in_channels=96, out_channels=96, kernel_size=4, stride=2, padding=1),
                                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.rnn2_1 = CGRU_cell(shape=(32,32), input_channels=96, filter_size=filter_size, num_features=96)
        self.deconv_2_leaky_deconv_2 =  nn.Sequential(nn.ConvTranspose2d(in_channels=96, out_channels=96, kernel_size=4, stride=2, padding=1),
                                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.rnn1_1 = CGRU_cell(shape=(64,64), input_channels=96, filter_size=filter_size, num_features=64)
        self.conv_3_leaky_5 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1),
                                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0),
                                            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        
        if self.attention_type == 'legacy_eaa':
            self.EAA = EfficientAdditiveAttnetion(in_dims=96, token_dim=96, num_heads=2)
        else:
            self.EAA = CTSFF(96, 16*16)

    def forward_by_stage_encoder(self, inputs, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))
        outputs_stage, state_stage = rnn(inputs, None)
        return outputs_stage, state_stage

    def forward_by_stage_decoder(self, inputs, state, subnet, rnn, seq_len=None):
        if seq_len is None:
            seq_len = self.seq_len
        inputs, state_stage = rnn(inputs, state, seq_len=seq_len)
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))
        return inputs

    def forward(self, inputs):
        # encoder part
        inputs = inputs.transpose(0, 1)  # to S,B,1,64,64
        hidden_states = []
        # stage-1
        inputs_1, state_stage_1 = self.forward_by_stage_encoder(inputs, self.conv_1_leaky_1, self.rnn1_0)
        hidden_states.append(state_stage_1)
        # stage-2
        inputs_2, state_stage_2 = self.forward_by_stage_encoder(inputs_1, self.conv_2_leaky_2, self.rnn2_0)
        hidden_states.append(state_stage_2)
        # stage-3
        inputs_3, state_stage_3 = self.forward_by_stage_encoder(inputs_2, self.conv_3_leaky_3, self.rnn3_0)

        # operage stage_stage_3
        state_stage_3 = state_stage_3.permute(0, 2, 3, 1) # 张量维度转换
        b, w, h, c = state_stage_3.size()
        # print(b,w,h,c)
        state_stage_3 = state_stage_3.view(b, -1, c)
        if self.attention_type == 'legacy_eaa':
            state_stage_3 = self.EAA(state_stage_3)
        else:
            state_stage_3 = self.EAA(state_stage_3,  w, h)
        state_stage_3 = state_stage_3.view(b, w, h, c)
        state_stage_3 = state_stage_3.permute(0, 3, 1, 2)

        hidden_states.append(state_stage_3)
        # print(hidden_states[0].shape, hidden_states[1].shape, hidden_states[2].shape)

        # decoder part
        inputs_1_ = self.forward_by_stage_decoder(None, hidden_states[-1], self.deconv_1_leaky_deconv_1, self.rnn3_1)
        # print(inputs_1_.shape, hidden_states[1].shape) # torch.Size([10, 8, 96, 32, 32]) torch.Size([8, 96, 32, 32])

        inputs_2_ = self.forward_by_stage_decoder(inputs_1_, hidden_states[1], self.deconv_2_leaky_deconv_2, self.rnn2_1)

        inputs_3_ = self.forward_by_stage_decoder(inputs_2_, hidden_states[0], self.conv_3_leaky_5, self.rnn1_1)

        result = inputs_3_.transpose(0, 1)  # to B,S,1,64,64

        return result
