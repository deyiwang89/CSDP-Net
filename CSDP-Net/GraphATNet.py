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
1, 复现trajgru-------------------done
2, 添加attention-----------------done 添加了通道注意力 eca 等待实验结果
3, 添加3d conv(实验)-------------不好用, 本文中已经将2d conv 整合到4维度张量中, 因此实际上2d conv起到的作用就是3d conv的作用
4, 增加DLA结构(实验)
5, 将gru更改为srnn---------------不好用, sru并不比gru好用, 
6, 添加gcn提高空间特征提取能力----done
'''

import torch
from torch import nn
from utils import make_layers
from ConvRNN import CGRU_cell, CLSTM_cell

import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
import scipy.sparse as sp

import torch.nn.functional as F

# 图卷积层
@torch.no_grad()
def getGaussianKernel(ksize, sigma=0):
    if sigma <= 0:
        # 根据 kernelsize 计算默认的 sigma，和 opencv 保持一致
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8 
    center = ksize // 2
    xs = (np.arange(ksize, dtype=np.float32) - center) # 元素与矩阵中心的横向距离
    kernel1d = (np.exp(-(xs ** 2) / (2 * sigma ** 2))) # 计算一维卷积核
    # 根据指数函数性质，利用矩阵乘法快速计算二维卷积核
    kernel = kernel1d[..., None] @ kernel1d[None, ...] 
    # kernel = kernel / kernel.sum() # 归一化
    kernel = preprocess_adj(kernel)
    kernel = sparse_mx_to_torch_sparse_tensor(kernel)
    # print(type(kernel))
    #kernel = torch.from_numpy(kernel)
    return kernel

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized



class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # print(input.shape, self.weight.shape)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

# ECA模块
class eca_layer(nn.Module):
    """Constructs a ECA module.
    参考论文: https://arxiv.org/abs/1910.03151
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__() # super类的作用是继承的时候，调用含super的哥哥的基类__init__函数。
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 全局平均池化
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False) # 一维卷积
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size() # b代表b个样本，c为通道数，h为高度，w为宽度
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        # torch.squeeze()这个函数主要对数据的维度进行压缩,torch.unsqueeze()这个函数 主要是对数据维度进行扩充
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion多尺度信息融合
        y = self.sigmoid(y)
        # 原网络中克罗内克积，也叫张量积，为两个任意大小矩阵间的运算
        return x * y.expand_as(x)





class my_trajgru(nn.Module):
    def __init__(self):
        super().__init__()
        #-----encoder
        # stage-1 
        # class torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

        # in_channels(int) – 输入信号的通道，就是输入中每帧图像的通道数
        # out_channels(int) – 卷积产生的通道，就是输出中每帧图像的通道数
        # kernel_size(int or tuple) - 过滤器的尺寸，假设为(a,b,c)，表示的是过滤器每次处理 a 帧图像，该图像的大小是b x c。
        # stride(int or tuple, optional) - 卷积步长，形状是三维的，假设为(x,y,z)，表示的是三维上的步长是x，在行方向上步长是y，在列方向上步长是z。
        # padding(int or tuple, optional) - 输入的每一条边补充0的层数，形状是三维的，假设是(l,m,n)，表示的是在输入的三维方向前后分别padding l 个全零二维矩阵，在输入的行方向上下分别padding m 个全零行向量，在输入的列方向左右分别padding n 个全零列向量。
        # dilation(int or tuple, optional) – 卷积核元素之间的间距，这个看看空洞卷积就okay了
        # groups(int, optional) – 从输入通道到输出通道的阻塞连接数；没用到，没细看
        # bias(bool, optional) - 如果bias=True，添加偏置；没用到，没细看

        self.conv_1_leaky_1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
                                             nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                             eca_layer(channel=16, k_size=3))
        self.rnn1_0 = CGRU_cell(shape=(64,64), input_channels=16, filter_size=5, num_features=64)
        # stage-2
        self.conv_2_leaky_2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
                                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                            eca_layer(channel=64, k_size=3))
        self.rnn2_0 = CGRU_cell(shape=(32,32), input_channels=64, filter_size=5, num_features=96)
        # stage-3
        self.conv_3_leaky_3 = nn.Sequential(nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=2, padding=1),
                                             nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                             eca_layer(channel=96, k_size=3))
        self.rnn3_0 = CGRU_cell(shape=(16,16), input_channels=96, filter_size=5, num_features=96)
        #-----decoder
        self.rnn3_1 = CGRU_cell(shape=(16,16), input_channels=96, filter_size=5, num_features=96)
        self.deconv_1_leaky_deconv_1 = nn.Sequential(nn.ConvTranspose2d(in_channels=96, out_channels=96, kernel_size=4, stride=2, padding=1),
                                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                    eca_layer(channel=96, k_size=3))

        self.rnn2_1 = CGRU_cell(shape=(32,32), input_channels=96, filter_size=5, num_features=96)
        self.deconv_2_leaky_deconv_2 =  nn.Sequential(nn.ConvTranspose2d(in_channels=96, out_channels=96, kernel_size=4, stride=2, padding=1),
                                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                    eca_layer(channel=96, k_size=3))

        self.rnn1_1 = CGRU_cell(shape=(64,64), input_channels=96, filter_size=5, num_features=64)
        self.conv_3_leaky_5 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1),
                                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0),
                                            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        #-----gcn
        self.gc1_0 = GraphConvolution(64*64*64, 16)
        self.gc1_1 = GraphConvolution(16, 64*64*64)

    def forward_by_stage_encoder(self, inputs, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        # print("forward_by_stage_encoder",inputs.shape)
        inputs = subnet(inputs)
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))
        outputs_stage, state_stage = rnn(inputs, None)

        return outputs_stage, state_stage

    def forward_by_stage_decoder(self, inputs, state, subnet, rnn):
        inputs, state_stage = rnn(inputs, state, seq_len=10)
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        # print("inputs.shape before:", inputs.shape) # [80, 64, 64, 64]
        inputs = subnet(inputs)
        # print("inputs.shape after:", inputs.shape) # [80, 1, 64, 64]
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))
        return inputs


    def forward(self, inputs):
        # encoder part
        inputs = inputs.transpose(0, 1)  # to S,B,1,64,64
        hidden_states = []
        # stage-1
        inputs_1, state_stage_1 = self.forward_by_stage_encoder(inputs, self.conv_1_leaky_1, self.rnn1_0)
        # 由于三个尺度下的hidden state大小不同, 需要重新定义对应的层
        # 在此添加gcn_1
        features = state_stage_1
        features = features.view(features.shape[0],-1)
        adj = getGaussianKernel(features.shape[0]).cuda() # torch.Size([8, 64, 64, 64])
        features = F.relu(self.gc1_0(features, adj))
        features = F.dropout(features, 0.5)
        features = F.relu(self.gc1_1(features, adj)).view(state_stage_1.shape)
        state_stage_1 = features

        hidden_states.append(state_stage_1)
        # stage-2
        inputs_2, state_stage_2 = self.forward_by_stage_encoder(inputs_1, self.conv_2_leaky_2, self.rnn2_0)
        hidden_states.append(state_stage_2)
        # stage-3
        inputs_3, state_stage_3 = self.forward_by_stage_encoder(inputs_2, self.conv_3_leaky_3, self.rnn3_0)
        hidden_states.append(state_stage_3)
        # print(hidden_states[0].shape, hidden_states[1].shape, hidden_states[2].shape)
        # torch.Size([8, 64, 64, 64]) torch.Size([8, 96, 32, 32]) torch.Size([8, 96, 16, 16])
        # hidden_states====list---(state_stage_1,state_stage_2,state_stage_3)
        # decoder part
        inputs_1_ = self.forward_by_stage_decoder(None, hidden_states[-1], self.deconv_1_leaky_deconv_1, self.rnn3_1)
        # print(inputs_1_.shape, hidden_states[1].shape) # torch.Size([10, 8, 96, 32, 32]) torch.Size([8, 96, 32, 32])

        inputs_2_ = self.forward_by_stage_decoder(inputs_1_, hidden_states[1], self.deconv_2_leaky_deconv_2, self.rnn2_1)

        inputs_3_ = self.forward_by_stage_decoder(inputs_2_, hidden_states[0], self.conv_3_leaky_5, self.rnn1_1)

        result = inputs_3_.transpose(0, 1)  # to B,S,1,64,64

        return result

def test2():
    print('===test===')
    from torch.autograd import Variable
    from torchsummary import summary 
    import torch

    features = Variable(torch.rand(4, 3, 256, 512)).cuda()
    features = features.view(features.shape[0],-1)
    model = GCN(nfeat=features.shape[1],
            nhid=16,
            nclass=512*256,
            dropout=0.5).cuda()

    adj = getGaussianKernel(features.shape[0]).cuda()
    
    print(adj.shape, features.shape)
    y = model(features,adj)
    print('output shape:', y.shape)
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

if __name__ == "__main__":
    from net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_decoder_params, convgru_encoder_params
    from data.mm import MovingMNIST
    from encoder import Encoder
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    trainFolder = MovingMNIST(is_train=True,
                              root='data/',
                              n_frames_input=10,
                              n_frames_output=10,
                              num_objects=[3])
    
    trainLoader = torch.utils.data.DataLoader(
        trainFolder,
        batch_size=8,
        shuffle=False,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = my_trajgru().cuda()

    for i, (idx, targetVar, inputVar, _, _) in enumerate(trainLoader):
        inputs = inputVar.to(device)  # B,S,1,64,64

        output = model(inputs)

        print(output.shape)  # B,S,1,64,64

        total_params = sum(p.numel() for p in model.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')
        break
