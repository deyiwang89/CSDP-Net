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

class FC_LSTM(nn.Module):
    def __init__(self):
        super(FC_LSTM, self).__init__()
        
        # 定义全连接层
        self.fc = nn.Sequential(
            nn.Linear(64*64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        
        # 定义LSTM层
        self.lstm = nn.LSTM(input_size=32, hidden_size=32, num_layers=2, batch_first=True)
        
        # 定义输出层
        self.out = nn.Linear(32, 10*64*64)
        
    def forward(self, x):
        b,p,i,w,h = x.shape
        # 将输入的特征进行reshape
        x = x.view(-1, 64*64)
        
        # 全连接层
        x = self.fc(x)
        
        # 将输出reshape为LSTM的输入格式
        x = x.view(-1, 10, 32)
        
        # LSTM层
        x, _ = self.lstm(x)
        
        # 取LSTM的最后一个时间步作为输出
        x = x[:, -1, :]
        
        # 输出层
        x = self.out(x)
        
        # 将输出reshape为与输入相同的维度
        x = x.view(-1, p,i,w,h )
        
        return x
