#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   decoder.py
@Time    :   2020/03/09
@Author  :   jhhuang96
@Mail    :   hjh096@126.com
@Version :   1.0
@Description:   decoder
'''

from torch import nn
from utils import make_layers
import torch


class Decoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)
        '''
        subnets:
            [
        OrderedDict({'deconv1_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [96, 96, 4, 2, 1]}),
        OrderedDict({
                'conv3_leaky_1': [64, 16, 3, 1, 1],
                'conv4_leaky_1': [16, 1, 1, 1, 0]
        }),
        ],
        ====================================================
        rnns:
        [
            CGRU_cell(shape=(16,16), input_channels=96, filter_size=5, num_features=96),
            CGRU_cell(shape=(32,32), input_channels=96, filter_size=5, num_features=96),
            CGRU_cell(shape=(64,64), input_channels=96, filter_size=5, num_features=64),
        ]
        '''

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks - index), rnn) # 2,1,0
            setattr(self, 'stage' + str(self.blocks - index),
                    make_layers(params))

    def forward_by_stage(self, inputs, state, subnet, rnn):
        inputs, state_stage = rnn(inputs, state, seq_len=10)
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))
        return inputs

        # input: 5D S*B*C*H*W

    def forward(self, hidden_states):
        inputs = self.forward_by_stage(None, hidden_states[-1],
                                       getattr(self, 'stage3'),
                                       getattr(self, 'rnn3'))
        for i in list(range(1, self.blocks))[::-1]:
            inputs = self.forward_by_stage(inputs, hidden_states[i - 1],
                                           getattr(self, 'stage' + str(i)),
                                           getattr(self, 'rnn' + str(i)))
        inputs = inputs.transpose(0, 1)  # to B,S,1,64,64
        return inputs


if __name__ == "__main__":
    from net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_decoder_params, convgru_encoder_params
    from data.mm import MovingMNIST
    from encoder import Encoder
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    encoder = Encoder(convgru_encoder_params[0],
                      convgru_encoder_params[1]).cuda()
    decoder = Decoder(convgru_decoder_params[0],
                      convgru_decoder_params[1]).cuda()
    # print(convgru_encoder_params[0], convgru_encoder_params[1])
    # print(encoder)
    # print(decoder)
    if torch.cuda.device_count() > 1:
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)

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
    for i, (idx, targetVar, inputVar, _, _) in enumerate(trainLoader):
        inputs = inputVar.to(device)  # B,S,1,64,64
        # print(type(inputVar), inputVar.shape, inputs.shape)
        # <class 'torch.Tensor'> torch.Size([8, 10, 1, 64, 64]) torch.Size([8, 10, 1, 64, 64])
        state = encoder(inputs)
        # print(len(state),len(state[1]),len(state[2]),state[0][0].shape)
        # 3 2 2 torch.Size([8, 64, 64, 64])
        break
    # print(decoder)
    output = decoder(state)
    print(output.shape)  # B,S,1,64,64
    # torch.Size([8, 10, 1, 64, 64]) 3 8 8 8*64*64*64
