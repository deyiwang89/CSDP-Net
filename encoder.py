#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   encoder.py
@Time    :   2020/03/09 18:47:50
@Author  :   jhhuang96
@Mail    :   hjh096@126.com
@Version :   1.0
@Description:   encoder
'''

from torch import nn
from utils import make_layers
import torch
import logging


class Encoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)
        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            # index sign from 1
            setattr(self, 'stage' + str(index), make_layers(params))
            setattr(self, 'rnn' + str(index), rnn)

    def forward_by_stage(self, inputs, subnet, rnn):
        # print("def forward_by_stage(self, inputs, subnet, rnn):",inputs.shape) 
        # # torch.Size([10, 8, 1, 64, 64])
        seq_number, batch_size, input_channel, height, width = inputs.size()
        inputs = torch.reshape(inputs, (-1, input_channel, height, width))
        inputs = subnet(inputs)
        # print("def forward_by_stage(self, inputs, subnet, rnn): inputs = subnet(inputs)",inputs.shape) 
        # # torch.Size([80, 16, 64, 64])
        inputs = torch.reshape(inputs, (seq_number, batch_size, inputs.size(1),
                                        inputs.size(2), inputs.size(3)))
        # print("def forward_by_stage(self, inputs, subnet, rnn): inputs = subnet(inputs) torch.reshape",inputs.shape) 
        # # torch.Size([10, 8, 16, 64, 64])
        outputs_stage, state_stage = rnn(inputs, None)
        # print("outputs_stage, state_stage = rnn(inputs, None)",outputs_stage.shape,state_stage.shape) 
        # # torch.Size([8, 64, 64, 64])
        return outputs_stage, state_stage

    def forward(self, inputs):
        inputs = inputs.transpose(0, 1)  # to S,B,1,64,64
        # print("def forward(self, inputs):",inputs.shape) # torch.Size([10, 8, 1, 64, 64])
        hidden_states = []
        logging.debug(inputs.size())
        for i in range(1, self.blocks + 1):
            inputs, state_stage = self.forward_by_stage(
                inputs, getattr(self, 'stage' + str(i)),
                getattr(self, 'rnn' + str(i)))
            hidden_states.append(state_stage)
        # print("forward hidden_states.append(state_stage)",hidden_states[0].shape,state_stage.shape) 
        # torch.Size([8, 64, 64, 64]) torch.Size([8, 96, 16, 16])
        return tuple(hidden_states)


if __name__ == "__main__":
    from net_params import convgru_encoder_params, convgru_decoder_params
    from data.mm import MovingMNIST

    encoder = Encoder(convgru_encoder_params[0],
                      convgru_encoder_params[1]).cuda()
    trainFolder = MovingMNIST(is_train=True,
                              root='data/',
                              n_frames_input=10,
                              n_frames_output=10,
                              num_objects=[3])
    trainLoader = torch.utils.data.DataLoader(
        trainFolder,
        batch_size=4,
        shuffle=False,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i, (idx, targetVar, inputVar, _, _) in enumerate(trainLoader):
        inputs = inputVar.to(device)  # B,S,1,64,64
        state = encoder(inputs)
