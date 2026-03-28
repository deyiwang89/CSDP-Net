from torch import nn
import torch.nn.functional as F
import torch
from utils import make_layers


class activation():

    def __init__(self, act_type, negative_slope=0.2, inplace=True):
        super().__init__()
        self._act_type = act_type
        self.negative_slope = negative_slope
        self.inplace = inplace

    def __call__(self, input):
        if self._act_type == 'leaky':
            return F.leaky_relu(input, negative_slope=self.negative_slope, inplace=self.inplace)
        elif self._act_type == 'relu':
            return F.relu(input, inplace=self.inplace)
        elif self._act_type == 'sigmoid':
            return torch.sigmoid(input)
        else:
            raise NotImplementedError


class CSDP_Net(nn.Module):
    def __init__(self, encoder_params, decoder_params):
        super(CSDP_Net, self).__init__()
        self.encoder = Encoder(encoder_params[0], encoder_params[1])
        self.decoder = Decoder(decoder_params[0], decoder_params[1])

    def forward(self, input):
        state = self.encoder(input)
        output = self.decoder(state)
        return output
