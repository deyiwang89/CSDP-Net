import torch
import torch.nn as nn
from torchvision import models
import numpy as np

#-----scse net
# copy from: https://www.cnblogs.com/pprp/p/12200334.html
class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U)  # U:[bs,c,h,w] to q:[bs,1,h,w]
        q = self.norm(q)
        return U * q  # 广播机制

class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels//2, in_channels, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)# shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z) # shape: [bs, c/2]
        z = self.Conv_Excitation(z) # shape: [bs, c]
        z = self.norm(z)
        return U * z.expand_as(U)

class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return U_cse+U_sse
        
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        #-----scse
        self.scse = scSE(out_channels * BasicBlock.expansion)
    def forward(self, x):
        x = nn.ReLU(inplace=True)(self.scse(self.residual_function(x))+ self.shortcut(x))
        return x


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )
        #-----scse
        self.scse = scSE(out_channels * BottleNeck.expansion)
    def forward(self, x):
        x = nn.ReLU(inplace=True)(self.scse(self.residual_function(x)) + self.shortcut(x))
        return x



class ResNet(nn.Module):

    def __init__(self,in_channel,out_channel, block, num_block):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size = 7, stride = 2, padding = 3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(384, 256)
        self.dconv_up2 = double_conv(192, 128)
        self.dconv_up1 = double_conv(96, 64)

        self.dconv_last=nn.Sequential(
            nn.Conv2d(80, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64,out_channel,1)
        )
        self.chunk_num = out_channel

        self.cus_up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.cus_up2 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.cus_up3 = nn.ConvTranspose2d(32, 16, 2, 2)

    def convlstm_cell(self, ai, af, ao, ag, new_c):
        # (ai, af, ao, ag) = torch.split(A, A.size()[1] // 4, dim=1)
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ag)
        g = torch.tanh(ag)
        new_c = f * new_c + i * g
        new_h = o * torch.tanh(new_c)
        return new_c , new_h

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.squeeze(x,2)
        conv1 = self.conv1(x)
        temp=self.maxpool(conv1)
        conv2 = self.conv2_x(temp)
        conv3 = self.conv3_x(conv2)
        conv4 = self.conv4_x(conv3)
        bottle = self.conv5_x(conv4)
        # output = self.avg_pool(output)
        # output = output.view(output.size(0), -1)
        # output = self.fc(output)
        x = self.upsample(bottle)
        #-----
        s1,s2,s3,s4 = torch.chunk(x, 4, dim=1)
        new_c, new_h = self.convlstm_cell(s1,s2,s3,s4,s1)
        x = new_h# ([4, 768, 4, 4])
        new_c = self.cus_up1(new_c)

        
        x = torch.cat([x, conv4], dim=1)# 768

        x = self.dconv_up3(x)
        x = self.upsample(x)

        s1,s2,s3,s4 = torch.chunk(x, 4, dim=1)
        new_c, new_h = self.convlstm_cell(s1,s2,s3,s4,new_c)
        x = new_h# ([4, 768, 4, 4])
        new_c = self.cus_up2(new_c)

        x = torch.cat([x, conv3], dim=1)# 384


        x = self.dconv_up2(x)
        x = self.upsample(x)

        s1,s2,s3,s4 = torch.chunk(x, 4, dim=1)
        new_c, new_h = self.convlstm_cell(s1,s2,s3,s4,new_c)
        x = new_h# ([4, 768, 4, 4])
        new_c = self.cus_up3(new_c)

        x = torch.cat([x, conv2], dim=1)# 192
        # print(x.shape)
        x = self.dconv_up1(x)
        x=self.upsample(x)

        s1,s2,s3,s4 = torch.chunk(x, 4, dim=1)
        new_c, new_h = self.convlstm_cell(s1,s2,s3,s4,new_c)
        x = new_h# ([4, 768, 4, 4])

        x=torch.cat([x,conv1],dim=1)# 128
        out=self.dconv_last(x)
        out = torch.unsqueeze(out,2)
        return out

    def load_pretrained_weights(self):

        model_dict=self.state_dict()
        resnet34_weights = models.resnet34(True).state_dict()
        count_res = 0
        count_my = 0

        reskeys = list(resnet34_weights.keys())
        mykeys = list(model_dict.keys())
        # print(self)
        # print(models.resnet34())
        # print(reskeys)
        # print(mykeys)

        corresp_map = []
        while (True):              # 后缀相同的放入list
            reskey = reskeys[count_res]
            mykey = mykeys[count_my]

            if "fc" in reskey:
                break

            while reskey.split(".")[-1] not in mykey:
                count_my += 1
                mykey = mykeys[count_my]

            corresp_map.append([reskey, mykey])
            count_res += 1
            count_my += 1

        for k_res, k_my in corresp_map:
            model_dict[k_my]=resnet34_weights[k_res]

        try:
            self.load_state_dict(model_dict)
            print("Loaded resnet34 weights in mynet !")
        except:
            print("Error resnet34 weights in mynet !")
            raise


def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])


def SeClstmresnet34(in_channel,out_channel,pretrain=False):
    """ return a ResNet 34 object
    """
    model=ResNet(in_channel,out_channel,BasicBlock, [3, 4, 6, 3])
    if pretrain:
        pretrained_dict = torch.load('./model_data/resnet34-333f7ec4.pth')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(pretrained_dict[k]) == np.shape(v)} 
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print('======================fint tune activated======================')
    return model


def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])


def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])


def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])

def test():
    print('===test===')
    from torch.autograd import Variable
    from torchsummary import summary 
    import torch
    net = SeClstmresnet34(10, 10, False).cuda()
    # print(net)
    x = Variable(torch.rand(4, 10, 1, 64, 64)).cuda()
    print(net.forward(x).shape)

# test()