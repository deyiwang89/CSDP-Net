#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2020/03/09
@Author  :   jhhuang96
@Mail    :   hjh096@126.com
@Version :   1.0
@Description:   
'''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from encoder import Encoder
from decoder import Decoder
from model import CSDP_Net
from net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_encoder_params, convgru_decoder_params
from data.mm import MovingMNIST
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
import sys
from earlystopping import EarlyStopping
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import argparse
import time
#---ssim
from ssim import SSIM

from SmaAt_UNet import SmaAt_UNet
# ----- 
from GraphATNet import my_trajgru
from fc_lstm import FC_LSTM

from CSDPNet import CSDPNet

#-----visualization
from torchvision.utils import make_grid

TIMESTAMP = str(time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time())))
parser = argparse.ArgumentParser()

parser.add_argument('-CSDPNet',
                    help='use CSDPNet as base cell',
                    action='store_true')
parser.add_argument('-GraphATNet',
                    help='use GraphAT-Net as base cell',
                    action='store_true')
parser.add_argument('-clstm',
                    '--convlstm',
                    help='use convlstm as base cell',
                    action='store_true')
parser.add_argument('-cgru',
                    '--convgru',
                    help='use convgru as base cell',
                    action='store_true')
parser.add_argument('-smatunet',
                    '--smatunet',
                    help='use smatunet',
                    action='store_true')
parser.add_argument('-seresunet',
                    '--seresunet',
                    help='use seresunet',
                    action='store_true')
parser.add_argument('-SeClstmresnet34',
                    '--SeClstmresnet34',
                    help='use seresunet',
                    action='store_true')
parser.add_argument('-pspnet',
                    '--pspnet',
                    help='use pspnet',
                    action='store_true')
parser.add_argument('-fclstm',
                    '--fclstm',
                    help='use fclstm as base cell',
                    action='store_true')
parser.add_argument('--batch_size',
                    default=8,
                    type=int,
                    help='mini-batch size')
parser.add_argument('-lr', default=1e-4, type=float, help='G learning rate')
parser.add_argument('-frames_input',
                    default=10,
                    type=int,
                    help='sum of input frames')
parser.add_argument('-frames_output',
                    default=10,
                    type=int,
                    help='sum of predict frames')
parser.add_argument('-epochs', default=100, type=int, help='sum of epochs')
parser.add_argument('-N', type=int, default=5, help='kernel size for CBS-Conv (N)')
args = parser.parse_args()

random_seed = 3407
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

save_dir = './save_model/' + TIMESTAMP

trainFolder = MovingMNIST(is_train=True,# True: use fake data; False use moving mnist to train
                          root='data/',
                          n_frames_input=args.frames_input,
                          n_frames_output=args.frames_output,
                          num_objects=[3])
validFolder = MovingMNIST(is_train=False,
                          root='data/',
                          n_frames_input=args.frames_input,
                          n_frames_output=args.frames_output,
                          num_objects=[3])
trainLoader = torch.utils.data.DataLoader(trainFolder,
                                          batch_size=args.batch_size,
                                          shuffle=False)
validLoader = torch.utils.data.DataLoader(validFolder,
                                          batch_size=args.batch_size,
                                          shuffle=False)

if args.convlstm:
    encoder_params = convlstm_encoder_params
    decoder_params = convlstm_decoder_params
if args.convgru:
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params
else:
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params


def train():
    '''
    main function to run the training
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
    decoder = Decoder(decoder_params[0], decoder_params[1]).cuda()
    net = CSDP_Net(encoder_params, decoder_params).to(device)
    
    if args.smatunet:
        net = SmaAt_UNet(n_channels = args.frames_input, n_classes = args.frames_output)
        print('using smat unet')
    elif args.GraphATNet:
        from GraphATNet import my_trajgru
        net = my_trajgru(N=args.N).to(device)
        print("use GraphAT-Net")
        folder_name = 'graphatnet'
    elif args.CSDPNet:
        from CSDPNet import CSDPNet
        net = CSDPNet(seqNum=args.frames_output, filter_size=args.N).to(device)
        print(f'use CSDP-Net with N={args.N}')
        folder_name = f'csdpnet_N{args.N}'
    elif getattr(args, 'gcnnet_old', False):
        net = gcnnet(sizevalue = 1)
        print('using gcnnet')
    elif args.seresunet:
        from seresunet import resnet34
        net = resnet34( args.frames_input, args.frames_output, False)
        print('using seresunet')
    elif args.fclstm:
        net = FC_LSTM()
        print('using FC_LSTM')
    elif args.pspnet:
        from nets.pspnet import PSPNet
        net = PSPNet(num_classes=10, backbone='resnet50', downsample_factor=16, pretrained=True, aux_branch=False)
        print('using PSPNet')
    elif args.SeClstmresnet34:
        from seresunetclstm import SeClstmresnet34
        net = SeClstmresnet34(in_channel=10,out_channel=10,pretrain=True)
        print('using SeClstmresnet34')

    print(net)
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameters: %.2fM" % (total/1e6))

    run_dir = './runs/' + TIMESTAMP
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    tb = SummaryWriter(run_dir)
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=args.epochs//3, verbose=True)

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device)

    if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
        # load existing model
        print('==> loading existing model')
        model_info = torch.load(os.path.join(save_dir, 'checkpoin.pth.tar'))
        net.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(net.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch'] + 1
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        cur_epoch = 0

    lossfunction = nn.MSELoss().to(device) 
    lossfunction_2 = SSIM().to(device)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                      factor=0.5,
                                                      patience=4,
                                                      verbose=True)

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # mini_val_loss = np.inf
    for epoch in range(cur_epoch, args.epochs + 1):
        ###################
        # train the model #
        ###################
        val_visual = []
        t = tqdm(trainLoader, leave=False, total=len(trainLoader))
        for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
            inputs = inputVar.to(device)  # B,S,C,H,W
            # print('input:  ',inputs.shape) # ([4, 10, 1, 64, 64])
            label = targetVar.to(device)  # B,S,C,H,W
            # print('output:  ',label.shape) # ([4, 10, 1, 64, 64])
            # print('output:  ',label.shape,type(label))

            optimizer.zero_grad()
            net.train()
            pred = net(inputs)  # B,S,C,H,W
            # print('pred:  ',pred.shape)
            # difference ssim
            loss = (lossfunction(pred, label) +(1-lossfunction_2(pred, label))*0.5)/2
            loss_aver = loss.item() / args.batch_size
            train_losses.append(loss_aver)
            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0)
            optimizer.step()
            t.set_postfix({
                'trainloss': '{:.6f}'.format(loss_aver),
                'epoch': '{:02d}'.format(epoch)
            })
 
        # tensor board
        tb.add_scalar('TrainLoss', loss_aver, epoch)
        # 显示的最后的10张预测图片
        val_visual.append([inputs[0], label[0],pred[0]])
        train_img = []
        for d, t, o in val_visual:
            # 图片是灰度图片，因此要按照
            train_img.extend(d)
            train_img.extend(t)
            train_img.extend(o)
        # print(train_img[0].shape)
        train_img = torch.stack(train_img, 0)
        # print(train_img.shape)
        train_img = make_grid(train_img.cpu(), nrow=args.frames_output, padding=3)
        # print(train_img.shape)
        tb.add_image(f'train', train_img, epoch)
        ######################
        # validate the model #
        ######################
        fps_rec = []
        val_visual = []
        with torch.no_grad():
            net.eval()
            t = tqdm(validLoader, leave=False, total=len(validLoader))
            for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
                if i == 3000:
                    break
                inputs = inputVar.to(device)
                label = targetVar.to(device)
                start_time = time.time()
                pred = net(inputs)
                fps_rec.append(1/(time.time() - start_time))# record fps
                loss = lossfunction(pred, label)
                loss_2 = lossfunction_2(pred, label)

                loss_aver = loss.item() / args.batch_size
                loss_aver_2 = loss_2.item() / args.batch_size
                # record validation loss
                valid_losses.append((loss_aver+0.5-loss_aver_2*0.5)*0.5)

                #print ("validloss: {:.6f},  epoch : {:02d}".format(loss_aver,epoch),end = '\r', flush=True)
                t.set_postfix({
                    'validloss_mse': '{:.6f}'.format(loss_aver),
                    'validloss_ssim': '{:.6f}'.format(loss_aver_2),
                    'epoch': '{:02d}'.format(epoch)
                })

        tb.add_scalar('ValidLoss_mse', loss_aver, epoch)
        tb.add_scalar('ValidLoss_ssim', loss_aver_2, epoch)
        # 显示的最后的10张预测图片
        val_visual.append([inputs[0], label[0],pred[0]])
        train_img = []
        for d, t, o in val_visual:
            # 图片是灰度图片，因此要按照
            train_img.extend(d)
            train_img.extend(t)
            train_img.extend(o)
        # print(train_img[0].shape)
        train_img = torch.stack(train_img, 0)
        # print(train_img.shape)
        train_img = make_grid(train_img.cpu(), nrow=args.frames_output, padding=3)
        # print(train_img.shape)
        tb.add_image(f'val', train_img, epoch)

        torch.cuda.empty_cache()
        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        fps_mean = np.average(fps_rec)

        epoch_len = len(str(args.epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{args.epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.6f} ' +
                     f'valid_loss: {valid_loss:.6f} ' + f'fps_mean: {fps_mean:.2f}')

        print(print_msg)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        pla_lr_scheduler.step(valid_loss)  # lr_scheduler
        model_dict = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        early_stopping(valid_loss.item(), model_dict, epoch, save_dir)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    with open("avg_train_losses.txt", 'wt') as f:
        for i in avg_train_losses:
            print(i, file=f)

    with open("avg_valid_losses.txt", 'wt') as f:
        for i in avg_valid_losses:
            print(i, file=f)


if __name__ == "__main__":
    train()
