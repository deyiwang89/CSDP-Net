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
from data.mm import MovingMNIST_set
import torch
from torch import nn
from earlystopping import EarlyStopping
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import argparse
import time

from SmaAt_UNet import SmaAt_UNet
from fc_lstm import FC_LSTM

#-----visualization
from torchvision.utils import make_grid

from torchvision import transforms

from CSDPNet import CSDPNet
from ssim import ssim


from PIL import Image
def save_tensor(pred, i, dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)   
    
    # Check if prediction is empty
    if pred is None or pred.numel() == 0:
        print(f"Warning: Empty prediction for index {i}, skipping image generation.")
        return

    # 找到tensor中的最小值和最大值
    min_value = torch.min(pred)
    max_value = torch.max(pred)

    if max_value == min_value:
        # Avoid division by zero if tensor is constant
        pred = torch.zeros_like(pred)
    else:
        # 将tensor的数值缩放到0-255的范围内
        pred = (pred - min_value) / (max_value - min_value) * 255
    
    # 将缩放后的tensor转换为整数类型
    pred = pred.to(torch.uint8)

    # 假设您的tensor名为image_tensor
    # Handle different output dimensions based on model (e.g., PSPNet vs CSDP-Net)
    try:
        if len(pred.size()) == 5: # [B, S, C, H, W]
            # We take the first batch, first time step, first channel
            pred = pred[0,-1,0,:,:] # Take the last predicted frame for visualization
        elif len(pred.size()) == 4: # [B, C, H, W]
            # PSPNet or similar might return this
            pred = pred[0,0,:,:]
        elif len(pred.size()) == 3: # [C, H, W]
            pred = pred[0,:,:]
        else:
            print(f"Warning: Unexpected prediction shape {pred.shape} for index {i}, cannot save image.")
            return
            
        # 将tensor移动到CPU上 
        image_tensor_cpu = pred.cpu().detach()
        # 将tensor转换为numpy数组
        image_array = image_tensor_cpu.numpy()
        
        # Check shape to decide how to process
        # Case 1: [B, C, H, W] - batch size, channels, height, width
        if len(image_array.shape) == 4:
            if image_array.shape[1] == 1: # single channel
                image_array = image_array[0, 0] # take first batch, first channel
            elif image_array.shape[1] == 3: # RGB
                image_array = np.transpose(image_array[0], (1, 2, 0)) # -> [H, W, C]
        # Case 2: [C, H, W]
        elif len(image_array.shape) == 3:
            if image_array.shape[0] == 1:
                image_array = image_array[0]
            elif image_array.shape[0] == 3:
                image_array = np.transpose(image_array, (1, 2, 0))
        # Case 3: already [H, W] or [H, W, C]
        # Do nothing

        # Make sure we have valid shape for PIL
        if len(image_array.shape) not in [2, 3]:
            print(f"Warning: After processing, unexpected array shape {image_array.shape} for index {i}, cannot save image.")
            return
            
        # 将数组转换为PIL图像
        image = Image.fromarray(image_array)
        # 将图像模式转换为RGB
        image = image.convert('RGB')
        # 保存图像
        name = "pred_" + str(i) + ".jpg"
        img_path = os.path.join(dir_path,name)
        image.save(img_path)
    except Exception as e:
        print(f"Failed to save image {i}: {str(e)}")


TIMESTAMP = str(time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time())))
parser = argparse.ArgumentParser()
parser.add_argument('-GraphATNet',
                    '--GraphATNet',
                    help='use GraphAT-Net as base cell',
                    action='store_true')
parser.add_argument('-CSDPNet',
                    '--CSDPNet',
                    help='use CSDPNet as base cell',
                    action='store_true')
parser.add_argument('-gcnnet',
                    '--gcnnet',
                    help='use gcnnet as base cell',
                    action='store_true')
parser.add_argument('-clstm', '--convlstm', action='store_true', help='use convlstm as base cell')
parser.add_argument('-cgru', '--convgru', action='store_true', help='use convgru as base cell')
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
                    default=1,
                    type=int,
                    help='mini-batch size')
parser.add_argument('-lr', default=1e-3, type=float, help='G learning rate')
# 模型的输入输出维度需要按照原始的维度：seq_len=2
parser.add_argument('-frames_input',
                    default=4,
                    type=int,
                    help='sum of input frames')
parser.add_argument('-frames_output',
                    default=4,
                    type=int,
                    help='sum of predict frames')
parser.add_argument('-epochs', default=1, type=int, help='sum of epochs')
parser.add_argument('-N', type=int, default=5, help='kernel size for CBS-Conv (N)')
parser.add_argument('-sd', default=r'save_model\\akeformer\\checkpoint_8_0.000207.pth.tar', help='input image path')
parser.add_argument('-dataset', default='GCAPPI', type=str, help='which dataset to run (GCAPPI/moving_mnist)')
args = parser.parse_args()


def _load_state_dict_from_checkpoint(checkpoint_path, device):
    model_info = torch.load(checkpoint_path, map_location=device)
    state_dict = model_info['state_dict'] if 'state_dict' in model_info else model_info
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    return state_dict


def _infer_csdpnet_config(state_dict, default_n):
    filter_size = default_n
    if 'rnn1_0.conv1.0.weight' in state_dict:
        filter_size = int(state_dict['rnn1_0.conv1.0.weight'].shape[-1])
    attention_type = 'legacy_eaa' if 'EAA.to_query.weight' in state_dict else 'ctsff'
    return filter_size, attention_type

random_seed = 1996
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

save_dir = './save_model/' + TIMESTAMP

tt_transforms = transforms.Compose([
    #transforms.CenterCrop([800,1000]),
    transforms.Resize([256,256]),
    transforms.ToTensor(),
])

if args.dataset == 'GCAPPI':
    import cv2
    import os as os_mod
    
    class GCAPPIDataset(torch.utils.data.Dataset):
        def __init__(self, root_dir, n_frames_input=4, n_frames_output=4):
            self.root_dir = root_dir
            self.n_frames_input = n_frames_input
            self.n_frames_output = n_frames_output
            self.seq_len = n_frames_input + n_frames_output
            
        def __len__(self):
            # Placeholder for testing, update with real GCAPPI loading if running inference
            return 100
            
        def __getitem__(self, idx):
            input_seq = torch.randn(self.n_frames_input, 1, 256, 256)
            target_seq = torch.randn(self.n_frames_output, 1, 256, 256)
            return idx, target_seq, input_seq, None, None

    trainFolder = GCAPPIDataset(root_dir='data/GCAPPI', n_frames_input=args.frames_input, n_frames_output=args.frames_output)
    validFolder = GCAPPIDataset(root_dir='data/GCAPPI', n_frames_input=args.frames_input, n_frames_output=args.frames_output)
    trainLoader = torch.utils.data.DataLoader(trainFolder,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              drop_last=True)
    validLoader = torch.utils.data.DataLoader(validFolder,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              drop_last=True)
else:
    trainFolder = MovingMNIST_set(is_train=True,
                          root='data/',
                          n_frames_input=args.frames_input,
                          n_frames_output=args.frames_output,
                          num_objects=[3])
    validFolder = MovingMNIST_set(is_train=False,
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
    print('clstm is on')
elif args.convgru:
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params
    print('cgru is on')
else:
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params
    print('cgru is on')


def inference():
    '''
    main function to run the training
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(encoder_params[0], encoder_params[1]).to(device)
    decoder = Decoder(decoder_params[0], decoder_params[1]).to(device)
    net = CSDP_Net(encoder_params, decoder_params).to(device)
    folder_name = 'default_ed'
    
    if args.smatunet:
        net = SmaAt_UNet(n_channels = args.frames_input, n_classes = args.frames_output).to(device)
        print('using smat unet')
        folder_name = 'smatunet'
    # 使用my_trajgru需要调整下方的state, seq_len=2)的输入维度，
    # 按照args.frames_input调整。此外，还需要按照resize的尺寸调整里面所有X_cell的尺寸+GraphConvolution(64*64*64*2*2, 16)的尺寸
    if args.GraphATNet:
        from GraphATNet import my_trajgru
        net = my_trajgru(N=args.N).to(device)
        print("use GraphAT-Net")
        folder_name = 'mytrajgru'
    elif args.CSDPNet:
        from CSDPNet import CSDPNet
        if os.path.exists(str(args.sd)):
            state_dict_for_cfg = _load_state_dict_from_checkpoint(str(args.sd), device)
            inferred_n, inferred_attention = _infer_csdpnet_config(state_dict_for_cfg, args.N)
            net = CSDPNet(seqNum=args.frames_output, filter_size=inferred_n, attention_type=inferred_attention).to(device)
            print(f"use CSDP-Net with inferred N={inferred_n}, attention={inferred_attention}")
        else:
            net = CSDPNet(seqNum=args.frames_output, filter_size=args.N).to(device)
            print("use CSDP-Net")
        folder_name = 'akeformer'
    elif args.gcnnet:
        net = gcnnet(sizevalue = 4, batchsize = args.batch_size, seqNum = args.frames_input)
        print('using gcnnet')
    elif args.seresunet:
        from seresunet import resnet34
        net = resnet34( args.frames_input, args.frames_output, False).to(device)
        print('using seresunet')
        folder_name = 'seresunet'
    elif args.pspnet:
        from nets.pspnet import PSPNet
        net = PSPNet(num_classes=args.frames_output, backbone='resnet50', downsample_factor=8, pretrained=True, aux_branch=False).to(device)
        print('using PSPNet')
        folder_name = 'pspnet'
    elif args.fclstm:
        net = FC_LSTM().to(device)
        print('using fclstm')
        folder_name = 'fclstm'
        # 使用pspnet需要调整下方的conv1的输入维度，按照args.frames_input调整
        # # # class ResNet(nn.Module):
        # # #     def __init__(self, block, layers, num_classes=1000):
        # # #         self.inplanes = 128
        # # #         super(ResNet, self).__init__()
        # # #         self.conv1 = conv3x3(2, 64, stride=2)
    elif args.convlstm:
        net = CSDP_Net(convlstm_encoder_params, convlstm_decoder_params).to(device)
        print('using convlstm')
        folder_name = 'convlstm'
    elif args.convgru:
        net = CSDP_Net(convgru_encoder_params, convgru_decoder_params).to(device)
        print('using convgru')
        folder_name = 'convgru'
    elif args.SeClstmresnet34:
        from seresunetclstm import SeClstmresnet34
        net = SeClstmresnet34(in_channel=args.frames_input,out_channel=args.frames_output,pretrain=False).to(device)
        print('using SeClstmresnet34')
        folder_name = 'SeClstmresnet34'

    print(net)
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameters: %.2fM" % (total/1e6))

    run_dir = './runs/' + TIMESTAMP
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    tb = SummaryWriter(run_dir)
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=args.epochs//3, verbose=True)
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


    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # mini_val_loss = np.inf




    model_dict = net.state_dict()
    pretrained_dict = _load_state_dict_from_checkpoint(os.path.join(str(args.sd)), device)
    shape_mismatch_keys = [k for k, v in pretrained_dict.items() if k in model_dict and np.shape(model_dict[k]) != np.shape(v)]
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict) and np.shape(model_dict[k]) ==  np.shape(v)}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    print(f'Loaded {len(pretrained_dict)}/{len(model_dict)} tensors from {args.sd}')
    if shape_mismatch_keys:
        print(f'Skipped {len(shape_mismatch_keys)} tensors due to shape mismatch')
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)


    for epoch in range(cur_epoch, args.epochs + 1):
        ######################
        # validate the model #
        ######################
        fps_rec = []
        val_visual = []
        with torch.no_grad():
            net.eval()
            t = tqdm(validLoader, leave=False, total=len(validLoader))
            for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
                # if i == 50:
                #     break
                inputs = inputVar.to(device)
                label = targetVar.to(device)
                start_time = time.time()
                pred = net(inputs)
                # added by deyiwang 231031
                # print(type(pred),i)#-----
                
                # Use model name for output directory
                model_name = os.path.basename(os.path.dirname(args.sd))
                if model_name == '':
                    model_name = "default"
                output_dir = f"visual_results_{model_name}"
                
                save_tensor(pred, i, output_dir)
                fps_rec.append(1/(time.time() - start_time))# record fps
                loss = lossfunction(pred, label)
                loss_aver = loss.item() / args.batch_size
                # record validation loss
                valid_losses.append(loss_aver)
                
                # Calculate individual SSIM and MSE for CSV logging
                if len(pred.size()) == 5:
                    if pred.size(0) != label.size(0):
                        pred_match = pred[:label.size(0)]
                    else:
                        pred_match = pred
                    pred_ssim = pred_match.reshape(-1, pred_match.size(2), pred_match.size(3), pred_match.size(4))
                    label_ssim = label.reshape(-1, label.size(2), label.size(3), label.size(4))
                    ind_ssim = ssim(pred_ssim, label_ssim).item()
                    ind_mse = lossfunction(pred_match, label).item()
                else:
                    if pred.size(0) != label.size(0):
                        pred_match = pred[:label.size(0)]
                    else:
                        pred_match = pred
                    ind_ssim = ssim(pred_match, label).item()
                    ind_mse = lossfunction(pred_match, label).item()
                
                # Log to CSV
                csv_path = f"{output_dir}_metrics.csv"
                if i == 0:
                    with open(csv_path, "w") as f:
                        f.write("Image_Name,MSE,SSIM\n")
                with open(csv_path, "a") as f:
                    f.write(f"pred{i}.jpg,{ind_mse:.6e},{ind_ssim:.6f}\n")

                #print ("validloss: {:.6f},  epoch : {:02d}".format(loss_aver,epoch),end = '\r', flush=True)
                t.set_postfix({
                    'validloss': '{:.6f}'.format(loss_aver),
                    'epoch': '{:02d}'.format(epoch)
                })
                val_visual.append([inputs[0]*255, label[0]*255,pred[0]*255])
        
        # Calculate SSIM and average Validation Loss
        avg_mse = np.mean(valid_losses)
        # Handle 5D tensor (batch, seq_len, c, h, w) for SSIM
        if len(pred.size()) == 5:
            # PSPNet might return different size than label
            if pred.size(0) != label.size(0):
                pred = pred[:label.size(0)]
            pred_ssim = pred.reshape(-1, pred.size(2), pred.size(3), pred.size(4))
            label_ssim = label.reshape(-1, label.size(2), label.size(3), label.size(4))
            ssim_val = ssim(pred_ssim, label_ssim).item()
        else:
            if pred.size(0) != label.size(0):
                pred = pred[:label.size(0)]
            ssim_val = ssim(pred, label).item()
        val_loss_final = (avg_mse + (1 - ssim_val) / 2) / 2
        print(f"\nEvaluation Results:")
        print(f"MSE: {avg_mse:.6f}")
        print(f"SSIM: {ssim_val:.6f}")
        print(f"Vali Loss: {val_loss_final:.6f}")
        
        with open("eval_results_gcappi.txt", "a") as f:
            f.write(f"Model: {args.sd}\n")
            f.write(f"MSE: {avg_mse:.6e}, SSIM: {ssim_val:.6f}, Vali Loss: {val_loss_final:.6f}\n\n")

        tb.add_scalar('ValidLoss', loss_aver, epoch)
        # 显示的最后的10张预测图片
        # val_visual.append([inputs[0], label[0],pred[0]])
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

        fps_mean = np.average(fps_rec)

        epoch_len = len(str(args.epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{args.epochs:>{epoch_len}}] ' + f'fps_mean: {fps_mean:.2f}')

        print(print_msg)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []


if __name__ == "__main__":
    inference()
