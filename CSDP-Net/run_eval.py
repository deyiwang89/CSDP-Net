import os
import torch
from torch import nn
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-GraphATNet', action='store_true', help='use GraphAT-Net as base cell')
parser.add_argument('-CSDPNet', action='store_true', help='use CSDPNet as base cell')
parser.add_argument('-clstm', '--convlstm', action='store_true', help='use convlstm as base cell')
parser.add_argument('-cgru', '--convgru', action='store_true', help='use convgru as base cell')
parser.add_argument('-smatunet', action='store_true', help='use smatunet')
parser.add_argument('-seresunet', action='store_true', help='use seresunet')
parser.add_argument('-SeClstmresnet34', action='store_true', help='use seresunet')
parser.add_argument('-pspnet', action='store_true', help='use pspnet')
parser.add_argument('-fclstm', action='store_true', help='use fclstm as base cell')
parser.add_argument('-N', type=int, default=5, help='kernel size for CBS-Conv (N)')
parser.add_argument('-dataset', default='GCAPPI', type=str, help='which dataset to run (GCAPPI/moving_mnist)')
args = parser.parse_args()

from encoder import Encoder
from decoder import Decoder
from model import CSDP_Net
from net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_encoder_params, convgru_decoder_params
from data.mm import MovingMNIST_set
from ssim import SSIM

from SmaAt_UNet import SmaAt_UNet
from GraphATNet import my_trajgru
from fc_lstm import FC_LSTM


def _load_state_dict_from_checkpoint(weight_path, device):
    model_info = torch.load(weight_path, map_location=device)
    state_dict = model_info['state_dict'] if 'state_dict' in model_info else model_info
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    return state_dict


def _infer_csdpnet_config(state_dict, default_n):
    filter_size = default_n
    if 'rnn1_0.conv1.0.weight' in state_dict:
        filter_size = int(state_dict['rnn1_0.conv1.0.weight'].shape[-1])
    attention_type = 'legacy_eaa' if 'EAA.to_query.weight' in state_dict else 'ctsff'
    return filter_size, attention_type


def eval_model(model_name, weight_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state_dict = _load_state_dict_from_checkpoint(weight_path, device)
    
    print(f"[{model_name}] Initializing model...")
    if args.GraphATNet or model_name == 'my_trajgru' or model_name == 'graphatnet':
        from GraphATNet import my_trajgru
        net = my_trajgru(N=args.N).to(device)
        print("use GraphAT-Net")
    elif args.CSDPNet or model_name == 'csdpnet' or model_name == 'ours':
        from CSDPNet import CSDPNet
        inferred_n, inferred_attention = _infer_csdpnet_config(state_dict, args.N)
        net = CSDPNet(seqNum=10, filter_size=inferred_n, attention_type=inferred_attention).to(device)
        print(f"use CSDP-Net with inferred N={inferred_n}, attention={inferred_attention}")
    elif args.convlstm or model_name == 'clstm':
        net = CSDP_Net(convlstm_encoder_params, convlstm_decoder_params).to(device)
    elif args.convgru or model_name == 'cgru':
        net = CSDP_Net(convgru_encoder_params, convgru_decoder_params).to(device)
    elif args.smatunet or model_name == 'smatunet':
        net = SmaAt_UNet(n_channels=10, n_classes=10).to(device)
    elif args.seresunet or model_name == 'seresunet':
        from seresunet import resnet34
        net = resnet34(10, 10, False).to(device)
    elif args.SeClstmresnet34 or model_name == 'seclstmresnet34':
        from seresunetclstm import SeClstmresnet34
        net = SeClstmresnet34(in_channel=10, out_channel=10, pretrain=False).to(device)
    elif args.pspnet or model_name == 'pspnet':
        from nets.pspnet import PSPNet
        net = PSPNet(num_classes=10, backbone='resnet50', downsample_factor=16, pretrained=False, aux_branch=False).to(device)
    elif args.fclstm or model_name == 'fclstm':
        net = FC_LSTM().to(device)
    else:
        print(f"Unknown model: {model_name}")
        return

    net = net.to(device)

    # Load weights
    try:
        pretrained = state_dict
        model_dict = net.state_dict()
        shape_mismatch = [k for k, v in pretrained.items() if k in model_dict and model_dict[k].shape != v.shape]
        pretrained = {k: v for k, v in pretrained.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained)
        net.load_state_dict(model_dict)
        print(f"[{model_name}] loaded {len(pretrained)}/{len(model_dict)} tensors")
        if shape_mismatch:
            print(f"[{model_name}] skipped {len(shape_mismatch)} shape-mismatched tensors")
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    except Exception as e:
        print(f"Error loading {weight_path}: {e}")
        return

    net.eval()
    
    if args.dataset == 'GCAPPI':
        print(f"[{model_name}] Initializing GCAPPI dataset...")
        import cv2
        import os as os_mod
        
        class GCAPPIDataset(torch.utils.data.Dataset):
            def __init__(self, root_dir, n_frames_input=4, n_frames_output=4):
                self.root_dir = root_dir
                self.n_frames_input = n_frames_input
                self.n_frames_output = n_frames_output
                self.seq_len = n_frames_input + n_frames_output
                self.data_files = sorted([os_mod.path.join(root_dir, f) for f in os_mod.listdir(root_dir) if f.endswith('.npy') or f.endswith('.png') or f.endswith('.jpg')])
                # Simplified dataset logic for evaluation script structure
                # In real scenario, load the preprocessed test set
                
            def __len__(self):
                # Placeholder
                return 100
                
            def __getitem__(self, idx):
                # Placeholder: return dummy data of correct shape [Seq, Channels, Height, Width]
                input_seq = torch.randn(self.n_frames_input, 1, 256, 256)
                target_seq = torch.randn(self.n_frames_output, 1, 256, 256)
                return idx, target_seq, input_seq, None, None
                
        validFolder = GCAPPIDataset(root_dir='data/GCAPPI', n_frames_input=4, n_frames_output=4)
    else:
        print(f"[{model_name}] Initializing MovingMNIST dataset...")
        validFolder = MovingMNIST_set(is_train=False,
                              root='data/',
                              n_frames_input=10,
                              n_frames_output=10,
                              num_objects=[3])
    validLoader = torch.utils.data.DataLoader(validFolder,
                                          batch_size=8,
                                          shuffle=False)

    lossfunction = nn.MSELoss().to(device)
    ssim_func = SSIM().to(device)

    mse_list = []
    ssim_list = []

    print(f"[{model_name}] Starting evaluation loop...")
    with torch.no_grad():
        t = validLoader
        for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
            if i % 10 == 0:
                print(f"[{model_name}] Batch {i}/{len(validLoader)}")
            inputs = inputVar.to(device)
            label = targetVar.to(device)
            
            pred = net(inputs)
            
            mse = lossfunction(pred, label).item()
            ssim = ssim_func(pred, label).item()
            
            mse_list.append(mse)
            ssim_list.append(ssim)

    avg_mse = np.mean(mse_list)
    avg_ssim = np.mean(ssim_list)
    result_str = f"{model_name} -> MSE: {avg_mse:.8e}, SSIM: {avg_ssim:.6f}\n"
    print(result_str)
    result_file = f"eval_results_{args.dataset}.txt"
    with open(result_file, "a") as f:
        f.write(result_str)

if __name__ == '__main__':
    print(f"Starting evaluation script for {args.dataset} dataset...")
    result_file = f"eval_results_{args.dataset}.txt"
    with open(result_file, "w") as f:
        f.write(f"{args.dataset} Evaluation Results:\n")
    models = {
        'ours': 'save_model/CSDPNet/checkpoint.pth.tar',
        'graphatnet': 'save_model/GraphATNet/checkpoint.pth.tar',
        'cgru': 'save_model/cgru/checkpoint.pth.tar',
        'clstm': 'save_model/clstm/checkpoint.pth.tar',
        'fclstm': 'save_model/fc_lstm/checkpoint.pth.tar',
        'seresunet': 'save_model/seresunet/checkpoint.pth.tar',
        'smatunet': 'save_model/smatunet/checkpoint.pth.tar',
        'seclstmresnet34': 'save_model/SeClstmresnet34/checkpoint.pth.tar',
        'pspnet': 'save_model/pspnet/checkpoint.pth.tar'
    }
    
    for name, path in models.items():
        if os.path.exists(path):
            print(f"Evaluating {name}...")
            eval_model(name, path)
        else:
            print(f"Weights for {name} not found at {path}")
