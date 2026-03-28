# CSDP-Net (English)

## Overview

CSDP-Net is a PyTorch project for spatiotemporal prediction.  
The repository includes the `CSDPNet` model and multiple baselines (ConvGRU, ConvLSTM, GraphATNet, FC_LSTM, PSPNet, etc.) for comparison.

## Repository Structure

- `main.py`: training entry
- `run_eval.py`: evaluation entry
- `inference.py`: inference and visualization entry
- `CSDPNet.py`: core CSDP-Net implementation
- `data/mm.py`: MovingMNIST data pipeline
- `requirements.txt`: Python dependency list

## Environment Setup

```bash
conda create -n csdp_net python=3.8
conda activate csdp_net
pip install -r requirements.txt
```

## Data Preparation

- Place MovingMNIST files in `data/` (common filenames already expected by code):
  - `train-images-idx3-ubyte.gz`
  - `mnist_test_seq.npy`
- For GCAPPI, prepare local data paths before running experiments.

## Common Commands

### 1) Train CSDP-Net

```bash
python main.py -CSDPNet -N 5 --batch_size 8 -frames_input 10 -frames_output 10
```

### 2) Evaluate CSDP-Net (MovingMNIST)

```bash
python run_eval.py -CSDPNet -dataset moving_mnist -N 5
```

### 3) Run inference and export visual results

```bash
python inference.py -CSDPNet -dataset moving_mnist -sd <checkpoint_path>
```

## Checkpoint Compatibility

- Current code automatically detects historical checkpoint variants:
  - If `EAA.to_query/to_key` exists, it loads with legacy attention mode.
  - `N` (kernel size) is inferred from `rnn1_0.conv1.0.weight`.
- This logic is already integrated in both `inference.py` and `run_eval.py`.
