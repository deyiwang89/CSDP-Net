# CSDP-Net（中文说明）

## 项目简介

CSDP-Net 是一个用于时空序列预测的 PyTorch 项目，核心模型为 `CSDPNet`，并包含多个基线模型（如 ConvGRU、ConvLSTM、GraphATNet、FC_LSTM、PSPNet 等）用于对比实验。

## 目录说明

- `main.py`：训练入口
- `run_eval.py`：评估入口
- `inference.py`：推理与可视化入口
- `CSDPNet.py`：CSDP-Net 主体实现
- `data/mm.py`：MovingMNIST 数据集构造
- `requirements.txt`：Python 依赖清单

## 环境安装

```bash
conda create -n csdp_net python=3.8
conda activate csdp_net
pip install -r requirements.txt
```

## 数据准备

- MovingMNIST 相关文件放在 `data/` 下（仓库已包含常用文件名）：
  - `train-images-idx3-ubyte.gz`
  - `mnist_test_seq.npy`
- GCAPPI 使用时请按你的本地路径准备数据。

## 常用命令

### 1) 训练 CSDP-Net

```bash
python main.py -CSDPNet -N 5 --batch_size 8 -frames_input 10 -frames_output 10
```

### 2) 评估 CSDP-Net（MovingMNIST）

```bash
python run_eval.py -CSDPNet -dataset moving_mnist -N 5
```

### 3) 推理并导出可视化

```bash
python inference.py -CSDPNet -dataset moving_mnist -sd <checkpoint_path>
```

## 权重兼容说明

- 当前代码支持自动识别历史权重的注意力版本与卷积核大小：
  - 若权重中存在 `EAA.to_query/to_key`，会按旧版注意力结构加载；
  - 会根据 `rnn1_0.conv1.0.weight` 自动推断 `N`（卷积核大小）。
- 推理与评估脚本已集成上述自动识别逻辑。
