# CSDP-Net

- 中文说明：见 [README_zh.md](README_zh.md)
- English documentation: see [README_en.md](README_en.md)

## Quick Start

```bash
conda create -n csdp_net python=3.8
conda activate csdp_net
pip install -r requirements.txt
```

```bash
python main.py -CSDPNet -N 5
python run_eval.py -CSDPNet -dataset moving_mnist -N 5
python inference.py -CSDPNet -dataset moving_mnist -sd <checkpoint_path>
```
