# CSDP-Net

Official implementation of **“Morphology-aware spatiotemporal prediction of cumulonimbus-related radar echoes using CSDP-Net”**, published in *Theoretical and Applied Climatology*.

* 中文说明：见 [README_zh.md](README_zh.md)
* English documentation: see [README_en.md](README_en.md)

## Paper

**Morphology-aware spatiotemporal prediction of cumulonimbus-related radar echoes using CSDP-Net**

Deyi Wang, Xu Wang, Zhenhong Cheng, Xin Chang, Hongping Yuan, Jixiang Yang, Yi Liang, Jiaming Hu, and Ting Zhang.

*Theoretical and Applied Climatology*, Volume 157, Issue 5, Article 324, 2026.

* Published: April 28, 2026
* DOI: [10.1007/s00704-026-06276-x](https://doi.org/10.1007/s00704-026-06276-x)
* ISSN: 1434-4483

## Abstract

Accurate prediction of cumulonimbus radar echo distribution remains challenging because most existing methods emphasize motion extrapolation or generic spatiotemporal learning while insufficiently representing the irregular boundaries and structural evolution of convective echoes. This limitation often leads to blurred boundaries, degraded structural consistency, and failure to capture nonlinear meteorological events.

To address these issues, we propose **CSDP-Net**, a boundary-aware spatiotemporal prediction network that combines **Cumulonimbus Boundary-Sensitive Convolution (CBS-Conv)** with a **Cumulonimbus Temporal-Spatial Feature Fusion (CTSFF)** module. CBS-Conv aligns with the morphology-aware characteristics of cumulonimbus-related echoes, and CTSFF refines deep semantic information to enhance prediction accuracy.

Extensive experiments demonstrate the effectiveness of CSDP-Net. On the GCAPPI-2.0 dataset, CSDP-Net achieves the lowest Mean Squared Error (MSE) while maintaining a highly competitive Structural Similarity Index (SSIM). Competitive performance is also observed on the Moving MNIST benchmark. Ablation studies further validate the efficacy of the proposed modules, underscoring their complementary roles in local boundary representation and global structural consistency.

## Quick Start

### Installation

```bash
conda create -n csdp_net python=3.8
conda activate csdp_net
pip install -r requirements.txt
```

### Training

```bash
python main.py -CSDPNet -N 5
```

### Evaluation

```bash
python run_eval.py -CSDPNet -dataset moving_mnist -N 5
```

### Inference

```bash
python inference.py -CSDPNet -dataset moving_mnist -sd <checkpoint_path>
```

## Citation

Please cite the following paper if you find this repository useful:

```bibtex
@article{Wang2026,
  author  = {Wang, Deyi and Wang, Xu and Cheng, Zhenhong and Chang, Xin and
             Yuan, Hongping and Yang, Jixiang and Liang, Yi and Hu, Jiaming and
             Zhang, Ting},
  title   = {Morphology-aware spatiotemporal prediction of cumulonimbus-related
             radar echoes using {CSDP-Net}},
  journal = {Theoretical and Applied Climatology},
  year    = {2026},
  volume  = {157},
  number  = {5},
  pages   = {324},
  doi     = {10.1007/s00704-026-06276-x},
  url     = {https://doi.org/10.1007/s00704-026-06276-x},
  issn    = {1434-4483}
}
```
