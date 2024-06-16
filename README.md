### <div align="center"> Reflected Flow Matching <div> 
### <div align="center"> ICML 2024 Poster <div> 

<div align="center">
  <a href="https://github.com/PixArt-alpha/PixArt-sigma/](https://github.com/tyuxie/RFM)"><img src="https://img.shields.io/static/v1?label=RFM Code&message=Github&color=blue&logo=github-pages"></a> &ensp;
  <a href="https://arxiv.org/abs/2405.16577"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv&color=red&logo=arxiv"></a> &ensp;
  <a href="https://openreview.net/forum?id=Sf5KYznS2G"><img src="https://img.shields.io/static/v1?label=Paper&message=Openreview&color=red&logo=arxiv"></a> &ensp;
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a> &ensp;
</div>

<div align="center">
Tianyu Xie*, Yu Zhu*, Longlin Yu*, Tong Yang, Ziheng Cheng, Shiyue Zhang, Xiangyu Zhang, Cheng Zhang
</div>

---

This repo contains PyTorch model definitions, pre-trained weights and inference/sampling code for Reflected Flow Matching.


## Installation and Running

### Setup `torch` Environment

To create the `torch` environment, use the following command:

```
conda env create -f torch.yml
```

**Note**: Please strictly follow this environment setup to ensure reproducibility of the results.

### Training and Testing
**Training and Testing Workflow**
To train and test the models, follow these steps:

```
# CIFAR10 Training
cd ./CIFAR10
./bash_train.sh

# CIFAR10 Testing
./bash_test.sh

# ImageNet64 Training
cd ./ImageNet64
./bash_train.sh

# ImageNet64 Testing
./bash_test.sh
```

**FID Calculation**
For CIFAR10, we use the statistics computed by cleanfid for FID calculation.
For ImageNet64, please follow the instructions below to recompute the statistics using cleanfid:
```
cd ./ImageNet64
python precompute_FID_statistics.py
```

**Note**: The ImageNet64 experiments require 64 A100 GPUs for training and 8 GPUs (2080Ti is sufficient) for testing. 

For CIFAR10, we employ all the experiments on 8 2080Ti GPUs. 

We provide all the pretrained weights at [Google Drive repository](https://drive.google.com/drive/folders/12m2FJiA2Jg9mej3os_wSyh6QLm2JL02Q?usp=sharing).


## References
[1] Tong, A., Malkin, N., Huguet, G., Zhang, Y., Rector-Brooks, J., Fatras, K., ... & Bengio, Y. (2023). Improving and generalizing flow-based generative models with minibatch optimal transport. arXiv preprint arXiv:2302.00482.

[2] Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., & Le, M. (2022). Flow matching for generative modeling. arXiv preprint arXiv:2210.02747.

[3] Chen, Ricky T. Q. torchdiffeq. 2018. https://github.com/rtqichen/torchdiffeq

---

If you find this repository useful in your research, please consider citing:

```
@inproceedings{
xie2024rfm,
title={Reflected Flow Matching},
author={Tianyu Xie and Yu Zhu and Longlin Yu and Tong Yang and Ziheng Cheng and Shiyue Zhang and Xiangyu Zhang and Cheng Zhang},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
}
```