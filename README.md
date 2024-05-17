# Reflected Flow Matching

## Authors

[Add authors here]

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

We provide all the pretrained weights[?].

## References
[1] Tong, A., Malkin, N., Huguet, G., Zhang, Y., Rector-Brooks, J., Fatras, K., ... & Bengio, Y. (2023). Improving and generalizing flow-based generative models with minibatch optimal transport. arXiv preprint arXiv:2302.00482.

[2]Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., & Le, M. (2022). Flow matching for generative modeling. arXiv preprint arXiv:2210.02747.