# Dataset Distillation with Pseudo-Label Training

Modified implementation based on the [WMDD (Dataset Distillation via the Wasserstein Metric)](https://github.com/Liu-Hy/WMDD) repository, adapted to support training with pseudo-labels on full datasets.

## Overview

This repository extends the original WMDD codebase to enable training models on full datasets using pseudo-labels derived from cluster assignments or other labeling schemes. The primary modification allows bypassing the dataset distillation synthesis and relabeling stages when working directly with pseudo-labeled full datasets.

### Key Modifications from Original WMDD

1. **Pseudo-Label Training Support**: Added `--pseudo-label-csv` parameter to `train_FKD.py` to load and apply pseudo-labels from CSV files during training
2. **Dataset-Specific Scripts**: Created convenience scripts for running experiments on CIFAR-100, ImageNette, and Tiny-ImageNet with pseudo-labels:
   - `run_cifar100_pseudo.sh`
   - `run_imagenette_pseudo.sh` 
   - `run_tinyimagenet_pseudo.sh`
   - `train_only_pseudo.sh`
3. **CSV-Based Label Mapping**: Training can now use external pseudo-label CSV files that map image paths to cluster-derived class assignments

### Use Cases

This modified version is designed for experiments that:
- Use cluster-based pseudo-labels instead of original ground-truth labels
- Train on full datasets rather than distilled synthetic data
- Skip the computationally expensive distillation synthesis and soft-label generation stages
- Evaluate model performance with alternative labeling schemes

### Original WMDD Reference

The base implementation comes from the ICCV 2025 paper:
> "Dataset Distillation via the Wasserstein Metric"  
> Haoyang Liu, Yijiang Li, Tiancheng Xing, Peiran Wang, Vibhu Dalal, Luwei Li, Jingrui He, Haohan Wang  
> UIUC, UC San Diego, NUS

[`[Paper]`](https://arxiv.org/abs/2311.18531) [`[Original Code]`](https://github.com/Liu-Hy/WMDD) [`[Website]`](https://liu-hy.github.io/WMDD/)

## Environment Setup

Create a conda environment with Python 3.10 and install the required packages:

```bash
conda create -n wmdd python=3.10 -y
conda activate wmdd
pip install -r requirements.txt
```

**Note**: If using the FKD (Fast Knowledge Distillation) training mode, modify the PyTorch source code according to [train/README.md](train/README.md).

## Usage

### Training with Pseudo-Labels

To train models using pseudo-labels on full datasets (bypassing distillation):

```bash
bash run_cifar100_pseudo.sh
bash run_imagenette_pseudo.sh
bash run_tinyimagenet_pseudo.sh
```

These scripts train directly on the full dataset using pseudo-labels from CSV files (`train_image_pseudo_labels_*.csv`).

**CSV Format**: The pseudo-label CSV should contain columns:
- `image_path`: Relative path to the image (e.g., "train/0/000002.png")
- `pseudo_label_class_index`: Integer class index to use for training
- Additional columns like `cluster_id`, `true_label_index`, etc. are optional

### Running Full WMDD Pipeline (Original)

To run the complete dataset distillation pipeline with pretrain, synthesis, relabel, and evaluation:

```bash
bash run.sh -x 1 -y 1 -d imagenette -u 0 -c 10 -r /home/user/data/ -n -w -b 10 -p
```

**Flags**:
- `-x`: Experiment ID for this run
- `-y`: Teacher model ID to use/create
- `-d`: Dataset name (imagenette, cifar100, tiny-imagenet)
- `-u`: GPU index (default 0)
- `-c`: Images per class (IPC) for distillation
- `-r`: Root directory containing datasets
- `-p`: Include to pretrain a teacher model from scratch
- `-n`: Enable per-class BatchNorm (recommended)
- `-w`: Enable Wasserstein barycenter computation
- `-b`: BatchNorm regularization coefficient

**Tips**:
- Prepare datasets in ImageFolder format in the directory specified by `-r`
- Pretrain teacher models once with `-p`, then reuse by matching `-y` values
- For pseudo-label training, edit the corresponding `run_*_pseudo.sh` script to set paths and parameters

## Project Structure

```
├── pretrain/          # Teacher model pretraining
├── recover/           # Synthetic data generation via Wasserstein barycenters
├── relabel/           # Soft label generation with FKD (Fast Knowledge Distillation)
├── train/             # Model training on distilled or pseudo-labeled data
├── models/            # Model architectures
├── datasets/          # Dataset storage
├── log/               # Experiment logs
└── *.csv              # Pseudo-label mappings for different datasets
```

## Citation

If you use the original WMDD method, please cite:

```bibtex
@misc{liu2025wmdd,
      title={Dataset Distillation via the Wasserstein Metric}, 
      author={Haoyang Liu and Yijiang Li and Tiancheng Xing and Peiran Wang and Vibhu Dalal and Luwei Li and Jingrui He and Haohan Wang},
      year={2025},
      eprint={2311.18531},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2311.18531}, 
}
```

