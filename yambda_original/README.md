# Original Reproducibility Guide

For kaggle training and submission see kaggle guide.

## Overview

This part of repo contains the implementation and experiments. This guide will help you reproduce the results using Docker or manual installation.

---

## Docker Setup (Recommended)

### 1. Build Docker Image

```bash
docker build -t yambda-image .
```

### 2. Run Container with GPU Support

```bash
docker run --gpus all \
           --runtime=nvidia \
           -it \
           -v </absolute/path/to/local/data>:/yambda/data \
           yambda-image
```

---

## Data Organization

Create following structure in mounted data directory:

```bash
data/
├── flat/
│   └── 50m/
│       ├── likes.parquet
│       ├── listens.parquet
│       └── ...
└── sequential/
    └── 50m/
        ├── likes.parquet
        ├── listens.parquet
        └── ...
```

Note:
    Sequential data is only needed for sasrec. You can build it from flat using scripts/transform2sequential.py or download

---

## Running Experiments

### General Usage

```bash
# For example random_rec

cd models/random_rec/

# Show help for main script
python main.py --help

# Basic execution
python main.py
```

### Specific Methods

#### BPR/ALS

```bash
cd models/bpr_als

python main.py --model bpr
python main.py --model als
```

#### SASRec

```bash
cd models/sasrec

# Local validation: train on first 299 days, validate on day 300
python train.py --exp_name local_val --train_days 299 --validate

# Final submission: train on full dataset
python train.py --exp_name final --full_train

# Evaluate on test set
python eval.py --exp_name local_val --train_days 299

# Generate candidates for kagglehub test users
python eval.py --exp_name final --use_kagglehub --full_train --output_path ./candidates.parquet

# Generate candidates for custom test users
python eval.py --exp_name exp1 --test_users_path ./test_users.parquet --output_path ./candidates.parquet
```
---

## Manual Installation (Not Recommedned)

### 1. Install Core Dependencies

```bash
pip install torch torchvision torchaudio
```

### 2. Install Implicit (CUDA 11.8 required)

Implicit works only with cuda<12. See reasons [here](https://github.com/NVIDIA/nvidia-docker/issues/700#issuecomment-381073278)

```bash
CUDACXX=/usr/local/cuda-11.8/bin/nvcc \
pip install implicit
```

### 3. Install SANSA

```bash
sudo apt-get install libsuitesparse-dev
git clone https://github.com/glami/sansa.git
cd sansa && \
SUITESPARSE_INCLUDE_DIR=/usr/include/suitesparse \
SUITESPARSE_LIBRARY_DIR=/usr/lib \
pip install .
```

### 4. Install Project Package

```bash
pip install .
```
