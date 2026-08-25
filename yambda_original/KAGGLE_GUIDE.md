# SASRec Training and Submission Guide

## Overview

This guide covers training SASRec for local validation and generating submissions for the competition.

---

## Quick Start

### Local Validation (train on first 299 days, validate on day 300)

```bash
python -m models.sasrec.train \
    --exp_name local_val \
    --train_days 299 \
    --validate \
    --num_epochs 100 \
    --device cuda:0
```

### Final Submission (train on full dataset, generate candidates from kagglehub)

```bash
# Step 1: Train on full dataset
python -m models.sasrec.train \
    --exp_name final \
    --full_train \
    --num_epochs 100 \
    --device cuda:0

# Step 2: Generate candidates for competition test users
python -m models.sasrec.eval \
    --exp_name final \
    --use_kagglehub \
    --full_train \
    --output_path ./submission.parquet \
    --device cuda:0
```

---

## Training Options

### train.py

| Option | Description |
|--------|-------------|
| `--train_days N` | Train on first N days, test on day N+1 (for local validation) |
| `--validate` | Run validation after training (requires `--train_days`) |
| `--full_train` | Train on full dataset without time split (for final submission) |

**Examples:**

```bash
# Local validation: train on first 299 days, validate on day 300
python -m models.sasrec.train --exp_name local_val --train_days 299 --validate

# Full training for submission
python -m models.sasrec.train --exp_name final --full_train

# Default: use Constants.TEST_TIMESTAMP for splitting
python -m models.sasrec.train --exp_name default
```

### generate_candidates.py

| Option | Description |
|--------|-------------|
| `--use_kagglehub` | Load test users from kagglehub (for competition submission) |
| `--full_train` | Use full dataset for training data (no time split) |
| `--test_users_path` | Path to custom test_users.parquet file |

**Examples:**

```bash
# Generate candidates for competition (kagglehub test users)
python -m models.sasrec.eval --exp_name final --use_kagglehub --full_train

# Generate candidates for custom test users
python -m models.sasrec.eval --exp_name exp1 --test_users_path ./test_users.parquet
```

---

## Recommended Workflow

```
1. Local Validation (Hyperparameter Tuning)
   ↓
   python -m models.sasrec.train --exp_name val --train_days 299 --validate
   ↓
   Try different: embedding_dim, num_layers, learning_rate
   ↓
   Choose best hyperparameters based on recall@100

2. Final Submission
   ↓
   python -m models.sasrec.train --exp_name final --full_train
   ↓
   python -m models.sasrec.eval --exp_name final --use_kagglehub --full_train
   ↓
   Submit submission.parquet to competition
```

---

## Hyperparameter Tuning Examples

```bash
# Try default parameters
python -m models.sasrec.train --exp_name val_default --train_days 299 --validate

# Try larger model
python -m models.sasrec.train --exp_name val_large --train_days 299 --validate \
    --embedding_dim 128 --num_layers 4

# Try different learning rate
python -m models.sasrec.train --exp_name val_lr --train_days 299 --validate \
    --learning_rate 5e-4
```

---

## Output Format

The generated `submission.parquet` contains:

| Column | Type | Description |
|--------|------|-------------|
| `uid` | int | User ID |
| `candidate_item_ids` | list[int] | Top-100 recommended item IDs |
| `candidate_scores` | list[float] | Recommendation scores |

---

## Requirements

```bash
pip install kagglehub  # For --use_kagglehub option
```

---

## Data Organization

```bash
data/
└── sequential/
    └── 50m/
        └── likes.parquet
```

Sequential data is required for SASRec. Convert from flat using:

```bash
python scripts/transform2sequential.py --input data/flat/50m/likes.parquet --output data/sequential/50m/likes.parquet
```

---

## Docker Setup

```bash
# Build
docker build -t yambda-image .

# Run with GPU
docker run --gpus all --runtime=nvidia -it \
    -v /path/to/data:/yambda/data \
    yambda-image
```

---

## Manual Installation

```bash
pip install torch torchvision torchaudio
pip install .
```
