# Human Depth Estimation Project

This repository contains a complete pipeline for **depth estimation of human-centered images**. The project explores several modelling strategies, loss functions, evaluation metrics, and qualitative visualization tools for predicting dense depth maps from RGB images.

The code supports both classic convolutional architectures such as **UNet** and transformer-based models such as **DeiT/Vision Transformer**. It also includes optional extensions for multi-task learning with human pose estimation and pose-guided depth consistency.

---

## Overview

The goal of the project is to estimate a depth map from a single RGB image:

```text
RGB image -> Model -> Depth map
```

The repository includes:

- Dataset preprocessing from synthetic human samples.
- RGB and depth map generation.
- 14-joint SMPL projection into image coordinates.
- Depth-only models.
- Multi-task depth and pose models.
- Data augmentation.
- Several depth loss functions.
- Quantitative metrics.
- Qualitative visualization scripts.

---

## Main Features

### Depth Estimation Models

The project implements different model families:

- **UNet** for encoder-decoder convolutional depth estimation.
- **UNet with joint prediction branch** for multi-task learning.
- **DeiT / Vision Transformer-based depth models**.
- **DeiT with joint prediction branch**.

### Training Modes

The code supports several training configurations through `loss_method`:

```python
loss_method = None
```

Standard depth estimation using a masked depth loss.

```python
loss_method = "normal"
```

Training with normal consistency loss.

```python
loss_method = "scale_invariant"
```

Training with scale-invariant depth loss.

```python
loss_method = "ranking"
```

Training with edge-guided ranking loss.

```python
loss_method = "multitask"
```

Joint training of depth estimation and 2D pose estimation.

```python
loss_method = "pose_guided"
```

Joint depth and pose training with an additional skeleton-depth consistency loss.

---

## Repository Structure

```text
.
├── main.py                  # Main training script
├── train.py                 # Training loop, losses, validation, checkpointing
├── utils.py                 # Dataset class, split loading, depth normalization
├── unet_task3.py            # UNet and UNet with joint prediction
├── deit_small.py            # DeiT/Vision Transformer depth models
├── metrics.py               # Evaluation metrics
├── ranking_loss.py          # Edge-guided ranking loss
├── preprocessing.py         # RGB/depth preprocessing
├── 14_joints.py             # SMPL 14-joint projection
├── dataaug.py               # Data augmentation methods
├── extract_frames.py        # Frame extraction utilities
└── data/
    ├── image/               # RGB images
    ├── depth/               # Depth maps
    ├── joints/              # 2D joint annotations
    ├── train.txt
    ├── validation.txt
    └── test.txt
```

---

## Dataset Format

The expected dataset structure is:

```text
data/
├── image/
│   ├── sample_0001.jpg
│   └── ...
├── depth/
│   ├── sample_0001.npy
│   └── ...
├── joints/
│   ├── sample_0001.npy
│   └── ...
├── train.txt
├── validation.txt
└── test.txt
```

Each RGB image has a corresponding depth map:

```text
data/image/<sample_name>.jpg
data/depth/<sample_name>.npy
```

For multi-task and pose-guided experiments, each sample also requires:

```text
data/joints/<sample_name>.npy
```

Each joint annotation contains 14 joints in the format:

```text
[x, y, valid]
```

where:

- `x` is the horizontal coordinate,
- `y` is the vertical coordinate,
- `valid` indicates whether the joint is valid inside the image/crop.

---

## Installation

Install the required dependencies:

```bash
pip install torch torchvision timm opencv-python numpy matplotlib pillow tqdm
```

---

## Preprocessing

### 1. Extract frames

If starting from video files, extract and merge RGB/segmentation frames using:

```bash
python extract_frames.py
```

### 2. Generate RGB images and depth maps

```bash
python preprocessing.py
```

This creates:

```text
data/image/
data/depth/
```

### 3. Generate 2D SMPL joints

```bash
python 14_joints.py
```

This creates:

```text
data/joints/
```

### 4. Prepare train/validation/test splits

The split files should be stored as:

```text
data/train.txt
data/validation.txt
data/test.txt
```

Each line should contain the sample name without extension.

---

## Training

Training is configured in `main.py`.

A typical UNet configuration is:

```python
lr = 1e-3
batch_size = 2
resolution = 384
n_epochs = 11

model_name = "unet"
partition = "validation"
loss_method = None

n_class = 1
f1 = 64
f2 = 128
f3 = 256
f4 = 512
f5 = 1024
activation = "gelu"
output_activation = "sigmoid"
```

Run training with:

```bash
python main.py
```

Outputs are saved in:

```text
results/
results/plots/
```

---

## Data Augmentation

The repository includes data augmentation utilities in `dataaug.py`, including:

- crop,
- shift,
- horizontal flip,
- rotation,
- shearing,
- grayscale conversion,
- degradation,
- uneven scaling.

These augmentations are useful for improving robustness in depth-only experiments.

