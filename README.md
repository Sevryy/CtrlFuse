# CtrlFuse: Mask-Prompt Guided Controllable Infrared and Visible Image Fusion (Official PyTorch Implementation)

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2601.08619)
[![Framework](https://img.shields.io/badge/PyTorch-%3E%3D2.0.0-orange)](https://pytorch.org/)
[![Stars](https://img.shields.io/github/stars/[GithubUsername]/[RepoName].svg?style=social)](https://github.com/[GithubUsername]/[RepoName])

This repository contains the official PyTorch implementation of the paper:
**"CtrlFuse: Mask-Prompt Guided Controllable Infrared and Visible Image Fusion"** (Accepted by AAAI 2026)

> **Authors**: Yiming Sun, Yuan Ruan, Qinghua Hu,Pengfei Zhu
> **Affiliation**: VisDrone Group

## 📢 News
- **[2026-01]**: Code and pre-trained models are released!
- **[2025-11-08]**: The paper is accepted by AAAI 2026.

## 📜 Abstract
Infrared and visible image fusion generates all-weather perception-capable images by combining complementary modalities, enhancing environmental awareness for intelligent unmanned systems. Existing methods either focus on pixel-level fusion while overlooking downstream task adaptability or implicitly learn rigid semantics through cascaded detection/segmentation models, unable to interactively address diverse semantic target perception needs. We propose CtrlFuse, a controllable image fusion framework that enables interactive dynamic fusion guided by mask prompts. The model integrates a multi-modal feature extractor, a reference prompt encoder (RPE), and a prompt-semantic fusion module(PSFM). The RPE dynamically encodes task-specific semantic prompts by fine-tuning pre-trained segmentation models with input mask guidance, while the PSFM explicitly injects these semantics into fusion features. Through synergistic optimization of parallel segmentation and fusion branches, our method achieves mutual enhancement between task performance and fusion quality. Experiments demonstrate state-ofthe-art results in both fusion controllability and segmentation accuracy, with the adapted task branch even outperforming the original segmentation model.

![Network Architecture](model.png)
*Figure 1: The overall architecture of our proposed CtrlFuse.*

## 🔨 Requirements
The code has been tested with Python 3.8 and PyTorch 2.0.0 .
checkpoints can be downloaded with the links below:
[![Baidu Yun]](https://pan.baidu.com/s/1hqUoh8TS2ZCn4IWTDbV_vQ?pwd=sh7h)
Additionally, you can download the ViT-H SAM model from the official Segment Anything website.

```bash
# 1. Create a conda environment
conda create -n ctrlfuse python=3.8
conda activate ctrlfuse

# 2. Install dependencies
pip install -r requirements.txt

# 3.  Segment-Anything-Model setting
cd ./segment-anything
pip install -v -e .
cd ..
```

## 📂 Data Preparation
Please organize your dataset as follows. Note: Ensure that the Visible and Infrared images are strictly aligned (registered) and have the same filenames.

```
Project_Root/
├── dataset/
│   ├── train/
│   │   ├── vi/             # Visible images (RGB)
│   │   │   ├── 1.jpg
│   │   │   └── ...
│   │   └── ir/           # Infrared images (Grayscale)
│   │   │   ├── 1.jpg
│   │   │   └── ...
│   │   └── mask/             # mask (Grayscale)
│   │       ├── 1.jpg
│   │       └── ...
│   └── test/
│   │   ├── vi/             # Visible images (RGB)
│   │   │   ├── 1.jpg
│   │   │   └── ...
│   │   └── ir/           # Infrared images (Grayscale)
│   │   │   ├── 1.jpg
│   │   │   └── ...
│   │   └── mask/             # mask (Grayscale)
│   │       ├── 1.jpg
│   │       └── ...
```

## 🚀 Usage

## 📊 Results

| **FMB Dataset**  | MSE   | PSNR  | Q<sub>abf</sub> | N<sub>abf</sub> | SSIM  | SCD   |
|---------------|-------|-------|------------------|------------------|-------|-------|
| LDFusion      | 0.061 | 60.71 | 0.51             | 0.112            | 0.514 | 1.549 |
| SwinFuse      | **0.042** | 62.334 | 0.577 | 0.029 | 0.905 | **1.900** |
| NestFuse      | 0.046 | 61.96 | 0.483            | 0.042            | 0.787 | 1.594 |
| CDDFuse       | 0.048 | 62.696 | 0.674 | 0.026 | **1.002** | 1.626 |
| DIDFuse       | 0.047 | 61.565 | 0.528            | 0.042            | 0.765 | 1.824 |
| SeAFusion     | 0.047 | 62.539 | 0.654 | 0.029 | 0.964 | 1.62  |
| PSFusion      | 0.051 | 61.517 | 0.627            | 0.056            | 0.836 | 1.875 |
| SDCFusion     | 0.048 | 62.456 | 0.693 | 0.031 | 0.906 | 1.657 |
| CtrlFuse(Ours)| 0.043 | **63.292** | **0.719** | **0.024** | 0.925 | 1.522 |

| **Drone Vehicle Dataset** | MSE   | PSNR  | Q<sub>abf</sub> | N<sub>abf</sub> | SSIM  | SCD     |
|---------------------------|-------|-------|------------------|------------------|-------|-------|
| LDFusion                  | 0.076 | 59.573 | 0.376            | 0.054            | 0.568 | 1.38  |
| SwinFuse                  | 0.084 | 59.165 | 0.202            | 0.069            | 0.558 | 1.295 |
| NestFuse                  | 0.071 | 59.786 | 0.307            | 0.052            | 0.486 | 1.413 |
| CDDFuse                   | 0.065 | 60.199 | 0.469 | **0.021** | 0.845 | 1.359 |
| DIDFuse                   | 0.067 | 59.988 | 0.265            | 0.062            | 0.466 | 1.459 |
| SeAFusion                 | 0.094 | 58.649 | 0.492            | 0.044            | **0.879** | 1.472 |
| PSFusion                  | 0.067 | 60.065 | 0.454 | 0.095 | 0.717 | 1.534 |
| SDCFusion                 | 0.078 | 59.443 | **0.534** | 0.035 | 0.853 | 1.316 |
| CtrlFuse(Ours)            | **0.063** | **60.317** | 0.496 | 0.035 | 0.779 | **1.552** |

| **MSRS Dataset** |  MSE   | PSNR  | Q<sub>abf</sub> | N<sub>abf</sub> | SSIM  | SCD    |
|------------------|-------|-------|------------------|------------------|-------|-------|
| LDFusion         | 0.056 | 61.05 | 0.438            | 0.116            | 0.541 | 1.515 |
| SwinFuse         | 0.038 | 63.69 | 0.178            | 0.026            | 0.343 | 1.033 |
| NestFuse         | 0.033 | 64.128 | 0.242            | 0.025            | 0.217 | 1.138 |
| CDDFuse          | 0.038 | 64.309 | 0.689 | 0.023            | **1.001** | 1.623 |
| DIDFuse          | **0.035** | 63.94 | 0.204            | 0.025            | 0.223 | 1.121 |
| SeAFusion        | 0.036 | 64.491 | 0.675          | 0.021        | 0.982 | 1.707 |
| PSFusion         | 0.037 | 64.001 | 0.676            | 0.042            | 0.917 | **1.812** |
| SDCFusion        | 0.039 | 64.003 | **0.712** | 0.023            | 0.957 | 1.739 |
| CtrlFuse(Ours)   | **0.035** | **64.75** | 0.685          | **0.018**        | 0.969 | 1.726 |

## 🤝 Citation

## 📧 Contact
If you have any other questions about the code, please email ruanyuan@seu.edu.cn.

