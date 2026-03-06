# Comparative Fine-Tuning of a Compact LLM for PubMedQA: FFT, QLoRA, Knowledge Distillation, and DPO on Qwen3-0.6B
**Module: 7043SCN — Generative AI and Reinforcement Learning - Task 1**

**Institution: Coventry University**

**Author: Mohamed Bseikri**

**Student ID: 12224702**

## Overview
This project investigates resource-efficient fine-tuning of a compact 0.6B parameter model (Qwen3-0.6B) for expert-level biomedical question answering on the PubMedQA benchmark. Four adaptation techniques are systematically compared:

- Baseline — Untuned Qwen3-0.6B

- Full Fine-Tuning (FFT) — All weights updated via supervised training

- QLoRA — 4-bit quantized low-rank adaptation

- Knowledge Distillation (KD) — Student trained from II-Medical-8B teacher

- Direct Preference Optimization (DPO) — Preference alignment from teacher/base pairs

<img width="8192" height="5233" alt="image" src="https://github.com/user-attachments/assets/37243eb6-eb59-4078-a99d-e4928a9a603a" />

### Results Summary
QLoRA achieved the best overall performance. DPO preserved QLoRA performance while enforcing strict formatting alignment. KD degraded below baseline, likely due to noisy pseudo-labels from the teacher model.

<img width="4170" height="2966" alt="image" src="https://github.com/user-attachments/assets/067cbe6b-338b-417e-9468-8c30978956b0" />

## Requirements
### Hardware used:

- GPU: NVIDIA RTX 3090 (24GB VRAM)
- CPU: AMD Ryzen 7 7800X3D
- RAM: 48GB
- OS: Arch Linux

### Install dependencies:

#### Core Deep Learning & NLP
- PyTorch: 2.10.0 (with CUDA 12.8 runtimes: nvidia-cuda-runtime-cu12==12.8.90)
- Transformers (Hugging Face): 4.57.6
- TRL (Transformer Reinforcement Learning): 0.24.0
- PEFT: 0.18.1
- Unsloth: 2026.2.1
- Tokenizers: 0.22.2
- Datasets: 2.21.0

#### Hardware Optimization & Memory
- Bitsandbytes: 0.49.2
- Xformers: 0.0.35
- Triton: 3.6.0

#### Evaluation & Metrics
- Evaluate (Hugging Face): 0.4.6
- Rouge-score: 0.1.2
- Sacrebleu: 2.6.0
- Scikit-learn: 1.8.0

#### Data Processing & Visualization
- Pandas: 3.0.1
- NumPy: 2.4.2
- SciPy: 1.17.1
- Matplotlib: 3.10.8
- Seaborn: 0.13.2

#### Dataset Configuration
- 50,000 training examples (PQA-A)
- 5,000 validation examples (PQA-A)
- 1,000 test examples (PQA-L (Train set)

## Interpreting Results


## Dataset
- PubMedQA (Jin et al., 2019) — https://huggingface.co/datasets/qiaojin/PubMedQA
- PQA-A (Artificial): Used for training and validation
- PQA-L (Labelled): Used for final evaluation only


## Model
- Qwen3-0.6B — https://huggingface.co/Qwen/Qwen3-0.6B
- Teacher Model: II-Medical-8B — https://huggingface.co/Intelligent-Internet/II-Medical-8B
