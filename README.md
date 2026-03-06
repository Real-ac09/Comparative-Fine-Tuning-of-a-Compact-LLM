Comparative Fine-Tuning of a Compact LLM for PubMedQA
QLoRA, Knowledge Distillation, and DPO on Qwen3-0.6B
Module: 7043SCN — Generative AI and Reinforcement Learning
Institution: Coventry University
Author: Mohamed Bseikri

Overview
This project investigates resource-efficient fine-tuning of a compact 0.6B parameter model (Qwen3-0.6B) for expert-level biomedical question answering on the PubMedQA benchmark. Four adaptation techniques are systematically compared:

Baseline — Untuned Qwen3-0.6B
Full Fine-Tuning (FFT) — All weights updated via supervised training
QLoRA — 4-bit quantized low-rank adaptation
Knowledge Distillation (KD) — Student trained from II-Medical-8B teacher
Direct Preference Optimization (DPO) — Preference alignment from teacher/base pairs

Results Summary
QLoRA achieved the best overall performance. DPO preserved QLoRA performance while enforcing strict formatting alignment. KD degraded below baseline, likely due to noisy pseudo-labels from the teacher model.

Requirements
Hardware used:

GPU: NVIDIA RTX 3090 (24GB VRAM)
CPU: AMD Ryzen 7 7800X3D
RAM: 48GB
OS: Arch Linux

Install dependencies:
bashpip install -r requirements.txt
requirements.txt includes:
torch
transformers
datasets
peft
trl
bitsandbytes
evaluate
accelerate
rouge-score
nltk
scipy

Dataset Configuration
50,000 training examples (PQA-A)
5,000 validation examples (PQA-A)
1,000 test examples (PQA-L (Train set)

Interpreting Results


Dataset
PubMedQA (Jin et al., 2019) — https://huggingface.co/datasets/qiaojin/PubMedQA

PQA-A (Artificial): Used for training and validation
PQA-L (Labelled): Used for final evaluation only


Model
Qwen3-0.6B — https://huggingface.co/Qwen/Qwen3-0.6B
Teacher Model: II-Medical-8B — https://huggingface.co/Intelligent-Internet/II-Medical-8B
