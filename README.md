Comparative Fine-Tuning of a Compact LLM for PubMedQA
QLoRA, Knowledge Distillation, and DPO on Qwen3-0.6B
Module: 7043SCN — Generative AI and Reinforcement Learning
Institution: Coventry University
Author: Mohamed Bseikri
Repository: https://github.com/Real-ac09/Comparative-Fine-Tuning-of-a-Compact-LLM

Overview
This project investigates resource-efficient fine-tuning of a compact 0.6B parameter model (Qwen3-0.6B) for expert-level biomedical question answering on the PubMedQA benchmark. Four adaptation techniques are systematically compared:

Baseline — Untuned Qwen3-0.6B
Full Fine-Tuning (FFT) — All weights updated via supervised training
QLoRA — 4-bit quantized low-rank adaptation
Knowledge Distillation (KD) — Student trained from II-Medical-8B teacher
Direct Preference Optimization (DPO) — Preference alignment from teacher/base pairs


Results Summary
ModelAccuracyMacro F1Format ComplianceBaseline50.6%0.27575%FFT57.2%—44.3%QLoRA68.2%0.455100%KD41.3%0.162100%DPO68.0%~0.455100%
QLoRA achieved the best overall performance. DPO preserved QLoRA performance while enforcing strict formatting alignment. KD degraded below baseline, likely due to noisy pseudo-labels from the teacher model.

Repository Structure
├── data/
│   └── preparation scripts for PQA-A and PQA-L splits
├── training/
│   ├── fft_train.py          # Full Fine-Tuning
│   ├── qlora_train.py        # QLoRA training
│   ├── kd_train.py           # Knowledge Distillation
│   └── dpo_train.py          # Direct Preference Optimization
├── evaluation/
│   └── evaluate.py           # Evaluation pipeline (accuracy, F1, ROUGE-L, BLEU-4)
├── results/
│   └── plots/                # Training curves and result visualisations
├── requirements.txt
└── README.md

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

How to Run
1. Prepare Data
bashpython data/prepare_data.py
Downloads PubMedQA from Hugging Face and creates:

50,000 training examples (PQA-A)
5,000 validation examples (PQA-A)
1,000 test examples (PQA-L)

2. Run Baseline Evaluation
bashpython evaluation/evaluate.py --model Qwen/Qwen3-0.6B
3. Full Fine-Tuning
bashpython training/fft_train.py
4. QLoRA Fine-Tuning
bashpython training/qlora_train.py
5. Knowledge Distillation
bashpython training/kd_train.py
Note: Requires pre-generated teacher outputs in JSONL format. See kd_train.py for details.
6. DPO
bashpython training/dpo_train.py
Note: Requires pre-generated preference pairs from II-Medical-8B teacher.
7. Evaluate All Models
bashpython evaluation/evaluate.py --model [model_path]

Experiments Conducted
ExperimentDatasetEpochsKey ParametersFFT50K PQA-A3lr=2e-5, batch=4, BF16QLoRA50K PQA-A3lr=2e-4, rank=64, NF4KD6K PQA-A—KL divergence, from QLoRA checkpointDPOPreference pairs—beta=0.1, lr=5e-6, from QLoRA checkpoint

Interpreting Results

Accuracy is the primary metric, evaluated on 1,000 expert-labelled PQA-L examples
Macro F1 accounts for class imbalance (yes ~55%, no ~39%, maybe ~6%)
ROUGE-L measures explanation quality against gold-standard long answers
Format compliance tracks adherence to the required Answer/Explanation output structure
McNemar's test (α=0.05) was used to validate statistical significance of differences between models


Dataset
PubMedQA (Jin et al., 2019) — https://huggingface.co/datasets/qiaojin/PubMedQA

PQA-A (Artificial): Used for training and validation
PQA-L (Labelled): Used for final evaluation only


Model
Qwen3-0.6B — https://huggingface.co/Qwen/Qwen3-0.6B
Teacher Model: II-Medical-8B — https://huggingface.co/Intelligent-Internet/II-Medical-8B

AI Use Declaration
ToolHow usedGemini 3.1 ProFormatting for tables and report, basic code generationChatGPT 5.2Research for literature reviewKimi K2.5Version control libraries
