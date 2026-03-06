#!/usr/bin/env python
import argparse
from typing import Dict, Tuple, Optional

from datasets import load_dataset
import torch

from unsloth import FastLanguageModel  # Unsloth QLoRA API for Qwen3
from transformers import TrainingArguments
from trl import SFTTrainer  # supervised fine-tuning trainer

SYSTEM_PROMPT = (
    "You are a careful medical AI assistant. You read biomedical research questions "
    "and their PubMed abstracts, then provide concise, evidence-based answers. "
    "Base your conclusion only on the given abstract. Clearly state the answer first "
    "(yes/no/maybe), then briefly justify it using findings from the abstract. "
    "Do not give clinical advice to patients; your answers are for clinicians and researchers."
)


def build_user_prompt(question: str, abstract: str) -> str:
    return f"""You will be given a biomedical research question and the abstract of a related PubMed article.

Question:
{question}

Abstract:
{abstract}

Task:
1. Decide whether the abstract’s findings answer the question with "yes", "no", or "maybe".
2. Then provide a concise, evidence-based explanation using the abstract.

Format your answer as:
Answer:
Explanation: <2–5 sentence justification based only on the abstract>"""


# ---------- PubMedQA mapping (qiaojin/PubMedQA) ----------


def map_pqaa_to_text(example: Dict, tokenizer) -> Dict:
    # PQA-A artificial subset
    question = example["question"]
    contexts = example["context"]["contexts"]
    abstract = " ".join(contexts)

    label = example.get("final_decision", "maybe")
    long_answer = example.get("long_answer", "")

    user_text = build_user_prompt(question, abstract)
    assistant_text = f"Answer: {label}\nExplanation: {long_answer}".strip()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    return {"text": text}


def map_pqal_to_text(example: Dict, tokenizer) -> Dict:
    # PQA-L expert-labeled subset
    question = example["question"]
    contexts = example["context"]["contexts"]
    abstract = " ".join(contexts)

    label = example.get("final_decision", "maybe")
    long_answer = example.get("long_answer", "")

    user_text = build_user_prompt(question, abstract)
    assistant_text = f"Answer: {label}\nExplanation: {long_answer}".strip()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    return {
        "text": text,
        "label": label,
        "reference_answer": long_answer,
    }


def load_pqaa_text_splits(
    tokenizer,
    samples: int,
    val_size: int,
    seed: int,
) -> Tuple:
    """
    Load PQA-A (artificial) and return (train_text, val_text).

    We shuffle, optionally sub-sample to `samples` (e.g. 55k),
    then take the first `val_size` as validation and the rest as train.
    """
    ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
    print(f"[PQA-A] Full train size: {len(ds)}")

    # Shuffle once for reproducibility
    ds = ds.shuffle(seed=seed)

    # Optional sub-sampling to a fixed subset (e.g. 55k)
    if samples and samples < len(ds):
        ds = ds.select(range(samples))
        print(f"[PQA-A] Subset for Stage 1: {len(ds)}")

    # Fixed-size validation split
    val_size = min(val_size, len(ds))
    if val_size > 0:
        val_raw = ds.select(range(val_size))
        train_raw = ds.select(range(val_size, len(ds)))
        print(f"[PQA-A] Train: {len(train_raw)}, Val: {len(val_raw)}")
    else:
        train_raw = ds
        val_raw = None
        print(f"[PQA-A] Train: {len(train_raw)}, Val: 0")

    train_mapped = train_raw.map(
        lambda ex: map_pqaa_to_text(ex, tokenizer),
        remove_columns=train_raw.column_names,
        desc="Mapping PQA-A train to chat text",
    )

    if val_raw is not None:
        val_mapped = val_raw.map(
            lambda ex: map_pqaa_to_text(ex, tokenizer),
            remove_columns=val_raw.column_names,
            desc="Mapping PQA-A val to chat text",
        )
    else:
        val_mapped = None

    return train_mapped, val_mapped


def load_pqal_text(tokenizer, val_frac: float, seed: int) -> Tuple:
    ds_full = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    print(f"[PQA-L] Full train size: {len(ds_full)}")
    ds_full = ds_full.shuffle(seed=seed)

    val_size = max(1, int(len(ds_full) * val_frac))
    val_raw = ds_full.select(range(val_size))
    train_raw = ds_full.select(range(val_size, len(ds_full)))
    print(f"[PQA-L] Train: {len(train_raw)}, Val: {len(val_raw)}")

    train_mapped = train_raw.map(
        lambda ex: map_pqal_to_text(ex, tokenizer),
        desc="Mapping PQA-L train to chat text",
    )

    val_mapped = val_raw.map(
        lambda ex: map_pqal_to_text(ex, tokenizer),
        desc="Mapping PQA-L val to chat text",
    )

    return train_mapped, val_mapped


# ---------- Main QLoRA training ----------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["both", "pqa_a", "pqa_l"],
        default="pqa_a",  # default to clean PQA-A-only
        help="Training mode: both stages, or individual stage.",
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Base model name or path.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="runs/qwen3-0.6b_qlora_pqaa_only",
        help="Root path for output checkpoints.",
    )

    # 55k PQA-A subset: 50k train + 5k val (to mirror FFT)
    parser.add_argument(
        "--pqa_a_samples",
        type=int,
        default=55000,
        help="Number of PQA-A examples to sample after shuffling.",
    )
    parser.add_argument(
        "--pqa_a_val_size",
        type=int,
        default=5000,
        help="Validation size taken from the (shuffled) PQA-A subset.",
    )
    parser.add_argument(
        "--pqa_l_val_frac",
        type=float,
        default=0.1,
        help="Fraction of PQA-L used for validation in Stage 2 (if used).",
    )

    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage1_epochs", type=float, default=1.0)
    parser.add_argument("--stage2_epochs", type=float, default=4.0)
    parser.add_argument("--stage1_lr", type=float, default=2e-4)
    parser.add_argument("--stage2_lr", type=float, default=1e-4)

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    # ---- Load Qwen3-0.6B in 4-bit with Unsloth ----
    print(f"Loading base model with Unsloth from {args.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,          # let Unsloth choose (bf16 if available)
        load_in_4bit=True,   # 4x VRAM reduction
    )

    # Attach LoRA adapters to key projection layers
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
    )

    root = args.output_root.rstrip("/")
    stage1_dir = root + "_qlora_stage1_pqaa"
    stage2_dir = root + "_qlora_stage2_pqal"

    # Helper to build a trainer
    def make_trainer(train_ds, val_ds, lr, num_epochs, out_dir: str):
        total_batch = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps
        )

        print(
            f"[QLoRA] Training: {len(train_ds)} examples, "
            f"epochs={num_epochs}, lr={lr}, eff batch={total_batch}"
        )

        training_args = TrainingArguments(
            output_dir=out_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=lr,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=50,
            save_strategy="epoch",
            eval_strategy="epoch" if val_ds is not None else "no",
            save_total_limit=2,
            bf16=torch.cuda.is_available(),
            fp16=False,
            gradient_checkpointing=True,
            report_to=["tensorboard"],
            seed=args.seed,
            data_seed=args.seed,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            packing=False,
        )

        return trainer

    # ---- Stage 1: PQA-A (artificial) ----
    if args.mode in ("both", "pqa_a"):
        pqaa_train, pqaa_val = load_pqaa_text_splits(
            tokenizer=tokenizer,
            samples=args.pqa_a_samples,
            val_size=args.pqa_a_val_size,
            seed=args.seed,
        )
        trainer1 = make_trainer(
            train_ds=pqaa_train,
            val_ds=pqaa_val,
            lr=args.stage1_lr,
            num_epochs=args.stage1_epochs,
            out_dir=stage1_dir,
        )

        trainer1.train()
        trainer1.save_model(stage1_dir)
        tokenizer.save_pretrained(stage1_dir)
        print(f"[QLoRA] Stage 1 finished, saved to {stage1_dir}")

    # ---- Stage 2: PQA-L (expert-labeled) ----
    if args.mode in ("both", "pqa_l"):
        # Continue from in-memory model (already trained on Stage 1 if mode==both)
        pqal_train, pqal_val = load_pqal_text(
            tokenizer, args.pqa_l_val_frac, args.seed
        )

        trainer2 = make_trainer(
            train_ds=pqal_train,
            val_ds=pqal_val,
            lr=args.stage2_lr,
            num_epochs=args.stage2_epochs,
            out_dir=stage2_dir,
        )

        trainer2.train()
        trainer2.save_model(stage2_dir)
        tokenizer.save_pretrained(stage2_dir)
        print(f"[QLoRA] Stage 2 finished, saved to {stage2_dir}")

    print(f"Done. Final QLoRA model (Stage 2) at: {stage2_dir}")


if __name__ == "__main__":
    main()
