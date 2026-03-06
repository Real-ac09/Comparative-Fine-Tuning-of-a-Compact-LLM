#!/usr/bin/env python
import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)

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


# ---------- Tokeniser with assistant-only labels ----------


@dataclass
class ChatTokeniserWithMask:
    tokenizer: AutoTokenizer
    max_length: int = 2048

    def __call__(self, example: Dict) -> Dict:
        # Full dialogue (system + user + assistant)
        full_text = self.tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        # Dialogue without assistant (system + user only)
        sys_user_text = self.tokenizer.apply_chat_template(
            example["messages"][:-1],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        # Pad to max_length so that input_ids and labels have uniform length
        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        input_ids = np.array(enc["input_ids"])
        labels = input_ids.copy()

        sys_user_enc = self.tokenizer(
            sys_user_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        boundary = len(sys_user_enc["input_ids"])

        # Mask system + user tokens
        labels[:boundary] = -100

        enc["labels"] = labels.tolist()
        return enc


# ---------- PubMedQA loaders (qiaojin/PubMedQA) ----------


def load_pqaa_splits(
    pqa_a_samples: int,
    val_size: int,
    seed: int,
) -> Tuple[Dict, Optional[Dict]]:
    """
    Load PQA-A (artificial) from qiaojin/PubMedQA and split into train / val.

    We first shuffle the full split, optionally sub-sample to `pqa_a_samples`,
    then carve out the first `val_size` as validation and the rest as train.
    """
    ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
    print(f"[PQA-A] Full train size: {len(ds)}")

    # Shuffle once for reproducibility
    ds = ds.shuffle(seed=seed)

    # Optional sub-sampling to a fixed subset (e.g. 55k total)
    if pqa_a_samples and pqa_a_samples < len(ds):
        ds = ds.select(range(pqa_a_samples))
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

    def _map(example: Dict) -> Dict:
        question = example["question"]
        # context is a dict with a 'contexts' list field
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

        return {"messages": messages}

    train_mapped = train_raw.map(
        _map,
        remove_columns=train_raw.column_names,
        desc="Mapping PQA-A train to messages",
    )

    if val_raw is not None:
        val_mapped = val_raw.map(
            _map,
            remove_columns=val_raw.column_names,
            desc="Mapping PQA-A val to messages",
        )
    else:
        val_mapped = None

    return train_mapped, val_mapped


def load_pqal_splits(val_frac: float, seed: int) -> Tuple[Dict, Dict]:
    """
    Load PQA-L (expert-labeled) from qiaojin/PubMedQA and split into train/val.
    """
    ds_full = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    print(f"[PQA-L] Full train size: {len(ds_full)}")
    ds_full = ds_full.shuffle(seed=seed)

    val_size = max(1, int(len(ds_full) * val_frac))
    val_raw = ds_full.select(range(val_size))
    train_raw = ds_full.select(range(val_size, len(ds_full)))
    print(f"[PQA-L] Train: {len(train_raw)}, Val: {len(val_raw)}")

    def _map(example: Dict) -> Dict:
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

        return {
            "messages": messages,
            "label": label,
            "reference_answer": long_answer,
        }

    train_mapped = train_raw.map(
        _map,
        desc="Mapping PQA-L train to messages",
    )

    val_mapped = val_raw.map(
        _map,
        desc="Mapping PQA-L val to messages",
    )

    return train_mapped, val_mapped


# ---------- Shared stage runner ----------


def run_stage(
    stage: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_ds_raw,
    val_ds_raw: Optional[Dict],
    args,
    output_dir: str,
    num_epochs: float,
    lr: float,
):
    print(f"=== Running stage {stage} ===")

    chat_tokeniser = ChatTokeniserWithMask(
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    if stage == "pqa_l":
        remove_cols = ["messages", "label", "reference_answer"]
    else:
        remove_cols = ["messages"]

    tokenised_train = train_ds_raw.map(
        chat_tokeniser,
        remove_columns=remove_cols,
        desc=f"Tokenising train ({stage})",
    )

    if val_ds_raw is not None:
        tokenised_val = val_ds_raw.map(
            chat_tokeniser,
            remove_columns=remove_cols,
            desc=f"Tokenising val ({stage})",
        )
    else:
        tokenised_val = None

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    total_batch_size = (
        args.per_device_train_batch_size * args.gradient_accumulation_steps
    )

    print(
        f"[{stage}] Effective batch size: {total_batch_size}, "
        f"epochs: {num_epochs}, lr: {lr}"
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,  # smaller eval batch to avoid OOM
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch" if tokenised_val is not None else "no",
        eval_accumulation_steps=4,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=True,
        report_to=["tensorboard"],
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised_train,
        eval_dataset=tokenised_val,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[{stage}] Finished. Saved to {output_dir}")


# ---------- Main ----------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["both", "pqa_a", "pqa_l"],
        default="pqa_a",  # default to PQA-A only
        help="Training mode: both stages, or individual stage.",
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Base model name or path.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="runs/qwen3-0.6b_fft_pqaa_only",
        help="Root path for output checkpoints.",
    )

    # 55k PQA-A subset: 50k train + 5k val
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

    parser.add_argument("--max_length", type=int, default=2048)
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
    parser.add_argument("--stage1_lr", type=float, default=2e-5)
    parser.add_argument("--stage2_lr", type=float, default=1e-5)

    args = parser.parse_args()
    set_seed(args.seed)

    print(f"Loading tokenizer+model from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    root = args.output_root.rstrip("/")
    stage1_dir = root + "_stage1_pqaa"
    stage2_dir = root + "_stage2_pqal"

    # Stage 1: PQA-A (artificial)
    if args.mode in ("both", "pqa_a"):
        pqaa_train, pqaa_val = load_pqaa_splits(
            pqa_a_samples=args.pqa_a_samples,
            val_size=args.pqa_a_val_size,
            seed=args.seed,
        )
        run_stage(
            stage="pqa_a",
            model=model,
            tokenizer=tokenizer,
            train_ds_raw=pqaa_train,
            val_ds_raw=pqaa_val,
            args=args,
            output_dir=stage1_dir,
            num_epochs=args.stage1_epochs,
            lr=args.stage1_lr,
        )

    # Stage 2: PQA-L (expert-labeled)
    if args.mode in ("both", "pqa_l"):
        if args.mode == "pqa_l":
            print(f"Reloading model from {stage1_dir} for Stage 2")
            model = AutoModelForCausalLM.from_pretrained(
                stage1_dir,
                torch_dtype=torch.bfloat16
                if torch.cuda.is_available()
                else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )

        pqal_train, pqal_val = load_pqal_splits(args.pqa_l_val_frac, args.seed)
        run_stage(
            stage="pqa_l",
            model=model,
            tokenizer=tokenizer,
            train_ds_raw=pqal_train,
            val_ds_raw=pqal_val,
            args=args,
            output_dir=stage2_dir,
            num_epochs=args.stage2_epochs,
            lr=args.stage2_lr,
        )

    print(f"Done. Final FFT model (Stage 2) at: {stage2_dir}")


if __name__ == "__main__":
    main()
