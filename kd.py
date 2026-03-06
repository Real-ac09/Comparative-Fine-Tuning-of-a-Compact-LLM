#!/usr/bin/env python
import argparse
import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import PeftModel

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

        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        input_ids = np.array(enc["input_ids"])
        labels = input_ids.copy()

        # Get true boundary length without padding so we only mask system+user tokens
        sys_user_ids = self.tokenizer(
            sys_user_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )["input_ids"]
        boundary = len(sys_user_ids)
        labels[:boundary] = -100  # ignore system+user tokens

        enc["labels"] = labels.tolist()
        return enc


# ---------- Dataset loading ----------


def load_pqaa_for_kd(
    samples: Optional[int],
    seed: int,
):
    """
    Load PQA-A artificial subset from qiaojin/PubMedQA (config 'pqa_artificial').
    Optionally subsample to `samples` examples.
    """
    ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
    print(f"[KD] PQA-A full train size: {len(ds)}")

    if samples is not None and samples > 0 and samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(samples))
        print(f"[KD] PQA-A subset for KD: {len(ds)}")
    else:
        print("[KD] Using full PQA-A train split for KD.")
    return ds


def load_pqal_train_val(val_frac: float, seed: int):
    """
    Load PQA-L (expert-labeled) from qiaojin/PubMedQA (config 'pqa_labeled')
    and split into train/val.
    """
    ds_full = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    print(f"[KD] PQA-L full size: {len(ds_full)}")
    ds_full = ds_full.shuffle(seed=seed)

    val_size = max(1, int(len(ds_full) * val_frac))
    val_raw = ds_full.select(range(val_size))
    train_raw = ds_full.select(range(val_size, len(ds_full)))
    print(f"[KD] PQA-L train: {len(train_raw)}, val: {len(val_raw)}")
    return train_raw, val_raw


# ---------- Teacher generation (sequence-level KD) ----------


def generate_teacher_answers(
    ds,
    tokenizer,
    teacher_model,
    max_seq_length: int,
    max_new_tokens: int,
    batch_size: int,
    jsonl_path: Optional[str] = None,
) -> List[Dict]:
    """
    For each example, use the teacher to generate an answer.
    Return a list of dicts with `messages` including teacher's answer.
    Optionally stream each example to a JSONL file at `jsonl_path`.
    """
    teacher_model.eval()
    device = next(teacher_model.parameters()).device

    all_examples: List[Dict] = []
    n = len(ds)

    # Open JSONL file if requested
    f = None
    if jsonl_path is not None:
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
        f = open(jsonl_path, "w", encoding="utf-8")

    for i in tqdm(
        range(0, n, batch_size),
        desc="[KD] Teacher generation",
        total=(n + batch_size - 1) // batch_size,
    ):
        batch_ds = ds.select(range(i, min(i + batch_size, n)))
        prompts: List[str] = []
        meta: List[Dict] = []

        for ex in batch_ds:
            question = ex["question"]
            contexts = ex["context"]["contexts"]
            abstract = " ".join(contexts)
            label = ex.get("final_decision", "maybe")
            long_answer = ex.get("long_answer", "")

            user_text = build_user_prompt(question, abstract)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
            ]

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            prompts.append(prompt)
            meta.append(
                {
                    "question": question,
                    "abstract": abstract,
                    "label": label,
                    "reference_answer": long_answer,
                }
            )

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        ).to(device)

        with torch.no_grad():
            outputs = teacher_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for meta_ex, full_text in zip(meta, decoded):
            pred_text = full_text
            idx = full_text.lower().rfind("answer:")
            if idx != -1:
                pred_text = full_text[idx:]
            assistant_text = pred_text.strip()

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_user_prompt(
                        meta_ex["question"], meta_ex["abstract"]
                    ),
                },
                {"role": "assistant", "content": assistant_text},
            ]

            example = {
                "messages": messages,
                "label": meta_ex["label"],
                "reference_answer": meta_ex["reference_answer"],
                "teacher_raw": full_text,
            }

            all_examples.append(example)

            if f is not None:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

    if f is not None:
        f.close()

    return all_examples


def make_trainer(
    student_model,
    tokenizer,
    train_ds: Dataset,
    val_ds: Optional[Dataset],
    learning_rate: float,
    num_epochs: float,
    output_dir: str,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    max_seq_length: int,
    seed: int,
):
    chat_tokeniser = ChatTokeniserWithMask(
        tokenizer=tokenizer,
        max_length=max_seq_length,
    )

    remove_cols = ["messages", "label", "reference_answer", "teacher_raw"]
    train_tok = train_ds.map(
        chat_tokeniser,
        remove_columns=[c for c in remove_cols if c in train_ds.column_names],
        desc="Tokenising KD train",
    )

    if val_ds is not None:
        val_tok = val_ds.map(
            chat_tokeniser,
            remove_columns=[c for c in remove_cols if c in val_ds.column_names],
            desc="Tokenising KD val",
        )
    else:
        val_tok = None

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    total_batch_size = per_device_train_batch_size * gradient_accumulation_steps
    print(
        f"[KD] Training KD student: {len(train_ds)} examples, "
        f"epochs={num_epochs}, lr={learning_rate}, eff batch={total_batch_size}"
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_strategy="epoch" if val_tok is not None else "no",
        eval_strategy="epoch" if val_tok is not None else "no",
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=True,
        report_to=["tensorboard"],
        seed=seed,
        data_seed=seed,
    )

    trainer = Trainer(
        model=student_model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
    )

    return trainer


# ---------- Main KD training ----------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["both", "pqaa", "pqal"],
        default="pqaa",
        help="KD mode: 'both' (PQA-A then PQA-L), or individual stage.",
    )

    parser.add_argument(
        "--teacher_model_path",
        type=str,
        default="Intelligent-Internet/II-Medical-8B",
        help="Path to teacher model (default: II-Medical-8B).",
    )

    parser.add_argument(
        "--student_init_model",
        type=str,
        default="runs/qwen3-0.6b_qlora_pqaa_only_qlora_stage1_pqaa",
        help="Base model to initialize the student (default: PQA-A-only QLoRA checkpoint).",
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default="runs/qwen3-0.6b_kd_iimed8b_qlora_v2", # Changed so it doesn't overwrite old broken run yet
        help="Root directory; stages will be saved under this.",
    )

    # Data settings
    parser.add_argument("--pqa_a_samples", type=int, default=6000)
    parser.add_argument("--pqa_l_val_frac", type=float, default=0.1)

    # Generation settings
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--teacher_batch_size", type=int, default=2)

    # Optimisation
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    parser.add_argument("--stage1_epochs", type=float, default=1.0)
    parser.add_argument("--stage2_epochs", type=float, default=4.0)
    parser.add_argument("--stage1_lr", type=float, default=2e-5)
    parser.add_argument("--stage2_lr", type=float, default=1e-5)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)
    root = args.output_root.rstrip("/")

    stage1_dir = os.path.join(root, "kd_stage1_pqaa")
    stage2_dir = os.path.join(root, "kd_stage2_pqal")
    os.makedirs(stage1_dir, exist_ok=True)
    os.makedirs(stage2_dir, exist_ok=True)

    print(f"[KD] Loading tokenizer from {args.student_init_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.student_init_model,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # We will load the teacher only if we actually need to generate KD data
    teacher_model = None

    print(f"[KD] Loading base model architecture: Qwen/Qwen3-0.6B")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"[KD] Merging QLoRA adapters from {args.student_init_model} into student model...")
    student_model = PeftModel.from_pretrained(base_model, args.student_init_model)
    student_model = student_model.merge_and_unload()

    print("[KD] Unfreezing all parameters for full fine-tuning KD...")
    for param in student_model.parameters():
        param.requires_grad = True
        
    student_model.enable_input_require_grads()
    trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    print(f"[KD] Trainable parameters for KD: {trainable_params:,}")

    # ----- Stage 1: PQA-A KD -----
    if args.mode in ("both", "pqaa"):
        print("[KD] ---- Stage 1: PQA-A KD ----")
        
        # Point this to the JSONL you already generated in the old run
        pqaa_jsonl = "runs/qwen3-0.6b_kd_iimed8b_qlora/kd_stage1_pqaa/pqaa_kd.jsonl"
        
        if os.path.exists(pqaa_jsonl):
            print(f"[KD] Found existing KD JSONL at {pqaa_jsonl}, loading it.")
            pqaa_train_ds = load_dataset(
                "json", data_files={"train": pqaa_jsonl}
            )["train"]
            print(f"[KD] Loaded {len(pqaa_train_ds)} KD examples from JSONL.")
        else:
            print(f"[KD] Loading teacher model from {args.teacher_model_path}")
            teacher_model = AutoModelForCausalLM.from_pretrained(
                args.teacher_model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
            
            pqaa_ds = load_pqaa_for_kd(args.pqa_a_samples, args.seed)
            print("[KD] Generating teacher answers for PQA-A...")
            pqaa_kd = generate_teacher_answers(
                ds=pqaa_ds,
                tokenizer=tokenizer,
                teacher_model=teacher_model,
                max_seq_length=args.max_seq_length,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.teacher_batch_size,
                jsonl_path=os.path.join(stage1_dir, "pqaa_kd.jsonl"),
            )
            print(
                f"[KD] Generated {len(pqaa_kd)} PQA-A KD examples. "
                f"JSONL written to {stage1_dir}/pqaa_kd.jsonl"
            )
            pqaa_train_ds = Dataset.from_list(pqaa_kd)

            # Free the 8B teacher before training to avoid OOM
            del teacher_model
            torch.cuda.empty_cache()

        trainer1 = make_trainer(
            student_model=student_model,
            tokenizer=tokenizer,
            train_ds=pqaa_train_ds,
            val_ds=None,
            learning_rate=args.stage1_lr,
            num_epochs=args.stage1_epochs,
            output_dir=stage1_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_seq_length=args.max_seq_length,
            seed=args.seed,
        )

        trainer1.train()
        trainer1.save_model(stage1_dir)
        tokenizer.save_pretrained(stage1_dir)
        print(f"[KD] Stage 1 (PQA-A KD) finished, saved to {stage1_dir}")

        # Continue training in-memory from Stage1 weights
        student_model = trainer1.model

    # ----- Stage 2: PQA-L KD -----
    if args.mode in ("both", "pqal"):
        print("[KD] ---- Stage 2: PQA-L KD ----")
        train_raw, val_raw = load_pqal_train_val(
            args.pqa_l_val_frac,
            args.seed,
        )

        if teacher_model is None:
            print(f"[KD] Loading teacher model from {args.teacher_model_path} for Stage 2...")
            teacher_model = AutoModelForCausalLM.from_pretrained(
                args.teacher_model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )

        print("[KD] Generating teacher answers for PQA-L train...")
        train_kd = generate_teacher_answers(
            ds=train_raw,
            tokenizer=tokenizer,
            teacher_model=teacher_model,
            max_seq_length=args.max_seq_length,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.teacher_batch_size,
            jsonl_path=None,
        )
        print(f"[KD] Generated {len(train_kd)} PQA-L train KD examples.")

        print("[KD] Generating teacher answers for PQA-L val...")
        val_kd = generate_teacher_answers(
            ds=val_raw,
            tokenizer=tokenizer,
            teacher_model=teacher_model,
            max_seq_length=args.max_seq_length,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.teacher_batch_size,
            jsonl_path=None,
        )
        print(f"[KD] Generated {len(val_kd)} PQA-L val KD examples.")

        del teacher_model
        torch.cuda.empty_cache()

        train_ds = Dataset.from_list(train_kd)
        val_ds = Dataset.from_list(val_kd)

        trainer2 = make_trainer(
            student_model=student_model,
            tokenizer=tokenizer,
            train_ds=train_ds,
            val_ds=val_ds,
            learning_rate=args.stage2_lr,
            num_epochs=args.stage2_epochs,
            output_dir=stage2_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_seq_length=args.max_seq_length,
            seed=args.seed,
        )

        trainer2.train()
        trainer2.save_model(stage2_dir)
        tokenizer.save_pretrained(stage2_dir)
        print(f"[KD] Stage 2 (PQA-L KD) finished, saved to {stage2_dir}")

    print("[KD] Done.")


if __name__ == "__main__":
    main()
