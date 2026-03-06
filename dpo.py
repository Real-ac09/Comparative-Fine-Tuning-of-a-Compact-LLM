#!/usr/bin/env python
import argparse
import json
import os
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)
from peft import PeftModel

from trl import DPOTrainer, DPOConfig

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


def build_chat_prompt(tokenizer, question: str, abstract: str) -> str:
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
    return prompt


def generate_responses(
    tokenizer,
    model,
    prompts: List[str],
    max_seq_length: int,
    max_new_tokens: int,
    batch_size: int,
    desc: str,
) -> List[str]:
    device = next(model.parameters()).device
    model.eval()
    all_out: List[str] = []

    for i in tqdm(range(0, len(prompts), batch_size), desc=desc):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_out.extend(decoded)

    return all_out


def build_prefs(
    chosen_model_path: str,
    rejected_model_path: str,
    output_jsonl: str,
    max_examples: int,
    max_seq_length: int,
    max_new_tokens: int,
    batch_size: int,
    seed: int,
    prefs_source: str,
    cached_chosen_jsonl: Optional[str] = None,
):
    set_seed(seed)
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    print("[M4] Loading tokenizer from rejected model path")
    tokenizer = AutoTokenizer.from_pretrained(
        rejected_model_path,
        trust_remote_code=True,
        use_fast=True,
        fix_mistral_regex=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts: List[str] = []
    meta: List[Dict] = []
    chosen_texts: List[str] = []

    # 1. Try loading cached chosen answers first
    if cached_chosen_jsonl and os.path.exists(cached_chosen_jsonl):
        print(f"[M4] Found cached chosen answers at {cached_chosen_jsonl}")
        with open(cached_chosen_jsonl, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        # Only take up to max_examples
        lines = lines[:max_examples] if max_examples else lines
        print(f"[M4] Loading {len(lines)} examples from cache...")

        for line in tqdm(lines, desc="Parsing cache"):
            data = json.loads(line)
            # Recover prompt pieces from the messages array stored by kd.py
            # The structure is: System, User, Assistant
            user_text = data["messages"][1]["content"]
            
            prompt = tokenizer.apply_chat_template(
                data["messages"][:2], # system + user
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompts.append(prompt)
            
            meta.append({
                "question": "N/A (from cache)",  
                "abstract": "N/A (from cache)",
                "label": data["label"],
                "reference_answer": data["reference_answer"],
                "user_prompt": user_text, 
            })
            
            # Use raw output if available, else assistant text
            chosen_texts.append(data.get("teacher_raw", data["messages"][-1]["content"]))

    else:
        # Fallback to standard PQA dataset loading and generation
        if prefs_source == "pqa_a":
            print("[M4] Loading PQA-A (pqa_artificial)...")
            ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
        elif prefs_source == "pqa_l":
            print("[M4] Loading PQA-L (pqa_labeled)...")
            ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        else:
            raise ValueError(f"Unknown prefs_source: {prefs_source}")

        print(f"[M4] Source dataset size ({prefs_source}): {len(ds)}")

        if max_examples is not None and max_examples > 0 and max_examples < len(ds):
            ds = ds.shuffle(seed=seed).select(range(max_examples))
            print(f"[M4] Using subset of {prefs_source}: {len(ds)} examples")

        print(f"[M4] Building prompts from {prefs_source}...")
        for ex in tqdm(ds, desc="Building prompts"):
            question = ex["question"]
            contexts = ex["context"]["contexts"]
            abstract = " ".join(contexts)
            label = ex.get("final_decision", "maybe")
            long_answer = ex.get("long_answer", "")

            user_text = build_user_prompt(question, abstract)
            prompt = build_chat_prompt(tokenizer, question, abstract)
            
            prompts.append(prompt)
            meta.append(
                {
                    "question": question,
                    "abstract": abstract,
                    "label": label,
                    "reference_answer": long_answer,
                    "user_prompt": user_text,
                }
            )

        print("[M4] Loading chosen model:", chosen_model_path)
        chosen_model = AutoModelForCausalLM.from_pretrained(
            chosen_model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

        print("[M4] Generating chosen (preferred) answers...")
        chosen_texts = generate_responses(
            tokenizer=tokenizer,
            model=chosen_model,
            prompts=prompts,
            max_seq_length=max_seq_length,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            desc="Chosen model generation",
        )

        del chosen_model
        torch.cuda.empty_cache()

    # 3. Generate rejected answers (always run)
    print("[M4] Loading rejected model:", rejected_model_path)
    rejected_model = AutoModelForCausalLM.from_pretrained(
        rejected_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    print("[M4] Generating rejected answers...")
    rejected_texts = generate_responses(
        tokenizer=tokenizer,
        model=rejected_model,
        prompts=prompts,
        max_seq_length=max_seq_length,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        desc="Rejected model generation",
    )

    del rejected_model
    torch.cuda.empty_cache()

    # 4. Save to JSONL
    def postprocess_answer(full_text: str) -> str:
        idx = full_text.lower().rfind("answer:")
        if idx != -1:
            return full_text[idx:].strip()
        return full_text.strip()

    print("[M4] Writing preferences to JSONL:", output_jsonl)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for m, ch, rj in tqdm(
            list(zip(meta, chosen_texts, rejected_texts)),
            desc="Writing JSONL",
        ):
            record = {
                "prompt": m["user_prompt"],
                "chosen": postprocess_answer(ch),
                "rejected": postprocess_answer(rj),
                "label": m["label"],
                "reference_answer": m["reference_answer"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("[M4] Preference data written to", output_jsonl)


def build_chat_prompt_for_dpo(tokenizer, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return text


def train_dpo(
    policy_model_path: str,
    output_dir: str,
    dpo_train_jsonl: str,
    dpo_val_jsonl: str,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: float,
    learning_rate: float,
    max_prompt_length: int,
    max_length: int,
    beta: float,
    seed: int,
):
    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed)

    print("[M4] Loading tokenizer from policy model")
    tokenizer = AutoTokenizer.from_pretrained(
        policy_model_path,
        trust_remote_code=True,
        use_fast=True,
        fix_mistral_regex=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[M4] Loading base model architecture: Qwen/Qwen3-0.6B")
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"[M4] Merging QLoRA adapters from {policy_model_path} into base model...")
    policy_model = PeftModel.from_pretrained(base_model, policy_model_path)
    policy_model = policy_model.merge_and_unload()

    print("[M4] Unfreezing all parameters for DPO full fine-tuning...")
    for param in policy_model.parameters():
        param.requires_grad = True

    # Ensure gradients flow for gradient checkpointing
    policy_model.enable_input_require_grads()
    
    trainable_params = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
    print(f"[M4] Trainable parameters for DPO: {trainable_params:,}")

    print("[M4] Loading preference data from JSONL")
    data_files = {"train": dpo_train_jsonl}
    if dpo_val_jsonl is not None:
        data_files["validation"] = dpo_val_jsonl

    ds = load_dataset("json", data_files=data_files)

    if "validation" not in ds:
        print("[M4] No explicit validation set; splitting 90/10 from train")
        split = ds["train"].train_test_split(test_size=0.1, seed=seed)
        train_ds = split["train"]
        val_ds = split["test"]
    else:
        train_ds = ds["train"]
        val_ds = ds["validation"]

    print(f"[M4] Train examples: {len(train_ds)}, Val examples: {len(val_ds)}")

    def format_for_dpo(batch):
        prompts = []
        for p in batch["prompt"]:
            prompts.append(build_chat_prompt_for_dpo(tokenizer, p))
        return {"prompt": prompts}

    print("[M4] Applying chat template to train prompts...")
    train_ds = train_ds.map(
        format_for_dpo,
        batched=True,
        desc="Formatting train prompts",
    )

    print("[M4] Applying chat template to val prompts...")
    val_ds = val_ds.map(
        format_for_dpo,
        batched=True,
        desc="Formatting val prompts",
    )

    dpo_config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=True,
        report_to=["tensorboard"],
        seed=seed,
        max_prompt_length=max_prompt_length,
        max_length=max_length,
        beta=beta,
    )

    dpo_trainer = DPOTrainer(
        model=policy_model,
        ref_model=None, # TRL will automatically handle the reference model copy
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    dpo_trainer.tokenizer = tokenizer

    print("[M4] Starting DPO training...")
    dpo_trainer.train()
    dpo_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[M4] Finished. DPO-aligned model saved to", output_dir)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        choices=["build_prefs", "train_dpo", "all"],
        default="build_prefs",
        help="Run only preference building, only DPO training, or both.",
    )

    parser.add_argument(
        "--prefs_source",
        type=str,
        choices=["pqa_a", "pqa_l"],
        default="pqa_a",
        help="Source dataset for preferences: 'pqa_a' (artificial) or 'pqa_l' (labeled).",
    )

    parser.add_argument(
        "--chosen_model_path",
        type=str,
        default="Intelligent-Internet/II-Medical-8B",
        help="Preferred model (default: II-Medical-8B teacher).",
    )

    parser.add_argument(
        "--cached_chosen_jsonl",
        type=str,
        default="runs/qwen3-0.6b_kd_iimed8b_qlora/kd_stage1_pqaa/pqaa_kd.jsonl",
        help="Path to pre-generated chosen answers to skip teacher loading.",
    )

    parser.add_argument(
        "--rejected_model_path",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Rejected model (default: base Qwen3-0.6B).",
    )

    parser.add_argument(
        "--prefs_output_jsonl",
        type=str,
        default="dpo_data/pqaa_iimed8b_vs_base.jsonl",
        help="Where to save preference JSONL.",
    )

    parser.add_argument("--prefs_max_examples", type=int, default=6000)
    parser.add_argument("--prefs_max_seq_length", type=int, default=2048)
    parser.add_argument("--prefs_max_new_tokens", type=int, default=128)
    parser.add_argument("--prefs_batch_size", type=int, default=2)

    parser.add_argument(
        "--policy_model_path",
        type=str,
        default="runs/qwen3-0.6b_qlora_pqaa_only_qlora_stage1_pqaa",
        help="Initial policy (e.g. QLoRA base model).",
    )

    parser.add_argument(
        "--dpo_output_dir",
        type=str,
        default="runs/qwen3-0.6b_dpo_iimed8b_pqaa",
        help="Where to save the DPO-aligned model.",
    )

    parser.add_argument(
        "--dpo_train_jsonl",
        type=str,
        default=None,
        help="Preference JSONL; if None, use prefs_output_jsonl.",
    )

    parser.add_argument(
        "--dpo_val_jsonl",
        type=str,
        default=None,
        help="Optional validation JSONL.",
    )

    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.mode in ("build_prefs", "all"):
        build_prefs(
            chosen_model_path=args.chosen_model_path,
            rejected_model_path=args.rejected_model_path,
            output_jsonl=args.prefs_output_jsonl,
            max_examples=args.prefs_max_examples,
            max_seq_length=args.prefs_max_seq_length,
            max_new_tokens=args.prefs_max_new_tokens,
            batch_size=args.prefs_batch_size,
            seed=args.seed,
            prefs_source=args.prefs_source,
            cached_chosen_jsonl=args.cached_chosen_jsonl,
        )

    if args.mode in ("train_dpo", "all"):
        dpo_train_jsonl = (
            args.dpo_train_jsonl if args.dpo_train_jsonl else args.prefs_output_jsonl
        )

        train_dpo(
            policy_model_path=args.policy_model_path,
            output_dir=args.dpo_output_dir,
            dpo_train_jsonl=dpo_train_jsonl,
            dpo_val_jsonl=args.dpo_val_jsonl,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            max_prompt_length=args.max_prompt_length,
            max_length=args.max_length,
            beta=args.beta,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
