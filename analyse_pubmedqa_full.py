#!/usr/bin/env python3
"""
Clean analysis excluding old/broken KD model.
"""
import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from scipy.stats import binomtest
import warnings
warnings.filterwarnings("ignore")

# Clean model mapping (no old m3)
MODEL_NAMES = {
    "m0_base_qwen3-0.6b": "Base Qwen3-0.6B",
    "m1_fft_pqaa_only": "FFT PQA-A only", 
    "m2_qlora_pqaa_only": "QLoRA PQA-A only",
    "m3_kd_iimed8b_v2": "KD v2 (failed)",
    "m4_dpo_iimed8b": "DPO II-Med-8B",
}

EXCLUDE_MODELS = ["m3_kd_iimed8b"]  # Old broken KD

def mcnemar_test_numpy(correct1, correct2):
    n12 = np.sum(correct1 & ~correct2)
    n21 = np.sum(~correct1 & correct2)
    if n12 + n21 == 0:
        return 1.0, "identical"
    result = binomtest(n12, n=n12 + n21, p=0.5, alternative='two-sided')
    return result.pvalue, "sig" if result.pvalue < 0.05 else "ns"

def load_results_csvs(results_dir="eval_results_full_v2"):
    csv_pattern = os.path.join(results_dir, "pubmedqa_full_*.csv")
    csv_files = sorted(glob.glob(csv_pattern))
    
    all_dfs = {}
    for csv_file in csv_files:
        model_name = Path(csv_file).stem.replace("pubmedqa_full_", "")
        
        # Skip excluded models and summary
        if model_name in EXCLUDE_MODELS or model_name == "summary":
            continue
            
        try:
            df = pd.read_csv(csv_file)
            required_cols = ["gold_label", "pred_label", "correct"]
            if all(col in df.columns for col in required_cols):
                df["model_name"] = model_name
                all_dfs[model_name] = df
                print(f"✅ Loaded {len(df)} examples for {model_name}")
            else:
                print(f"⚠️  Skipping {model_name}: missing columns")
        except Exception as e:
            print(f"❌ Failed to load {csv_file}: {e}")
    
    return all_dfs

def compute_per_class_f1(df):
    y_true = df["gold_label"]
    y_pred = df["pred_label"]
    report = classification_report(y_true, y_pred, labels=["yes", "no", "maybe"], 
                                   output_dict=True, zero_division=0)
    return {
        "yes_f1": report["yes"]["f1-score"],
        "no_f1": report["no"]["f1-score"], 
        "maybe_f1": report["maybe"]["f1-score"],
        "macro_f1": report["macro avg"]["f1-score"],
    }

def analyze_all_results(results_dir="eval_results_full_v2"):
    print("🔍 Loading clean results (excluding old m3)...")
    all_dfs = load_results_csvs(results_dir)
    
    # Load summary and filter
    summary_path = os.path.join(results_dir, "pubmedqa_full_summary.csv")
    summary_df = pd.read_csv(summary_path)
    summary_df = summary_df[~summary_df["model_name"].isin(EXCLUDE_MODELS)]
    
    # Per-class F1
    per_class_f1 = []
    for model_name, df in all_dfs.items():
        f1_scores = compute_per_class_f1(df)
        f1_scores["model_name"] = model_name
        per_class_f1.append(f1_scores)
    per_class_df = pd.DataFrame(per_class_f1)
    
    # Safety metrics from summary
    safety_cols = ["correct_format_pct", "has_disclaimer_pct", "patient_direct_advice_pct"]
    safety_df = summary_df[["model_name"] + safety_cols].round(1)
    
    # McNemar vs QLoRA
    qlora_key = "m2_qlora_pqaa_only"
    if qlora_key in all_dfs:
        qlora_df = all_dfs[qlora_key]
        mcnemar_results = []
        for model_name, df in all_dfs.items():
            if model_name == qlora_key:
                continue
            pval, sig = mcnemar_test_numpy(qlora_df["correct"], df["correct"])
            mcnemar_results.append({"model_name": model_name, "vs_qlora_pval": pval, "vs_qlora_sig": sig})
        mcnemar_df = pd.DataFrame(mcnemar_results)
    else:
        mcnemar_df = pd.DataFrame()
    
    # Save clean results
    per_class_df.to_csv(os.path.join(results_dir, "pubmedqa_clean_per_class_f1.csv"), index=False)
    safety_df.to_csv(os.path.join(results_dir, "pubmedqa_clean_safety.csv"), index=False)
    summary_df.to_csv(os.path.join(results_dir, "pubmedqa_clean_summary.csv"), index=False)
    
    print("\n📈 Per-class F1 (clean):")
    print(per_class_df[["model_name", "yes_f1", "no_f1", "maybe_f1", "macro_f1"]].round(3))
    print("\n🛡️ Safety (1000 examples):")
    print(safety_df)
    
    if not mcnemar_df.empty:
        print("\n🔬 McNemar vs QLoRA:")
        print(mcnemar_df.round(4))
    
    return summary_df, per_class_df, safety_df, mcnemar_df, all_dfs

def plot_results(summary_df, per_class_df, safety_df, results_dir="eval_results_full_v2"):
    os.makedirs(os.path.join(results_dir, "plots_clean"), exist_ok=True)
    
    # Clean order: base → FFT → QLoRA → KD v2 → DPO
    order = ["m0_base_qwen3-0.6b", "m1_fft_pqaa_only", "m2_qlora_pqaa_only", 
             "m3_kd_iimed8b_v2", "m4_dpo_iimed8b"]
    
    # 1. Main accuracy + safety plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy
    acc_data = summary_df.set_index("model_name")["accuracy"].reindex(order)
    colors = ['gray', 'orange', 'green', 'red', 'blue']
    bars = axes[0,0].bar(range(len(acc_data)), acc_data.values, 
                        color=colors, alpha=0.8, edgecolor='black')
    axes[0,0].set_ylim(0, 0.75)
    axes[0,0].set_ylabel("Accuracy")
    axes[0,0].set_title("PubMedQA Accuracy")
    axes[0,0].set_xticks(range(len(acc_data)))
    axes[0,0].set_xticklabels([MODEL_NAMES.get(m, m) for m in order], rotation=45, ha='right')
    for i, bar in enumerate(bars):
        axes[0,0].text(bar.get_x()+0.5, bar.get_height()+0.01, f'{bar.get_height():.3f}', 
                      ha='center', va='bottom', fontweight='bold')
    
    # ROUGE-L
    rouge_data = summary_df.set_index("model_name")["rougeL_expl"].reindex(order)
    bars = axes[0,1].bar(range(len(rouge_data)), rouge_data.values, 
                        color=colors, alpha=0.8, edgecolor='black')
    axes[0,1].set_ylim(0.15, 0.30)
    axes[0,1].set_ylabel("ROUGE-L")
    axes[0,1].set_title("Explanation Quality")
    axes[0,1].set_xticks(range(len(rouge_data)))
    axes[0,1].set_xticklabels([MODEL_NAMES.get(m, m) for m in order], rotation=45, ha='right')
    for i, bar in enumerate(bars):
        axes[0,1].text(bar.get_x()+0.5, bar.get_height()+0.005, f'{bar.get_height():.3f}', 
                      ha='center', va='bottom')
    
    # Safety metrics (stacked)
    safety_pct = safety_df.set_index("model_name")[["correct_format_pct", "has_disclaimer_pct", "patient_direct_advice_pct"]].reindex(order) / 100
    safety_pct.plot(kind='bar', stacked=True, ax=axes[1,0], color=['green', 'orange', 'red'], alpha=0.8)
    axes[1,0].set_ylim(0, 1.1)
    axes[1,0].set_ylabel("Proportion")
    axes[1,0].set_title("Safety Metrics")
    axes[1,0].legend(["Correct Format", "Disclaimer", "Patient Advice"])
    axes[1,0].set_xticklabels([MODEL_NAMES.get(m, m) for m in order], rotation=45, ha='right')
    
    # Per-class F1 heatmap
    f1_metrics = ["yes_f1", "no_f1", "maybe_f1"]
    f1_subset = per_class_df.set_index("model_name")[f1_metrics].reindex(order)
    sns.heatmap(f1_subset.T, annot=True, cmap="YlOrRd", fmt='.3f', ax=axes[1,1],
                cbar_kws={'label': 'F1 Score'})
    axes[1,1].set_title("Per-Class F1")
    axes[1,1].set_xlabel("Model")
    axes[1,1].set_xticklabels([MODEL_NAMES.get(m, m) for m in order], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plots_clean", "pubmedqa_full_analysis.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(results_dir, "plots_clean", "pubmedqa_full_analysis.pdf"), bbox_inches='tight')
    plt.close()
    
    print("✅ Clean plots saved to eval_results_full_v2/plots_clean/")
    print("📊 Clean CSVs saved!")

if __name__ == "__main__":
    summary_df, per_class_df, safety_df, mcnemar_df, all_dfs = analyze_all_results()
    plot_results(summary_df, per_class_df, safety_df)
