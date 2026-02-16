# eval_compare_models.py
# モデルを教師データを用いて評価するコード。生成したcsvをcheck_top1labels.pyやplot_confusion_top1.pyでテキストにしたり、ヒートマップにしたりできる

import os
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from typing import List, Tuple

# ---- settings ----
INPUT_CSV = "data/wrime_emotag8_b_WR.csv"
MODEL_DIRS = [
    "results_models/emoji8_0105_b_model_result/final",
]
OUT_NAME = "emo8_0105_b"
OUT_DIR = f"eval_results/{OUT_NAME}"
BATCH_SIZE = 32
TOP_K = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# ---- load csv and prepare label encoder ----
df_all = pd.read_csv(INPUT_CSV)
# 安全にstrip
df_all['Train/Dev/Test'] = df_all['Train/Dev/Test'].astype(str).str.strip()

# 1) test データだけ取り出す（欠損best_emojiは評価から外す）
df_test = df_all[df_all['Train/Dev/Test'].str.lower().str.strip() == 'test'].copy()
print(f"Found {len(df_test)} test rows in {INPUT_CSV}")

# If you want to evaluate even when best_emoji is missing, change below logic.
df_eval = df_test.dropna(subset=['best_emoji']).reset_index(drop=True)
print(f"Evaluating on {len(df_eval)} rows with non-null best_emoji")

# LabelEncoder must be fit the same way as training used it.
# We fit on ALL best_emoji values from the original csv to reproduce mapping.
le = LabelEncoder()
all_labels_for_fit = df_all['best_emoji'].fillna("##NA##").astype(str)
le.fit(all_labels_for_fit)
emoji_labels = le.classes_  # index -> emoji
num_labels = len(emoji_labels)
print("Number of label classes:", num_labels)

# ground-truth indices for eval rows
true_labels = le.transform(df_eval['best_emoji'].fillna("##NA##").astype(str))

# ---- helper: batch predict for a single model ----
def predict_with_model(model_dir: str,
                       texts: List[str],
                       top_k: int = 5,
                       batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    returns (top_indices (n, top_k), top_probs (n, top_k))
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(DEVICE)
    model.eval()

    n = len(texts)
    top_indices_all = []
    top_probs_all = []

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_texts = texts[i:i+batch_size]
            enc = tokenizer(batch_texts, truncation=True, padding=True, return_tensors="pt", max_length=128)
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            outputs = model(**enc)
            logits = outputs.logits  # (B, num_labels)
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
            # get top-k indices and probs
            top_idx = np.argsort(probs, axis=1)[:, ::-1][:, :top_k]  # (B, top_k)
            top_pr = np.take_along_axis(probs, top_idx, axis=1)      # (B, top_k)
            top_indices_all.append(top_idx)
            top_probs_all.append(top_pr)

    top_indices = np.vstack(top_indices_all)
    top_probs = np.vstack(top_probs_all)
    return top_indices, top_probs

# ---- run predictions for each model and evaluate ----
summary_rows = []

texts = df_eval['Sentence'].astype(str).tolist()

for model_dir in MODEL_DIRS:
    print("\n=== Evaluating model:", model_dir)
    if not os.path.isdir(model_dir):
        print("Model dir not found:", model_dir)
        continue

    top_idx, top_pr = predict_with_model(model_dir, texts, top_k=TOP_K, batch_size=BATCH_SIZE)

    # top-1 preds
    preds_top1_idx = top_idx[:, 0]
    preds_top1_labels = [emoji_labels[i] for i in preds_top1_idx]

    # compute metrics vs true_labels
    acc_top1 = accuracy_score(true_labels, preds_top1_idx)

    # top-K accuracy (true label in top-K)
    in_topk = [true_labels[i] in top_idx[i] for i in range(len(true_labels))]
    topk_acc = np.mean(in_topk)

    # classification report (top1)
    target_names = [str(c) for c in emoji_labels]
    # classification_report expects label ids present; produce mapping only for labels present in true set
    try:
        cls_report = classification_report(true_labels, preds_top1_idx, target_names=target_names, zero_division=0)
    except Exception:
        # fallback: don't pass target_names
        cls_report = classification_report(true_labels, preds_top1_idx, zero_division=0)

    # Compare with provided 'top_k_emojis' if column exists
    match_with_provided_topk = None
    if 'top_k_emojis' in df_eval.columns:
        # parse provided column (assumed comma-separated string)
        provided_lists = df_eval['top_k_emojis'].fillna("").astype(str).tolist()
        provided_parsed = [p.split(",") if p != "" else [] for p in provided_lists]
        match_with_provided_topk = np.mean([any(emoji_labels[idx] in provided_parsed[i] for idx in top_idx[i]) for i in range(len(top_idx))])
    # Save per-row result dataframe
    out_df = df_eval.copy()
    out_df['pred_top1_emoji'] = preds_top1_labels
    out_df['pred_top1_idx'] = preds_top1_idx
    out_df['pred_top1_score'] = [float(s) for s in top_pr[:, 0]]
    # top-k as strings
    out_df['pred_top_k_emojis'] = [",".join([emoji_labels[j] for j in row]) for row in top_idx]
    out_df['pred_top_k_scores'] = [",".join([f"{s:.4f}" for s in row]) for row in top_pr]
    out_df['match_top1'] = (preds_top1_idx == true_labels)
    out_df['match_topk_with_true'] = in_topk
    if match_with_provided_topk is not None:
        out_df['match_topk_with_provided'] = [int(any(emoji_labels[idx] in provided_parsed[i] for idx in top_idx[i])) for i in range(len(top_idx))]

    # save per-model predictions
    model_name_safe = os.path.basename(model_dir.rstrip("/"))
    per_model_file = os.path.join(OUT_DIR, f"predictions_{model_name_safe}.csv")
    out_df.to_csv(per_model_file, index=False, encoding="utf-8")
    print("Saved per-row predictions to:", per_model_file)

    # save summary
    summary = {
        "model": model_dir,
        "model_name": model_name_safe,
        "n_eval": len(df_eval),
        "top1_accuracy": float(acc_top1),
        f"top{TOP_K}_accuracy": float(topk_acc),
        "match_with_provided_topk_rate": float(match_with_provided_topk) if match_with_provided_topk is not None else None,
    }
    summary_rows.append(summary)

    # save detailed classification report
    report_file = os.path.join(OUT_DIR, f"classification_report_{model_name_safe}.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_dir}\n")
        f.write(f"Top-1 accuracy: {acc_top1:.4f}\n")
        f.write(f"Top-{TOP_K} accuracy: {topk_acc:.4f}\n\n")
        f.write("Classification report (top1 preds):\n")
        f.write(cls_report)
    print("Saved classification report to:", report_file)

# save summary CSV
summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(OUT_DIR, "models_summary.csv")
summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
print("\nSaved summary to", summary_csv)
print("Done.")
