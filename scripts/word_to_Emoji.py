# map_wrime_to_emotag.py
# WRIMEの文章データとEmoTagの絵文字データをベクトル化して、コサイン類似度に基づいて最も近い絵文字をWRIMEの各文章に割り当てたデータセットを作成するコード

# pip install pandas numpy scikit-learn tqdm
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ---- settings ----
DATA_DIR = "data"
EMOTAG_CSV = os.path.join(DATA_DIR, "EmoTag49-scores.csv")
WRIME_TSV  = os.path.join(DATA_DIR, "wrime-ver2.tsv")
OUT_CSV    = os.path.join(DATA_DIR, "wrime_emotag_verR.csv")

# How many top emojis to keep
TOP_K = 5
# Minimum cosine similarity to accept a match; below this we set best_emoji to blank (optional)
MIN_SIMILARITY = 0.0  # set to e.g. 0.15 to require a minimum similarity

# ---- read files ----
df_emotag = pd.read_csv(EMOTAG_CSV)
# wrime-ver2.tsv: ensure it's read with correct separator and encoding
df_wrime = pd.read_csv(WRIME_TSV, sep="\t", encoding="utf-8", lineterminator="\n", dtype=str)

# ---- specify emotion order ----
# WRIME column names (as given by you)
#wrime_cols = ["Writer_Joy","Writer_Sadness","Writer_Anticipation","Writer_Surprise","Writer_Anger","Writer_Fear","Writer_Disgust","Writer_Trust"]
wrime_cols = ["Avg. Readers_Joy","Avg. Readers_Sadness","Avg. Readers_Anticipation","Avg. Readers_Surprise","Avg. Readers_Anger","Avg. Readers_Fear","Avg. Readers_Disgust","Avg. Readers_Trust"]
# canonical short names
emotion_order = ["joy","sadness","anticipation","surprise","anger","fear","disgust","trust"]

# Check columns exist in wrime
if not all(c in df_wrime.columns for c in wrime_cols):
    # Maybe writer columns are lowercase or slightly different; try case-insensitive match
    lowered = {c.lower():c for c in df_wrime.columns}
    rewired = []
    for c in wrime_cols:
        key = c.lower()
        if key in lowered:
            rewired.append(lowered[key])
        else:
            raise ValueError(f"Expected WRIME column '{c}' not found in wrime-ver2.tsv; actual columns: {df_wrime.columns.tolist()}")
    wrime_cols = rewired

# Convert writer columns to numeric (coerce errors to NaN)
for c in wrime_cols:
    df_wrime[c] = pd.to_numeric(df_wrime[c], errors="coerce")

# ---- prepare writer vectors (scale 0..3 -> 0..1) ----
writer_vectors = df_wrime[wrime_cols].to_numpy(dtype=float)
# If values are already in 0..1, you can detect and skip scaling:
#if np.nanmax(writer_vectors) > 1.01:
writer_vectors = writer_vectors / 3.0

# ---- prepare emoji vectors ----
# EmoTag columns in file are: anger, anticipation, disgust, fear, joy, sadness, surprise, trust
emotag_cols_actual = ["anger","anticipation","disgust","fear","joy","sadness","surprise","trust"]
if not all(c in df_emotag.columns for c in emotag_cols_actual):
    raise ValueError("EmoTag CSV missing expected emotion columns. Found: " + ", ".join(df_emotag.columns.tolist()))

# reorder emotag columns to match emotion_order
reorder_idx = [emotag_cols_actual.index(e) for e in emotion_order]
emoji_vectors = df_emotag[emotag_cols_actual].to_numpy(dtype=float)[:, reorder_idx]

# ---- similarity calculation in chunks if large ----
n_texts = writer_vectors.shape[0]
n_emojis = emoji_vectors.shape[0]

# Prepare output columns
df_wrime["best_emoji"] = pd.NA
df_wrime["best_emoji_name"] = pd.NA
df_wrime["best_score"] = pd.NA
df_wrime["top_k_emojis"] = pd.NA
df_wrime["top_k_scores"] = pd.NA
for emo in emotion_order:
    df_wrime[f"emoji_{emo}"] = pd.NA

# compute similarities in one go if memory allows
sims = cosine_similarity(writer_vectors, emoji_vectors)  # shape (n_texts, n_emojis)

# top-k
top_idx_all = np.argsort(sims, axis=1)[:, ::-1]  # descending
topk_idx = top_idx_all[:, :TOP_K]
topk_scores = np.take_along_axis(sims, topk_idx, axis=1)
top1_idx = topk_idx[:, 0]
top1_scores = topk_scores[:, 0]

# fill output
for i in range(n_texts):
    if np.isnan(top1_scores[i]) or top1_scores[i] < MIN_SIMILARITY:
        continue
    df_wrime.at[i, "best_emoji"] = df_emotag.loc[top1_idx[i], "emoji"]
    # optional: emoji name column if present
    if "name" in df_emotag.columns:
        df_wrime.at[i, "best_emoji_name"] = df_emotag.loc[top1_idx[i], "name"]
    df_wrime.at[i, "best_score"] = float(top1_scores[i])
    df_wrime.at[i, "top_k_emojis"] = ",".join(df_emotag.loc[topk_idx[i], "emoji"].astype(str).tolist())
    df_wrime.at[i, "top_k_scores"] = ",".join([f"{s:.4f}" for s in topk_scores[i]])
    # per-emotion scores of best emoji
    for j, emo in enumerate(emotion_order):
        df_wrime.at[i, f"emoji_{emo}"] = float(emoji_vectors[top1_idx[i], j])

# save
df_wrime.to_csv(OUT_CSV, index=False)
print("Saved:", OUT_CSV)
