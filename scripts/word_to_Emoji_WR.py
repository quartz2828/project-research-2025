# map_wrime_to_emotag.py
# WRIMEの文章データとEmoTagの絵文字データをベクトル化して、コサイン類似度に基づいて最も近い絵文字をWRIMEの各文章に割り当てたデータセットを作成するコード
# writerとreaderのaverage対応版

# pip install pandas numpy scikit-learn tqdm
import csv
import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ---- settings ----
DATA_DIR = "data"
EMOTAG_CSV = os.path.join(DATA_DIR, "EmoTag8-scores.csv") # ファイル名は適宜変更
WRIME_TSV  = os.path.join(DATA_DIR, "wrime-ver2.tsv")
OUT_CSV    = os.path.join(DATA_DIR, "wrime_emotag8_WR.csv")

# 上位何件の絵文字を残すか
TOP_K = 5
# マッチングを採用する最小の類似度（これ以下は空欄にする）
MIN_SIMILARITY = 0.0

# ---- read files ----
print("Loading data...")
df_emotag = pd.read_csv(EMOTAG_CSV, comment="#")

# wrime-ver2.tsv: 読み込み設定
df_wrime = pd.read_csv(WRIME_TSV, sep="\t", encoding="utf-8", dtype=str)

# ---- specify emotion order ----
# 感情の並び順（基準）
emotion_order = ["joy","sadness","anticipation","surprise","anger","fear","disgust","trust"]

# WRIMEのカラム名定義（WriterとReader）
cols_writer = ["Writer_Joy","Writer_Sadness","Writer_Anticipation","Writer_Surprise","Writer_Anger","Writer_Fear","Writer_Disgust","Writer_Trust"]
cols_reader = ["Avg. Readers_Joy","Avg. Readers_Sadness","Avg. Readers_Anticipation","Avg. Readers_Surprise","Avg. Readers_Anger","Avg. Readers_Fear","Avg. Readers_Disgust","Avg. Readers_Trust"]

# 数値型に変換 (エラーはNaNに)
print("Processing emotion columns...")
for c in cols_writer + cols_reader:
    if c in df_wrime.columns:
        df_wrime[c] = pd.to_numeric(df_wrime[c], errors="coerce")
    else:
        print(f"Warning: Column {c} not found in input CSV.")

# ---- prepare writer vectors (scale 0..3 -> 0..1) ----
# WriterとReaderの値をそれぞれ取得
vals_writer = df_wrime[cols_writer].to_numpy(dtype=float)
vals_reader = df_wrime[cols_reader].to_numpy(dtype=float)

# ★ここが変更点: 2つの平均をとる (Writer + Reader) / 2
# Writerだけ、あるいはReaderだけの行がある場合、NaNになる可能性があります。
# 欠損を0埋めしたい場合は .fillna(0) を to_numpy() の前に入れてください。
target_vectors = (vals_writer + vals_reader) / 2.0

# 0..3 のスケールを 0..1 に正規化
target_vectors = target_vectors / 3.0

# ---- prepare emoji vectors ----
# EmoTagのカラム定義
emotag_cols_actual = ["anger","anticipation","disgust","fear","joy","sadness","surprise","trust"]
if not all(c in df_emotag.columns for c in emotag_cols_actual):
    raise ValueError("EmoTag CSV missing expected emotion columns.")

# emotion_orderの順序に合わせてEmoTagのカラムを並べ替え
reorder_idx = [emotag_cols_actual.index(e) for e in emotion_order]
emoji_vectors = df_emotag[emotag_cols_actual].to_numpy(dtype=float)[:, reorder_idx]

# ---- similarity calculation ----
n_texts = target_vectors.shape[0]
n_emojis = emoji_vectors.shape[0]

# 出力用カラムの準備
df_wrime["best_emoji"] = pd.NA
df_wrime["best_emoji_name"] = pd.NA
df_wrime["best_score"] = pd.NA
df_wrime["top_k_emojis"] = pd.NA
df_wrime["top_k_scores"] = pd.NA

# ベクトルに欠損(NaN)が含まれている行は計算できないため除外するためのマスク
# (WriterまたはReaderのどちらかが欠損していると平均もNaNになるため)
valid_mask = ~np.isnan(target_vectors).any(axis=1)

print(f"Computing cosine similarity for {np.sum(valid_mask)} valid rows...")

# 有効な行のみ計算
if np.sum(valid_mask) > 0:
    # 一括計算
    valid_vectors = target_vectors[valid_mask]
    sims = cosine_similarity(valid_vectors, emoji_vectors)  # shape (n_valid, n_emojis)

    # 上位K件のインデックスを取得
    top_idx_all = np.argsort(sims, axis=1)[:, ::-1]
    topk_idx = top_idx_all[:, :TOP_K]
    topk_scores = np.take_along_axis(sims, topk_idx, axis=1)
    
    # 元のDataFrameのインデックスに対応させるためのマッピング
    valid_indices = np.where(valid_mask)[0]

    # 結果をDataFrameに格納
    print("Mapping results to dataframe...")
    for i, original_idx in enumerate(tqdm(valid_indices)):
        top1_sc = topk_scores[i, 0]
        
        # スコアが閾値未満ならスキップ
        if top1_sc < MIN_SIMILARITY:
            continue
            
        best_idx = topk_idx[i, 0]
        
        df_wrime.at[original_idx, "best_emoji"] = df_emotag.loc[best_idx, "emoji"]
        if "name" in df_emotag.columns:
            df_wrime.at[original_idx, "best_emoji_name"] = df_emotag.loc[best_idx, "name"]
        
        df_wrime.at[original_idx, "best_score"] = float(top1_sc)
        
        # Top-K情報を文字列で保存
        k_emojis = df_emotag.loc[topk_idx[i], "emoji"].astype(str).tolist()
        k_scores = [f"{s:.4f}" for s in topk_scores[i]]
        
        df_wrime.at[original_idx, "top_k_emojis"] = ",".join(k_emojis)
        df_wrime.at[original_idx, "top_k_scores"] = ",".join(k_scores)

# save
print(f"Saving to {OUT_CSV}...")
df_wrime.to_csv(OUT_CSV, index=False, quoting=csv.QUOTE_ALL)
print("Done.")