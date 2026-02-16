# plot_confusion_topk.py
# eval_compare_models.py で生成された predictions_final.csv を読み込み、真ラベルと予測ラベルの分布を確認するためのコード(topk版、k=5で使用)
# さらに、混同行列を描画して、どのクラスがどのクラスと混同されやすいかを視覚的に確認できるように

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 設定
# ==========================================
INPUT_RESULT = "emo8_0105_B"
INPUT_CSV = f"eval_results/{INPUT_RESULT}/predictions_final.csv"
OUT_DIR = f"eval_results/{INPUT_RESULT}"
FIG_PREFIX = "confmat_top5"

# 評価したいKの値（CSVの列がそれ以上持っている必要があります）
K = 5

# 表示するクラス数（Noneなら全クラス）
TOP_N_CLASSES = None

os.makedirs(OUT_DIR, exist_ok=True)

# ==========================================
# データの読み込みと前処理
# ==========================================
print(f"Loading {INPUT_CSV} ...")
df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)

# Testデータのみ抽出
df['Train/Dev/Test'] = df['Train/Dev/Test'].astype(str).str.strip()
df_eval = df[df['Train/Dev/Test'].str.lower() == 'test'].copy()
print(f"Test rows: {len(df_eval)}")

# 必要な列のクリーニング
df_eval['best_emoji'] = df_eval['best_emoji'].str.strip()
df_eval['pred_top_k_emojis'] = df_eval['pred_top_k_emojis'].str.strip()

# 空データを除外
df_eval = df_eval[
    (df_eval['best_emoji'] != '') & 
    (df_eval['pred_top_k_emojis'] != '')
].copy()
print(f"Valid rows: {len(df_eval)}")

# ==========================================
# ラベルの定義と行列の作成
# ==========================================

# 1. 全ての正解ラベルと、Top-Kに含まれる全ての予測ラベルを収集してユニオンをとる
#    (計算コスト削減のため、まずは正解ラベルの頻度上位に絞るか決定する)

# 正解ラベルの頻度計算
label_counts = df_eval['best_emoji'].value_counts()
all_true_labels = label_counts.index.tolist()

if TOP_N_CLASSES is not None:
    # 頻度上位N件をターゲットにする
    target_labels = all_true_labels[:TOP_N_CLASSES]
else:
    target_labels = all_true_labels

print(f"Target labels count: {len(target_labels)}")
print("Labels:", target_labels)

# ラベル → インデックスの辞書
label_to_idx = {label: i for i, label in enumerate(target_labels)}
n_labels = len(target_labels)

# 2. Top-K 混同行列の初期化 (ゼロ行列)
# 行: 正解ラベル (True)
# 列: 予測ラベル (Predicted in Top-K)
cm_accumulated = np.zeros((n_labels, n_labels), dtype=int)

# 3. 集計ループ
#    sklearnのconfusion_matrixは1対1専用なので、手動で集計します
count_included = 0 # データが含まれた数

for _, row in df_eval.iterrows():
    true_label = row['best_emoji']
    
    # ターゲット外の正解ラベルならスキップ
    if true_label not in label_to_idx:
        continue
    
    true_idx = label_to_idx[true_label]
    count_included += 1
    
    # 文字列 "😉,😨,😭..." をリストに分解
    # CSVのフォーマットによっては引用符などが残る場合があるので注意してsplit
    preds_str = row['pred_top_k_emojis']
    # カンマで分割し、空白除去。K個まで取得
    pred_list = [p.strip() for p in preds_str.split(',') if p.strip()][:K]
    
    # Top-K個の予測それぞれについてカウントアップ
    for p_label in pred_list:
        if p_label in label_to_idx:
            pred_idx = label_to_idx[p_label]
            cm_accumulated[true_idx, pred_idx] += 1

print(f"Aggregated {count_included} samples into Top-{K} Matrix.")

# ==========================================
# 正規化 (Normalization)
# ==========================================
# 行ごとの合計（その正解ラベルの出現回数）で割る
# 注意: Top-Kなので、行の合計値は「サンプル数 × K」になりません。
# 「サンプル数」で割ることで、確率は以下の意味になります。
# 対角成分 (i, i) => 正解ラベルiがTop-Kに含まれた確率 (Recall@K)
# 非対角成分(i, j) => 正解がiのとき、誤ってjがTop-Kに入ってきた確率

# 各正解ラベルの実際の出現回数（support）を計算
# df_evalの中で、target_labelsに含まれるものだけ再集計
support = np.zeros(n_labels)
for i, label in enumerate(target_labels):
    support[i] = len(df_eval[df_eval['best_emoji'] == label])

# ゼロ除算回避
support[support == 0] = 1 
# shapeを(N, 1)にしてブロードキャスト
cm_norm = cm_accumulated.astype('float') / support[:, None]

# ==========================================
# プロット
# ==========================================
plt.rcParams.update({'font.size': 10})
# クラス数に応じてサイズ自動調整
figsize = (max(10, n_labels * 0.8), max(8, n_labels * 0.6))

plt.figure(figsize=figsize)
sns.heatmap(
    cm_norm, 
    annot=True, 
    fmt='.2f', 
    xticklabels=target_labels, 
    yticklabels=target_labels, 
    cmap='Blues',
    vmin=0, vmax=1.0  # 確率は0~1の範囲
)
plt.xlabel(f'Predicted in Top-{K}')
plt.ylabel('True label')
plt.title(f'Top-{K} Accumulated Confusion Matrix (Normalized by Support)\nDiagonal represents Recall@{K}')
plt.tight_layout()

out_path = os.path.join(OUT_DIR, f"{FIG_PREFIX}_norm.png")
plt.savefig(out_path, dpi=200)
plt.close()

print(f"Saved figure to: {out_path}")

# ==========================================
# テキストレポート: クラスごとのRecall@K
# ==========================================
report_path = os.path.join(OUT_DIR, f"{FIG_PREFIX}_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"Class-wise Recall@{K} (Probability that True Label is in Top-{K})\n")
    f.write("-" * 50 + "\n")
    for i, label in enumerate(target_labels):
        recall_at_k = cm_norm[i, i]
        count = int(support[i])
        f.write(f"{label} : {recall_at_k:.4f} (n={count})\n")

print(f"Saved report to: {report_path}")