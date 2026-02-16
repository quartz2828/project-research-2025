# plot_confusion_top1.py
# テスト用、eval_compare_models.py で生成された predictions_final.csv を読み込み、真ラベルと予測ラベルの分布を確認するためのコード(top1版、pred_top1_emojiを使用)

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Headless 環境用
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# 設定（必要に応じて編集）
INPUT_RESULT = "emo8_0105_b"
INPUT_CSV = f"eval_results/{INPUT_RESULT}/predictions_final.csv"   # あなたのCSVパス
OUT_DIR = f"eval_results/{INPUT_RESULT}"
FIG_PREFIX = "confmat"
TOP_N_CLASSES = None   # None = 全クラス描画。多すぎる場合は整数で上位Nクラスに制限。
SAVE_PNG = True

os.makedirs(OUT_DIR, exist_ok=True)

# --- load csv ---
df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)
# 安全にstripしておく
df['Train/Dev/Test'] = df['Train/Dev/Test'].astype(str).str.strip()

# filter only test rows (or change as needed)
df_test = df[df['Train/Dev/Test'].str.lower().str.strip() == 'test'].copy()
print("test rows:", len(df_test))

# drop rows without ground truth or prediction
df_eval = df_test[(df_test['best_emoji'].astype(str).str.strip() != '') & (df_test['pred_top1_emoji'].astype(str).str.strip() != '')].copy()
print("evaluatable rows:", len(df_eval))

# --- label encoding: create consistent class list ---
# Use union of true and predicted so confusion matrix covers all seen labels
labels_union = sorted(list(set(df_eval['best_emoji'].unique()).union(set(df_eval['pred_top1_emoji'].unique()))))
print("classes count:", len(labels_union))

# optionally restrict to top N frequent true classes
if TOP_N_CLASSES is not None and TOP_N_CLASSES < len(labels_union):
    # choose TOP_N_CLASSES by support (frequency in true labels)
    top_labels_by_support = df_eval['best_emoji'].value_counts().index[:TOP_N_CLASSES].tolist()
    labels = [l for l in labels_union if l in top_labels_by_support]
    print("Plotting top labels:", labels)
else:
    labels = labels_union

# map to indices
le = LabelEncoder()
le.fit(labels)   # ensure mapping only for chosen labels

# filter rows to chosen labels only (if we limited)
df_eval = df_eval[df_eval['best_emoji'].isin(labels)].copy()

y_true = le.transform(df_eval['best_emoji'])
y_pred = le.transform(df_eval['pred_top1_emoji'])

# --- confusion matrix (counts) ---
cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))

# --- optional normalization (row-wise: recall per true class) ---
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
# handle division by zero (when a true class has zero support)
cm_norm = np.nan_to_num(cm_norm)

# --- plot settings ---
plt.rcParams.update({'font.size': 10})
figsize = (max(8, len(labels)*0.25*1.2), max(6, len(labels)*0.25))  # auto-scale

# counts heatmap
plt.figure(figsize=figsize)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix (counts)')
plt.tight_layout()
if SAVE_PNG:
    plt.savefig(os.path.join(OUT_DIR, f"{FIG_PREFIX}_counts.png"), dpi=200)
plt.close()

# normalized heatmap (row-wise)
plt.figure(figsize=figsize)
sns
