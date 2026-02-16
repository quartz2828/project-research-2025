# check_top1labels.py
# eval_compare_models.py で生成された predictions_final.csv を読み込み、真ラベルと予測ラベルの分布を確認するためのコード

import pandas as pd
import numpy as np
from collections import Counter

df = pd.read_csv("eval_results/emo48_0105_W/predictions_final.csv", dtype=str).fillna("")

# 基本分布
true = df['best_emoji'].tolist()
pred = df['pred_top1_emoji'].tolist()

print("総評価行数:", len(true))
print("真ラベルのユニーク数:", len(set(true)))
print("予測ラベルのユニーク数:", len(set(pred)))

# support（真ラベル分布）
support = Counter(true)
print("Top 20 true supports:")
print(support.most_common(20))

# pred counts（モデルが何回各クラスを予測したか）
pred_counts = Counter(pred)
print("Top 20 predicted counts:")
print(pred_counts.most_common(20))

# モデルが一度も予測していない真ラベル
never_predicted = [cls for cls in set(true) if pred_counts.get(cls,0) == 0]
print("モデルが一度も予測しなかった真ラベル数:", len(never_predicted))
print("例:", never_predicted[:30])

# TP/FP/FN 簡易集計
from collections import defaultdict
tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
for t,p in zip(true,pred):
    if t == p:
        tp[t] += 1
    else:
        fp[p] += 1
        fn[t] += 1

zero_recall = [c for c in set(true) if tp.get(c,0) == 0]
zero_precision = [c for c in set(pred) if tp.get(c,0) == 0 and pred_counts.get(c,0)>0]

print("recall==0 のクラス数:", len(zero_recall))
print("precision==0 のクラス数 (predictedあるがTP0):", len(zero_precision))
