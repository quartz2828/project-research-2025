# labels.py
# 学習時に使用したラベルエンコーダーを保存するコード。
# predict_emoji.pyで使う上で、モデルにラベルの情報を入れていなかったので、これを実行し、predict_emoji_manual.py で同じラベルエンコーダーを読み込むことができる。

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# 学習時と同じCSVを読み込む
INPUT_CSV = os.path.join('data', 'wrime_emotag8_WR.csv')
df = pd.read_csv(INPUT_CSV)

# 学習時と全く同じ手順でLabelEncoderを作る
le = LabelEncoder()
le.fit(df['best_emoji'])

# Streamlitが探しに行っているパスに保存する
SAVE_DIR = "results_models/emoji8_model_result/final"
os.makedirs(SAVE_DIR, exist_ok=True)
joblib.dump(le, os.path.join(SAVE_DIR, "label_encoder.pkl"))
