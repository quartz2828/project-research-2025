# inference.py
# テスト用、学習したモデルを使って文章から絵文字を予測する

import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------
# 1. 設定と準備
# ---------------------------------------------------------
# 学習済みモデルのパス
#MODEL_PATH = "final_emoji_model" # train_emoji.pyで保存したフォルダ名
MODEL_PATH = os.path.join('results_models', 'emoji8_0105_w_model_result', 'final')

DATA_PATH = os.path.join('data', 'wrime_emotag8_WR.csv')

# GPU設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------------------
# 2. ラベル情報の復元
# ---------------------------------------------------------
# モデルは「数字(0, 1, 2...)」で予測するので、それを「絵文字」に戻すための辞書が必要です。
# 学習時と同じ手順でラベルリストを作成します。
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"データファイルが見つかりません: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
le = LabelEncoder()
le.fit(df['best_emoji'])
emoji_labels = le.classes_

print("ラベル情報をロードしました。")

# ---------------------------------------------------------
# 3. モデルとトークナイザの読み込み
# ---------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"学習済みモデルが見つかりません: {MODEL_PATH}\n先に train_emoji.py を実行してください。")

print("モデルを読み込んでいます...")
# 保存されたフォルダから読み込む
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()
print("準備完了！")

# ---------------------------------------------------------
# 4. 予測関数
# ---------------------------------------------------------
def generate_emoji(text, top_k=8):
    # トークナイズ
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 推論
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 確率計算
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
    
    # 上位K個を取得
    top_indices = probs.argsort()[-top_k:][::-1]
    
    # 結果表示
    print(f"\n📝 入力: {text}")
    print("---------------------------------")
    
    results = []
    for i in top_indices:
        emoji = emoji_labels[i]
        score = probs[i]
        print(f"{emoji}  (確率: {score:.1%})")
        results.append(emoji)
        
    return results[0] # 最も確率が高い絵文字を返す

# ---------------------------------------------------------
# 5. インタラクティブ実行ループ
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*40)
    print("  ✨ 絵文字生成AI (終了するには 'q' を入力) ✨")
    print("="*40 + "\n")

    while True:
        try:
            # ユーザーからの入力を受け付け
            input_text = input("文章を入力してください: ")
            
            if input_text.lower() in ['q', 'exit', 'quit']:
                print("終了します。")
                break
            
            if not input_text.strip():
                continue

            # 生成実行
            generate_emoji(input_text)
            
        except KeyboardInterrupt:
            print("\n終了します。")
            break
        except Exception as e:
            print(f"エラーが発生しました: {e}")