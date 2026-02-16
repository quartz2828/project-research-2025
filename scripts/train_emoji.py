# train_emoji.py
# 作成した文章→絵文字のデータセットを用いて重み付けをせず文章から絵文字を予測するモデルの学習コード

import os
import pandas as pd
import numpy as np
import torch
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder

# 【重要】SSH(Headless)環境用の設定
import matplotlib
matplotlib.use('Agg') # 画面表示しないバックエンドを指定
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

# ---------------------------------------------------------
# setting
# ---------------------------------------------------------
INPUT_CSV = os.path.join('data', 'wrime_emotag8_WR_b.csv')
EXP_NAME = "emoji8_0105_b_model_result"

OUTPUT_DIR = f"results_models/{EXP_NAME}/checkpoints"
FINAL_DIR  = f"results_models/{EXP_NAME}/final"

# ---------------------------------------------------------
# 1. 設定
# ---------------------------------------------------------
# WandBのログイン要求を無効化
os.environ["WANDB_DISABLED"] = "true"

# GPUチェック
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Device: GPU (CUDA) - {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Device: CPU")

# ---------------------------------------------------------
# 2. データ読み込みと前処理
# ---------------------------------------------------------
file_path = INPUT_CSV

if not os.path.exists(file_path):
    raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")

print(f"Loading data from: {file_path}")
df = pd.read_csv(file_path)

# 必要な列を抽出
df = df[['Sentence', 'Train/Dev/Test', 'best_emoji']]

# 絵文字を数値IDに変換 (Label Encoding)
le = LabelEncoder()
df['label'] = le.fit_transform(df['best_emoji'])
emoji_labels = le.classes_ # IDから絵文字に戻すためのリスト
num_labels = len(emoji_labels)

print(f"ターゲット絵文字数: {num_labels}種類")
# print(emoji_labels) # 全種類の確認用

# データの分割
df_train = df[df['Train/Dev/Test'] == 'train']
df_test = df[df['Train/Dev/Test'].isin(['test', 'dev'])]

print(f"学習データ数: {len(df_train)}")
print(f"テストデータ数: {len(df_test)}")

# ---------------------------------------------------------
# 3. Dataset作成
# ---------------------------------------------------------
checkpoint = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(batch):
    return tokenizer(batch['Sentence'], truncation=True, padding='max_length', max_length=128)

# Datasetオブジェクトへ変換
train_dataset = Dataset.from_pandas(df_train[['Sentence', 'label']])
test_dataset = Dataset.from_pandas(df_test[['Sentence', 'label']])

# トークナイズ処理
print("トークナイズを実行中...")
train_tokenized = train_dataset.map(tokenize_function, batched=True)
test_tokenized = test_dataset.map(tokenize_function, batched=True)

# ---------------------------------------------------------
# 4. モデル学習設定
# ---------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
model.to(device)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 学習パラメータ
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    #パラメータは適宜変えて テスト時→8,8,1.0
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5.0, 

    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,

    save_total_limit=2,

    logging_steps=50,
    learning_rate=2e-5,
    load_best_model_at_end=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    compute_metrics=compute_metrics,

    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# ---------------------------------------------------------
# 5. 学習実行
# ---------------------------------------------------------
print(">>> 学習を開始します...")
trainer.train()
print(">>> 学習完了。モデルを保存します。")
trainer.save_model(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)

# ---------------------------------------------------------
# 6. 絵文字生成（推論）関数
# ---------------------------------------------------------
def predict_emoji(text, top_k=3, output_img_name=None):
    model.eval()
    
    # 入力テキストの変換
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 確率に変換 (Softmax)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
    
    # 確率が高い順にソートしてTop-Kを取得
    top_indices = probs.argsort()[-top_k:][::-1]
    
    print(f"\n入力: {text}")
    print("-" * 30)
    
    results = []
    for i in top_indices:
        emoji = emoji_labels[i]
        score = probs[i]
        print(f"{emoji}  : {score:.4f}")
        results.append((emoji, score))
    
    # グラフ保存 (49次元ではあまり意味ない。絵文字のフォントに未対応)
    if output_img_name:
        plt.figure(figsize=(8, 4))
        # ここでは確率バーのみ表示
        
        # データフレーム作成
        plot_df = pd.DataFrame(results, columns=['Emoji', 'Probability'])
        
        try:
            sns.barplot(x='Emoji', y='Probability', data=plot_df)
            plt.title(f'Prediction: {text[:15]}...')
            plt.ylim(0, 1)
            plt.savefig(output_img_name)
            print(f">>> グラフを保存しました: {output_img_name}")
        except Exception as e:
            print(f"グラフ保存中にエラーが発生しました: {e}")
        finally:
            plt.close()

# ---------------------------------------------------------
# 7. 指定された推論の実行
# ---------------------------------------------------------
#グラフの生成は不要なのでコメントアウト
if __name__ == "__main__":
    print("\n========== 推論テスト ==========")

    # 1つ目
    predict_emoji(
        "今日から長期休暇だぁーーー！！！", 
        #output_img_name="pred_1.png"
    )

    # 2つ目
    predict_emoji(
        "この書類にはコーヒーかかってなくて良かった…。不幸中の幸いだ。", 
        #output_img_name="pred_2.png"
    )

    # 3つ目
    predict_emoji(
        "なんで自分だけこんな目に遭うんだ……", 
        #output_img_name="pred_3.png"
    )