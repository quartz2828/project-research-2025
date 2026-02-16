# emotion_train.py
# テスト用、文章から感情値を出し、表示させるテスト
# 参考 https://qiita.com/izaki_shin/items/2b4573ee7fbea5ec8ed6

import os
import requests
import pandas as pd
import numpy as np
import torch
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# 【重要】SSH(Headless)環境用の設定
import matplotlib
matplotlib.use('Agg') # 画面表示しないバックエンドを指定
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

# ---------------------------------------------------------
# setting
# ---------------------------------------------------------
EXP_NAME = "emotion48_model_result"

OUTPUT_DIR = f"results_models/{EXP_NAME}/checkpoints"
FINAL_DIR  = f"results_models/{EXP_NAME}/final"

# ---------------------------------------------------------
# 1. 設定
# ---------------------------------------------------------
os.environ["WANDB_DISABLED"] = "true"

# GPUチェック
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Device: GPU (CUDA) - {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Device: CPU")

# ---------------------------------------------------------
# 2. データ準備
# ---------------------------------------------------------
file_path = os.path.join('data', 'wrime-ver2.tsv')

# ファイルの存在確認
if not os.path.exists(file_path):
    raise FileNotFoundError(f"エラー: データファイルが見つかりません。\nパス: {file_path}\n'data'フォルダを作成し、その中に'wrime-ver2.tsv'を配置してください。")

print(f"データセットを読み込んでいます: {file_path}")
df_wrime = pd.read_table(file_path)

emotion_names = ['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust']
emotion_names_jp = ['喜び', '悲しみ', '期待', '驚き', '怒り', '恐れ', '嫌悪', '信頼']
num_labels = len(emotion_names)

df_wrime['readers_emotion_intensities'] = df_wrime.apply(lambda x: [x['Avg. Readers_' + name] for name in emotion_names], axis=1)
is_target = df_wrime['readers_emotion_intensities'].map(lambda x: max(x) >= 2)
df_wrime_target = df_wrime[is_target]

df_groups = df_wrime_target.groupby('Train/Dev/Test')
df_train = df_groups.get_group('train')
df_test = pd.concat([df_groups.get_group('dev'), df_groups.get_group('test')])

# ---------------------------------------------------------
# 3. 前処理
# ---------------------------------------------------------
from datasets import Dataset
checkpoint = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(batch):
    tokenized_batch = tokenizer(batch['Sentence'], truncation=True, padding='max_length', max_length=128)
    tokenized_batch['labels'] = [np.array(x) / np.sum(x) for x in batch['readers_emotion_intensities']]
    return tokenized_batch

train_dataset = Dataset.from_pandas(df_train[['Sentence', 'readers_emotion_intensities']])
test_dataset = Dataset.from_pandas(df_test[['Sentence', 'readers_emotion_intensities']])

train_tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
test_tokenized_dataset = test_dataset.map(tokenize_function, batched=True)

# ---------------------------------------------------------
# 4. 学習
# ---------------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
model.to(device)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    label_ids = np.argmax(labels, axis=-1)
    return metric.compute(predictions=predictions, references=label_ids)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8, # GPUメモリに応じて調整 (T4なら16-32)
    per_device_eval_batch_size=8,
    num_train_epochs=1.0,           # 本番用に3エポック
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    logging_steps=100,
    learning_rate=2e-5,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_dataset,
    eval_dataset=test_tokenized_dataset,
    compute_metrics=compute_metrics,
)

print(">>> 学習を開始します...")
trainer.train()
print(">>> 学習完了。モデルを保存します。")
trainer.save_model(FINAL_DIR) # モデルを保存
tokenizer.save_pretrained(FINAL_DIR)

# ---------------------------------------------------------
# 5. 推論と結果の画像保存
# ---------------------------------------------------------
def analyze_emotion(text, output_filename="result.png"):
    model.eval()
    tokens = tokenizer(text, truncation=True, return_tensors="pt")
    tokens.to(device)
    
    with torch.no_grad():
        preds = model(**tokens)
    
    prob = np.exp(preds.logits.cpu().detach().numpy()[0])
    prob = prob / np.sum(prob)
    
    # コンソール出力
    print(f"\n入力テキスト: {text}")
    print("-" * 30)
    for n, p in zip(emotion_names_jp, prob):
        print(f"{n}: {p:.4f}")
    
    # グラフ保存 (Headless対応)
    plt.figure(figsize=(10, 5))
    df_prob = pd.DataFrame(list(zip(emotion_names_jp, prob)), columns=['Emotion', 'Probability'])
    sns.barplot(x='Emotion', y='Probability', data=df_prob)
    plt.title(f'Text: {text}')
    plt.ylim(0, 1)
    
    plt.savefig(output_filename) # 画像として保存
    print(f"\n>>> グラフを保存しました: {output_filename}")

# テスト実行
if __name__ == "__main__":
    #test_text = "Ubuntuサーバーでの学習、意外とスムーズにいけた！"
    #analyze_emotion(test_text, "emotion_result.png")
    analyze_emotion('今日から長期休暇だぁーーー！！！', "emotion_result2.png")
    analyze_emotion('この書類にはコーヒーかかってなくて良かった…。不幸中の幸いだ。', "emotion_result3.png")
    analyze_emotion('なんで自分だけこんな目に遭うんだ……', "emotion_result4.png")