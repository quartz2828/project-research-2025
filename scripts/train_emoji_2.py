# train_emoji_2.py
# 作成した文章→絵文字のデータセットを用いて重み付きで文章から絵文字を予測するモデルの学習コード

import os
import pandas as pd
import numpy as np
import torch
from torch import nn # 追加
import evaluate
# EarlyStoppingCallback を追加
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight # 追加

# 【重要】SSH(Headless)環境用の設定
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

# ---------------------------------------------------------
# setting
# ---------------------------------------------------------
INPUT_CSV = os.path.join('data', 'wrime_emotag8_WR.csv')
EXP_NAME = "emoji8_0105_weighted_model_result" 

OUTPUT_DIR = f"results_models/{EXP_NAME}/checkpoints"
FINAL_DIR  = f"results_models/{EXP_NAME}/final"

# ---------------------------------------------------------
# 1. 設定
# ---------------------------------------------------------
os.environ["WANDB_DISABLED"] = "true"

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

df = df[['Sentence', 'Train/Dev/Test', 'best_emoji']]

le = LabelEncoder()
df['label'] = le.fit_transform(df['best_emoji'])
emoji_labels = le.classes_
num_labels = len(emoji_labels)

print(f"ターゲット絵文字数: {num_labels}種類")

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

train_dataset = Dataset.from_pandas(df_train[['Sentence', 'label']])
test_dataset = Dataset.from_pandas(df_test[['Sentence', 'label']])

print("トークナイズを実行中...")
train_tokenized = train_dataset.map(tokenize_function, batched=True)
test_tokenized = test_dataset.map(tokenize_function, batched=True)

# ---------------------------------------------------------
# ★変更点: クラスの重みを計算（少数派クラスの重視）
# ---------------------------------------------------------
print("クラスの重みを計算中...")
# sklearnで均衡になる重みを計算
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(df_train['label']),
    y=df_train['label']
)
# Tensorに変換してGPUへ
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

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

# ★変更点: 重み付きLossを使うカスタムTrainer
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # ここで重み付きのCrossEntropyLossを使用
        loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5.0, 

    eval_strategy="steps",
    eval_steps=500,        
    save_strategy="steps",
    save_steps=500,
    
    save_total_limit=2,

    logging_steps=50,
    learning_rate=2e-5,
    load_best_model_at_end=True,
    
    # ★追加: LossではなくAccuracyでベストモデルを決める
    metric_for_best_model="accuracy", 
    greater_is_better=True,
    
    report_to="none"
)

# Trainerを WeightedTrainer に変更
trainer = WeightedTrainer(
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
def predict_emoji(text, top_k=3):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
    top_indices = probs.argsort()[-top_k:][::-1]
    
    print(f"\n入力: {text}")
    print("-" * 30)
    for i in top_indices:
        print(f"{emoji_labels[i]}  : {probs[i]:.4f}")

if __name__ == "__main__":
    print("\n========== 推論テスト ==========")
    predict_emoji("今日から長期休暇だぁーーー！！！")
    predict_emoji("この書類にはコーヒーかかってなくて良かった…。不幸中の幸いだ。")
    predict_emoji("なんで自分だけこんな目に遭うんだ……")