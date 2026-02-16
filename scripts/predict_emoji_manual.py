# predict_emoji_manual.py
# テスト用、streamlitと用意した学習済モデルを用いてブラウザに入力された文章から予測し絵文字を出力
# 実行 streamlit run predict_emoji_manual.py

import streamlit as st
import pandas as pd
import numpy as np
import os, time, signal
import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---- 設定 ----
MODEL_DIR = "results_models/emoji48_0105_w_model_result/final"
# 評価スクリプトの TOP_K に合わせる
DEFAULT_TOP_K = 5 

# --- 1. リソースの読み込み ---
@st.cache_resource
def load_prediction_resources():
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # A. トークナイザーとモデルのロード
    if not os.path.exists(MODEL_DIR):
        st.error(f"モデルディレクトリが見つかりません: {MODEL_DIR}")
        st.stop()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    
    # B. ラベル情報のロード (手動で用意した label_encoder.pkl を使用)
    le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    if os.path.exists(le_path):
        le = joblib.load(le_path)
        emoji_labels = le.classes_
    else:
        st.error(f"ラベルエンコーダーが見つかりません: {le_path}")
        st.stop()
    
    return tokenizer, model, device, emoji_labels

# リソースのロード
tokenizer, model, device, emoji_labels = load_prediction_resources()

# --- 2. UI部分 ---
st.title("✨絵文字予測")
st.write(f"モデル: `{os.path.basename(os.path.dirname(MODEL_DIR))}`")

# 候補数の設定
top_k = st.slider("表示する候補数 (Top-K)", 1, len(emoji_labels), DEFAULT_TOP_K)

# テキスト入力
user_input = st.text_area("文章を入力してください", placeholder="今日から長期休暇だぁーーー！！！", height=100)

# --- 3. 予測実行 (eval_compare_models.py のロジックを移植) ---
if st.button("予測実行"):
    if user_input.strip() == "":
        st.warning("文章を入力してください。")
    else:
        with st.spinner("推論中..."):
            # 評価スクリプトの predict_with_model 内の処理を1件用に適用
            enc = tokenizer([user_input], truncation=True, padding=True, return_tensors="pt", max_length=128)
            enc = {k: v.to(device) for k, v in enc.items()}
            
            with torch.no_grad():
                outputs = model(**enc)
                logits = outputs.logits
                # Softmaxで確率算出
                probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
            
            # 評価スクリプトと同じソート順 (降順で上位K個)
            top_indices = np.argsort(probs)[::-1][:top_k]
            top_probs = probs[top_indices]

            # --- 結果表示 ---
            st.subheader("🔮 予測結果")
            
            # 最上位の表示
            best_emoji = emoji_labels[top_indices[0]]
            st.markdown(f"<div style='text-align: center; font-size: 80px; margin: 20px;'>{best_emoji}</div>", unsafe_allow_html=True)
            st.metric(label="Top-1 予測", value=best_emoji, delta=f"信頼度: {top_probs[0]:.2%}")

            # Top-K 一覧
            st.write(f"Top-{top_k} 候補:")
            res_df = pd.DataFrame({
                "絵文字": [emoji_labels[i] for i in top_indices],
                "確率": [f"{p:.4%}" for p in top_probs]
            })
            st.table(res_df)

# --- 4. 終了・管理 ---
st.divider()
st.caption(f"Device: {device} | Labels: {len(emoji_labels)} classes")

if st.button("終了"):
    st.info("ブラウザを閉じてください。")
    time.sleep(1)
    os.kill(os.getpid(), signal.SIGINT)
    os._exit(0)