# predict_emoji.py
# 完成版、streamlitと用意した複数の学習済モデルを用いて、ブラウザに
# 入力された文章から予測し絵文字を出力
# モデルを保存した場所に応じて、MODEL_DICTのパスを変更して使用してください。
# 実行 streamlit run predict_emoji.py

import streamlit as st
import pandas as pd
import numpy as np
import os, time, signal, torch, joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from st_keyup import st_keyup

# ---- 設定 ----
MODEL_DIR = "results_models/emoji48_0105_w_model_result/final"

# 使用モデルの設定
MODEL_DICT = {
    "モデルA (Emo48)": "results_models/emoji48_0105_model_result/final",
    "モデルA (Emo48 Weighted)": "results_models/emoji48_0105_w_model_result/final",
    "モデルB (Emo8)": "results_models/emoji8_model_result/final",
    "モデルB (Emo8 Weighted)": "results_models/emoji8_0105_w_model_result/final",
    "モデルB (Emo8 Balanced)": "results_models/emoji8_0105_b_model_result/final",

    # "表示名": "フォルダパス",
}

# デフォルトで選択されるモデル（辞書のキー）
DEFAULT_MODEL_NAME = list(MODEL_DICT.keys())[0]

# --- 終了処理 ---
if "exit_app" in st.session_state and st.session_state.exit_app:
    st.empty()
    st.sidebar.empty()
    st.markdown("""
        <div style='text-align: center; margin-top: 100px;'>
            <h1>終了しました 👋</h1>
            <p>このタブを閉じてください。</p>
        </div>
    """, unsafe_allow_html=True)
    time.sleep(2)
    os.kill(os.getpid(), signal.SIGINT)
    st.stop()

# --- 1. リソースの読み込み ---
@st.cache_resource(max_entries=2)
def load_prediction_resources(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    le = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
    emoji_labels = le.classes_
    return tokenizer, model, device, emoji_labels

# --- 2. 予測ロジック ---
def get_prediction(text, k=5):
    if not text.strip():
        return None, None
    enc = tokenizer([text], truncation=True, padding=True, return_tensors="pt", max_length=128)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = model(**enc)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    top_indices = np.argsort(probs)[::-1][:k]
    return top_indices, probs[top_indices]

# --- 3. セッション状態の初期化 ---
if "accumulated_text" not in st.session_state:
    st.session_state.accumulated_text = ""
if "manual_preds" not in st.session_state:
    st.session_state.manual_preds = None
if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0

# --- 4. UI/サイドバー設定 ---
st.title("絵文字予測くん")

with st.sidebar:
    st.header("設定")
    #　モデル
    selected_model_name = st.selectbox("使用するモデル(cache:2models)", list(MODEL_DICT.keys()))
    current_model_path = MODEL_DICT[selected_model_name]
    st.divider() 

    # モード選択
    is_live_mode = st.toggle("リアルタイム予測（自動）", value=True)
    top_k_val = st.slider("手動モードの候補数", 1, 48, 5)
    st.divider()

    # 文章消去
    if st.button("文章をリセット", type="primary"):
        st.session_state.accumulated_text = ""
        st.session_state.manual_preds = None
        st.rerun()

tokenizer, model, device, emoji_labels = load_prediction_resources(current_model_path)

# --- 5. 入力エリア ---
col_input, col_action = st.columns([3, 1])

with col_input:
    if is_live_mode:
        current_input = st_keyup(
            "文章を入力:", 
            placeholder="入力すると自動予測...", 
            key=f"live_input_{st.session_state.reset_counter}",
            debounce=400
        )
    else:
        current_input = st.text_input(
            "文章を入力:", 
            placeholder="入力してボタンを押してください", 
            key=f"static_input_{st.session_state.reset_counter}"
        )

# --- 6. モード別の予測処理 ---

# A. リアルタイムモード（Top-1を横に出して決定）
if is_live_mode and current_input:
    indices, scores = get_prediction(current_input, k=1)
    if indices is not None:
        best_emoji = emoji_labels[indices[0]]
        with col_action:
            st.markdown(f"<h1 style='font-size: 60px; margin: 0;'>{best_emoji}</h1>", unsafe_allow_html=True)
        if st.button("決定 ➔", type="primary"):
            st.session_state.accumulated_text += f" {current_input}{best_emoji}"
            st.session_state.reset_counter += 1
            st.rerun()

# B. 手動モード（ボタンを押すとTop-Kを並べる）
elif not is_live_mode:
    if current_input:
        indices, scores = get_prediction(current_input, k=top_k_val)
        st.session_state.manual_preds = (indices, scores)

    # 予測結果表示
    if st.session_state.manual_preds and current_input:
        indices, scores = st.session_state.manual_preds
        
        st.write("--- 候補 ---")

        cols = st.columns(len(indices))
        for i, idx in enumerate(indices):
            emoji = emoji_labels[idx]
            prob = scores[i]
            
            # 絵文字ボタン
            if cols[i].button(f"{emoji}\n{prob:.1%}", key=f"btn_{idx}"):
                st.session_state.accumulated_text += f" {current_input}{emoji}"
                
                st.session_state.manual_preds = None
                st.session_state.reset_counter += 1  
                st.rerun()

# 文章全体の履歴を表示
if st.session_state.accumulated_text:
    st.caption(f"**文章：**")
    st.code(st.session_state.accumulated_text, language=None, wrap_lines=True)

# --- 7. 管理機能 ---
if st.button("終了"):
    st.session_state.exit_app = True
    st.rerun()

st.divider()
st.caption(f" Model: `{selected_model_name}` | ⚡ Device: {device} ")