# count_train_emojis.py
# 作成した文章→絵文字のデータセットを用いて、trainデータにおける絵文字の出現数を集計するコード

import pandas as pd
import os

# ==============================
# 設定
# ==============================
# 集計したいCSVファイルのパス
INPUT_CSV = os.path.join('data', 'wrime_emotag48_WR_balanced.csv')
OUTPUT_CSV = "data/counts_emotag48_b.txt"

# ==============================
# 実行処理
# ==============================
def main():
    if not os.path.exists(INPUT_CSV):
        print(f"エラー: ファイルが見つかりません -> {INPUT_CSV}")
        return

    print(f"Loading {INPUT_CSV} ...")
    
    # CSV読み込み
    # quotechar='"' がデフォルトなので、囲まれた値も正しく読み込まれます
    df = pd.read_csv(INPUT_CSV, dtype=str)

    # 列名の余分な空白を除去（念のため）
    df.columns = df.columns.str.strip()

    # 'Train/Dev/Test' 列で 'train' の行だけを抽出
    # データ内の空白も除去して比較
    target_col = 'Train/Dev/Test'
    emoji_col = 'best_emoji'
    
    if target_col not in df.columns:
        print(f"エラー: '{target_col}' 列が見つかりません。")
        print("現在の列名:", df.columns.tolist())
        return

    # フィルタリング (trainのみ)
    df_train = df[df[target_col].str.strip() == 'train'].copy()
    
    print(f"Total rows: {len(df)}")
    print(f"Train rows: {len(df_train)}")

    # best_emoji の出現数を集計
    counts = df_train[emoji_col].value_counts()

    print("\n=== Trainデータにおける best_emoji 出現数ランキング ===")
    print(f"ユニークな絵文字の種類数: {len(counts)}")
    print("-" * 40)
    print(f"{'Emoji':<6} | {'Count':<6} | {'Percentage'}")
    print("-" * 40)

    total_train = len(df_train)
    
    for emoji, count in counts.items():
        percent = (count / total_train) * 100
        print(f"{emoji:<6} | {count:<6} | {percent:.2f}%")


    # 結果をCSVにも保存
    out_path = OUTPUT_CSV
    counts.to_csv(out_path, header=["count"], index_label="emoji")
    print(f"\n集計結果を保存しました: {out_path}")

if __name__ == "__main__":
    main()