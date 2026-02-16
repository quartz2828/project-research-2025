# make_balanced_dataset.py
# 作成した文章→絵文字のデータセットを用いて、trainデータを絵文字ごとに均衡になるようにサンプリングするコード("b"とついたデータの作成)

import pandas as pd
import numpy as np
import os

# ==============================
# 設定
# ==============================
INPUT_CSV = os.path.join('data', 'wrime_emotag8_WR.csv')
OUTPUT_CSV = os.path.join('data', 'wrime_emotag8_b_WR.csv')

# 目標件数 (None にすると 中央値 を自動採用します)
TARGET_N = None

# ==============================
# 処理
# ==============================
def main():
    if not os.path.exists(INPUT_CSV):
        print("入力ファイルがありません")
        return

    print("データを読み込み中...")
    df = pd.read_csv(INPUT_CSV)
    
    # Trainデータのみ抽出して加工対象にする
    # (Dev/Testデータは絶対に加工してはいけません！評価の正当性がなくなるため)
    df_train = df[df['Train/Dev/Test'] == 'train'].copy()
    df_others = df[df['Train/Dev/Test'] != 'train'].copy()
    
    # 分布の確認
    counts = df_train['best_emoji'].value_counts()
    stats = counts.describe()
    
    print("\n--- 元データの分布統計 ---")
    print(stats)
    
    # 目標件数の決定
    if TARGET_N is None:
        target_n = int(stats['50%']) # 中央値を採用
        print(f"\n>> 目標件数をデータの「中央値 ({target_n}件)」に設定しました。")
    else:
        target_n = TARGET_N
        print(f"\n>> 目標件数を「指定値 ({target_n}件)」に設定しました。")

    # サンプリング実行
    print(f"\nサンプリング処理を実行中 (Target: {target_n})...")
    
    def sampling_func(group):
        if len(group) < target_n:
            # データが足りない -> 重複ありで増やす (Up-sampling)
            return group.sample(n=target_n, replace=True, random_state=42)
        else:
            # データが多い -> ランダムに選んで減らす (Down-sampling)
            return group.sample(n=target_n, replace=False, random_state=42)

    df_train_balanced = df_train.groupby('best_emoji', group_keys=False).apply(sampling_func)
    
    # シャッフル
    df_train_balanced = df_train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"--- 処理後データ数: {len(df_train_balanced)} ---")

    # 元のDev/Testデータと結合して保存
    # (学習時は Train/Dev/Test カラムで再度フィルタリングされるので、1ファイルにまとめてOK)
    df_final = pd.concat([df_train_balanced, df_others], axis=0)
    
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"\n保存完了: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()