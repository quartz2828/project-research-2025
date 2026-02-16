# count_sentence_ave.py
# WRIMEの文章の平均文字数を計算するコード

import pandas as pd

# ---------------------------------------------------------
# 1. ファイルパスの設定
# ---------------------------------------------------------
# お手元のファイル名をここに指定してください
file_path = './data/wrime-ver2.tsv' 

# ---------------------------------------------------------
# 2. データの読み込み
# ---------------------------------------------------------
try:
    # まずWRIMEの標準形式である「タブ区切り(.tsv)」として読み込みを試みます
    df = pd.read_csv(file_path, sep='\t')
    
    # もし列が1つしか認識されなかった場合、カンマ区切り(.csv)の可能性があります
    if len(df.columns) < 2:
        df = pd.read_csv(file_path, sep=',')
        print("カンマ区切り(CSV)として読み込みました。")
    else:
        print("タブ区切り(TSV)として読み込みました。")

except FileNotFoundError:
    print(f"エラー: ファイル '{file_path}' が見つかりません。パスを確認してください。")
    exit()

# ---------------------------------------------------------
# 3. 平均文字数の計算
# ---------------------------------------------------------
# WRIMEのテキストデータは通常 'Sentence' というカラムに入っています
target_column = 'Sentence'

if target_column in df.columns:
    # 文字数を計算（欠損値NaNは空文字として扱い、計算から除外または0文字とする処理）
    # ここでは単純に文字列として変換して長さを測ります
    lengths = df[target_column].astype(str).str.len()
    
    # 平均値を計算
    avg_length = lengths.mean()
    max_length = lengths.max()
    min_length = lengths.min()
    
    print("-" * 30)
    print(f"対象カラム: {target_column}")
    print(f"データ件数: {len(df)}")
    print(f"平均文字数: {avg_length:.2f} 文字")
    print(f"最大文字数: {max_length} 文字")
    print(f"最小文字数: {min_length} 文字")
    print("-" * 30)
    
else:
    print(f"エラー: '{target_column}' という列が見つかりません。カラム名を確認してください。")
    print("現在のカラム一覧:", df.columns.tolist())

# ---------------------------------------------------------
# (オプション) 分布の可視化
# ---------------------------------------------------------
# ヒストグラムを表示したい場合はコメントアウトを外してください
"""
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(lengths, bins=50, color='skyblue', edgecolor='black')
plt.title('Character Count Distribution (WRIME)')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.show()
"""