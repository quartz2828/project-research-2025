# [title]

> **Note**: 本リポジトリは、大学の課題提出用として作成されました。
[explain]

## 📋 概要 (Overview)
このリポジトリは、学習済みのモデルを使用して推論を行ったり、追加の評価を行ったりするためのコードを含んでいます。

### ⚠️ 注意事項
- 個人の練習的側面が強く、predict_emoji.py以外は実行の再確認ができていません
- データセットやモデルは GoogleDrive で管理、ここにあるのはコードのみです。
- scriptsに入っているプログラムを使うにはコード内の各ディレクトリを指定し直すか、scriptsから一旦取り出して使う必要があります
- 手元の環境(mac)でpredict_emoji.pyを最低限実行できるようにrequirements.txtを調整済み、コメントアウト部分を無くせば元の状態になります

## 🚀 使い方 (Usage)

```bash
# リポジトリのクローン
git clone [https://github.com/](https://github.com/)[quartz2828]/[project-research-2025].git
cd [project-research-2025]

# 各種ライブラリのインストール
pip install -r requirements.txt

# 実行
# GoogleDriveからダウンロードした各種データがあれば、そのまま実行できます。
# 別の場所にモデルを保存した場合は保存先のディレクトリを指定し直してください。
streamlit run predict_emoji.py
```

## 📂 ディレクトリ構成 (Directory Structure)
```text
.
├── .gitignore/ 
├── predict_emoji.py     # 完成したメインスクリプト
├── scripts/             # 利用したコード群
├── data/                # データセット (GoogleDriveで管理)
├── results_models/      # 学習済みモデル (GoogleDriveで管理)
├── eval_results/        # 評価結果 (GoogleDriveで管理)
├── requirements.txt     # 依存ライブラリ
└── README.md            # このファイル
```

