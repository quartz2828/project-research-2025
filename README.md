# [title]

[explain]

## 📋 概要 (Overview)
このリポジトリは、学習済みのモデルを使用して推論を行ったり、追加の評価を行ったりするためのコードを含んでいます。
※ 注:データセットやモデルのファイルは GoogleDrive で管理、ここにあるのはコードのみです。
※ scriptsに入っているプログラムを使うにはコード内の各ディレクトリを指定し直すか、scriptsから一旦取り出して使う必要があります

## 🚀 使い方 (Usage)

```bash
# リポジトリのクローン
git clone [https://github.com/](https://github.com/)[quartz2828]/[project-research-2025].git
cd [project-research-2025]

# 各種ライブラリのインストール
pip install -r requirements.txt
※ 手元の環境(mac)でpredict_emoji.pyを最低限実行できるようにrequirements.txtを調整済み、コメントアウト部分を無くせば元の状態になります

# aa

## 📂 ディレクトリ構成 (Directory Structure)
.
├── .gitignore/ 
├── predict_emoji.py     # 完成したメインスクリプト
├── scripts/             # 利用したコード群
├── data/                # データセット (GoogleDriveで管理)
├── results_models/      # 学習済みモデル (GoogleDriveで管理)
├── eval_results/        # 評価結果 (GoogleDriveで管理)
├── requirements.txt     # 依存ライブラリ
└── README.md            # このファイル

