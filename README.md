# Pコン跡の判定

富士フイルム AI-FDE ハッカソン（2026-11-26）前半課題のコードです。
256×256のコンクリート壁の画像に、Pコン跡が写っているか（1/0）を判定します。

配布された学習用画像だけを使って、小さめのResNetを一から学習しています。
事前学習済みモデル、外部データ、APIは使っていません。

## 環境

- macOS (Apple M3 Pro, 36GB) / Python 3.12 / PyTorch 2.15 (MPS)
- CUDA、CPUでも動きます
- バージョンは `requirements.txt`

## データの置き場所

```
DATA_Maguire_20180517_ALL/SDNET2018/W/{CW,UW}/*.jpg
dist/train_label.csv
dist/test.txt
dist/sample_submission.csv
```

## 実行

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
mkdir -p work out

# 学習（1回40〜60分）
.venv/bin/python src/train.py --tag v2 --zoom_out 2.5
.venv/bin/python src/train.py --tag v3 --zoom_out 2.5 --cmin 0.35 --seed 1
.venv/bin/python src/train.py --tag v4 --zoom_out 2.5 --cmin 0.35 --seed 2 --epochs 64

# 提出ファイル
.venv/bin/python src/make_submission.py --tags v2 v3 v4 --th 0.8
```

`out/submission_v2+v3+v4.csv` ができます。
初回は画像を `work/*.npy` にまとめるので少し時間がかかります。

## やっていること

- 1枚の写真が18×14に切られているので、検証は写真単位の5分割（同じ穴が学習側と検証側に入らないように）
- 「あり」が2%しかないので、各バッチの25%を「あり」にする
- データ拡張: 反転・90度回転、縮小（余白は鏡像で埋める）、明るさ・コントラスト・色味、ぼかし、ノイズ
  - 平行移動や拡大は、端で切れた穴が枠の外に出てラベルが変わるので使っていない
- 予測は反転・回転8通りの平均
- 設定の違う3回 × 5分割 = 15モデルの確率を平均して、0.8で切る

## 結果

| | 手元のF1（写真単位CV） | 画面のスコア |
|---|---|---|
| v1（縮小の拡張なし） | 0.877 | 0.809 |
| v2 | 0.893 | 0.813 |
| v2+v3+v4 の平均 | 0.897 | 0.813 |

構造を変えた版（`--hires`、最初の縮小を1段減らす）も試しましたが、手元で0.877と良くならなかったので最終版には入れていません。

手元と画面で0.08くらい差があります。写真が変わると落ちる、というのが今のところの理解です。

## 分析用のスクリプト

学習用データのOOF予測（その写真を学習に使っていないモデルの予測）で計算します。

| ファイル | 内容 |
|---|---|
| `src/analyze.py` | 穴単位で何個見つけたか。確率を3段階に分けたとき人が確認する枚数 |
| `src/rubric_demo.py` | 写真ごとに「Pコンのうち近くにひび割れがある割合」を出す（沈みひび割れの評価を想定） |
| `src/per_photo.py` | 写真ごとの誤りの数と、F1のばらつき |

これらは元の写真を復元して使いますが、提出する予測には使っていません。

## ファイル

| ファイル | 内容 |
|---|---|
| `src/common.py` | 読み込み、モデル、データ拡張 |
| `src/train.py` | 学習と予測 |
| `src/make_submission.py` | 集計と提出ファイルの作成 |
