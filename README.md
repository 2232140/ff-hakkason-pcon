# Pコン跡の画像認識（富士フイルム AI-FDE ハッカソン 前半課題）

256×256 のコンクリート壁画像に Pコン跡が含まれるか（1/0）を判定するプログラムです。
配布された学習用画像だけを使い、CNN をゼロから学習します。事前学習済みモデル、外部データ、API は使っていません。

※ 手順と結果は作業の進行に合わせて更新中です。

## 動作環境

- macOS（Apple M3 Pro、メモリ 36GB）、Python 3.12
- PyTorch（MPS で学習。CUDA / CPU でも動作）
- ライブラリのバージョンは `requirements.txt` のとおり

## データの配置

```
DATA_Maguire_20180517_ALL/SDNET2018/W/{CW,UW}/*.jpg   # SDNET2018 の壁画像
dist/train_label.csv                                  # 配布物
dist/test.txt
dist/sample_submission.csv
```

## 実行手順

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
mkdir -p work out
.venv/bin/python src/train.py --tag v1      # 写真単位の5分割交差検証で学習し、予測確率を work/ に保存
```

## ファイル

| ファイル | 内容 |
|---|---|
| `src/common.py` | データ読み込み、モデル定義、データ拡張 |
| `src/train.py` | 交差検証での学習と予測 |
