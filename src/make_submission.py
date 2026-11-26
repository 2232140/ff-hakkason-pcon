"""交差検証の結果を集計し、閾値を決めて提出用CSVを作る。

usage: python src/make_submission.py --tags v1 [--th 0.5]
"""
import argparse
import glob

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from common import ROOT, WORK, read_test, read_train

ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", default=["v1"])
ap.add_argument("--th", type=float, default=None, help="指定しなければOOFでF1最大の閾値")
ap.add_argument("--out", default=None)
args = ap.parse_args()

paths, y = read_train()
tpaths = read_test()

# OOF（各画像を、その画像を学習に使っていないモデルで予測した確率）をタグ間で平均
oofs, tests = [], []
for tag in args.tags:
    for seed_files in sorted({f.split("_f")[0] for f in glob.glob(f"{WORK}/oof_{tag}_s*_f*.npy")}):
        o = np.full(len(y), np.nan)
        for f in sorted(glob.glob(seed_files + "_f*.npy")):
            idx, p = np.load(f)
            o[idx.astype(int)] = p
        oofs.append(o)
    tests += [np.load(f) for f in sorted(glob.glob(f"{WORK}/test_{tag}_s*_f*.npy"))]
oof = np.nanmean(oofs, 0)
test = np.mean(tests, 0)
m = ~np.isnan(oof)
print(f"models: oof runs={len(oofs)} test preds={len(tests)}  oof coverage={m.mean():.2f}")

ths = np.round(np.arange(0.05, 0.96, 0.05), 2)
print(" th    F1    P     R   pred+   test+率")
best = (0, 0.5)
for t in ths:
    pr = oof[m] > t
    f = f1_score(y[m], pr, zero_division=0)
    print(f"{t:.2f} {f:.3f} {precision_score(y[m], pr, zero_division=0):.3f} {recall_score(y[m], pr):.3f} {pr.sum():5d}  {(test > t).mean():.4f}")
    best = max(best, (f, t))
th = args.th if args.th is not None else best[1]
print(f"OOF best F1 {best[0]:.4f} @ {best[1]:.2f} / 採用閾値 {th:.2f} / 学習データの陽性率 {y.mean():.4f}")

np.save(f"{WORK}/oof_ens_{'+'.join(args.tags)}.npy", oof)
out = args.out or f"{ROOT}/out/submission_{'+'.join(args.tags)}.csv"
with open(out, "w") as f:
    for p, s in zip(tpaths, test):
        f.write(f"{p},{int(s > th)}\n")
print("wrote", out, "rows", len(tpaths), "positives", int((test > th).sum()))
