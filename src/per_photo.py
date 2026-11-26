"""写真ごとの成績のばらつきを出す。新しい現場（写真）に持ち込んだときの性能の振れ幅の目安。

usage: python src/per_photo.py --tags v2 v3 --th 0.85
"""
import argparse
import collections

import numpy as np

from common import WORK, parse, read_train

ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", default=["v2", "v3"])
ap.add_argument("--th", type=float, default=0.85)
args = ap.parse_args()

paths, y = read_train()
oof = np.load(f"{WORK}/oof_ens_{'+'.join(args.tags)}.npy")
pred = oof > args.th
src = np.array([parse(p)[0] for p in paths])

rows = []
for s in sorted(set(src)):
    m = src == s
    tp = int((pred & (y == 1) & m).sum())
    fn = int((~pred & (y == 1) & m).sum())
    fp = int((pred & (y == 0) & m).sum())
    rows.append((s, tp, fn, fp))
clean = sum(1 for _, _, fn, fp in rows if fn + fp == 0)
print(f"写真 {len(rows)}枚中、誤り0の写真 {clean}枚、誤り1〜2枚 {sum(1 for _,_,fn,fp in rows if 1 <= fn+fp <= 2)}枚、3枚以上 {sum(1 for _,_,fn,fp in rows if fn+fp >= 3)}枚")
tot_err = sum(fn + fp for _, _, fn, fp in rows)
worst = sorted(rows, key=lambda r: -(r[2] + r[3]))[:6]
print("誤りの多い写真 (写真, 正解, 見落とし, 空振り):", worst)
print(f"誤り全{tot_err}枚のうち、上位5写真が {sum(r[2]+r[3] for r in worst[:5])}枚 ({sum(r[2]+r[3] for r in worst[:5])/tot_err:.0%})")

# 写真を無作為に半分選んだときのF1の振れ幅（評価用の一部だけで計算される画面スコアの粗さの目安）
rng = np.random.default_rng(0)
R = np.array([r[1:] for r in rows])
f = []
for _ in range(2000):
    k = rng.choice(len(R), len(R) // 2, replace=False)
    tp, fn, fp = R[k].sum(0)
    f.append(2 * tp / (2 * tp + fn + fp))
print(f"全体F1 {2*R[:,0].sum()/(2*R[:,0].sum()+R[:,1].sum()+R[:,2].sum()):.3f} / 写真を半分選んだときのF1: 5%点 {np.percentile(f,5):.3f}, 中央 {np.median(f):.3f}, 95%点 {np.percentile(f,95):.3f}")
