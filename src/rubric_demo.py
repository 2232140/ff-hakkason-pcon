"""沈みひび割れ評価の試算: 「Pコンの何割の近傍にひび割れがあるか」を写真ごとに出す。

中国地方整備局の手引き（施工編 p.18）の4段階評価は、Pコン総数に対する発生割合で決まる。
SDNET2018 にはタイルごとのひび割れ有無（CW/UW フォルダ）が元から付いているので、
Pコン跡の検出結果と組み合わせると、この割合を自動で出す流れを試せる。

学習用データのOOF予測だけを使う。写真の復元は用途の試算のためで、提出する予測には使わない。
ひび割れの種類（沈みひび割れかどうか）と長さは判定していないので、点数は参考値。

usage: python src/rubric_demo.py --tags v1 --th 0.6
"""
import argparse
import collections

import numpy as np

from common import WORK, parse, read_train

ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", default=["v1"])
ap.add_argument("--th", type=float, default=0.6)
args = ap.parse_args()

paths, y = read_train()
oof = np.load(f"{WORK}/oof_ens_{'+'.join(args.tags)}.npy")
crack = np.array(["/CW/" in p for p in paths])
print(f"ひび割れありの割合: Pコン跡ありタイル {crack[y == 1].mean():.3f} / なしタイル {crack[y == 0].mean():.3f}")

by = collections.defaultdict(dict)
for i, p in enumerate(paths):
    s, n = parse(p)
    by[s][divmod(n - 1, 18)] = i


def holes_of(g, flag):
    """隣接（斜め含む）する flag=True のタイルを1つの穴にまとめる。"""
    seen, out = set(), []
    for rc, i in g.items():
        if not flag[i] or rc in seen:
            continue
        comp, stack = [], [rc]
        seen.add(rc)
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in g and q not in seen and flag[g[q]]:
                        seen.add(q)
                        stack.append(q)
        out.append(comp)
    return out


def cracked(g, comp):
    """穴のタイル自身か、その周囲1タイル以内にひび割れありのタイルがあるか。"""
    return any(
        (r + dr, c + dc) in g and crack[g[(r + dr, c + dc)]]
        for r, c in comp for dr in (-1, 0, 1) for dc in (-1, 0, 1)
    )


def score(ratio):
    return 4 if ratio == 0 else 3 if ratio < 0.5 else 2  # 1/5・1/2 の境目を簡略化した参考値


rows, agree = [], 0
for s, g in sorted(by.items()):
    if len(g) < 200:
        continue
    res = []
    for flag in (y == 1, oof > args.th):
        hs = holes_of(g, flag)
        n = len(hs)
        k = sum(cracked(g, h) for h in hs)
        res.append((n, k))
    (n_t, k_t), (n_p, k_p) = res
    r_t, r_p = (k_t / n_t if n_t else 0), (k_p / n_p if n_p else 0)
    agree += score(r_t) == score(r_p)
    rows.append((s, n_t, k_t, n_p, k_p))
print("写真  正解:Pコン数/ひび近傍  AI:Pコン数/ひび近傍")
for s, n_t, k_t, n_p, k_p in rows:
    print(f"{s:>6}  {n_t:2d}/{k_t:2d}   {n_p:2d}/{k_p:2d}" + ("" if (n_t, k_t) == (n_p, k_p) else "   *"))
T = np.array([r[1:] for r in rows])
print(f"合計: 正解 Pコン{T[:,0].sum()}個中 ひび近傍{T[:,1].sum()}個 ({T[:,1].sum()/T[:,0].sum():.2f}) / AI Pコン{T[:,2].sum()}個中 {T[:,3].sum()}個 ({T[:,3].sum()/T[:,2].sum():.2f})")
print(f"Pコン数が正解と一致した写真: {(T[:,0]==T[:,2]).sum()}/{len(rows)}、参考点数が一致: {agree}/{len(rows)}")
