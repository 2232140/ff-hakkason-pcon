"""現場目線の評価: 穴単位の発見率と、確信度3段階での人の確認枚数を出す。

学習用データのOOF予測（その画像を学習に使っていないモデルの予測）だけを使う。
写真の復元はこの分析のためだけに行い、予測には使わない。

usage: python src/analyze.py --tags v1 --th 0.6
"""
import argparse
import collections

import numpy as np

from common import WORK, parse, read_train

ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", default=["v1"])
ap.add_argument("--th", type=float, default=0.6)
ap.add_argument("--lo", type=float, default=0.05)
ap.add_argument("--hi", type=float, default=0.95)
args = ap.parse_args()

paths, y = read_train()
oof = np.load(f"{WORK}/oof_ens_{'+'.join(args.tags)}.npy")
pred = oof > args.th

# 写真ごとに 18列のグリッドへ戻し、隣接（斜め含む）する「あり」タイルを1つの穴としてまとめる
by = collections.defaultdict(dict)
for i, p in enumerate(paths):
    s, n = parse(p)
    by[s][divmod(n - 1, 18)] = i
holes = []
for s, g in by.items():
    seen = set()
    for rc, i in g.items():
        if y[i] != 1 or rc in seen:
            continue
        comp, stack = [], [rc]
        seen.add(rc)
        while stack:
            r, c = stack.pop()
            comp.append(g[(r, c)])
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in g and q not in seen and y[g[q]] == 1:
                        seen.add(q)
                        stack.append(q)
        holes.append((s, comp))

found = [any(pred[i] for i in comp) for _, comp in holes]
print(f"画像単位: あり{int(y.sum())}枚中 {int((pred & (y == 1)).sum())}枚を検出 (再現率 {(pred & (y == 1)).sum() / y.sum():.3f}) / 空振り {int((pred & (y == 0)).sum())}枚")
print(f"穴単位  : {len(holes)}個中 {sum(found)}個を検出 (発見率 {np.mean(found):.3f})")
print("見落とした穴:", [(s, [parse(paths[i])[1] for i in c], [round(float(oof[i]), 2) for i in c]) for (s, c), f in zip(holes, found) if not f])
sz = collections.Counter(len(c) for _, c in holes)
print("1つの穴がまたがる枚数の分布:", dict(sorted(sz.items())))

# 写真ごとの穴の数え上げ誤差（検査では Pコンの総数が評価の分母になる）
fp_by = collections.Counter(parse(paths[i])[0] for i in np.where(pred & (y == 0))[0])
print("空振りが多い写真:", fp_by.most_common(5))

# 3段階運用: hi以上=自動で「あり」、lo未満=自動で「なし」、その間=人が確認
auto_pos, auto_neg = oof >= args.hi, oof < args.lo
review = ~auto_pos & ~auto_neg
print(f"\n3段階運用 (あり>={args.hi}, なし<{args.lo})")
print(f"  自動あり {auto_pos.sum()}枚 (うち誤り {int((auto_pos & (y == 0)).sum())})")
print(f"  自動なし {auto_neg.sum()}枚 (うち見落とし {int((auto_neg & (y == 1)).sum())})")
print(f"  人が確認 {review.sum()}枚 = 全体の {review.mean() * 100:.1f}% (この中の「あり」 {int((review & (y == 1)).sum())}枚)")
