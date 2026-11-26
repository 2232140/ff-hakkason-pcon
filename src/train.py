"""写真単位の交差検証で学習し、OOF予測と評価用画像の予測確率を保存する。

usage: python src/train.py --tag v1 --folds 0 1 2 3 4
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from common import DEV, MEAN, STD, WORK, Net, augment, dihedral, load_images, parse, read_test, read_train

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="v1")
ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--steps", type=int, default=70)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--pos_frac", type=float, default=0.25)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--full", action="store_true", help="全データで学習（検証なし）")
args = ap.parse_args()

paths, y = read_train()
tpaths = read_test()
X = torch.from_numpy(load_images(paths, "train"))
XT = torch.from_numpy(load_images(tpaths, "test"))
groups = np.array([parse(p)[0] for p in paths])
splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(paths, y, groups))


@torch.no_grad()
def predict(model, Xu, tta=True):
    model.eval()
    out = []
    for i in range(0, len(Xu), 256):
        x = (Xu[i:i + 256].to(DEV).permute(0, 3, 1, 2).float() / 255 - MEAN) / STD
        ks = range(8) if tta else [0]
        p = torch.stack([torch.sigmoid(model(dihedral(x, k))) for k in ks]).mean(0)
        out.append(p.cpu())
    return torch.cat(out).numpy()


def best_f1(yt, p):
    ths = np.linspace(0.05, 0.95, 37)
    f = [f1_score(yt, p > t, zero_division=0) for t in ths]
    return max(f), ths[int(np.argmax(f))]


def fit(tr_idx, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Net().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-2)
    total = args.epochs * args.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=total, pct_start=0.15)
    pos, neg = tr_idx[y[tr_idx] == 1], tr_idx[y[tr_idx] == 0]
    npos = int(args.bs * args.pos_frac)
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for _ in range(args.steps):
            idx = np.concatenate([rng.choice(pos, npos), rng.choice(neg, args.bs - npos, replace=False)])
            x = X[idx].to(DEV).permute(0, 3, 1, 2).float() / 255
            t = torch.from_numpy(y[idx]).float().to(DEV)
            x = (augment(x) - MEAN) / STD
            loss = F.binary_cross_entropy_with_logits(model(x), t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        yield ep, tot / args.steps, model


oof = np.full(len(y), np.nan, np.float32)
for fold in args.folds:
    tr_idx, va_idx = splits[fold]
    if args.full:
        tr_idx, va_idx = np.arange(len(y)), splits[fold][1]
    t0 = time.time()
    for ep, loss, model in fit(tr_idx, args.seed * 100 + fold):
        if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            p = predict(model, X[va_idx], tta=False)
            f, th = best_f1(y[va_idx], p)
            print(f"fold{fold} ep{ep+1} loss{loss:.4f} valF1 {f:.3f}@{th:.2f} npos={int(y[va_idx].sum())} {time.time()-t0:.0f}s", flush=True)
    sfx = f"{args.tag}_s{args.seed}_f{fold}" + ("_full" if args.full else "")
    if not args.full:
        oof[va_idx] = predict(model, X[va_idx])
        np.save(f"{WORK}/oof_{sfx}.npy", np.stack([va_idx, oof[va_idx]]))
        f, th = best_f1(y[va_idx], oof[va_idx])
        print(f"fold{fold} TTA valF1 {f:.3f}@{th:.2f}", flush=True)
    np.save(f"{WORK}/test_{sfx}.npy", predict(model, XT))
    torch.save(model.state_dict(), f"{WORK}/model_{sfx}.pt")

m = ~np.isnan(oof)
if m.sum():
    f, th = best_f1(y[m], oof[m])
    print(f"OOF F1 {f:.4f} @ th {th:.2f} (n={m.sum()}, pos={int(y[m].sum())})")
