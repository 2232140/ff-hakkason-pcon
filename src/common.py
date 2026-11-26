"""共通処理: ラベル読み込み、画像キャッシュ、モデル、データ拡張。"""
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "DATA_Maguire_20180517_ALL", "SDNET2018")
DIST = os.path.join(ROOT, "dist")
WORK = os.path.join(ROOT, "work")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def parse(path):
    """'W/UW/7079-45.jpg' -> (元写真ID, タイル番号)。'7107-28_2.jpg' は別写真 '7107_2' 扱い。"""
    m = re.match(r"(\d+)-(\d+)(_\d+)?\.jpg", os.path.basename(path))
    return m.group(1) + (m.group(3) or ""), int(m.group(2))


def read_train():
    rows = [l.strip().split(",") for l in open(os.path.join(DIST, "train_label.csv")) if l.strip()]
    return [r[0] for r in rows], np.array([int(r[1]) for r in rows])


def read_test():
    return [l.strip() for l in open(os.path.join(DIST, "test.txt")) if l.strip()]


def load_images(paths, name):
    """全画像を uint8 配列 (N,256,256,3) にして work/ にキャッシュする。"""
    cache = os.path.join(WORK, f"{name}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    arr = np.zeros((len(paths), 256, 256, 3), np.uint8)
    for i, p in enumerate(paths):
        im = Image.open(os.path.join(IMG, p)).convert("RGB")
        if im.size != (256, 256):
            im = im.resize((256, 256), Image.BILINEAR)
        arr[i] = np.asarray(im)
    np.save(cache, arr)
    return arr


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.sc = None
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)), inplace=True)
        y = self.b2(self.c2(y))
        return F.relu(y + (x if self.sc is None else self.sc(x)), inplace=True)


class Net(nn.Module):
    """ゼロから学習する小型ResNet。最後は位置ごとのスコアの最大値と平均を使う。"""

    def __init__(self, w=(32, 64, 128, 192, 256), in_ch=3, hires=False):
        super().__init__()
        if hires:
            # 最初の縮小を1段減らし、細かい模様（端のわずかな弧、気泡との違い）を残す
            w = (16, 32, 64, 128, 192, 256)
            self.stem = nn.Sequential(nn.Conv2d(in_ch, w[0], 5, 2, 2, bias=False), nn.BatchNorm2d(w[0]), nn.ReLU(inplace=True))
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, 24, 5, 2, 2, bias=False), nn.BatchNorm2d(24), nn.ReLU(inplace=True),
                nn.Conv2d(24, w[0], 3, 2, 1, bias=False), nn.BatchNorm2d(w[0]), nn.ReLU(inplace=True),
            )
        layers, cin = [], w[0]
        for i, c in enumerate(w):
            layers += [Block(cin, c, 1 if i == 0 else 2), Block(c, c, 1)]
            cin = c
        self.body = nn.Sequential(*layers)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(cin * 2, 1)

    def forward(self, x):
        f = self.body(self.stem(x))
        f = torch.cat([f.amax((2, 3)), f.mean((2, 3))], 1)
        return self.fc(self.drop(f)).squeeze(1)


MEAN, STD = 0.66, 0.10


def to_input(x_u8):
    """uint8 (B,H,W,C) の Tensor -> 正規化済み float (B,C,H,W)。"""
    return (x_u8.permute(0, 3, 1, 2).float() / 255.0 - MEAN) / STD


def dihedral(x, k):
    """反転と90度回転の8通り。端で切れた穴のラベルが変わらない変換だけを使う。"""
    if k & 4:
        x = x.transpose(2, 3)
    if k & 1:
        x = x.flip(2)
    if k & 2:
        x = x.flip(3)
    return x


def random_zoom_out(x, zmax, p=0.5):
    """撮影距離の違いを模した縮小。画像を 1/z 倍に縮め、余白は鏡像で埋める。

    元の画像全体が必ず枠内に残るので、端で切れた穴も含めてラベルは変わらない。
    """
    B, dev = x.shape[0], x.device
    z = torch.exp(torch.empty(B, device=dev).uniform_(0, float(np.log(zmax))))
    z = torch.where(torch.rand(B, device=dev) < p, z, torch.ones_like(z))
    t = (torch.rand(B, 2, device=dev) * 2 - 1) * (z - 1).unsqueeze(1)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = z
    theta[:, 1, 1] = z
    theta[:, :, 2] = t
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def augment(x, zoom_out=1.0, cmin=0.7):
    """x: 正規化前 float (B,C,H,W) 0..1。反転・回転、縮小、明るさ・コントラスト・色味・ぼかし・ノイズ。"""
    B = x.shape[0]
    k = torch.randint(0, 8, (1,)).item()
    x = dihedral(x, k)
    dev = x.device
    if zoom_out > 1:
        x = random_zoom_out(x, zoom_out)
    gamma = torch.exp(torch.empty(B, 1, 1, 1, device=dev).uniform_(-0.35, 0.35))
    x = x.clamp(1e-4, 1).pow(gamma)
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * torch.empty(B, 1, 1, 1, device=dev).uniform_(cmin, 1.4) + m
    x = x + torch.empty(B, 1, 1, 1, device=dev).uniform_(-0.12, 0.12)
    x = x * torch.empty(B, 3, 1, 1, device=dev).uniform_(0.93, 1.07)
    if torch.rand(1).item() < 0.3:
        x = F.avg_pool2d(x, 3, 1, 1)
    x = x + torch.randn_like(x) * torch.empty(B, 1, 1, 1, device=dev).uniform_(0, 0.03)
    return x.clamp(0, 1)
