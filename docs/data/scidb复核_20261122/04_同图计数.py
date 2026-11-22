# -*- coding: utf-8 -*-
"""统计 SciDB 各测试条件与训练集的“同一图像”数量（dHash 候选 + NCC 复核）"""
import os, json
import numpy as np
from PIL import Image

BASE = r"D:\新版柑橘\复现_论文\scidb_det\images"
BC = r"D:\新版柑橘\方法路线扩展_20261031\统一检测数据\images"
splits = {"testA": os.path.join(BASE, "test"), "testB": os.path.join(BC, "testB"), "testC": os.path.join(BC, "testC")}
TRAIN = os.path.join(BASE, "train")

def dhash(path):
    a = np.asarray(Image.open(path).convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return int("".join("1" if b else "0" for b in (a[:, 1:] > a[:, :-1]).flatten()), 2)

def ncc(path_a, path_b, size=(160, 120)):
    a = np.asarray(Image.open(path_a).convert("RGB").resize(size, Image.BILINEAR), dtype=np.float32)
    b = np.asarray(Image.open(path_b).convert("RGB").resize(size, Image.BILINEAR), dtype=np.float32)
    x, y = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))

train_files = sorted(f for f in os.listdir(TRAIN) if f.lower().endswith((".jpg", ".jpeg", ".png")))
train_h = np.array([dhash(os.path.join(TRAIN, f)) for f in train_files], dtype=np.uint64)

def popcount(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)

res = {}
for sp, d in splits.items():
    qs = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    same, near, checked = [], [], 0
    for q in qs:
        qh = np.uint64(dhash(os.path.join(d, q)))
        dist = popcount(np.bitwise_xor(np.full_like(train_h, qh), train_h))
        k = int(dist.argmin())
        if int(dist[k]) <= 4:
            checked += 1
            v = ncc(os.path.join(d, q), os.path.join(TRAIN, train_files[k]))
            if v >= 0.999:
                same.append((q, train_files[k], round(v, 5)))
            elif v >= 0.99:
                near.append((q, train_files[k], round(v, 5)))
    res[sp] = {"样本数": len(qs), "dHash候选": checked, "同一图像(NCC>=0.999)": len(same),
               "高度相似(0.99<=NCC<0.999)": len(near), "同一图像样例": same[:6]}
print(json.dumps(res, ensure_ascii=False, indent=1))
with open(r"D:\新版柑橘\_tmp_verify\scidb_same_image.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
