# -*- coding: utf-8 -*-
"""SciDB 近重复（同叶/连拍）泄漏审计：dHash + 汉明距离"""
import os, json
from collections import Counter
import numpy as np
from PIL import Image

BASE = r"D:\新版柑橘\复现_论文\scidb_det"
BC = r"D:\新版柑橘\方法路线扩展_20261031\统一检测数据"
splits = {
    "train": os.path.join(BASE, "images", "train"),
    "val": os.path.join(BASE, "images", "val"),
    "testA": os.path.join(BASE, "images", "test"),
    "testB": os.path.join(BC, "images", "testB"),
    "testC": os.path.join(BC, "images", "testC"),
}
label_dirs = {
    "train": os.path.join(BASE, "labels", "train"),
    "val": os.path.join(BASE, "labels", "val"),
    "testA": os.path.join(BASE, "labels", "test"),
    "testB": os.path.join(BC, "labels", "testB"),
    "testC": os.path.join(BC, "labels", "testC"),
}

def dhash(path):
    im = Image.open(path).convert("L").resize((9, 8), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return "".join("1" if b else "0" for b in bits)

def bits_to_int(s):
    return int(s, 2)

store = {}
for sp, d in splits.items():
    names, ints = [], []
    for f in sorted(os.listdir(d)):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        names.append(f)
        ints.append(bits_to_int(dhash(os.path.join(d, f))))
    store[sp] = (names, np.array(ints, dtype=np.uint64))
    print(sp, len(names), flush=True)

def popcount(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)

res = {}
for sp in ("val", "testA", "testB", "testC"):
    tr_names, tr_h = store["train"]
    q_names, q_h = store[sp]
    counts = {4: 0, 8: 0}
    worst = []
    for i in range(0, len(q_h), 64):
        block = q_h[i:i + 64]
        d = popcount(np.bitwise_xor(block[:, None], tr_h[None, :]))
        mn = d.min(axis=1)
        counts[4] += int((mn <= 4).sum())
        counts[8] += int((mn <= 8).sum())
        for j, v in enumerate(mn):
            if v <= 8:
                k = int(d[j].argmin())
                worst.append((int(v), q_names[i + j], tr_names[k]))
    worst.sort(key=lambda x: x[0])
    res[sp] = {"样本数": len(q_names), "dHash<=4 命中数": counts[4],
               "dHash<=8 命中数": counts[8], "最近样例": worst[:8]}

# 每个划分缺失的类别
missing = {}
for sp, d in label_dirs.items():
    cs = set()
    for f in os.listdir(d):
        if f.endswith(".txt"):
            for line in open(os.path.join(d, f), encoding="utf-8", errors="ignore"):
                p = line.split()
                if len(p) >= 5:
                    cs.add(int(float(p[0])))
    missing[sp] = sorted(set(range(15)) - cs)
res["缺失类别id"] = missing

with open(r"D:\新版柑橘\_tmp_verify\scidb_neardup.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
