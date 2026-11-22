# -*- coding: utf-8 -*-
"""对疑似近重复的 train/test 配对做像素级复核，并输出并排对照图"""
import os
import numpy as np
from PIL import Image

BASE = r"D:\新版柑橘\复现_论文\scidb_det\images"
BC = r"D:\新版柑橘\方法路线扩展_20261031\统一检测数据\images"
OUT = r"D:\新版柑橘\_tmp_verify"

pairs = [
    ("testA_t0_29.jpg", os.path.join(BASE, "test"), "Btrain_3-HLB-1-185.jpg", os.path.join(BASE, "train")),
    ("testA_t3_26.jpg", os.path.join(BASE, "test"), "Ctrain_3-HLB-9-111.jpg", os.path.join(BASE, "train")),
    ("0_2_1_172.JPG", os.path.join(BC, "testC"), "Ctrain_3-HLB-9-221.jpg", os.path.join(BASE, "train")),
    ("15_2_1_1.JPG", os.path.join(BC, "testC"), "Ctrain_3-HLB-9-470.jpg", os.path.join(BASE, "train")),
    ("t0_1_23.jpg", os.path.join(BC, "testB"), "Ctrain_3-HLB-9-594.jpg", os.path.join(BASE, "train")),
]

def load(p, size=(320, 240)):
    return Image.open(p).convert("RGB").resize(size, Image.BILINEAR)

rows = []
for qn, qd, tn, td in pairs:
    a, b = load(os.path.join(qd, qn)), load(os.path.join(td, tn))
    aa = np.asarray(a, dtype=np.float32) / 255.0
    bb = np.asarray(b, dtype=np.float32) / 255.0
    # 归一化互相关（去均值后）
    x = aa.ravel() - aa.mean()
    y = bb.ravel() - bb.mean()
    ncc = float((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))
    mse = float(((aa - bb) ** 2).mean())
    rows.append((qn, tn, round(ncc, 4), round(mse, 4)))
    canvas = Image.new("RGB", (640, 240), "white")
    canvas.paste(a, (0, 0))
    canvas.paste(b, (320, 0))
    canvas.save(os.path.join(OUT, f"pair_{len(rows)}.png"))

print("query | train | NCC | MSE")
for r in rows:
    print(" | ".join(map(str, r)))
