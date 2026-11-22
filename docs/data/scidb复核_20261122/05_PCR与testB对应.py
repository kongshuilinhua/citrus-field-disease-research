# -*- coding: utf-8 -*-
"""建立 PCR 结果与 testB 图像的对应关系，并与症状类别标签交叉统计"""
import zipfile, io, os, json
import numpy as np
from PIL import Image
from collections import Counter, defaultdict

IMG = r"D:\新版柑橘\方法路线扩展_20261031\统一检测数据\images\testB"
LBL = r"D:\新版柑橘\方法路线扩展_20261031\统一检测数据\labels\testB"
PCR = r"D:\病虫害论文\福建 SciDB 柑橘田间集\PCRs.zip"
NAMES = ['S0 others', 'S1 Healthy fruit', 'S2 Healthy leaf', 'S3 Blotchy mottling', 'S4 Red nose fruit',
         'S5 Zinc deficiency', 'S6 Vein yellowing', 'S7 Uniform yellowing', 'S8 Magnesium deficiency',
         'S9 Boron deficiency', 'S10 Anthracnose', 'S11 Citrus greasy spot', 'S14 Canker',
         'S13 Sooty mould', 'S12 Algal spot and moss']

def dhash_img(im):
    a = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return int("".join("1" if x else "0" for x in (a[:, 1:] > a[:, :-1]).flatten()), 2)

def popcount(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)

def ncc(im_a, im_b, size=(160, 120)):
    a = np.asarray(im_a.convert("RGB").resize(size, Image.BILINEAR), dtype=np.float32)
    b = np.asarray(im_b.convert("RGB").resize(size, Image.BILINEAR), dtype=np.float32)
    x, y = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))

files = sorted(f for f in os.listdir(IMG) if f.lower().endswith((".jpg", ".jpeg", ".png")))
hashes = np.array([dhash_img(Image.open(os.path.join(IMG, f))) for f in files], dtype=np.uint64)

def cls_of(f):
    p = os.path.join(LBL, os.path.splitext(f)[0] + ".txt")
    if not os.path.exists(p):
        return None
    ids = []
    for line in open(p, encoding="utf-8", errors="ignore"):
        t = line.split()
        if len(t) >= 5:
            ids.append(int(float(t[0])))
    return ids[0] if ids else None

z = zipfile.ZipFile(PCR)
pcr_of = {}
unmatched = []
for name in ["PCR+.zip", "PCR-.zip"]:
    label = "PCR+" if "+" in name else "PCR-"
    inner = zipfile.ZipFile(io.BytesIO(z.read(name)))
    for n in inner.namelist():
        if not n.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        im = Image.open(io.BytesIO(inner.read(n)))
        qh = np.uint64(dhash_img(im))
        dist = popcount(np.bitwise_xor(np.full_like(hashes, qh), hashes))
        k = int(dist.argmin())
        if int(dist[k]) <= 4 and ncc(im, Image.open(os.path.join(IMG, files[k]))) >= 0.99:
            pcr_of[files[k]] = (label, os.path.basename(n))
        else:
            unmatched.append((label, os.path.basename(n), int(dist[k])))

tab = defaultdict(Counter)
for f in files:
    c = cls_of(f)
    lab = pcr_of.get(f, ("未匹配", ""))[0]
    tab[lab][NAMES[c] if c is not None else "无标签"] += 1

out = {
    "testB图像数": len(files),
    "PCR匹配数": len(pcr_of),
    "PCR+匹配": sum(1 for v in pcr_of.values() if v[0] == "PCR+"),
    "PCR-匹配": sum(1 for v in pcr_of.values() if v[0] == "PCR-"),
    "未匹配数": len(unmatched),
    "未匹配样例": unmatched[:8],
    "交叉表": {k: dict(v) for k, v in tab.items()},
}
print(json.dumps(out, ensure_ascii=False, indent=1))
with open(r"D:\新版柑橘\数据集复核_20261122\pcr_testB映射.json", "w", encoding="utf-8") as fp:
    json.dump(out, fp, ensure_ascii=False, indent=1)
