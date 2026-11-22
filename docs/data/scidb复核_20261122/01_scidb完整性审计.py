# -*- coding: utf-8 -*-
"""SciDB 数据完整性/泄漏/条件差异的可复核审计（只读，不修改原始数据）"""
import hashlib, json, os, random
from collections import Counter, defaultdict
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

def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

report = {}
hashes = {}
counts = {}
for sp, d in splits.items():
    files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    counts[sp] = len(files)
    for p in files:
        hashes.setdefault(md5(p), []).append((sp, os.path.basename(p)))
report["图像数"] = counts

# 跨划分重复
dups = {h: v for h, v in hashes.items() if len({s for s, _ in v}) > 1}
report["跨划分完全重复组数"] = len(dups)
report["跨划分重复样例"] = [
    {"md5": h, "出现": v} for h, v in list(dups.items())[:10]
]

# 标签类别分布与非法 id
cls_all = Counter()
invalid = 0
for sp, d in label_dirs.items():
    for f in os.listdir(d):
        if not f.endswith(".txt"):
            continue
        with open(os.path.join(d, f), "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 5:
                    continue
                cid = int(float(parts[0]))
                cls_all[(sp, cid)] += 1
                if cid < 0 or cid > 14:
                    invalid += 1
report["非法类别id行数"] = invalid
report["每划分类别数"] = {sp: len({c for s, c in cls_all if s == sp}) for sp in splits}
report["类别id范围"] = {"min": min(c for _, c in cls_all), "max": max(c for _, c in cls_all)}

# 条件差异：亮度/饱和度/尺寸（随机抽样，固定种子）
random.seed(0)
cond = {}
for sp in ("testA", "testB", "testC"):
    d = splits[sp]
    files = sorted(os.listdir(d))
    pick = random.sample(files, min(60, len(files)))
    L, S = [], []
    for f in pick:
        im = Image.open(os.path.join(d, f)).convert("RGB").resize((256, 256))
        px = list(im.getdata())
        lum = [0.299 * r + 0.587 * g + 0.114 * b for r, g, b in px]
        mx = [max(p) for p in px]
        mn = [min(p) for p in px]
        sat = [(a - b) / a if a else 0 for a, b in zip(mx, mn)]
        L.append(sum(lum) / len(lum))
        S.append(sum(sat) / len(sat))
    im0 = Image.open(os.path.join(d, files[0]))
    cond[sp] = {
        "亮度均值": round(sum(L) / len(L), 2),
        "饱和度均值": round(sum(S) / len(S), 4),
        "示例尺寸": list(im0.size),
    }
report["条件差异"] = cond

with open(r"D:\新版柑橘\_tmp_verify\scidb_audit.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=1)
print(json.dumps(report, ensure_ascii=False, indent=1))
