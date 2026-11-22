# -*- coding: utf-8 -*-
"""校验 PCR→testB 映射是否一对一，并输出可视化对照"""
import zipfile, io, os, json
import numpy as np
from PIL import Image
from collections import Counter

IMG = r"D:\新版柑橘\方法路线扩展_20261031\统一检测数据\images\testB"
PCR = r"D:\病虫害论文\福建 SciDB 柑橘田间集\PCRs.zip"
OUT = r"D:\新版柑橘\数据集复核_20261122"

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

pairs, used, scores = [], Counter(), []
z = zipfile.ZipFile(PCR)
shown = 0
for name in ["PCR+.zip", "PCR-.zip"]:
    inner = zipfile.ZipFile(io.BytesIO(z.read(name)))
    for n in inner.namelist():
        if not n.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        im = Image.open(io.BytesIO(inner.read(n)))
        qh = np.uint64(dhash_img(im))
        dist = popcount(np.bitwise_xor(np.full_like(hashes, qh), hashes))
        order = np.argsort(dist)[:3]
        best = None
        for k in order:
            if int(dist[k]) > 4:
                break
            s = ncc(im, Image.open(os.path.join(IMG, files[int(k)])))
            if s >= 0.99:
                best = (int(k), float(s))
                break
        if best:
            k, s = best
            used[files[k]] += 1
            pairs.append((name, os.path.basename(n), files[k], round(s, 5)))
            scores.append(s)
            if name == "PCR+.zip" and shown < 3 and used[files[k]] == 1:
                a = im.convert("RGB").resize((320, 240))
                b = Image.open(os.path.join(IMG, files[k])).convert("RGB").resize((320, 240))
                c = Image.new("RGB", (640, 240), "white")
                c.paste(a, (0, 0)); c.paste(b, (320, 0))
                c.save(os.path.join(OUT, f"PCR对照_{shown+1}.png"))
                shown += 1

dup = {k: v for k, v in used.items() if v > 1}
print(json.dumps({
    "匹配对数": len(pairs),
    "被使用testB图像数": len(used),
    "一对多冲突数": len(dup),
    "NCC最小值": round(min(scores), 5) if scores else None,
    "NCC中位数": round(float(np.median(scores)), 5) if scores else None,
    "冲突样例": list(dup.items())[:5],
}, ensure_ascii=False, indent=1))
with open(os.path.join(OUT, "pcr_映射一对一校验.json"), "w", encoding="utf-8") as f:
    json.dump({"匹配对数": len(pairs), "被使用testB图像数": len(used), "一对多冲突": dup,
               "样例": pairs[:10]}, f, ensure_ascii=False, indent=1)
