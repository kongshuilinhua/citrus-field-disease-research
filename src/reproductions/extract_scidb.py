# -*- coding: utf-8 -*-
"""
把福建 SciDB 柑橘田间检测集(嵌套 zip)解压成 ultralytics YOLO 检测标准结构。
- 训练: A~E 各 *-train 合并 -> train ; 各 *-val 合并 -> val
- 测试: testA -> test
- 标注已是 YOLO 格式(class cx cy w h), 直接沿用; 15 类见 label_object.json
输出: 复现_论文/scidb_det/{images,labels}/{train,val,test}/  + data.yaml
"""
import zipfile, os, io, json

BASE = r"D:\病虫害论文\福建 SciDB 柑橘田间集"
OUT = r"C:\Users\Windows11\Desktop\新版柑橘\复现_论文\scidb_det"
CLASSES = ["S0 others", "S1 Healthy fruit", "S2 Healthy leaf", "S3 Blotchy mottling",
           "S4 Red nose fruit", "S5 Zinc deficiency", "S6 Vein yellowing", "S7 Uniform yellowing",
           "S8 Magnesium deficiency", "S9 Boron deficiency", "S10 Anthracnose",
           "S11 Citrus greasy spot", "S14 Canker", "S13 Sooty mould", "S12 Algal spot and moss"]


def ensure(*p):
    d = os.path.join(*p); os.makedirs(d, exist_ok=True); return d


def dump_inner(inner_zip, split, domain_tag):
    """把一个内层 zip 里的 images/labels 拷进 OUT/{images,labels}/{split}, 文件名加域前缀防撞"""
    n_img = 0
    img_names = {}   # stem -> ext
    for name in inner_zip.namelist():
        if name.endswith("/"):
            continue
        low = name.lower()
        parts = name.split("/")
        fn = parts[-1]
        stem, ext = os.path.splitext(fn)
        new = f"{domain_tag}_{stem}"
        if low.endswith((".jpg", ".jpeg", ".png")):
            dst = os.path.join(ensure(OUT, "images", split), new + ext)
            with open(dst, "wb") as f:
                f.write(inner_zip.read(name))
            n_img += 1
        elif low.endswith(".txt"):
            dst = os.path.join(ensure(OUT, "labels", split), new + ".txt")
            with open(dst, "wb") as f:
                f.write(inner_zip.read(name))
    return n_img


def main():
    plan = [
        ("trains.zip", ["A-train.zip", "B-train.zip", "C-train.zip", "D-train.zip", "E-train.zip"], "train"),
        ("trains.zip", ["A-val.zip", "B-val.zip", "C-val.zip", "D-val.zip", "E-val.zip"], "val"),
        ("tests.zip", ["testA.zip"], "test"),
    ]
    counts = {}
    for outer, inners, split in plan:
        z = zipfile.ZipFile(os.path.join(BASE, outer))
        for inner in inners:
            iz = zipfile.ZipFile(io.BytesIO(z.read(inner)))
            tag = inner.replace(".zip", "").replace("-", "")
            n = dump_inner(iz, split, tag)
            counts[inner] = n
            print(f"  {inner} -> {split}: {n} 图", flush=True)
    # data.yaml
    yaml = (f"path: {OUT}\n"
            f"train: images/train\nval: images/val\ntest: images/test\n"
            f"nc: {len(CLASSES)}\n"
            f"names: {CLASSES}\n")
    with open(os.path.join(OUT, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml)
    for split in ["train", "val", "test"]:
        d = os.path.join(OUT, "images", split)
        print(f"{split}: {len(os.listdir(d)) if os.path.isdir(d) else 0} 图")
    print("data.yaml ->", os.path.join(OUT, "data.yaml"))


if __name__ == "__main__":
    main()
