# -*- coding: utf-8 -*-
"""
M0 跨域评测
----------------
用"干净背景(Mendeley 柑橘叶)"训练好的分类器, 直接在 CitrusUAT
(orange 品种, 异域数据, qPCR 确诊 HLB) 上测, 量化跨数据集/跨品种掉点。

只测与训练类别重叠的类:
    CitrusUAT "Healthy"        -> mendeley "healthy"
    CitrusUAT "HLB/Huanglong"  -> mendeley "greening"(黄龙病, 同病)
(CitrusUAT 的 Greasy spot / 各缺素 / 虫害 训练集里没有, 不计入严格评测,
 但会单列"最易被误判成哪个训练类", 观察田间未知类的行为。)

报告:
  1) 严格 5-way top-1 accuracy(模型可输出全部 5 类, 最真实)
  2) 每个源类的预测去向(看掉点掉到哪)
  3) 受限 2-way(只在 healthy / greening 间比), 排除其它类干扰
对照基线(Mendeley val, 干净同分布): healthy F1=1.00, greening F1≈0.94
"""
import os, glob, json, argparse
from collections import Counter, defaultdict
import numpy as np
import torch, torch.nn as nn
from torchvision import transforms, models
from PIL import Image

ROOT = r"C:\Users\Windows11\Desktop\新版柑橘"
RESULTS = os.path.join(ROOT, "复现_baseline", "results")
MENDELEY = ["black_spot", "canker", "greening", "healthy", "melanose"]
IDX = {c: i for i, c in enumerate(MENDELEY)}
MODELS = ["efficientnet_b0", "resnet50", "mobilenet_v3_large"]

tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def build_model(name):
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 5)
    elif name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 5)
    elif name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 5)
    sd = torch.load(os.path.join(RESULTS, f"best_{name}.pt"), map_location="cpu")
    m.load_state_dict(sd)
    return m


def map_folder(fn):
    """CitrusUAT 文件夹名 -> mendeley 重叠类; 非重叠返回 None"""
    f = fn.lower()
    if "healthy" in f:
        return "healthy"
    if "hlb" in f or "huanglong" in f:
        return "greening"
    return None


def scan_images(d):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    return [p for p in glob.glob(os.path.join(d, "**", "*"), recursive=True)
            if p.lower().endswith(exts)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_root", required=True, help="CitrusUAT 解压后, 含各类子文件夹的根目录")
    ap.add_argument("--include_unknown", action="store_true", help="同时统计非重叠类被误判去向")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 收集重叠类样本 (path, mendeley_label, src_folder)
    overlap, unknown = [], []
    for d in sorted(os.listdir(args.test_root)):
        dp = os.path.join(args.test_root, d)
        if not os.path.isdir(dp):
            continue
        mc = map_folder(d)
        imgs = scan_images(dp)
        if mc is not None:
            for p in imgs:
                overlap.append((p, mc, d))
        else:
            for p in imgs:
                unknown.append((p, None, d))

    log = []
    def L(s): print(s, flush=True); log.append(str(s))

    L(f"device={device}  test_root={args.test_root}")
    L(f"重叠类样本={len(overlap)}  分布={dict(Counter([m for _,m,_ in overlap]))}")
    L(f"非重叠(未知)类样本={len(unknown)}  来源={dict(Counter([s for *_,s in unknown]))}")
    L("对照基线(Mendeley val, 干净同分布): healthy≈100%, greening≈94%\n")

    # 预加载重叠类图像张量
    def load_tensors(items):
        xs, meta = [], []
        for p, mc, src in items:
            try:
                xs.append(tf(Image.open(p).convert("RGB")))
                meta.append((mc, src))
            except Exception as e:
                pass
        return torch.stack(xs) if xs else torch.empty(0), meta

    X_ov, meta_ov = load_tensors(overlap)
    summary = {}

    for name in MODELS:
        m = build_model(name).to(device).eval()
        # ---- 重叠类推理 ----
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_ov), 64):
                out = m(X_ov[i:i+64].to(device))
                preds.extend(out.argmax(1).cpu().numpy().tolist())
        pred_names = [MENDELEY[i] for i in preds]
        gt_names = [mc for mc, _ in meta_ov]

        # 1) 严格 5-way top-1
        correct = sum(p == g for p, g in zip(pred_names, gt_names))
        acc5 = correct / len(gt_names) if gt_names else 0.0

        # 2) 每源类去向
        byclass = defaultdict(Counter)
        for (g, p) in zip(gt_names, pred_names):
            byclass[g][p] += 1

        # 3) 受限 2-way (仅 healthy/greening 两个 logit 比大小)
        gi, hi = IDX["greening"], IDX["healthy"]
        with torch.no_grad():
            corr2, tot2 = 0, 0
            for i in range(0, len(X_ov), 64):
                out = m(X_ov[i:i+64].to(device))
                sub = out[:, [gi, hi]]  # 列0=greening,列1=healthy
                choice = sub.argmax(1).cpu().numpy()  # 0->greening,1->healthy
                for j, c in enumerate(choice):
                    g = meta_ov[i+j][0]
                    pred2 = "greening" if c == 0 else "healthy"
                    corr2 += int(pred2 == g); tot2 += 1
        acc2 = corr2 / tot2 if tot2 else 0.0

        L(f"===== {name} =====")
        L(f"[严格5-way] 重叠类 top-1 accuracy = {acc5*100:.2f}%  ({correct}/{len(gt_names)})")
        for g in ["healthy", "greening"]:
            if g in byclass:
                tot = sum(byclass[g].values())
                hit = byclass[g][g]
                L(f"   源类 {g:9s} n={tot:3d}  正确={hit:3d} ({hit/tot*100:.1f}%)  预测去向={dict(byclass[g])}")
        L(f"[受限2-way healthy/greening] accuracy = {acc2*100:.2f}%  ({corr2}/{tot2})")

        # 未知类去向(可选)
        if args.include_unknown and unknown:
            Xu, meta_u = load_tensors(unknown)
            upred = []
            with torch.no_grad():
                for i in range(0, len(Xu), 64):
                    out = m(Xu[i:i+64].to(device))
                    upred.extend(out.argmax(1).cpu().numpy().tolist())
            un_by = defaultdict(Counter)
            for (mc_src, pi) in zip([s for _, s in meta_u], upred):
                un_by[mc_src][MENDELEY[pi]] += 1
            L("   [未知类被误判去向]")
            for src in sorted(un_by):
                L(f"      {src:22s} -> {dict(un_by[src])}")
        L("")

        summary[name] = {"strict5_acc": acc5, "restricted2_acc": acc2,
                         "n_overlap": len(gt_names),
                         "byclass": {g: dict(byclass[g]) for g in byclass}}

    out_json = os.path.join(RESULTS, "m0_crossdomain_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(RESULTS, "m0_crossdomain_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    L("Saved -> " + out_json)


if __name__ == "__main__":
    main()
