# -*- coding: utf-8 -*-
"""
柑橘叶病分类 baseline 复现
---------------------------------
数据: Mendeley 柑橘叶 5 类 (black_spot / canker / greening / healthy / melanose), 实验室干净背景
模型: ImageNet 预训练骨干 (默认 EfficientNet-B0) 微调
目的: 坐实"实验室干净背景分类已接近饱和"的基线, 作为后续
      跨品种(橙橘->胡柚) + 实验室->田间 域适应的对照起点。
指标: Accuracy / 宏 F1 / 各类 P-R-F1 / 混淆矩阵
输出: results/ 下保存最优权重、日志、json 摘要
"""
import os, json, random, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# ---- 绝对路径, 避免中文相对路径 / cwd 带来的歧义 ----
ROOT = r"C:\Users\Windows11\Desktop\新版柑橘"
DATA_DIR = os.path.join(ROOT, "数据集_整理", "源域_Mendeley叶5类")
OUT_DIR  = os.path.join(ROOT, "复现_baseline", "results")


def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_model(name, num_classes):
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    else:
        raise ValueError("unknown model: " + name)
    return m


def evaluate(model, dl, device):
    model.eval(); preds, gts = [], []
    with torch.no_grad():
        for x, y in dl:
            out = model(x.to(device))
            preds.extend(out.argmax(1).cpu().numpy().tolist())
            gts.extend(y.numpy().tolist())
    return np.array(gts), np.array(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img", type=int, default=224)
    args = ap.parse_args()

    set_seed(42)
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    val_tf = transforms.Compose([
        transforms.Resize(int(args.img * 1.14)),
        transforms.CenterCrop(args.img),
        transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   val_tf)
    classes = train_ds.classes
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True,  num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False, num_workers=0, pin_memory=True)

    model = build_model(args.model, len(classes)).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    log_lines = []
    def log(s):
        print(s, flush=True); log_lines.append(str(s))

    log(f"device={device} | model={args.model} | classes={classes}")
    log(f"train={len(train_ds)} val={len(val_ds)}")
    # 打印各类样本数, 暴露长尾 (melanose 极少)
    from collections import Counter
    tr_cnt = Counter([classes[t] for _, t in train_ds.samples])
    va_cnt = Counter([classes[t] for _, t in val_ds.samples])
    log("train per-class: " + str(dict(tr_cnt)))
    log("val   per-class: " + str(dict(va_cnt)))

    best_f1, best_state, best_epoch = -1.0, None, -1
    for ep in range(1, args.epochs + 1):
        model.train(); tl, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); opt.step()
            tl += loss.item() * x.size(0); n += x.size(0)
        sched.step()
        gts, preds = evaluate(model, val_dl, device)
        acc = accuracy_score(gts, preds)
        mf1 = f1_score(gts, preds, average="macro", zero_division=0)
        log(f"ep{ep:02d} loss={tl/n:.4f} val_acc={acc:.4f} val_macroF1={mf1:.4f}")
        if mf1 > best_f1:
            best_f1, best_epoch = mf1, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ---- 用最优权重出最终报告 ----
    model.load_state_dict(best_state)
    gts, preds = evaluate(model, val_dl, device)
    acc = accuracy_score(gts, preds)
    mf1 = f1_score(gts, preds, average="macro", zero_division=0)
    rep = classification_report(gts, preds, target_names=classes, digits=4, zero_division=0)
    cm = confusion_matrix(gts, preds)

    log(f"\n===== BEST (epoch {best_epoch}) =====")
    log(f"Accuracy = {acc:.4f}   MacroF1 = {mf1:.4f}")
    log("\n" + rep)
    log("Confusion matrix (rows=true, cols=pred), labels order = " + " ".join(classes))
    log(str(cm))

    torch.save(best_state, os.path.join(OUT_DIR, f"best_{args.model}.pt"))
    with open(os.path.join(OUT_DIR, f"log_{args.model}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    with open(os.path.join(OUT_DIR, f"summary_{args.model}.json"), "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "classes": classes, "best_epoch": best_epoch,
                   "accuracy": acc, "macro_f1": mf1,
                   "train_n": len(train_ds), "val_n": len(val_ds),
                   "train_per_class": dict(tr_cnt), "val_per_class": dict(va_cnt)},
                  f, ensure_ascii=False, indent=2)
    log("\nSaved -> " + OUT_DIR)


if __name__ == "__main__":
    main()
