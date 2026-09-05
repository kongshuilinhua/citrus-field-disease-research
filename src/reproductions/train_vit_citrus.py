# -*- coding: utf-8 -*-
"""
复现 [21]《多阶段深度集成 + ViT 柑橘病害分类》的核心分支：ViT 分类。
说明：论文是"多阶段集成 + ViT"，代码未开源；本脚本复现其**ViT 主干分类**这一核心，
      在本地同源的 4 类橙叶(源域_4类增广: black_spot/canker/greening/healthy)上微调。
      如已下载论文原 Kaggle 数据(橙2240+柠208)到 data_kaggle_citrus/，把 DATA 改过去即可。
模型：torchvision ViT-B/16 (ImageNet 预训练) 微调。指标：Accuracy / 宏 F1 / 各类 F1。
"""
import os, json, random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, f1_score, accuracy_score

ROOT = r"C:\Users\Windows11\Desktop\新版柑橘"
DATA = os.path.join(ROOT, "数据集_整理", "源域_4类增广")   # 有 train/ 与 test/
OUT = os.path.join(ROOT, "复现_论文", "results_vit")


def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def main():
    set_seed(42); os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tf_tr = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                                transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                                transforms.ToTensor(), transforms.Normalize(mean, std)])
    tf_te = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                transforms.ToTensor(), transforms.Normalize(mean, std)])
    tr = datasets.ImageFolder(os.path.join(DATA, "train"), tf_tr)
    te = datasets.ImageFolder(os.path.join(DATA, "test"), tf_te)
    classes = tr.classes
    tr_dl = DataLoader(tr, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    te_dl = DataLoader(te, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads.head = nn.Linear(model.heads.head.in_features, len(classes))
    model = model.to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)   # ViT 用更小 lr
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)

    logs = []
    def L(s): print(s, flush=True); logs.append(str(s))
    L(f"device={device} classes={classes} train={len(tr)} test={len(te)}")

    best_f1, best_state, best_ep = -1, None, -1
    for ep in range(1, 26):
        model.train()
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        sched.step()
        model.eval(); preds, gts = [], []
        with torch.no_grad():
            for x, y in te_dl:
                preds.extend(model(x.to(device)).argmax(1).cpu().numpy().tolist()); gts.extend(y.numpy().tolist())
        acc = accuracy_score(gts, preds); mf1 = f1_score(gts, preds, average="macro", zero_division=0)
        L(f"ep{ep:02d} test_acc={acc:.4f} macroF1={mf1:.4f}")
        if mf1 > best_f1:
            best_f1, best_ep = mf1, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state); model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for x, y in te_dl:
            preds.extend(model(x.to(device)).argmax(1).cpu().numpy().tolist()); gts.extend(y.numpy().tolist())
    L(f"\n===== BEST ep{best_ep} =====")
    L(f"Accuracy={accuracy_score(gts,preds):.4f} MacroF1={f1_score(gts,preds,average='macro',zero_division=0):.4f}")
    L("\n" + classification_report(gts, preds, target_names=classes, digits=4, zero_division=0))
    with open(os.path.join(OUT, "vit_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(logs))
    torch.save(best_state, os.path.join(OUT, "best_vit_b16.pt"))
    L("Saved -> " + OUT)


if __name__ == "__main__":
    main()
