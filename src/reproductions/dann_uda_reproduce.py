# -*- coding: utf-8 -*-
"""
复现 [35] MSUN《实验室→田间 UDA》的核心思想（论文无官方代码 → 忠实重实现 DANN）
--------------------------------------------------------------------------
- 源域 SRC：Mendeley 柑橘叶 5 类（有标签，干净背景）
- 目标域 TGT：CitrusUAT（无标签，orange 异域）——标准转导式 UDA，只用图不用标签
- 方法：域对抗（DANN + 梯度反转 GRL）对齐源/目标特征
- 目的：看无监督域适应能否把 M0 实测的跨域掉点补回来
- 对照：同架构同轮数下 source-only（域权重=0） vs DANN，评测 CitrusUAT 重叠类
        (Healthy→healthy, HLB→greening，143 张) 的严格 5 类 + 2 类准确率
"""
import os, glob, random, json
import numpy as np
import torch, torch.nn as nn
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

ROOT = r"C:\Users\Windows11\Desktop\新版柑橘"
SRC_DIR = os.path.join(ROOT, "数据集_整理", "源域_Mendeley叶5类", "train")
TGT_DIR = os.path.join(ROOT, "复现_baseline", "field_test_CitrusUAT")   # 全部 CitrusUAT 作无标签目标域
RUN_ROOT = os.environ.get("CITRUS_RUN_ROOT")
OUT_DIR = os.path.join(RUN_ROOT, "dann") if RUN_ROOT else os.path.join(ROOT, "复现_论文", "results_dann")
MENDELEY = ["black_spot", "canker", "greening", "healthy", "melanose"]
IDX = {c: i for i, c in enumerate(MENDELEY)}
OVERLAP = {"Healthy": "healthy", "HLB": "greening"}   # CitrusUAT 文件夹 -> mendeley 类

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05), transforms.ToTensor(), transforms.Normalize(mean, std)])
eval_tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)])


def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class GRL(Function):
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb
        return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        return -ctx.lamb * g, None


class UnlabeledDS(Dataset):
    def __init__(self, root, tf):
        self.paths = [p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
                      if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        self.tf = tf
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), 0


class OverlapDS(Dataset):
    """CitrusUAT 重叠类带标签(仅评测用)"""
    def __init__(self, root, tf):
        self.items = []
        for folder, mc in OVERLAP.items():
            for p in glob.glob(os.path.join(root, folder, "*")):
                if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    self.items.append((p, IDX[mc]))
        self.tf = tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


class DANN(nn.Module):
    def __init__(self, ncls=5):
        super().__init__()
        b = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features, self.pool = b.features, b.avgpool
        self.cls = nn.Linear(1280, ncls)
        self.dom = nn.Sequential(nn.Linear(1280, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 2))
    def forward(self, x, lamb=0.0):
        f = self.pool(self.features(x)).flatten(1)
        return self.cls(f), self.dom(GRL.apply(f, lamb))


@torch.no_grad()
def eval_overlap(model, dl, device):
    model.eval(); preds, gts, g2 = [], [], []
    gi, hi = IDX["greening"], IDX["healthy"]
    for x, y in dl:
        out, _ = model(x.to(device))
        preds.extend(out.argmax(1).cpu().numpy().tolist()); gts.extend(y.numpy().tolist())
        sub = out[:, [gi, hi]].argmax(1).cpu().numpy()  # 0->greening 1->healthy
        for j, c in enumerate(sub):
            g2.append(int((gi if c == 0 else hi) == y[j].item()))
    acc5 = accuracy_score(gts, preds)
    acc2 = float(np.mean(g2))
    return acc5, acc2


def run(mode, epochs, device, log):
    """mode='source_only' or 'dann'"""
    set_seed(42)
    src = datasets.ImageFolder(SRC_DIR, train_tf)
    # 对齐 ImageFolder 类顺序到 MENDELEY
    assert src.classes == MENDELEY, f"class order mismatch: {src.classes}"
    tgt = UnlabeledDS(TGT_DIR, train_tf)
    ov = OverlapDS(TGT_DIR, eval_tf)
    src_dl = DataLoader(src, batch_size=32, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
    tgt_dl = DataLoader(tgt, batch_size=32, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
    ov_dl = DataLoader(ov, batch_size=64, shuffle=False, num_workers=0)

    model = DANN(5).to(device)
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    n_iter = max(len(src_dl), len(tgt_dl))
    total = epochs * n_iter

    for ep in range(1, epochs + 1):
        model.train()
        si, ti = iter(src_dl), iter(tgt_dl)
        for it in range(n_iter):
            try: xs, ys = next(si)
            except StopIteration: si = iter(src_dl); xs, ys = next(si)
            try: xt, _ = next(ti)
            except StopIteration: ti = iter(tgt_dl); xt, _ = next(ti)
            xs, ys, xt = xs.to(device), ys.to(device), xt.to(device)
            p = ((ep - 1) * n_iter + it) / total
            lamb = (2.0 / (1.0 + np.exp(-10 * p)) - 1.0) if mode == "dann" else 0.0
            ys_out, ds_out = model(xs, lamb)
            loss = ce(ys_out, ys)
            if mode == "dann":
                _, dt_out = model(xt, lamb)
                d_src = ce(ds_out, torch.zeros(xs.size(0), dtype=torch.long, device=device))
                d_tgt = ce(dt_out, torch.ones(xt.size(0), dtype=torch.long, device=device))
                loss = loss + d_src + d_tgt
            opt.zero_grad(); loss.backward(); opt.step()
        a5, a2 = eval_overlap(model, ov_dl, device)
        log(f"  [{mode}] ep{ep:02d} lamb={lamb:.2f} 目标域重叠 严格5类={a5*100:.1f}% 2类={a2*100:.1f}%")
    a5, a2 = eval_overlap(model, ov_dl, device)
    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return a5, a2, final_state


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logs = []
    def log(s): print(s, flush=True); logs.append(str(s))
    log(f"device={device}  源域={SRC_DIR}")
    log(f"目标域(无标签)图数={len(UnlabeledDS(TGT_DIR, eval_tf))}  评测重叠={len(OverlapDS(TGT_DIR, eval_tf))}")
    EP = int(os.environ.get("CITRUS_EPOCHS", "15"))
    log(f"\n>>> 配置A: source-only（域权重=0，同架构同轮数），{EP} epoch")
    so5, so2, so_state = run("source_only", EP, device, log)
    log(f"\n>>> 配置B: DANN 域对抗，{EP} epoch")
    da5, da2, da_state = run("dann", EP, device, log)
    log("\n===== 复现结论：DANN 能否补回跨域掉点 =====")
    log(f"source-only : 严格5类={so5*100:.1f}%  2类(healthy/greening)={so2*100:.1f}%")
    log(f"DANN 域适应  : 严格5类={da5*100:.1f}%  2类(healthy/greening)={da2*100:.1f}%")
    log(f"提升        : 严格5类 {(da5-so5)*100:+.1f}pt   2类 {(da2-so2)*100:+.1f}pt")
    with open(os.path.join(OUT_DIR, "dann_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(logs))
    torch.save(so_state, os.path.join(OUT_DIR, "source_only_final.pt"))
    torch.save(da_state, os.path.join(OUT_DIR, "dann_final.pt"))
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": 42,
                "epochs": EP,
                "source_only": {"strict5_acc": so5, "restricted2_acc": so2},
                "dann": {"strict5_acc": da5, "restricted2_acc": da2},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
