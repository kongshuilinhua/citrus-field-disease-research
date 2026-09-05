# -*- coding: utf-8 -*-
"""
复现 [55] PlantCLR《对比自监督预训练做可泛化植病检测》的核心思路(SimCLR 式)。
- 自监督预训练：在无标签柑橘图池(Mendeley + 4类橙叶 + CitrusUAT, 约 2600 张)上做 SimCLR 对比学习
- 线性探针评测：冻结主干, 只训一个线性分类头(Mendeley 5类 train), 在 Mendeley val 上测
- 对照：SimCLR 预训练 vs ImageNet 预训练 vs 随机初始化 —— 看自监督表示是否有用
主干 ResNet18, 温度 0.5, NT-Xent 损失。
"""
import os, glob, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

ROOT = r"C:\Users\Windows11\Desktop\新版柑橘"
# Mendeley 只允许 train 进入无标签预训练池，避免线性探针 val 泄漏。
POOLS = [os.path.join(ROOT, "数据集_整理", "源域_Mendeley叶5类", "train"),
         os.path.join(ROOT, "数据集_整理", "源域_4类增广"),
         os.path.join(ROOT, "复现_baseline", "field_test_CitrusUAT")]
MEND = os.path.join(ROOT, "数据集_整理", "源域_Mendeley叶5类")   # 带标签, 线性探针用
RUN_ROOT = os.environ.get("CITRUS_RUN_ROOT")
OUT = os.path.join(RUN_ROOT, "simclr") if RUN_ROOT else os.path.join(ROOT, "复现_论文", "results_simclr")
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def all_images(dirs):
    ps = []
    for d in dirs:
        ps += [p for p in glob.glob(os.path.join(d, "**", "*"), recursive=True)
               if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    return sorted(set(ps))


sim_aug = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.4, 1.0)), transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(23, (0.1, 2.0))], p=0.5),
    transforms.ToTensor(), transforms.Normalize(mean, std)])
eval_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                              transforms.ToTensor(), transforms.Normalize(mean, std)])


class TwoView(Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        return sim_aug(im), sim_aug(im)


class SimCLRNet(nn.Module):
    def __init__(self):
        super().__init__()
        b = models.resnet18(weights=None)
        self.dim = b.fc.in_features
        b.fc = nn.Identity(); self.enc = b
        self.proj = nn.Sequential(nn.Linear(self.dim, 512), nn.ReLU(), nn.Linear(512, 128))
    def forward(self, x):
        h = self.enc(x)
        return F.normalize(self.proj(h), dim=1)


def nt_xent(z1, z2, t=0.5):
    N = z1.size(0)
    z = torch.cat([z1, z2], 0)                       # 2N x d
    sim = z @ z.t() / t                              # 2N x 2N
    sim.fill_diagonal_(-9e15)
    targets = torch.arange(N, device=z.device)
    targets = torch.cat([targets + N, targets], 0)   # 正样本索引
    return F.cross_entropy(sim, targets)


@torch.no_grad()
def extract(enc, dl, device):
    enc.eval(); X, Y = [], []
    for x, y in dl:
        X.append(enc(x.to(device)).cpu()); Y.append(y)
    return torch.cat(X), torch.cat(Y)


def linear_probe(enc, device, tag, log):
    """冻结主干, 训线性头, 评测 Mendeley val"""
    tr = datasets.ImageFolder(os.path.join(MEND, "train"), eval_tf)
    va = datasets.ImageFolder(os.path.join(MEND, "val"), eval_tf)
    trX, trY = extract(enc, DataLoader(tr, 64), device)
    vaX, vaY = extract(enc, DataLoader(va, 64), device)
    clf = nn.Linear(trX.size(1), len(tr.classes)).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    trX, trY, vaX, vaY = trX.to(device), trY.to(device), vaX.to(device), vaY.to(device)
    for _ in range(200):
        clf.train(); opt.zero_grad(); loss = ce(clf(trX), trY); loss.backward(); opt.step()
    clf.eval()
    pred = clf(vaX).argmax(1).cpu().numpy()
    acc = accuracy_score(vaY.cpu().numpy(), pred)
    mf1 = f1_score(vaY.cpu().numpy(), pred, average="macro", zero_division=0)
    log(f"[线性探针·{tag}] Mendeley val: acc={acc*100:.2f}% macroF1={mf1*100:.2f}%")
    return acc, mf1


def main():
    set_seed(42); os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logs = []
    def L(s): print(s, flush=True); logs.append(str(s))

    pool = all_images(POOLS)
    L(f"device={device}  无标签预训练池={len(pool)} 张")

    # ---- 对照1: ImageNet 预训练主干的线性探针 ----
    enc_in = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1); enc_in.fc = nn.Identity()
    linear_probe(enc_in.to(device), device, "ImageNet初始化", L)
    # ---- 对照2: 随机初始化主干 ----
    enc_rand = models.resnet18(weights=None); enc_rand.fc = nn.Identity()
    linear_probe(enc_rand.to(device), device, "随机初始化", L)

    # ---- SimCLR 自监督预训练 ----
    net = SimCLRNet().to(device)
    dl = DataLoader(TwoView(pool), batch_size=128, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    EP = int(os.environ.get("CITRUS_EPOCHS", "60"))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EP)
    L(f"\n>>> SimCLR 预训练 {EP} epoch, batch=128")
    for ep in range(1, EP + 1):
        net.train(); tot = 0; nb = 0
        for v1, v2 in dl:
            v1, v2 = v1.to(device), v2.to(device)
            loss = nt_xent(net(v1), net(v2))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if ep % 10 == 0 or ep == 1:
            L(f"  ep{ep:02d} NT-Xent loss={tot/nb:.4f}")

    # ---- SimCLR 主干的线性探针 ----
    L("")
    linear_probe(net.enc.to(device), device, "SimCLR自监督", L)
    torch.save(net.enc.state_dict(), os.path.join(OUT, "simclr_resnet18_encoder.pt"))
    with open(os.path.join(OUT, "simclr_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(logs))
    L("Saved -> " + OUT)


if __name__ == "__main__":
    main()
