# -*- coding: utf-8 -*-
"""
Probe-C0: 病理感知的“叶片双侧非对称拓扑 + 光照不变高频病斑”双流动态协同分类先导验证
严格对照矩阵（统一骨干 MobileNetV3-Large、统一优化器、统一20轮预算、源val选best、3随机种子 0/1/2）：
1. B1_SingleStream: 标准单流 224x224 裁图微调基线
2. B3_MultiScale: 多尺度随机裁剪训练强对照（对应 DECISIONS.md 规定的唯一证伪基线 B3）
3. B4_LocalGlobalConcat: 全局整叶裁图 + 中心局部放大斑块双尺度拼接（对应 GPT 建议的 B4 对照）
4. Ours_SymTex_Dynamic: 叶片双侧非对称空间差分 (BASD) + 局部亮度归一化高频残差流 (II-HFR) + 预测熵动态路由门控
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '2'

import json
import math
import time
import random
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_large
from sklearn.metrics import f1_score

ROOT = Path(r'D:\新版柑橘')
BASE_CACHE = ROOT / '基线补齐_20261026'
OLD_LIST = ROOT / '本周科研_20261014' / '裁图清单.json'
CITRUS_UAT_DIR = ROOT / '复现_baseline' / 'field_test_CitrusUAT'
MBV3_WEIGHT = Path(r'C:\Users\Windows11\.cache\torch\hub\checkpoints\mobilenet_v3_large-5c1a4163.pth')
OUT_DIR = ROOT / '多路线创新筛选_20261124' / '探针C_对称拓扑与高频病斑' / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 黄化与类黄龙病易混淆子类索引 (S3斑驳2, S5缺锌4, S6脉黄5, S7均黄6, S8缺镁7, S9缺硼8)
YELLOW_CLUSTER_IDS = [2, 4, 5, 6, 7, 8]
# 坏死与表生局部病斑类索引 (S10炭疽9, S11脂点10, S14溃疡11, S13煤烟12, S12藻斑13)
LESION_CLUSTER_IDS = [9, 10, 11, 12, 13]


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def gaussian_blur_2d(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - (kernel_size - 1) / 2.0
    g1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g1d = g1d / g1d.sum()
    g2d = (g1d[:, None] * g1d[None, :]).unsqueeze(0).unsqueeze(0).repeat(x.shape[1], 1, 1, 1)
    pad = kernel_size // 2
    return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode='reflect'), g2d, groups=x.shape[1])


def compute_illumination_invariant_hf(rgb_01: torch.Tensor) -> torch.Tensor:
    """
    计算局部亮度归一化多尺度高频残差图 II-HFR:
    对乘性曝光/色温偏移（如 testC 白板补光导致 S12 饱和度/明度剧变）保持数学近似不变。
    rgb_01: (B, 3, H, W) in [0, 1]
    """
    low_fine = gaussian_blur_2d(rgb_01, 5, 1.2)
    low_coarse = gaussian_blur_2d(rgb_01, 13, 3.5)
    lum = low_coarse.mean(dim=1, keepdim=True).clamp_min(0.12)
    hf_micro = (rgb_01 - low_fine) / lum
    hf_meso = (low_fine - low_coarse) / lum
    return torch.cat([hf_micro, hf_meso], dim=1).clamp(-3.0, 3.0)  # (B, 6, H, W)


class HighFreqDetailEncoder(nn.Module):
    """轻量高频病斑微观纹理编码器 (~0.09M 参数)，捕获溃疡火山口隆起、脂点、藻斑等高频边缘与晕环。"""
    def __init__(self, in_ch: int = 6, out_dim: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, hf_map: torch.Tensor) -> torch.Tensor:
        f = self.net(hf_map)  # (B, out_dim, 14, 14)
        # 同时聚合均值能量与局部峰值响应（针对极小点状病斑）
        return 0.5 * f.mean(dim=(2, 3)) + 0.5 * f.amax(dim=(2, 3))


class PathologyModel(nn.Module):
    def __init__(self, arch: str, num_classes: int = 14):
        super().__init__()
        assert arch in ['B1_SingleStream', 'B3_MultiScale', 'B4_LocalGlobalConcat', 'Ours_SymTex_Dynamic']
        self.arch = arch
        base = mobilenet_v3_large(weights=None)
        base.load_state_dict(torch.load(MBV3_WEIGHT, map_location='cpu', weights_only=True))
        self.features = base.features          # output: (B, 960, 7, 7)
        self.neck_proj = nn.Sequential(
            nn.Linear(960, 384),
            nn.LayerNorm(384),
            nn.Hardswish(inplace=True),
        )

        if arch in ['B1_SingleStream', 'B3_MultiScale']:
            self.cls_head = nn.Linear(384, num_classes)
        elif arch == 'B4_LocalGlobalConcat':
            self.cls_head = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(384 * 2, num_classes),
            )
        elif arch == 'Ours_SymTex_Dynamic':
            # 1. 双侧非对称拓扑分支 (编码跨中脉/跨轴非对称斑驳 vs 对称缺素黄化)
            self.asym_proj = nn.Sequential(
                nn.Linear(960 * 2, 256),
                nn.LayerNorm(256),
                nn.GELU(),
            )
            # 2. 光照不变高频残差病斑分支
            self.hf_enc = HighFreqDetailEncoder(in_ch=6, out_dim=192)
            self.hf_proj = nn.Sequential(
                nn.Linear(192, 256),
                nn.LayerNorm(256),
                nn.GELU(),
            )
            # 各子流独立辅助头 + 预测熵感知动态门控融合
            self.sem_head = nn.Linear(384, num_classes)
            self.asym_head = nn.Linear(256, num_classes)
            self.hf_head = nn.Linear(256, num_classes)
            # 门控网络接收 [f_sem_detach, sem_entropy, asym_energy, hf_energy] -> 3路自适应权重
            self.gate_net = nn.Sequential(
                nn.Linear(384 + 3, 64),
                nn.GELU(),
                nn.Linear(64, 3),
            )

    def forward_spatial(self, x_norm: torch.Tensor) -> torch.Tensor:
        return self.features(x_norm)  # (B, 960, 7, 7)

    def forward(self, rgb_01: torch.Tensor):
        mean = rgb_01.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = rgb_01.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x_norm = (rgb_01 - mean) / std

        fmap = self.forward_spatial(x_norm)  # (B, 960, 7, 7)
        f_sem = self.neck_proj(fmap.mean(dim=(2, 3)))

        if self.arch in ['B1_SingleStream', 'B3_MultiScale']:
            return self.cls_head(f_sem), {'fmap': fmap}

        if self.arch == 'B4_LocalGlobalConcat':
            # Center 128x128 local lesion region resized to 224x224
            c_patch = F.interpolate(x_norm[:, :, 48:176, 48:176], size=(224, 224), mode='bilinear', align_corners=False)
            f_loc = self.neck_proj(self.forward_spatial(c_patch).mean(dim=(2, 3)))
            logits = self.cls_head(torch.cat([f_sem, f_loc], dim=1))
            return logits, {'fmap': fmap}

        # Ours_SymTex_Dynamic:
        # 1) Canonical Bilateral Asymmetry Descriptor: 水平/垂直镜像对称差分 + 空间标准差异质性
        diff_h = torch.abs(fmap - torch.flip(fmap, dims=[3])).mean(dim=(2, 3))
        diff_v = torch.abs(fmap - torch.flip(fmap, dims=[2])).mean(dim=(2, 3))
        asym_raw = 0.5 * (diff_h + diff_v)
        spatial_std = fmap.std(dim=(2, 3))
        f_asym = self.asym_proj(torch.cat([asym_raw, spatial_std], dim=1))

        # 2) Illumination-Invariant High-Frequency Residual Stream
        hf_map = compute_illumination_invariant_hf(rgb_01)
        f_hf = self.hf_proj(self.hf_enc(hf_map))

        logits_sem = self.sem_head(f_sem)
        logits_asym = self.asym_head(f_asym)
        logits_hf = self.hf_head(f_hf)

        # Entropy-guided dynamic routing
        probs_sem = logits_sem.detach().softmax(dim=1)
        ent_sem = -(probs_sem * (probs_sem.clamp_min(1e-8)).log()).sum(dim=1, keepdim=True) / math.log(14.0)
        e_asym = f_asym.detach().norm(dim=1, keepdim=True) / 16.0
        e_hf = f_hf.detach().norm(dim=1, keepdim=True) / 16.0
        gate_in = torch.cat([f_sem.detach(), ent_sem, e_asym, e_hf], dim=1)
        weights = F.softmax(self.gate_net(gate_in), dim=1)  # (B, 3)

        final_logits = (
            weights[:, 0:1] * logits_sem
            + weights[:, 1:2] * logits_asym
            + weights[:, 2:3] * logits_hf
        )
        return final_logits, {
            'fmap': fmap,
            'logits_sem': logits_sem,
            'logits_asym': logits_asym,
            'logits_hf': logits_hf,
            'weights': weights,
        }


def gpu_augment_batch(x_01: torch.Tensor, multi_scale: bool = False) -> torch.Tensor:
    """快速 GPU 张量增强，确保四组模型除显式多尺度外拥有完全对等的随机翻转与色彩扰动。"""
    B, C, H, W = x_01.shape
    out = x_01.clone()
    # Random horizontal & vertical flips
    if random.random() < 0.5:
        out = torch.flip(out, dims=[3])
    if random.random() < 0.3:
        out = torch.flip(out, dims=[2])
    if multi_scale:
        # B3 多尺度随机缩放裁剪 (0.55 ~ 1.0)
        scale = random.uniform(0.55, 1.0)
        ch, cw = int(round(H * scale)), int(round(W * scale))
        top = random.randint(0, H - ch)
        left = random.randint(0, W - cw)
        out = F.interpolate(out[:, :, top:top + ch, left:left + cw], size=(H, W), mode='bilinear', align_corners=False)
    # Mild brightness/contrast jitter (±15%)
    b_factor = 1.0 + (torch.rand((B, 1, 1, 1), device=x_01.device) - 0.5) * 0.3
    out = (out * b_factor).clamp(0.0, 1.0)
    return out


def load_citrus_uat_with_leaf_mask():
    """
    读取 CitrusUAT 共有类 (Healthy, Greasy_spot) 及 HLB 样本；
    同时利用 CitrusUAT 受控均匀深灰底背景的显著色度边界生成高精度前景叶片区域掩膜，校验空间激活聚焦度。
    """
    tf = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
    ])
    imgs, cls_names, fg_masks = [], [], []
    for cls_dir in sorted([d for d in CITRUS_UAT_DIR.iterdir() if d.is_dir()], key=lambda p: p.name):
        for img_p in sorted(cls_dir.glob('*.*')):
            bgr = cv2.imdecode(np.fromfile(str(img_p), dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            arr224 = np.asarray(tf(Image.fromarray(rgb)))  # (224, 224, 3)
            # CitrusUAT 官方拍摄于深灰色背景板 (R≈G≈B, 低饱和度)，叶片前景在 HSV 饱和度 S 与绿/黄色度通道显著分离
            hsv = cv2.cvtColor(arr224, cv2.COLOR_RGB2HSV)
            # Otsu automatic foreground segmentation on Saturation + Lab |a|+|b| chroma
            lab = cv2.cvtColor(arr224, cv2.COLOR_RGB2LAB).astype(np.float32)
            chroma = np.sqrt((lab[:, :, 1] - 128.0) ** 2 + (lab[:, :, 2] - 128.0) ** 2)
            chroma_u8 = np.clip(chroma * 4.0, 0, 255).astype(np.uint8)
            _, mask_bin = cv2.threshold(chroma_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Downsample to 7x7 spatial feature resolution
            m7 = cv2.resize((mask_bin > 0).astype(np.float32), (7, 7), interpolation=cv2.INTER_AREA)
            imgs.append(arr224.transpose(2, 0, 1))
            cls_names.append(cls_dir.name)
            fg_masks.append(m7)
    return np.stack(imgs, axis=0), cls_names, np.stack(fg_masks, axis=0)


@torch.no_grad()
def eval_split(model: PathologyModel, x_u8_gpu: torch.Tensor, y_np: np.ndarray):
    model.eval()
    preds = []
    weights_log = []
    for start in range(0, len(x_u8_gpu), 128):
        xb = x_u8_gpu[start:start + 128].float() / 255.0
        logits, aux = model(xb)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        if 'weights' in aux:
            weights_log.append(aux['weights'].cpu().numpy())
    p = np.concatenate(preds, axis=0)
    acc = float((p == y_np).mean()) * 100.0
    mf1 = float(f1_score(y_np, p, average='macro')) * 100.0
    # Yellowing cluster Macro-F1
    ym = np.isin(y_np, YELLOW_CLUSTER_IDS)
    yf1 = float(f1_score(y_np[ym], p[ym], labels=YELLOW_CLUSTER_IDS, average='macro', zero_division=0)) * 100.0
    # Lesion cluster Macro-F1
    lm = np.isin(y_np, LESION_CLUSTER_IDS)
    lf1 = float(f1_score(y_np[lm], p[lm], labels=LESION_CLUSTER_IDS, average='macro', zero_division=0)) * 100.0
    # S12 Algal spot (class 13) recall
    s12_m = (y_np == 13)
    s12_rec = float((p[s12_m] == 13).mean()) * 100.0 if s12_m.any() else 0.0

    gate_info = None
    if weights_log:
        w_all = np.concatenate(weights_log, axis=0)
        gate_info = {
            'yellow_cluster_gate_mean': np.round(w_all[ym].mean(axis=0), 4).tolist(),
            'lesion_cluster_gate_mean': np.round(w_all[lm].mean(axis=0), 4).tolist(),
        }
    return {
        'acc': round(acc, 2),
        'macro_f1': round(mf1, 2),
        'yellow_f1': round(yf1, 2),
        'lesion_f1': round(lf1, 2),
        's12_recall': round(s12_rec, 2),
        'gate_routing': gate_info,
    }


@torch.no_grad()
def eval_uat(model: PathologyModel, uat_u8_gpu: torch.Tensor, uat_cls: list, fg_masks: np.ndarray):
    model.eval()
    preds = []
    eim_list = []
    for start in range(0, len(uat_u8_gpu), 128):
        xb = uat_u8_gpu[start:start + 128].float() / 255.0
        logits, aux = model(xb)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        # Compute spatial feature activation energy inside foreground leaf mask (Energy-Inside-Mask)
        fmap = aux['fmap']  # (B, 960, 7, 7)
        energy = (fmap ** 2).sum(dim=1).cpu().numpy()  # (B, 7, 7)
        m7 = fg_masks[start:start + 128]
        fg_e = (energy * m7).sum(axis=(1, 2))
        tot_e = energy.sum(axis=(1, 2)) + 1e-8
        eim_list.append(fg_e / tot_e)

    p = np.concatenate(preds, axis=0)
    eim = float(np.concatenate(eim_list, axis=0).mean()) * 100.0
    cls_arr = np.array(uat_cls)

    m_h = (cls_arr == 'Healthy')
    m_g = (cls_arr == 'Greasy_spot')
    m_hlb = (cls_arr == 'HLB')
    acc_h = float((p[m_h] == 1).mean()) * 100.0
    acc_g = float((p[m_g] == 10).mean()) * 100.0
    m_known = m_h | m_g
    yk = np.where(cls_arr[m_known] == 'Healthy', 1, 10)
    acc_known = float((p[m_known] == yk).mean()) * 100.0
    # HLB qPCR+ leaves mapped into SciDB suspected HLB yellowing symptoms {S3(2), S5(4), S6(5), S7(6)}
    hlb_yellow_hit = float(np.isin(p[m_hlb], [2, 5, 6]).mean()) * 100.0

    return {
        'known_compat_acc': round(acc_known, 2),
        'healthy_acc': round(acc_h, 2),
        'greasy_spot_acc': round(acc_g, 2),
        'hlb_mottling_vein_hit': round(hlb_yellow_hit, 2),
        'energy_inside_leaf_mask_pct': round(eim, 2),
    }


def run_one(arch: str, seed: int, gpu_splits: dict, splits_y: dict, uat_gpu: torch.Tensor, uat_cls: list, uat_masks: np.ndarray):
    set_seed(seed)
    device = 'cuda:0'
    model = PathologyModel(arch, num_classes=14).to(device)

    # Backbone lower lr, new heads higher lr (standard fine-tuning protocol matching CropStudent)
    feat_ids = {id(p) for p in model.features.parameters()}
    backbone_params = [p for p in model.parameters() if id(p) in feat_ids]
    head_params = [p for p in model.parameters() if id(p) not in feat_ids]
    opt = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 1.5e-4},
        {'params': head_params, 'lr': 1.5e-3},
    ], weight_decay=5e-4)
    EPOCHS = 20
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda')

    x_tr = gpu_splits['train']
    y_tr = torch.from_numpy(splits_y['train']).to(device)
    best_val_f1 = -1.0
    best_state = None
    best_ep = 0

    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(x_tr), device=device)
        for ids in perm.split(128):
            opt.zero_grad(set_to_none=True)
            xb = x_tr[ids].float() / 255.0
            xb = gpu_augment_batch(xb, multi_scale=(arch == 'B3_MultiScale'))
            yb = y_tr[ids]
            with torch.amp.autocast('cuda'):
                logits, aux = model(xb)
                loss = F.cross_entropy(logits, yb)
                if arch == 'Ours_SymTex_Dynamic':
                    # Auxiliary deep supervision so each specialized stream learns clean discriminative features
                    loss_aux = (
                        0.3 * F.cross_entropy(aux['logits_sem'], yb)
                        + 0.3 * F.cross_entropy(aux['logits_asym'], yb)
                        + 0.3 * F.cross_entropy(aux['logits_hf'], yb)
                    )
                    loss = loss + loss_aux
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()

        val_res = eval_split(model, gpu_splits['val'], splits_y['val'])
        if val_res['macro_f1'] > best_val_f1:
            best_val_f1 = val_res['macro_f1']
            best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    res = {
        'seed': seed,
        'best_epoch': best_ep,
        'params_M': round(sum(p.numel() for p in model.parameters()) / 1e6, 3),
        'val': eval_split(model, gpu_splits['val'], splits_y['val']),
        'testB': eval_split(model, gpu_splits['testB'], splits_y['testB']),
        'testC': eval_split(model, gpu_splits['testC'], splits_y['testC']),
        'citrus_uat': eval_uat(model, uat_gpu, uat_cls, uat_masks),
    }
    del model
    torch.cuda.empty_cache()
    return res


def main():
    t0 = time.monotonic()
    recs = json.loads(OLD_LIST.read_text(encoding='utf-8-sig'))
    splits_y = {sp: np.array([r['类别'] - 1 for r in recs[sp]], dtype=np.int64) for sp in ['train', 'val', 'testB', 'testC']}

    print("[1/2] 载入 SciDB 四划分图像缓存至显存 + CitrusUAT 前景掩膜...", flush=True)
    gpu_splits = {}
    for sp in ['train', 'val', 'testB', 'testC']:
        arr = np.load(BASE_CACHE / '图像缓存' / f'{sp}.npy')
        gpu_splits[sp] = torch.from_numpy(arr).to('cuda:0')
    uat_imgs, uat_cls, uat_masks = load_citrus_uat_with_leaf_mask()
    uat_gpu = torch.from_numpy(uat_imgs).to('cuda:0')
    print(f"  数据载入完成 ({time.monotonic() - t0:.1f}s)", flush=True)

    archs = ['B1_SingleStream', 'B3_MultiScale', 'B4_LocalGlobalConcat', 'Ours_SymTex_Dynamic']
    summary = {}

    for arch in archs:
        t_a = time.monotonic()
        print(f"\n================ 训练与评测架构: {arch} (3 Seeds) ================", flush=True)
        runs = [run_one(arch, s, gpu_splits, splits_y, uat_gpu, uat_cls, uat_masks) for s in [0, 1, 2]]

        def ms(fn):
            vs = [fn(r) for r in runs]
            return f"{np.mean(vs):.2f}±{np.std(vs, ddof=1):.2f}", round(float(np.mean(vs)), 2), round(float(np.std(vs, ddof=1)), 2)

        agg = {
            'params_M': runs[0]['params_M'],
            'val_f1': ms(lambda r: r['val']['macro_f1']),
            'testB_f1': ms(lambda r: r['testB']['macro_f1']),
            'testC_f1': ms(lambda r: r['testC']['macro_f1']),
            'testB_yellow_f1': ms(lambda r: r['testB']['yellow_f1']),
            'testC_yellow_f1': ms(lambda r: r['testC']['yellow_f1']),
            'testC_s12_recall': ms(lambda r: r['testC']['s12_recall']),
            'uat_known_acc': ms(lambda r: r['citrus_uat']['known_compat_acc']),
            'uat_eim_pct': ms(lambda r: r['citrus_uat']['energy_inside_leaf_mask_pct']),
            'seeds': runs,
        }
        summary[arch] = agg
        print(f"  [{arch}] ({time.monotonic() - t_a:.1f}s, {agg['params_M']}M) "
              f"val F1={agg['val_f1'][0]} | testB F1={agg['testB_f1'][0]} (黄化={agg['testB_yellow_f1'][0]}) | "
              f"testC F1={agg['testC_f1'][0]} (黄化={agg['testC_yellow_f1'][0]}, S12召回={agg['testC_s12_recall'][0]}) | "
              f"UAT共有类Acc={agg['uat_known_acc'][0]} | 叶片Mask能量占比={agg['uat_eim_pct'][0]}%",
              flush=True)

    out_path = OUT_DIR / 'probe_c0_symtex_summary.json'
    out_path.write_text(json.dumps({
        '实验名称': 'Probe-C0_双侧非对称拓扑与光照不变高频病斑协同分类',
        '总耗时秒': round(time.monotonic() - t0, 2),
        '结果汇总': summary,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n[完成] Probe-C0 结果已保存至: {out_path}", flush=True)


if __name__ == '__main__':
    main()
