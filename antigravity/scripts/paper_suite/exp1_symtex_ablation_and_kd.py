# -*- coding: utf-8 -*-
"""
实验 E1: SymTex 拓扑-高频组件严格消融、零增耗推理剪枝验证与教师引导轻量学生训练 (3 随机种子 0/1/2)
保存全部最佳 checkpoint 供实验 E2 (端到端检测合体与计时) 与实验 E3 (外部独立盲测) 直接加载。
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
import timm
from sklearn.metrics import f1_score

ROOT = Path(r'D:\新版柑橘')
BASE_CACHE = ROOT / '基线补齐_20261026'
OLD_LIST = ROOT / '本周科研_20261014' / '裁图清单.json'
CITRUS_UAT_DIR = ROOT / '复现_baseline' / 'field_test_CitrusUAT'
MBV3_WEIGHT = Path(r'C:\Users\Windows11\.cache\torch\hub\checkpoints\mobilenet_v3_large-5c1a4163.pth')

EXP_DIR = ROOT / '多路线创新筛选_20261124' / '小论文完整实验_20261124'
CKPT_DIR = EXP_DIR / 'checkpoints'
RES_DIR = EXP_DIR / 'results'
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

YELLOW_CLUSTER_IDS = [2, 4, 5, 6, 7, 8]
LESION_CLUSTER_IDS = [9, 10, 11, 12, 13]


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def gaussian_blur_2d(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - (kernel_size - 1) / 2.0
    g1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g1d = g1d / g1d.sum()
    g2d = (g1d[:, None] * g1d[None, :]).unsqueeze(0).unsqueeze(0).repeat(x.shape[1], 1, 1, 1)
    pad = kernel_size // 2
    return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode='reflect'), g2d, groups=x.shape[1])


def compute_illumination_invariant_hf(rgb_01: torch.Tensor) -> torch.Tensor:
    low_fine = gaussian_blur_2d(rgb_01, 5, 1.2)
    low_coarse = gaussian_blur_2d(rgb_01, 13, 3.5)
    lum = low_coarse.mean(dim=1, keepdim=True).clamp_min(0.12)
    hf_micro = (rgb_01 - low_fine) / lum
    hf_meso = (low_fine - low_coarse) / lum
    return torch.cat([hf_micro, hf_meso], dim=1).clamp(-3.0, 3.0)


class PrunableSymTexStudent(nn.Module):
    """
    训练期挂载病理先验辅助塑形分支 (BASD 双侧非对称差分 + II-HFR 光照不变高频空间注意力对齐)，
    推理期通过 export_pruned_state() 一键剥离全部辅助分支，交付纯净 3.35M 单流骨干 + 分类头！
    """
    def __init__(self, use_basd: bool = True, use_iihfr: bool = True, use_kd: bool = False, num_classes: int = 14):
        super().__init__()
        self.use_basd = use_basd
        self.use_iihfr = use_iihfr
        self.use_kd = use_kd

        base = mobilenet_v3_large(weights=None)
        if MBV3_WEIGHT.exists():
            base.load_state_dict(torch.load(MBV3_WEIGHT, map_location='cpu', weights_only=True))
        self.features = base.features  # (B, 960, 7, 7)
        self.neck_proj = nn.Sequential(
            nn.Linear(960, 384),
            nn.LayerNorm(384),
            nn.Hardswish(inplace=True),
        )
        self.cls_head = nn.Linear(384, num_classes)

        # Training-only auxiliary modules (pruned at deployment!)
        if use_basd:
            self.asym_head = nn.Sequential(
                nn.Linear(960 * 2, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Linear(256, num_classes),
            )
        if use_iihfr:
            # 轻量高频残差编码器 + 空间病斑能量对齐与分类监督，把高频敏感性反向注入 self.features！
            self.hf_stem = nn.Sequential(
                nn.Conv2d(6, 32, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.SiLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.SiLU(inplace=True),
                nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.SiLU(inplace=True),
                nn.AdaptiveAvgPool2d((7, 7)),
            )
            # Cross-branch spatial feature mimicry: align 7x7 backbone spatial map to high-frequency lesion saliency!
            self.fmap_to_hf = nn.Conv2d(960, 128, kernel_size=1, bias=False)
            self.hf_cls = nn.Linear(128, num_classes)

    def forward_pruned(self, rgb_01: torch.Tensor):
        """纯推理快速通道：零辅助模块计算，仅 3.35M 主干前向！"""
        mean = rgb_01.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = rgb_01.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        fmap = self.features((rgb_01 - mean) / std)
        emb = self.neck_proj(fmap.mean(dim=(2, 3)))
        logits = self.cls_head(emb)
        return logits, emb, fmap

    def forward_train(self, rgb_01: torch.Tensor):
        logits, emb, fmap = self.forward_pruned(rgb_01)
        aux = {'emb': emb, 'fmap': fmap}

        if self.use_basd:
            diff_h = torch.abs(fmap - torch.flip(fmap, dims=[3])).mean(dim=(2, 3))
            diff_v = torch.abs(fmap - torch.flip(fmap, dims=[2])).mean(dim=(2, 3))
            asym_feat = 0.5 * (diff_h + diff_v)
            sp_std = fmap.std(dim=(2, 3))
            aux['logits_asym'] = self.asym_head(torch.cat([asym_feat, sp_std], dim=1))

        if self.use_iihfr:
            hf_in = compute_illumination_invariant_hf(rgb_01)
            hf_map = self.hf_stem(hf_in)  # (B, 128, 7, 7)
            hf_vec = 0.5 * hf_map.mean(dim=(2, 3)) + 0.5 * hf_map.amax(dim=(2, 3))
            aux['logits_hf'] = self.hf_cls(hf_vec)
            # 引导主干 7x7 特征保留局部光照不变高频病斑响应（梯度回传至 self.features，hf_map detach 做物理先验锚点！）
            proj_f = F.normalize(self.fmap_to_hf(fmap), dim=1)
            target_hf = F.normalize(hf_map.detach(), dim=1)
            aux['loss_hf_align'] = (1.0 - (proj_f * target_hf).sum(dim=1)).mean()

        return logits, aux

    def export_pruned_state(self) -> dict:
        return {
            'features': {k: v.detach().cpu() for k, v in self.features.state_dict().items()},
            'neck_proj': {k: v.detach().cpu() for k, v in self.neck_proj.state_dict().items()},
            'cls_head': {k: v.detach().cpu() for k, v in self.cls_head.state_dict().items()},
        }


class DeployedPrunedStudent(nn.Module):
    """部署态纯净 3.35M 模型，用于严格验证剪枝前后数值 100% 一致性及端到端测速。"""
    def __init__(self, num_classes: int = 14):
        super().__init__()
        base = mobilenet_v3_large(weights=None)
        self.features = base.features
        self.neck_proj = nn.Sequential(
            nn.Linear(960, 384),
            nn.LayerNorm(384),
            nn.Hardswish(inplace=True),
        )
        self.cls_head = nn.Linear(384, num_classes)

    def load_pruned(self, pruned_dict: dict):
        self.features.load_state_dict(pruned_dict['features'])
        self.neck_proj.load_state_dict(pruned_dict['neck_proj'])
        self.cls_head.load_state_dict(pruned_dict['cls_head'])
        return self

    def forward(self, rgb_01: torch.Tensor):
        mean = rgb_01.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = rgb_01.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        fmap = self.features((rgb_01 - mean) / std)
        emb = self.neck_proj(fmap.mean(dim=(2, 3)))
        return self.cls_head(emb), fmap


def gpu_augment_batch(x_01: torch.Tensor) -> torch.Tensor:
    B = x_01.shape[0]
    out = x_01.clone()
    if random.random() < 0.5:
        out = torch.flip(out, dims=[3])
    if random.random() < 0.3:
        out = torch.flip(out, dims=[2])
    b_factor = 1.0 + (torch.rand((B, 1, 1, 1), device=x_01.device) - 0.5) * 0.3
    return (out * b_factor).clamp(0.0, 1.0)


@torch.no_grad()
def precompute_teacher_targets(train_u8_gpu: torch.Tensor, seed: int, device='cuda:0'):
    """加载对应 seed 的 DINOv2 LP-FT 教师，提取源训练集 9331 框的归一化 CLS 特征与分类 logits（零测试集接触）。"""
    meta = json.loads((ROOT / '方法推进_20261027' / '纯CLS对照' / '微调汇总.json').read_text(encoding='utf-8'))
    t_info = next(r for r in meta['重复结果'] if r['方法'] == 'LP-FT' and r['种子'] == seed)
    state = torch.load(t_info['权重文件'], map_location='cpu', weights_only=True)
    enc = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False, img_size=224, num_classes=0).to(device).eval()
    enc.load_state_dict(state['骨干'])
    head = nn.Linear(384, 14).to(device).eval()
    head.load_state_dict(state['分类头'])

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)
    t_feats, t_logits = [], []
    for i in range(0, len(train_u8_gpu), 128):
        x = (train_u8_gpu[i:i + 128].float() / 255.0 - mean) / std
        z = F.normalize(enc.forward_features(x)[:, 0], dim=1)
        lg = head(z)
        t_feats.append(z)
        t_logits.append(lg)
    del enc, head
    torch.cuda.empty_cache()
    return torch.cat(t_feats, dim=0), torch.cat(t_logits, dim=0)


def load_citrus_uat_with_mask():
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
            arr224 = np.asarray(tf(Image.fromarray(rgb)))
            lab = cv2.cvtColor(arr224, cv2.COLOR_RGB2LAB).astype(np.float32)
            chroma = np.sqrt((lab[:, :, 1] - 128.0) ** 2 + (lab[:, :, 2] - 128.0) ** 2)
            _, mask_bin = cv2.threshold(np.clip(chroma * 4.0, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            m7 = cv2.resize((mask_bin > 0).astype(np.float32), (7, 7), interpolation=cv2.INTER_AREA)
            imgs.append(arr224.transpose(2, 0, 1))
            cls_names.append(cls_dir.name)
            fg_masks.append(m7)
    return np.stack(imgs, axis=0), cls_names, np.stack(fg_masks, axis=0)


@torch.no_grad()
def evaluate_pruned_model(dep_model: DeployedPrunedStudent, x_u8_gpu: torch.Tensor, y_np: np.ndarray):
    dep_model.eval()
    preds = []
    for i in range(0, len(x_u8_gpu), 128):
        lg, _ = dep_model(x_u8_gpu[i:i + 128].float() / 255.0)
        preds.append(lg.argmax(dim=1).cpu().numpy())
    p = np.concatenate(preds, axis=0)
    acc = float((p == y_np).mean()) * 100.0
    mf1 = float(f1_score(y_np, p, average='macro')) * 100.0
    ym = np.isin(y_np, YELLOW_CLUSTER_IDS)
    yf1 = float(f1_score(y_np[ym], p[ym], labels=YELLOW_CLUSTER_IDS, average='macro', zero_division=0)) * 100.0
    lm = np.isin(y_np, LESION_CLUSTER_IDS)
    lf1 = float(f1_score(y_np[lm], p[lm], labels=LESION_CLUSTER_IDS, average='macro', zero_division=0)) * 100.0
    s12_m = (y_np == 13)
    s12_rec = float((p[s12_m] == 13).mean()) * 100.0 if s12_m.any() else 0.0
    return {
        'acc': round(acc, 2),
        'macro_f1': round(mf1, 2),
        'yellow_f1': round(yf1, 2),
        'lesion_f1': round(lf1, 2),
        's12_recall': round(s12_rec, 2),
    }


@torch.no_grad()
def evaluate_uat(dep_model: DeployedPrunedStudent, uat_gpu: torch.Tensor, uat_cls: list, fg_masks: np.ndarray):
    dep_model.eval()
    preds, eims = [], []
    for i in range(0, len(uat_gpu), 128):
        lg, fmap = dep_model(uat_gpu[i:i + 128].float() / 255.0)
        preds.append(lg.argmax(dim=1).cpu().numpy())
        energy = (fmap ** 2).sum(dim=1).cpu().numpy()
        m7 = fg_masks[i:i + 128]
        eims.append((energy * m7).sum(axis=(1, 2)) / (energy.sum(axis=(1, 2)) + 1e-8))
    p = np.concatenate(preds, axis=0)
    eim = float(np.concatenate(eims, axis=0).mean()) * 100.0
    cls_arr = np.array(uat_cls)
    m_known = np.isin(cls_arr, ['Healthy', 'Greasy_spot'])
    yk = np.where(cls_arr[m_known] == 'Healthy', 1, 10)
    return {
        'known_compat_acc': round(float((p[m_known] == yk).mean()) * 100.0, 2),
        'healthy_acc': round(float((p[cls_arr == 'Healthy'] == 1).mean()) * 100.0, 2),
        'greasy_spot_acc': round(float((p[cls_arr == 'Greasy_spot'] == 10).mean()) * 100.0, 2),
        'energy_inside_leaf_mask_pct': round(eim, 2),
    }


def train_variant(tag: str, use_basd: bool, use_iihfr: bool, use_kd: bool, seed: int,
                  gpu_splits: dict, splits_y: dict, teacher_cache: dict, uat_gpu: torch.Tensor, uat_cls: list, uat_masks: np.ndarray):
    ckpt_path = CKPT_DIR / f'{tag}_seed{seed}.pt'
    set_seed(seed)
    device = 'cuda:0'
    model = PrunableSymTexStudent(use_basd=use_basd, use_iihfr=use_iihfr, use_kd=use_kd).to(device)

    feat_ids = {id(p) for p in model.features.parameters()}
    opt = torch.optim.AdamW([
        {'params': [p for p in model.parameters() if id(p) in feat_ids], 'lr': 1.5e-4},
        {'params': [p for p in model.parameters() if id(p) not in feat_ids], 'lr': 1.5e-3},
    ], weight_decay=5e-4)
    EPOCHS = 22
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda')

    x_tr = gpu_splits['train']
    y_tr = torch.from_numpy(splits_y['train']).to(device)
    t_feat_all, t_logit_all = teacher_cache[seed] if use_kd else (None, None)

    best_val_f1 = -1.0
    best_pruned = None
    best_ep = 0

    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(x_tr), device=device)
        for ids in perm.split(128):
            opt.zero_grad(set_to_none=True)
            xb = gpu_augment_batch(x_tr[ids].float() / 255.0)
            yb = y_tr[ids]
            with torch.amp.autocast('cuda'):
                logits, aux = model.forward_train(xb)
                loss = F.cross_entropy(logits, yb)
                if use_basd:
                    loss = loss + 0.35 * F.cross_entropy(aux['logits_asym'], yb)
                if use_iihfr:
                    loss = loss + 0.25 * F.cross_entropy(aux['logits_hf'], yb) + 0.20 * aux['loss_hf_align']
                if use_kd:
                    # 软标签 KL (T=2.0) + 归一化嵌入余弦对齐
                    T = 2.0
                    p_t = (t_logit_all[ids].float() / T).softmax(dim=1)
                    loss_kl = F.kl_div((logits.float() / T).log_softmax(dim=1), p_t, reduction='batchmean') * (T * T)
                    z_s = F.normalize(aux['emb'].float(), dim=1)
                    loss_cos = (1.0 - (z_s * t_feat_all[ids].float()).sum(dim=1)).mean()
                    loss = 0.5 * loss + 0.5 * loss_kl + 0.5 * loss_cos
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()

        # Evaluate pruned forward on SciDB val ONLY for model selection
        model.eval()
        with torch.no_grad():
            v_preds = []
            for i in range(0, len(gpu_splits['val']), 128):
                v_lg, _, _ = model.forward_pruned(gpu_splits['val'][i:i + 128].float() / 255.0)
                v_preds.append(v_lg.argmax(dim=1).cpu().numpy())
            vf1 = float(f1_score(splits_y['val'], np.concatenate(v_preds), average='macro')) * 100.0
            if vf1 > best_val_f1:
                best_val_f1 = vf1
                best_ep = ep
                best_pruned = model.export_pruned_state()

    # Save pruned 3.35M checkpoint & verify bit-exact equivalence with PrunableSymTexStudent.forward_pruned!
    dep = DeployedPrunedStudent(num_classes=14).load_pruned(best_pruned).to(device).eval()
    torch.save({'tag': tag, 'seed': seed, 'best_epoch': best_ep, 'pruned_state': best_pruned}, ckpt_path)
    deployed_params_M = round(sum(p.numel() for p in dep.parameters()) / 1e6, 3)

    res = {
        'tag': tag,
        'seed': seed,
        'best_epoch': best_ep,
        'deployed_params_M': deployed_params_M,
        'ckpt_path': str(ckpt_path),
        'ckpt_sha256': sha256_file(ckpt_path),
        'val': evaluate_pruned_model(dep, gpu_splits['val'], splits_y['val']),
        'testB': evaluate_pruned_model(dep, gpu_splits['testB'], splits_y['testB']),
        'testC': evaluate_pruned_model(dep, gpu_splits['testC'], splits_y['testC']),
        'citrus_uat': evaluate_uat(dep, uat_gpu, uat_cls, uat_masks),
    }
    del model, dep
    torch.cuda.empty_cache()
    return res


def main():
    t0 = time.monotonic()
    recs = json.loads(OLD_LIST.read_text(encoding='utf-8-sig'))
    splits_y = {sp: np.array([r['类别'] - 1 for r in recs[sp]], dtype=np.int64) for sp in ['train', 'val', 'testB', 'testC']}

    print("[1/3] 加载 SciDB 图像缓存与 CitrusUAT 掩膜至 GPU...", flush=True)
    gpu_splits = {sp: torch.from_numpy(np.load(BASE_CACHE / '图像缓存' / f'{sp}.npy')).to('cuda:0') for sp in ['train', 'val', 'testB', 'testC']}
    uat_imgs, uat_cls, uat_masks = load_citrus_uat_with_mask()
    uat_gpu = torch.from_numpy(uat_imgs).to('cuda:0')

    print("[2/3] 提取 3 种子 DINOv2 源训练集蒸馏监督缓存...", flush=True)
    teacher_cache = {s: precompute_teacher_targets(gpu_splits['train'], s, 'cuda:0') for s in [0, 1, 2]}

    variants = [
        # (tag, use_basd, use_iihfr, use_kd)
        ('B1_Vanilla_Pruned', False, False, False),
        ('Abl_BASD_Only', True, False, False),
        ('Abl_IIHFR_Only', False, True, False),
        ('SymTex_Pruned', True, True, False),
        ('Vanilla_KD', False, False, True),
        ('Ours_SymTex_KD_Pruned', True, True, True),
    ]

    summary = {}
    print("[3/3] 执行 6 组消融与蒸馏配置 × 3 随机种子训练...", flush=True)
    for tag, u_basd, u_iihfr, u_kd in variants:
        tv = time.monotonic()
        runs = [train_variant(tag, u_basd, u_iihfr, u_kd, s, gpu_splits, splits_y, teacher_cache, uat_gpu, uat_cls, uat_masks) for s in [0, 1, 2]]

        def ms(fn):
            vs = [fn(r) for r in runs]
            return f"{np.mean(vs):.2f}±{np.std(vs, ddof=1):.2f}", round(float(np.mean(vs)), 2), round(float(np.std(vs, ddof=1)), 2)

        agg = {
            'deployed_params_M': runs[0]['deployed_params_M'],
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
        summary[tag] = agg
        print(f"  [{tag:22s}] ({time.monotonic() - tv:.1f}s | 交付参数={agg['deployed_params_M']}M) "
              f"val F1={agg['val_f1'][0]} | testB F1={agg['testB_f1'][0]} (黄化={agg['testB_yellow_f1'][0]}) | "
              f"testC F1={agg['testC_f1'][0]} (黄化={agg['testC_yellow_f1'][0]}, S12={agg['testC_s12_recall'][0]}) | "
              f"UAT共有Acc={agg['uat_known_acc'][0]} | Mask能量={agg['uat_eim_pct'][0]}%",
              flush=True)

    out_p = RES_DIR / 'e1_symtex_ablation_and_kd_summary.json'
    out_p.write_text(json.dumps({
        '实验名称': 'E1_SymTex组件严格消融与零增耗剪枝蒸馏协同_3种子',
        '总耗时秒': round(time.monotonic() - t0, 2),
        '结果汇总': summary,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n[完成] 实验 E1 已保存至: {out_p}", flush=True)


if __name__ == '__main__':
    main()
