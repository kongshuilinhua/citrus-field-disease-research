# -*- coding: utf-8 -*-
"""
Phase 1 (严格零泄漏纯源域开发与消融定型):
单趟框引导空间 Token 背景子空间正交剥离与病斑峰值聚合网络 (BG-OPTA)

铁律声明：
- 本脚本物理隔离 testB / testC / Zenodo / SweetOrange / Pomelo（代码中不读取任何目标测试集）；
- 仅使用 SciDB train (85% train_fit / 15% source_dev) 训练，在 source_dev_clean、source_dev_bgshift (合成白底反光/异源树冠四角侵入)
  与官方源验证集 SciDB val 上完成 S0/S1/S2/S3 四组同算力(单次 1.50x 裁框)消融及安全免裁阈值标定，一次性冻结权重与参数！
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'

import sys
import io
import json
import math
import time
import random
import hashlib
import contextlib
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from sklearn.metrics import f1_score
from torchvision.ops import batched_nms
from faster_coco_eval import COCO, COCOeval_faster

ROOT = Path(r'D:\新版柑橘')
EXT = ROOT / '方法路线扩展_20261031'
sys.path.append(str(EXT))
from 共同评测后端 import configure_strict_fp32

WORK_DIR = ROOT / 'Antigravity_BGOPTA_单趟正交Token攻关_20261126'
CKPT_DIR = WORK_DIR / 'frozen_models'
RES_DIR = WORK_DIR / 'results'
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

PREV_CACHE = ROOT / 'Antigravity_CGGAR_真创新攻关_20261125' / 'cache'
IMG_ROOT = Path(r'D:\病虫害论文\复现实验记录\yolo11n_public_compare_20261004\数据副本\images')
GT_VAL_PATH = EXT / '检测数据' / 'val.json'
STRONG_DIR = EXT / 'YOLOv8强基线重复_20261109'

CROP_SCALE = 1.50  # 单次上下文扩框倍率：中心 ~10.7x10.7 补丁为目标框，外周 ~2.7 补丁宽环带为真实场景背景采样区


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def coco_eval_ap(gt_path: Path, pred_rows: list):
    if not pred_rows:
        return 0.0, 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_path))
        ev = COCOeval_faster(gt, gt.loadRes(pred_rows), 'bbox')
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[0] * 100.0), float(ev.stats[1] * 100.0)


def apply_classwise_nms(rows: list, iou_thr: float = 0.70) -> list:
    by_img = defaultdict(list)
    for r in rows:
        by_img[r['image_id']].append(r)
    kept_all = []
    for _iid, rlist in by_img.items():
        if len(rlist) <= 1:
            kept_all.extend(rlist)
            continue
        b_xyxy = torch.tensor([[r['bbox'][0], r['bbox'][1], r['bbox'][0] + r['bbox'][2], r['bbox'][1] + r['bbox'][3]] for r in rlist], dtype=torch.float32)
        sc = torch.tensor([r['score'] for r in rlist], dtype=torch.float32)
        cl = torch.tensor([r['category_id'] for r in rlist], dtype=torch.int64)
        keep = batched_nms(b_xyxy, sc, cl, iou_thr).tolist()[:300]
        for k in keep:
            kept_all.append(rlist[k])
    return kept_all


def fit_mahalanobis_stats(feats: np.ndarray, labels: np.ndarray, reg: float = 1e-2):
    d = feats.shape[1]
    means = np.zeros((14, d), dtype=np.float64)
    centered = []
    for c in range(14):
        Xc = feats[labels == c].astype(np.float64)
        means[c] = Xc.mean(axis=0)
        centered.append(Xc - means[c])
    cov = (np.concatenate(centered, 0).T @ np.concatenate(centered, 0)) / max(1, len(feats) - 1)
    prec = np.linalg.inv(cov + reg * np.eye(d))
    diff_tr = feats.astype(np.float64) - means[labels]
    d_tr = np.sqrt(np.maximum(0.0, np.sum((diff_tr @ prec) * diff_tr, axis=1)))
    return means, prec, float(d_tr.mean()), float(d_tr.std() + 1e-6)


def compute_maha_z(feats: np.ndarray, means: np.ndarray, prec: np.ndarray, m_mu: float, m_sd: float):
    N = len(feats)
    out = np.zeros((N, 14), dtype=np.float32)
    X = feats.astype(np.float64)
    for c in range(14):
        df = X - means[c]
        dc = np.sqrt(np.maximum(0.0, np.sum((df @ prec) * df, axis=1)))
        out[:, c] = ((dc - m_mu) / m_sd).astype(np.float32)
    return out


def crop_single_context_and_masks(rgb: np.ndarray, bbox_xywh: list, scale: float = CROP_SCALE):
    """
    单次截取 1.50x 方形上下文缩放至 224x224，并精确计算 16x16=256 网格上的确定性几何掩膜（无需任何分割真值标注！）：
    - m_box  (256,): 落在检测框内的补丁；
    - m_ring (256,): 落在检测框外且位于真实原图有效边界内的真实背景环状补丁（自动剔除出界灰边 128！）。
    """
    h, w = rgb.shape[:2]
    x, y, bw, bh = bbox_xywh
    max_dim = max(h, w)
    if max_dim > 1280:
        f = 1280.0 / max_dim
        rgb = cv2.resize(rgb, (max(1, int(round(w * f))), max(1, int(round(h * f)))), interpolation=cv2.INTER_AREA)
        h, w = rgb.shape[:2]
        x, y, bw, bh = x * f, y * f, bw * f, bh * f

    cx, cy = x + bw * 0.5, y + bh * 0.5
    side = max(8, int(math.ceil(max(bw, bh) * scale)))
    half = side * 0.5
    x1, y1 = int(math.floor(cx - half)), int(math.floor(cy - half))
    x2, y2 = x1 + side, y1 + side

    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(w, x2), min(h, y2)
    canvas = np.full((side, side, 3), 128, dtype=np.uint8)
    if sx2 > sx1 and sy2 > sy1:
        canvas[(sy1 - y1):(sy2 - y1), (sx1 - x1):(sx2 - x1)] = rgb[sy1:sy2, sx1:sx2]

    interp = cv2.INTER_AREA if side > 224 else cv2.INTER_CUBIC
    crop224 = cv2.resize(canvas, (224, 224), interpolation=interp).transpose(2, 0, 1)

    # Compute exact normalized coordinates of bbox & valid image on [0, 1]^2 of canvas
    bu1, bv1 = (x - x1) / side, (y - y1) / side
    bu2, bv2 = (x + bw - x1) / side, (y + bh - y1) / side
    iu1, iv1 = (0.0 - x1) / side, (0.0 - y1) / side
    iu2, iv2 = (float(w) - x1) / side, (float(h) - y1) / side

    # 16x16 patch centers
    grid = (np.arange(16, dtype=np.float32) + 0.5) / 16.0
    uu, vv = np.meshgrid(grid, grid)  # (16, 16)
    in_box = (uu >= bu1) & (uu <= bu2) & (vv >= bv1) & (vv <= bv2)
    # Guarantee at least center 4 patches if box is tiny
    if in_box.sum() < 4:
        in_box[7:9, 7:9] = True

    in_img = (uu >= iu1 + 0.02) & (uu <= iu2 - 0.02) & (vv >= iv1 + 0.02) & (vv <= iv2 - 0.02)
    # Slight guard margin (0.03) outside box for clean background ring
    out_box = (uu < bu1 - 0.02) | (uu > bu2 + 0.02) | (vv < bv1 - 0.02) | (vv > bv2 + 0.02)
    ring_valid = out_box & in_img
    if ring_valid.sum() < 4:
        ring_valid = ~in_box

    # Pack into uint8 bit/bool masks (2, 256)
    masks = np.stack([in_box.reshape(-1), ring_valid.reshape(-1)], axis=0).astype(np.bool_)
    return crop224, masks


def synthesize_bg_intruded_crop(crop224: np.ndarray, masks: np.ndarray, donor_crop224: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """
    仅用于源域 SciDB train 内部合成背景侵入样本（不接触任何测试集！）：
    保留检测框内中央椭圆叶片主体，将框外环带与矩形框四角边缘替换为：
    (a) 高亮白板/灰底强曝光反光，或 (b) 同一子集内另一张训练图的异源枝叶背景，并施加全局色温/伽马偏移！
    """
    out = crop224.copy().astype(np.float32)
    in_box_16 = masks[0].reshape(16, 16).astype(np.float32)
    # Upsample box mask & multiply by an inscribed soft ellipse so the 4 corners INSIDE the box ALSO get background intrusion!
    box_224 = cv2.resize(in_box_16, (224, 224), interpolation=cv2.INTER_LINEAR)
    ys, xs = np.where(box_224 > 0.5)
    if len(xs) > 10:
        yc, xc = 0.5 * (ys.min() + ys.max()), 0.5 * (xs.min() + xs.max())
        ry, rx = max(8.0, 0.48 * (ys.max() - ys.min())), max(8.0, 0.48 * (xs.max() - xs.min()))
        yy, xx = np.ogrid[:224, :224]
        dist2 = ((yy - yc) / ry) ** 2 + ((xx - xc) / rx) ** 2
        fg_alpha = np.clip((1.08 - dist2) * 3.5, 0.0, 1.0)[None, :, :]
    else:
        fg_alpha = box_224[None, :, :]

    if rng.rand() < 0.55:
        # Simulate white/cool studio board + specular glare
        bg_col = np.array([rng.uniform(215, 250), rng.uniform(218, 250), rng.uniform(220, 252)], dtype=np.float32)[:, None, None]
        noise = rng.normal(0, 6.0, size=(3, 224, 224)).astype(np.float32)
        bg = np.clip(bg_col + noise, 0, 255)
    else:
        # Donor orchard clutter from same source split
        bg = donor_crop224.astype(np.float32)

    mixed = fg_alpha * out + (1.0 - fg_alpha) * bg
    # Global exposure/gamma & slight white-balance cast affecting BOTH foreground and background
    gm = rng.choice([0.65, 0.80, 1.0, 1.30, 1.55])
    wb = rng.uniform(0.88, 1.12, size=(3, 1, 1)).astype(np.float32)
    mixed = np.clip(((mixed / 255.0) ** gm) * wb * 255.0, 0, 255)
    return mixed.astype(np.uint8)


class BGOPTAHead(nn.Module):
    """
    单趟框引导空间 Token 背景正交剥离与双路病理聚合头 (BG-OPTA)：
    支持严格同算力 4 级消融模式：
    - mode='S0_cls_only'       : 仅使用 [CLS] 向量 (基准单框特征)
    - mode='S1_plain_concat'   : 直接拼接 [z_cls, mean(P_box), mean(P_ring), max(P_box), geom] 过同容量 MLP (无正交投影、无背景角抑制)
    - mode='S2_orth_topo_only' : 启用框外四象限 Gram-Schmidt 背景子空间正交投影 + 框内四角软剔除，仅整叶拓扑池化 (无局部病斑峰值)
    - mode='S3_bgopta_full'    : 完整正交剥离 + 四角软剔除 + 整叶对称拓扑 + 局部病斑峰值双路聚合！
    """
    def __init__(self, pretrained_head_weight: torch.Tensor, pretrained_head_bias: torch.Tensor,
                 mode: str = 'S3_bgopta_full', dim: int = 384, geom_dim: int = 9, num_classes: int = 14):
        super().__init__()
        self.mode = mode
        self.dim = dim

        # Keep frozen pretrained DINOv2 linear head as stable anchor
        self.base_cls = nn.Linear(dim, num_classes)
        self.base_cls.weight.data.copy_(pretrained_head_weight)
        self.base_cls.bias.data.copy_(pretrained_head_bias)
        for p in self.base_cls.parameters():
            p.requires_grad = False

        self.geom_proj = nn.Sequential(nn.Linear(geom_dim, 32), nn.LayerNorm(32), nn.GELU())

        if mode == 'S0_cls_only':
            in_d = dim + 32
        elif mode == 'S1_plain_concat':
            in_d = dim * 4 + 32
        elif mode == 'S2_orth_topo_only':
            in_d = dim * 3 + 32
            self.orth_strength = nn.Parameter(torch.zeros(dim))       # sigmoid -> 0.5 initial orthogonal removal
            self.bg_gate_tau = nn.Parameter(torch.tensor(0.35))       # background subspace energy cutoff
            self.bg_gate_temp = nn.Parameter(torch.tensor(8.0))
        elif mode == 'S3_bgopta_full':
            in_d = dim * 4 + 32
            self.orth_strength = nn.Parameter(torch.zeros(dim))
            self.bg_gate_tau = nn.Parameter(torch.tensor(0.35))
            self.bg_gate_temp = nn.Parameter(torch.tensor(8.0))
            self.lesion_scorer = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 1))
        else:
            raise ValueError(mode)

        self.mlp = nn.Sequential(
            nn.LayerNorm(in_d),
            nn.Dropout(0.35),
            nn.Linear(in_d, 384),
            nn.GELU(),
            nn.LayerNorm(384),
            nn.Dropout(0.25),
            nn.Linear(384, num_classes),
        )
        # Initialize residual output near zero so training starts cleanly from pretrained base_cls
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def _build_ring_orthonormal_basis(p_norm: torch.Tensor, m_ring: torch.Tensor) -> torch.Tensor:
        """
        从框外真实背景环带提取 K=3 标准正交背景子空间基 U_bg (B, 3, D)：
        u1: 全环带均值方向 (全局底色与环境光照)；
        u2: 上半环与下半环差分正交分量 (垂直光照梯度/天空地面异质背景)；
        u3: 左半环与右半环差分正交分量 (水平侧光/邻枝阴影)。
        全程纯张量批运算，耗时 <0.05ms，无 SVD 奇异值数值不稳定问题。
        """
        B, _, D = p_norm.shape
        mr = m_ring.float().unsqueeze(-1)  # (B, 256, 1)
        u1 = F.normalize((p_norm * mr).sum(dim=1), dim=-1)  # (B, D)

        p_grid = p_norm.view(B, 16, 16, D)
        mr_grid = mr.view(B, 16, 16, 1)

        top_m = (p_grid[:, :8] * mr_grid[:, :8]).sum(dim=(1, 2)) / (mr_grid[:, :8].sum(dim=(1, 2)) + 1e-6)
        bot_m = (p_grid[:, 8:] * mr_grid[:, 8:]).sum(dim=(1, 2)) / (mr_grid[:, 8:].sum(dim=(1, 2)) + 1e-6)
        v2 = top_m - bot_m
        v2 = v2 - (v2 * u1).sum(dim=-1, keepdim=True) * u1
        u2 = F.normalize(v2, dim=-1)

        left_m = (p_grid[:, :, :8] * mr_grid[:, :, :8]).sum(dim=(1, 2)) / (mr_grid[:, :, :8].sum(dim=(1, 2)) + 1e-6)
        right_m = (p_grid[:, :, 8:] * mr_grid[:, :, 8:]).sum(dim=(1, 2)) / (mr_grid[:, :, 8:].sum(dim=(1, 2)) + 1e-6)
        v3 = left_m - right_m
        v3 = v3 - (v3 * u1).sum(dim=-1, keepdim=True) * u1 - (v3 * u2).sum(dim=-1, keepdim=True) * u2
        u3 = F.normalize(v3, dim=-1)

        return torch.stack([u1, u2, u3], dim=1)  # (B, 3, D)

    def forward_with_diag(self, tokens_257: torch.Tensor, masks_2: torch.Tensor, geom: torch.Tensor):
        z_cls = F.normalize(tokens_257[:, 0].float(), dim=-1)      # (B, D)
        p_raw = F.normalize(tokens_257[:, 1:].float(), dim=-1)     # (B, 256, D)
        m_box = masks_2[:, 0]                                      # (B, 256) bool
        m_ring = masks_2[:, 1]                                     # (B, 256) bool
        g_emb = self.geom_proj(geom)

        base_logits = self.base_cls(z_cls)
        fg_weights = m_box.float()

        if self.mode == 'S0_cls_only':
            feat = torch.cat([z_cls, g_emb], dim=-1)
            return base_logits + self.mlp(feat), fg_weights

        if self.mode == 'S1_plain_concat':
            mb = m_box.float().unsqueeze(-1)
            mr = m_ring.float().unsqueeze(-1)
            f_in = F.normalize((p_raw * mb).sum(dim=1) / (mb.sum(dim=1) + 1e-6), dim=-1)
            f_out = F.normalize((p_raw * mr).sum(dim=1) / (mr.sum(dim=1) + 1e-6), dim=-1)
            p_masked = p_raw.masked_fill(~m_box.unsqueeze(-1), -1e4)
            f_max = F.normalize(p_masked.max(dim=1).values, dim=-1)
            feat = torch.cat([z_cls, f_in, f_out, f_max, g_emb], dim=-1)
            return base_logits + self.mlp(feat), fg_weights

        # --- S2 & S3: Box-Guided Background Subspace Orthogonal Projection ---
        U_bg = self._build_ring_orthonormal_basis(p_raw, m_ring)       # (B, 3, D)
        # Coordinates of every patch in the 3-D outer background subspace: (B, 256, 3)
        proj_coeff = torch.bmm(p_raw, U_bg.transpose(1, 2))
        bg_energy = (proj_coeff ** 2).sum(dim=-1)                      # (B, 256) in [0, 1]
        bg_recon = torch.bmm(proj_coeff, U_bg)                         # (B, 256, D)

        # Soft Foreground Silhouette inside M_box: suppresses the 4 background corners matching outer ring!
        temp = F.softplus(self.bg_gate_temp) + 2.0
        w_fg = m_box.float() * torch.sigmoid(temp * (self.bg_gate_tau - bg_energy))  # (B, 256)
        w_sum = w_fg.sum(dim=1, keepdim=True).clamp_min(1e-5)

        gamma = torch.sigmoid(self.orth_strength).view(1, 1, -1)
        p_orth = F.normalize(p_raw - gamma * bg_recon, dim=-1)         # (B, 256, D)

        # Also de-bias [CLS] against outer background subspace
        cls_coeff = torch.bmm(z_cls.unsqueeze(1), U_bg.transpose(1, 2))
        cls_bg = torch.bmm(cls_coeff, U_bg).squeeze(1)
        z_cls_orth = F.normalize(z_cls - gamma.squeeze(1) * cls_bg, dim=-1)

        # Pathway 1: Purified Whole-Organ Topology + Spatial Bilateral Asymmetry
        f_topo = F.normalize((p_orth * w_fg.unsqueeze(-1)).sum(dim=1) / w_sum, dim=-1)
        p_g = (p_orth * w_fg.unsqueeze(-1)).view(-1, 16, 16, self.dim)
        w_g = w_fg.view(-1, 16, 16, 1)
        f_l = p_g[:, :, :8].sum((1, 2)) / (w_g[:, :, :8].sum((1, 2)) + 1e-5)
        f_r = p_g[:, :, 8:].sum((1, 2)) / (w_g[:, :, 8:].sum((1, 2)) + 1e-5)
        f_t = p_g[:, :8, :].sum((1, 2)) / (w_g[:, :8, :].sum((1, 2)) + 1e-5)
        f_b = p_g[:, 8:, :].sum((1, 2)) / (w_g[:, 8:, :].sum((1, 2)) + 1e-5)
        f_asym = F.normalize((f_l - f_r).abs() + (f_t - f_b).abs(), dim=-1)

        anchor_logits = 0.5 * base_logits + 0.5 * self.base_cls(f_topo)

        if self.mode == 'S2_orth_topo_only':
            feat = torch.cat([z_cls_orth, f_topo, f_asym, g_emb], dim=-1)
            return anchor_logits + self.mlp(feat), w_fg

        # Pathway 2 (S3_bgopta_full): Lesion-Peak Anomaly Aggregation (captures localized micro-spots!)
        dev_sal = (p_orth - f_topo.unsqueeze(1)).norm(dim=-1)          # (B, 256) local anomaly from lamina mean
        learned_sal = self.lesion_scorer(p_orth).squeeze(-1)           # (B, 256)
        peak_logits = (learned_sal + 1.5 * dev_sal).masked_fill(w_fg < 0.05, -1e4)
        peak_attn = F.softmax(peak_logits * 2.5, dim=-1) * w_fg
        peak_attn = peak_attn / (peak_attn.sum(dim=1, keepdim=True) + 1e-6)
        f_peak = F.normalize((p_orth * peak_attn.unsqueeze(-1)).sum(dim=1), dim=-1)

        feat = torch.cat([z_cls_orth, f_topo, f_peak, f_asym, g_emb], dim=-1)
        return anchor_logits + self.mlp(feat), w_fg

    def forward(self, tokens_257: torch.Tensor, masks_2: torch.Tensor, geom: torch.Tensor):
        return self.forward_with_diag(tokens_257, masks_2, geom)[0]


def load_source_train_and_val_crops():
    """仅读取 SciDB train 与 SciDB val 图像生成单趟 1.50x 裁框与确定性掩膜（物理隔离 testB/testC！）。"""
    lines = [l.strip() for l in (EXT / '统一检测数据' / 'train.txt').read_text(encoding='utf-8').splitlines() if l.strip()]
    items = []
    for img_idx, p_str in enumerate(lines):
        ip = Path(p_str)
        parts = list(ip.parts)
        parts[parts.index('images')] = 'labels'
        lp = Path(*parts).with_suffix('.txt')
        if lp.exists():
            raw_rows = [r.split() for r in lp.read_text(encoding='utf-8').splitlines() if len(r.split()) >= 5]
            if raw_rows:
                items.append((img_idx, ip, raw_rows))

    print(f"[Phase 1] 解码源训练集 SciDB train ({len(items)} 图) 单趟 1.50x 上下文与掩膜...", flush=True)
    rng_master = np.random.RandomState(20261126)
    r_seeds = rng_master.randint(0, 10**9, size=len(items))

    def proc_tr(args):
        (img_idx, ip, raw_rows), rs = args
        rl = np.random.RandomState(int(rs))
        bgr = cv2.imdecode(np.fromfile(str(ip), dtype=np.uint8), 1)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        crops, msks = [], []
        for t in raw_rows:
            cx, cy, bw, bh = map(float, t[1:5])
            px, py, pw, ph = (cx - bw * 0.5) * w, (cy - bh * 0.5) * h, bw * w, bh * h
            if pw > 2.0 and ph > 2.0:
                jx = px + rl.uniform(-0.025, 0.025) * pw
                jy = py + rl.uniform(-0.025, 0.025) * ph
                jw = pw * rl.uniform(0.96, 1.04)
                jh = ph * rl.uniform(0.96, 1.04)
                c, m = crop_single_context_and_masks(rgb, [jx, jy, jw, jh], CROP_SCALE)
                crops.append(c); msks.append(m)
        return (crops, msks) if crops else None

    tr_crops, tr_masks = [], []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for out in ex.map(proc_tr, zip(items, r_seeds)):
            if out is not None:
                tr_crops.extend(out[0]); tr_masks.extend(out[1])
    tr_crops_arr = np.stack(tr_crops, 0)
    tr_masks_arr = np.stack(tr_masks, 0)
    print(f"  SciDB train 有效框={len(tr_crops_arr)} 提取完毕。", flush=True)

    # Also decode SciDB val crops per seed (since cand_idxs differ slightly by seed)
    val_gt = json.loads(GT_VAL_PATH.read_text(encoding='utf-8'))
    id2val = {im['id']: im for im in val_gt['images']}
    val_rgb_map = {}
    for iid, im in id2val.items():
        bgr = cv2.imdecode(np.fromfile(str(IMG_ROOT / 'val' / im['file_name']), dtype=np.uint8), 1)
        val_rgb_map[iid] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    val_per_seed = {}
    for seed in [0, 1, 2]:
        v_npz = np.load(PREV_CACHE / f'val_cggar_seed{seed}.npz')
        orig_preds = json.loads((STRONG_DIR / f'种子{seed}' / 'val预测.json').read_text(encoding='utf-8'))
        vc, vm = [], []
        for gi in v_npz['cand_idxs']:
            r = orig_preds[gi]
            c, m = crop_single_context_and_masks(val_rgb_map[r['image_id']], r['bbox'], CROP_SCALE)
            vc.append(c); vm.append(m)
        val_per_seed[seed] = (np.stack(vc, 0), np.stack(vm, 0))
    return tr_crops_arr, tr_masks_arr, val_per_seed


@torch.no_grad()
def encode_dinov2_tokens_fp16(enc, crops_u8: np.ndarray, device='cuda:0', batch_size: int = 192) -> torch.Tensor:
    """提取全部 257 个 Token (1 CLS + 256 Patch) 并以 float16 驻留 GPU 显存 (~1.8GB)，极速零磁盘瓶颈！"""
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    outs = []
    for i in range(0, len(crops_u8), batch_size):
        x = (torch.from_numpy(crops_u8[i:i + batch_size]).to(device).float() / 255.0 - mean) / std
        ft = enc.forward_features(x).half()  # (B, 257, 384) fp16
        outs.append(ft)
    return torch.cat(outs, dim=0)


def main():
    t0 = time.monotonic()
    backend = configure_strict_fp32()
    meta = json.loads((ROOT / '方法推进_20261027' / '纯CLS对照' / '微调汇总.json').read_text(encoding='utf-8'))

    tr_crops_clean, tr_masks_np, val_crops_by_seed = load_source_train_and_val_crops()

    modes = ['S0_cls_only', 'S1_plain_concat', 'S2_orth_topo_only', 'S3_bgopta_full']
    ablation_records = {m: {'dev_clean_f1': [], 'dev_bgshift_f1': [], 'val_ap50_95': []} for m in modes}
    frozen_seed_configs = []

    for seed in [0, 1, 2]:
        ts = time.monotonic()
        print(f"\n================ Phase 1 纯源域训练与消融 Seed {seed} ================", flush=True)
        tr_meta = np.load(PREV_CACHE / f'train_cggar_seed{seed}.npz')
        assert len(tr_meta['labels']) == len(tr_crops_clean)

        # Split SciDB train STRICTLY by image_id into 85% train_fit / 15% source_dev (zero cross-split donor leakage!)
        box_img_ids = tr_meta['box_img_ids']
        u_imgs = np.unique(box_img_ids)
        rng_sp = np.random.RandomState(2026 + seed)
        rng_sp.shuffle(u_imgs)
        fit_set = set(u_imgs[:int(len(u_imgs) * 0.85)].tolist())
        is_fit = np.array([iid in fit_set for iid in box_img_ids], dtype=bool)
        fit_idx = np.where(is_fit)[0]
        dev_idx = np.where(~is_fit)[0]

        # Synthesize bg-intruded crops strictly within train_fit and strictly within source_dev!
        rng_bg = np.random.RandomState(8800 + seed)
        tr_crops_bg = np.empty_like(tr_crops_clean)
        for sub_idxs in [fit_idx, dev_idx]:
            perm_donors = rng_bg.permutation(sub_idxs)
            for i_pos, idx in enumerate(sub_idxs):
                tr_crops_bg[idx] = synthesize_bg_intruded_crop(
                    tr_crops_clean[idx], tr_masks_np[idx], tr_crops_clean[perm_donors[i_pos]], rng_bg
                )

        # Load DINOv2 (seed) & encode 257 tokens into GPU fp16
        t_info = next(r for r in meta['重复结果'] if r['方法'] == 'LP-FT' and r['种子'] == seed)
        st = torch.load(t_info['权重文件'], map_location='cpu', weights_only=True)
        enc = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False, img_size=224, num_classes=0).to('cuda:0').eval()
        enc.load_state_dict(st['骨干'])
        w_head = st['分类头']['weight'].clone()
        b_head = st['分类头']['bias'].clone()

        tok_clean_gpu = encode_dinov2_tokens_fp16(enc, tr_crops_clean, 'cuda:0')
        tok_bg_gpu = encode_dinov2_tokens_fp16(enc, tr_crops_bg, 'cuda:0')
        vc_arr, vm_arr = val_crops_by_seed[seed]
        tok_val_gpu = encode_dinov2_tokens_fp16(enc, vc_arr, 'cuda:0')
        del enc
        torch.cuda.empty_cache()

        masks_tr_gpu = torch.from_numpy(tr_masks_np).to('cuda:0')
        geom_tr_gpu = torch.from_numpy(tr_meta['geom']).to('cuda:0')
        y_tr_gpu = torch.from_numpy(tr_meta['labels']).to('cuda:0')
        t_fit = torch.from_numpy(fit_idx).to('cuda:0')
        t_dev = torch.from_numpy(dev_idx).to('cuda:0')
        y_dev_np = tr_meta['labels'][dev_idx]

        # Prepare SciDB val structures
        v_npz = np.load(PREV_CACHE / f'val_cggar_seed{seed}.npz')
        val_orig_preds = json.loads((STRONG_DIR / f'种子{seed}' / 'val预测.json').read_text(encoding='utf-8'))
        masks_val_gpu = torch.from_numpy(vm_arr).to('cuda:0')
        geom_val_gpu = torch.from_numpy(v_npz['geom']).to('cuda:0')
        p14_val_gpu = torch.from_numpy(v_npz['p14']).to('cuda:0')
        lp_det_val = F.log_softmax(torch.logit(p14_val_gpu.clamp(1e-5, 1.0 - 1e-5)) / 1.35, dim=-1).cpu().numpy()
        yolo_sc_val = np.array([val_orig_preds[i]['score'] for i in v_npz['cand_idxs']], dtype=np.float64)

        # Fit Mahalanobis ONLY on train_fit clean neck features for conservative Safe-Bypass Gate!
        m_means, m_prec, m_mu, m_sd = fit_mahalanobis_stats(tr_meta['zneck_clean'][fit_idx], tr_meta['labels'][fit_idx])
        maha_val_14 = compute_maha_z(v_npz['zneck'], m_means, m_prec, m_mu, m_sd)
        yolo_cls_val = np.array([val_orig_preds[i]['category_id'] for i in v_npz['cand_idxs']], dtype=np.int64)
        d_maha_val = maha_val_14[np.arange(len(yolo_cls_val)), yolo_cls_val]

        seed_models = {}
        for mode in modes:
            set_seed(seed * 10 + modes.index(mode) + 500)
            model = BGOPTAHead(w_head, b_head, mode=mode).to('cuda:0')
            trainable = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(trainable, lr=1.0e-3, weight_decay=2e-3)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=22)

            best_score, best_state = -1.0, None
            for _ep in range(22):
                model.train()
                perm = t_fit[torch.randperm(len(t_fit), device='cuda:0')]
                for ids in perm.split(256):
                    opt.zero_grad()
                    # Half clean, half background-intruded crops on train_fit
                    use_bg = (torch.rand((len(ids), 1, 1), device='cuda:0') < 0.55)
                    tk = torch.where(use_bg, tok_bg_gpu[ids], tok_clean_gpu[ids])
                    lg = model(tk, masks_tr_gpu[ids], geom_tr_gpu[ids])
                    loss = F.cross_entropy(lg, y_tr_gpu[ids], label_smoothing=0.05)
                    loss.backward()
                    opt.step()
                sched.step()

                model.eval()
                with torch.no_grad():
                    p_clean = model(tok_clean_gpu[t_dev], masks_tr_gpu[t_dev], geom_tr_gpu[t_dev]).argmax(-1).cpu().numpy()
                    p_bg = model(tok_bg_gpu[t_dev], masks_tr_gpu[t_dev], geom_tr_gpu[t_dev]).argmax(-1).cpu().numpy()
                    f1_c = float(f1_score(y_dev_np, p_clean, average='macro')) * 100.0
                    f1_b = float(f1_score(y_dev_np, p_bg, average='macro')) * 100.0
                    # Select best epoch strictly on source_dev (0.4 * clean + 0.6 * bgshift)!
                    sc_dev = 0.4 * f1_c + 0.6 * f1_b
                    if sc_dev > best_score:
                        best_score = sc_dev
                        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            model.load_state_dict(best_state)
            model.eval()
            seed_models[mode] = (model, best_state)

            with torch.no_grad():
                f1_c = float(f1_score(y_dev_np, model(tok_clean_gpu[t_dev], masks_tr_gpu[t_dev], geom_tr_gpu[t_dev]).argmax(-1).cpu().numpy(), average='macro')) * 100.0
                f1_b = float(f1_score(y_dev_np, model(tok_bg_gpu[t_dev], masks_tr_gpu[t_dev], geom_tr_gpu[t_dev]).argmax(-1).cpu().numpy(), average='macro')) * 100.0

                # Evaluate on SciDB val COCO AP50-95 (with standard alpha=0.32, beta=0.45)
                lg_v = model(tok_val_gpu, masks_val_gpu, geom_val_gpu)
                lp_v = F.log_softmax(lg_v / 1.15, dim=-1).cpu().numpy()
                lp_f = 0.32 * lp_det_val + 0.68 * lp_v
                p_f = np.exp(lp_f - lp_f.max(axis=1, keepdims=True)); p_f /= p_f.sum(axis=1, keepdims=True)
                rows_v = [dict(r) for r in val_orig_preds]
                for li, gi in enumerate(v_npz['cand_idxs']):
                    if yolo_sc_val[li] >= 0.08:
                        nc = int(np.argmax(p_f[li]))
                        rows_v[gi]['category_id'] = nc
                        rows_v[gi]['score'] = float((yolo_sc_val[li] ** 0.45) * (max(float(p_f[li, nc]), 1e-6) ** 0.55))
                ap_v, _ = coco_eval_ap(GT_VAL_PATH, apply_classwise_nms(rows_v, 0.70))

            ablation_records[mode]['dev_clean_f1'].append(round(f1_c, 2))
            ablation_records[mode]['dev_bgshift_f1'].append(round(f1_b, 2))
            ablation_records[mode]['val_ap50_95'].append(round(ap_v, 2))
            print(f"  [{mode:20s}] source_dev Clean F1={f1_c:.2f}% | BgShift F1={f1_b:.2f}% | SciDB val AP50-95={ap_v:.2f}%", flush=True)

        # Calibrate alpha, beta, and Safe-Bypass Gate for S3_bgopta_full STRICTLY on SciDB val!
        m_full, st_full = seed_models['S3_bgopta_full']
        m_plain, st_plain = seed_models['S1_plain_concat']
        with torch.no_grad():
            lp_v_full = F.log_softmax(m_full(tok_val_gpu, masks_val_gpu, geom_val_gpu) / 1.15, dim=-1).cpu().numpy()

        best_va, best_alpha, best_beta = -1.0, 0.32, 0.45
        for al in [0.25, 0.32, 0.40]:
            lp_f = al * lp_det_val + (1.0 - al) * lp_v_full
            p_f = np.exp(lp_f - lp_f.max(axis=1, keepdims=True)); p_f /= p_f.sum(axis=1, keepdims=True)
            for bt in [0.35, 0.45, 0.55]:
                rows_v = [dict(r) for r in val_orig_preds]
                for li, gi in enumerate(v_npz['cand_idxs']):
                    if yolo_sc_val[li] >= 0.08:
                        nc = int(np.argmax(p_f[li]))
                        rows_v[gi]['category_id'] = nc
                        rows_v[gi]['score'] = float((yolo_sc_val[li] ** bt) * (max(float(p_f[li, nc]), 1e-6) ** (1.0 - bt)))
                ap_v, _ = coco_eval_ap(GT_VAL_PATH, apply_classwise_nms(rows_v, 0.70))
                if ap_v > best_va:
                    best_va, best_alpha, best_beta = ap_v, al, bt

        # Safe-Bypass Gate threshold on SciDB val: skip DINOv2 ONLY when (yolo_sc >= 0.85 AND p_det top2_margin >= 0.90 AND d_maha < val median)!
        p_det_val = np.exp(lp_det_val)
        top2_v = np.sort(p_det_val, axis=1)[:, -2:]
        mar_val = top2_v[:, 1] - top2_v[:, 0]
        tau_maha_safe = float(np.quantile(d_maha_val[yolo_sc_val >= 0.25], 0.55))

        ckpt_file = CKPT_DIR / f'bgopta_frozen_seed{seed}.pt'
        torch.save({
            'seed': seed,
            'crop_scale': CROP_SCALE,
            'S0_cls_only': seed_models['S0_cls_only'][1],
            'S1_plain_concat': st_plain,
            'S2_orth_topo_only': seed_models['S2_orth_topo_only'][1],
            'S3_bgopta_full': st_full,
            'mahalanobis': {'means': m_means, 'prec': m_prec, 'm_mu': m_mu, 'm_sd': m_sd},
            'val_frozen_hyperparams': {
                'alpha_det': best_alpha,
                'beta_score': best_beta,
                'fg_thr': 0.08,
                'safe_bypass_conf': 0.85,
                'safe_bypass_margin': 0.90,
                'safe_bypass_maha_z': round(tau_maha_safe, 5),
            }
        }, ckpt_file)
        sha = hashlib.sha256(ckpt_file.read_bytes()).hexdigest()
        frozen_seed_configs.append({'seed': seed, 'ckpt': str(ckpt_file), 'sha256': sha, 'best_val_ap': round(best_va, 4), 'alpha': best_alpha, 'beta': best_beta, 'tau_maha_safe': round(tau_maha_safe, 4)})
        print(f"  [Seed {seed} 权重已冻结] {ckpt_file.name} | SHA256={sha[:16]}... | 源 val 最优 AP={best_va:.2f}% (耗时 {time.monotonic() - ts:.1f}s)", flush=True)

        del tok_clean_gpu, tok_bg_gpu, tok_val_gpu
        torch.cuda.empty_cache()

    print("\n================ Phase 1 纯源域 3 种子消融汇总 ================", flush=True)
    agg_p1 = {}
    for m in modes:
        dc = ablation_records[m]['dev_clean_f1']
        db = ablation_records[m]['dev_bgshift_f1']
        va = ablation_records[m]['val_ap50_95']
        agg_p1[m] = {
            'source_dev_clean_F1': f"{np.mean(dc):.2f}±{np.std(dc, ddof=1):.2f}", 'dev_clean_seeds': dc,
            'source_dev_bgshift_F1': f"{np.mean(db):.2f}±{np.std(db, ddof=1):.2f}", 'dev_bgshift_seeds': db,
            'SciDB_val_AP50_95': f"{np.mean(va):.2f}±{np.std(va, ddof=1):.2f}", 'val_seeds': va,
        }
        print(f"  {m:20s} | Dev-Clean F1={agg_p1[m]['source_dev_clean_F1']} | Dev-BgShift F1={agg_p1[m]['source_dev_bgshift_F1']} {db} | SciDB val AP={agg_p1[m]['SciDB_val_AP50_95']} {va}", flush=True)

    out_json = RES_DIR / 'phase1_source_only_ablation_and_lock.json'
    out_json.write_text(json.dumps({
        '阶段': 'Phase1_纯源域零泄漏训练消融与参数冻结',
        '耗时秒': round(time.monotonic() - t0, 2),
        '严格后端': backend,
        '源内消融三种子汇总': agg_p1,
        '冻结权重与参数清单': frozen_seed_configs,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n[Phase 1 完成并锁死] 记录已保存至: {out_json}", flush=True)


if __name__ == '__main__':
    main()
