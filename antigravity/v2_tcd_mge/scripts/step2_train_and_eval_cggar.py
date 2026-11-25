# -*- coding: utf-8 -*-
"""
Step 2: 训练与严格评估反事实收益感知跨粒度证据仲裁网络 (CG-GAR, 3 随机种子 0/1/2)
严格遵守学术诚信与预注册门禁：
1. 所有神经网络参数 (目标-环境相对差分头 R_psi、第一级反事实收益门控 G_theta、第二级跨粒度冲突仲裁器 A_phi)
   100% 仅在源训练集 SciDB train (按图像茎分组划分 85% train_fit / 15% train_dev) 上训练；
2. 温度校准系数与门控预算分位数 100% 仅在 SciDB val (866 图) 上标定并冻结；
3. 严禁接触或利用 testB / testC 调整任何参数；最后在同进程严格 FP32 COCO AP50-95 口径下一次性评估全部基线与消融，
   并逐种子检验是否 3/3 种子全部严格战胜 B0(原生)、B1(全量DINOv2)、B2(MGS-v1) 与 B3(静态后验融合)！
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms
from faster_coco_eval import COCO, COCOeval_faster

ROOT = Path(r'D:\新版柑橘')
EXT = ROOT / '方法路线扩展_20261031'
sys.path.append(str(EXT))
from 共同评测后端 import configure_strict_fp32

WORK_DIR = ROOT / 'Antigravity_CGGAR_真创新攻关_20261125'
CACHE_DIR = WORK_DIR / 'cache'
CKPT_DIR = WORK_DIR / 'models'
RES_DIR = WORK_DIR / 'results'
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

GT_DIR = EXT / '检测数据'
STRONG_DIR = EXT / 'YOLOv8强基线重复_20261109'


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def coco_eval_full(gt_path: Path, pred_rows: list):
    if not pred_rows:
        return [0.0] * 12, [0.0] * 14
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_path))
        dt = gt.loadRes(pred_rows)
        ev = COCOeval_faster(gt, dt, 'bbox')
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        prec = ev.eval['precision']
        per_cls = []
        for k in range(14):
            v = prec[:, :, k, 0, -1]
            v = v[v > -1]
            per_cls.append(round(float(v.mean() * 100.0), 2) if v.size else 0.0)
    return ev.stats.tolist(), per_cls


def apply_classwise_nms(rows: list, iou_thr: float = 0.70) -> list:
    """对同图候选框执行与 YOLOv8n 原生一致的同类别 NMS (iou=0.70)，消除跨类改判引发的同叶重复假阳性。"""
    by_img = defaultdict(list)
    for r in rows:
        by_img[r['image_id']].append(r)
    kept_all = []
    for _iid, rlist in by_img.items():
        if len(rlist) <= 1:
            kept_all.extend(rlist)
            continue
        boxes_xyxy = torch.tensor([[r['bbox'][0], r['bbox'][1], r['bbox'][0] + r['bbox'][2], r['bbox'][1] + r['bbox'][3]] for r in rlist], dtype=torch.float32)
        scores = torch.tensor([r['score'] for r in rlist], dtype=torch.float32)
        idxs = torch.tensor([r['category_id'] for r in rlist], dtype=torch.int64)
        keep = batched_nms(boxes_xyxy, scores, idxs, iou_thr).tolist()
        if len(keep) > 300:
            keep = keep[:300]
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
    # Compute train distribution mean & std for z-score normalization
    diff_tr = feats.astype(np.float64) - means[labels]
    d_tr = np.sqrt(np.maximum(0.0, np.sum((diff_tr @ prec) * diff_tr, axis=1)))
    return means, prec, float(d_tr.mean()), float(d_tr.std() + 1e-6)


def compute_maha_z(feats: np.ndarray, means: np.ndarray, prec: np.ndarray, m_mu: float, m_sd: float):
    """返回每个样本对全部 14 类的标准化马氏距离矩阵 (N, 14)。"""
    N = len(feats)
    out = np.zeros((N, 14), dtype=np.float32)
    X = feats.astype(np.float64)
    for c in range(14):
        df = X - means[c]
        dc = np.sqrt(np.maximum(0.0, np.sum((df @ prec) * df, axis=1)))
        out[:, c] = ((dc - m_mu) / m_sd).astype(np.float32)
    return out


def p14_to_norm_prob(p14_sig: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    """将 YOLOv8n 14 类 Sigmoid 响应转换为温度校准后的 14 类分类后验分布。"""
    p_clip = p14_sig.clamp(1e-5, 1.0 - 1e-5)
    logits = torch.logit(p_clip) / temp
    return F.softmax(logits, dim=-1)


def entropy_norm(probs: torch.Tensor) -> torch.Tensor:
    p = probs.clamp_min(1e-7)
    return -(p * p.log()).sum(dim=-1, keepdim=True) / math.log(14.0)


def top2_margin(probs: torch.Tensor) -> torch.Tensor:
    top2 = torch.topk(probs, k=2, dim=-1).values
    return (top2[:, 0:1] - top2[:, 1:2])


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = p.clamp_min(1e-7)
    q = q.clamp_min(1e-7)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1, keepdim=True)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1, keepdim=True)
    return 0.5 * (kl_pm + kl_qm) / math.log(2.0)


class TargetContextRelationalHead(nn.Module):
    """
    目标-近邻环境相对差分分类头 R_psi：
    显式计算 z_core - z_ctx 与 z_std - z_ctx 剥离白板反光/果园环境光色偏，结合框尺度几何描述子 g(b) 输出关系分类 logits。
    """
    def __init__(self, emb_dim: int = 384, geom_dim: int = 9, num_classes: int = 14):
        super().__init__()
        # 输入维度: z_std(384) + z_core(384) + (z_core - z_ctx)(384) + (z_std - z_ctx)(384) + geom_proj(32) = 1568
        self.geom_proj = nn.Sequential(
            nn.Linear(geom_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
        )
        self.net = nn.Sequential(
            nn.LayerNorm(emb_dim * 4 + 32),
            nn.Dropout(0.35),
            nn.Linear(emb_dim * 4 + 32, 384),
            nn.GELU(),
            nn.LayerNorm(384),
            nn.Dropout(0.25),
            nn.Linear(384, num_classes),
        )

    def forward(self, z_core: torch.Tensor, z_std: torch.Tensor, z_ctx: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
        d_core_ctx = F.normalize(z_core - z_ctx, dim=-1)
        d_std_ctx = F.normalize(z_std - z_ctx, dim=-1)
        g = self.geom_proj(geom)
        feat = torch.cat([z_std, z_core, d_core_ctx, d_std_ctx, g], dim=-1)
        return self.net(feat)


class Stage1GainRiskGate(nn.Module):
    """
    第一级零裁图开销反事实收益感知门控 G_theta：
    同时预测当前框原始检测出错风险 r_cls 与触发二阶段重审的预期交叉熵净收益 Delta_U。
    """
    def __init__(self, in_dim: int = 29):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.GELU(),
        )
        self.head_err = nn.Linear(32, 1)   # logit of P(YOLO wrong)
        self.head_gain = nn.Linear(32, 1)  # predicted expected gain E[L_det - L_ref]

    def forward(self, u1: torch.Tensor):
        h = self.mlp(u1)
        r_logit = self.head_err(h)
        gain_pred = self.head_gain(h)
        return r_logit, gain_pred


class CounterfactualCrossGranularityArbitrator(nn.Module):
    """
    第二级反事实收益感知跨粒度证据仲裁器 A_phi：
    仅依赖视角不变的不确定性、跨视角分歧度、特征空间余弦一致性、流形偏移与框几何 (24 维元特征，杜绝源域类别过拟合)，
    动态预测整图检测分布与局部多粒度视图的自适应对数后验融合权重！
    """
    def __init__(self, meta_dim: int = 24):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(meta_dim),
            nn.Linear(meta_dim, 64),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 4),  # adaptive residual log-weights for [det, std, core, rel]
        )
        # 初始化偏置使起始状态平滑接近稳健先验
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, meta_f: torch.Tensor, escalated: torch.Tensor,
                lp_det: torch.Tensor, lp_std: torch.Tensor, lp_core: torch.Tensor, lp_rel: torch.Tensor) -> torch.Tensor:
        raw = self.net(meta_f)
        # Base prior logits: [det=-0.35, std=0.25, core=-0.25, rel=0.35]
        prior = raw.new_tensor([-0.35, 0.25, -0.25, 0.35]).unsqueeze(0)
        logits_w = prior + 0.85 * torch.tanh(raw)
        # 若未升级 Level-2 (escalated == 0)，屏蔽 core 与 rel 分支，仅在 det 与 std 间动态仲裁！
        mask_no_esc = (escalated.squeeze(-1) < 0.5)
        if mask_no_esc.any():
            logits_w = logits_w.clone()
            logits_w[mask_no_esc, 2] = -1e9
            logits_w[mask_no_esc, 3] = -1e9
        w = F.softmax(logits_w, dim=-1)  # (N, 4)
        fused_lp = (
            w[:, 0:1] * lp_det
            + w[:, 1:2] * lp_std
            + w[:, 2:3] * lp_core
            + w[:, 3:4] * lp_rel
        )
        return F.log_softmax(fused_lp, dim=-1), w


def build_u1_features(p14_sig: torch.Tensor, p_det: torch.Tensor, maha_14: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
    """构建第一级门控 29 维特征 u1。"""
    s_obj = p14_sig.max(dim=-1, keepdim=True).values
    s_sum = p14_sig.sum(dim=-1, keepdim=True)
    ent = entropy_norm(p_det)
    mar = top2_margin(p_det)
    c_hat = p_det.argmax(dim=-1)
    d_pred = maha_14.gather(1, c_hat.unsqueeze(1)).clamp(-3.0, 15.0) / 5.0
    d_min = maha_14.min(dim=-1, keepdim=True).values.clamp(-3.0, 15.0) / 5.0
    onehot = F.one_hot(c_hat, num_classes=14).float()
    return torch.cat([s_obj, s_sum, ent, mar, d_pred, d_min, geom, onehot], dim=-1)  # 6 + 9 + 14 = 29


def build_meta_features(p14_sig: torch.Tensor, p_det: torch.Tensor, p_std: torch.Tensor, p_core: torch.Tensor, p_rel: torch.Tensor,
                        z_core: torch.Tensor, z_std: torch.Tensor, z_ctx: torch.Tensor, maha_14: torch.Tensor, geom: torch.Tensor,
                        escalated: torch.Tensor) -> torch.Tensor:
    """构建第二级仲裁器 24 维视角不变诊断元特征。"""
    s_obj = p14_sig.max(dim=-1, keepdim=True).values
    h_det = entropy_norm(p_det)
    h_std = entropy_norm(p_std)
    h_core = entropy_norm(p_core) * escalated
    h_rel = entropy_norm(p_rel) * escalated

    m_det = top2_margin(p_det)
    m_std = top2_margin(p_std)
    m_rel = top2_margin(p_rel) * escalated

    dis_det_std = (p_det.argmax(-1) != p_std.argmax(-1)).float().unsqueeze(1)
    js_det_std = js_divergence(p_det, p_std)
    js_std_rel = js_divergence(p_std, p_rel) * escalated
    js_det_rel = js_divergence(p_det, p_rel) * escalated

    cos_cs = (z_core * z_std).sum(dim=-1, keepdim=True) * escalated
    cos_sx = (z_std * z_ctx).sum(dim=-1, keepdim=True) * escalated

    c_hat = p_det.argmax(dim=-1)
    d_pred = maha_14.gather(1, c_hat.unsqueeze(1)).clamp(-3.0, 15.0) / 5.0
    return torch.cat([
        s_obj, h_det, h_std, h_core, h_rel,
        m_det, m_std, m_rel,
        dis_det_std, js_det_std, js_std_rel, js_det_rel,
        cos_cs, cos_sx, d_pred, geom
    ], dim=-1)  # 15 + 9 = 24


def train_cggar_for_seed(seed: int, device='cuda:0'):
    set_seed(seed + 100)
    tr = np.load(CACHE_DIR / f'train_cggar_seed{seed}.npz')
    N = len(tr['labels'])
    box_img_ids = tr['box_img_ids']
    unique_imgs = np.unique(box_img_ids)
    rng = np.random.RandomState(2026 + seed)
    rng.shuffle(unique_imgs)
    n_fit = int(len(unique_imgs) * 0.85)
    fit_img_set = set(unique_imgs[:n_fit].tolist())
    is_fit = np.array([iid in fit_img_set for iid in box_img_ids], dtype=bool)
    fit_idx = np.where(is_fit)[0]
    dev_idx = np.where(~is_fit)[0]

    # Fit Mahalanobis ONLY on train_fit clean neck features!
    means, prec, m_mu, m_sd = fit_mahalanobis_stats(tr['zneck_clean'][fit_idx], tr['labels'][fit_idx])
    maha_clean = compute_maha_z(tr['zneck_clean'], means, prec, m_mu, m_sd)
    maha_deg = compute_maha_z(tr['zneck_deg'], means, prec, m_mu, m_sd)

    # Move tensors to GPU
    y_all = torch.from_numpy(tr['labels']).to(device)
    geom_all = torch.from_numpy(tr['geom']).to(device)
    zc_all = torch.from_numpy(tr['z_core'].astype(np.float32)).to(device)
    zs_all = torch.from_numpy(tr['z_std'].astype(np.float32)).to(device)
    zx_all = torch.from_numpy(tr['z_ctx'].astype(np.float32)).to(device)
    zcf_all = torch.from_numpy(tr['z_cf_bad'].astype(np.float32)).to(device)

    lgc_all = torch.from_numpy(tr['lg_core']).to(device)
    lgs_all = torch.from_numpy(tr['lg_std']).to(device)
    lgcf_all = torch.from_numpy(tr['lg_cf_bad']).to(device)

    p14_c = torch.from_numpy(tr['p14_clean']).to(device)
    p14_d = torch.from_numpy(tr['p14_deg']).to(device)
    mc_gpu = torch.from_numpy(maha_clean).to(device)
    md_gpu = torch.from_numpy(maha_deg).to(device)

    t_fit = torch.from_numpy(fit_idx).to(device)
    t_dev = torch.from_numpy(dev_idx).to(device)

    # ---------------- 1. Train Target-Context Relational Head R_psi ----------------
    rel_head = TargetContextRelationalHead().to(device)
    opt_r = torch.optim.AdamW(rel_head.parameters(), lr=1.2e-3, weight_decay=2e-3)
    sched_r = torch.optim.lr_scheduler.CosineAnnealingLR(opt_r, T_max=20)
    best_rel_loss = 1e9
    best_rel_state = None

    for ep in range(20):
        rel_head.train()
        perm = t_fit[torch.randperm(len(t_fit), device=device)]
        for ids in perm.split(256):
            opt_r.zero_grad()
            zc_b, zs_b, zx_b = zc_all[ids], zs_all[ids], zx_all[ids]
            # Common-mode background/illumination shift injection: shared perturbation cancels in (z_core - z_ctx)!
            cm_noise = torch.randn((len(ids), 384), device=device) * 0.08
            zc_in = F.normalize(zc_b + cm_noise, dim=-1)
            zs_in = F.normalize(zs_b + cm_noise, dim=-1)
            zx_in = F.normalize(zx_b + cm_noise, dim=-1)
            # On 25% of batch, simulate standard crop border corruption while core & ctx remain informative
            corrupt_m = torch.rand(len(ids), device=device) < 0.25
            if corrupt_m.any():
                zs_in = torch.where(corrupt_m.unsqueeze(1), zcf_all[ids], zs_in)
            lg_r = rel_head(zc_in, zs_in, zx_in, geom_all[ids])
            loss_r = F.cross_entropy(lg_r, y_all[ids], label_smoothing=0.05)
            loss_r.backward()
            opt_r.step()
        sched_r.step()

        rel_head.eval()
        with torch.no_grad():
            dev_lg = rel_head(zc_all[t_dev], zs_all[t_dev], zx_all[t_dev], geom_all[t_dev])
            # Validate on mixture of clean & border-corrupted dev samples
            dev_lg_cf = rel_head(zc_all[t_dev], zcf_all[t_dev], zx_all[t_dev], geom_all[t_dev])
            v_loss = 0.5 * F.cross_entropy(dev_lg, y_all[t_dev]).item() + 0.5 * F.cross_entropy(dev_lg_cf, y_all[t_dev]).item()
            if v_loss < best_rel_loss:
                best_rel_loss = v_loss
                best_rel_state = {k: v.detach().cpu() for k, v in rel_head.state_dict().items()}

    rel_head.load_state_dict(best_rel_state)
    rel_head.eval()
    with torch.no_grad():
        lgr_clean_all = rel_head(zc_all, zs_all, zx_all, geom_all)
        lgr_cf_all = rel_head(zc_all, zcf_all, zx_all, geom_all)

    # ---------------- 2. Train Stage-1 Counterfactual Gain Gate G_theta & Stage-2 Arbitrator A_phi ----------------
    # Note: To match out-of-sample calibration, soften clean in-sample train logits slightly (T_in=1.6) so train entropies match val!
    T_DET, T_DINO, T_REL = 1.35, 1.15, 1.20
    gate = Stage1GainRiskGate(in_dim=29).to(device)
    arb = CounterfactualCrossGranularityArbitrator(meta_dim=24).to(device)
    opt_ga = torch.optim.AdamW(list(gate.parameters()) + list(arb.parameters()), lr=1.5e-3, weight_decay=1e-3)
    sched_ga = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ga, T_max=24)

    def assemble_counterfactual_batch(ids: torch.Tensor, training: bool = True):
        """
        为每个样本在 4 种反事实模态中均衡采样，生成具有真实正负纠错收益 Delta_U 的训练四元组！
        Regime 0 (25%): Clean Det + Clean Crop (Mode A, 零或微收益)
        Regime 1 (35%): Degraded Blur/Gamma Det + Clean Crop (Mode B, 显著正收益 Delta_U > 0)
        Regime 2 (25%): Clean Det + Bad Truncated/Glare Std Crop (Mode C, 单紧框负收益, 需靠 Det 与 Core/Rel 挽救!)
        Regime 3 (15%): Degraded Det + Noisy Std Crop + Intact Core/Rel (Mode D, 双视角冲突需升级三级仲裁!)
        """
        B = len(ids)
        if training:
            reg = torch.randint(0, 100, (B,), device=device)
        else:
            reg = (torch.arange(B, device=device) * 29) % 100

        m_b = (reg >= 25) & (reg < 60)
        m_c = (reg >= 60) & (reg < 85)
        m_d = (reg >= 85)

        p14_use = torch.where((m_b | m_d).unsqueeze(1), p14_d[ids], p14_c[ids])
        maha_use = torch.where((m_b | m_d).unsqueeze(1), md_gpu[ids], mc_gpu[ids])

        # Soften clean in-sample train logits slightly so confidence distribution matches unseen val
        t_det_eff = torch.where((m_b | m_d).unsqueeze(1), torch.full((B, 1), T_DET, device=device), torch.full((B, 1), T_DET * 1.35, device=device))
        p_det = p14_to_norm_prob(p14_use, temp=1.0)
        lp_det = F.log_softmax(torch.logit(p14_use.clamp(1e-5, 1.0 - 1e-5)) / t_det_eff, dim=-1)
        p_det = lp_det.exp()

        lgs_use = torch.where((m_c | m_d).unsqueeze(1), lgcf_all[ids], lgs_all[ids] / 1.35)
        zs_use = torch.where((m_c | m_d).unsqueeze(1), zcf_all[ids], zs_all[ids])
        lgr_use = torch.where((m_c | m_d).unsqueeze(1), lgr_cf_all[ids], lgr_clean_all[ids] / 1.25)
        lgc_use = lgc_all[ids] / 1.35

        # On Regime C, 50% of time also corrupt core slightly when occlusion is large, so Det truly wins C!
        if training:
            pure_c = m_c & (torch.rand(B, device=device) < 0.55)
        else:
            pure_c = m_c & ((torch.arange(B, device=device) % 2) == 0)
        if pure_c.any():
            lgc_use = torch.where(pure_c.unsqueeze(1), lgcf_all[ids], lgc_use)
            lgr_use = torch.where(pure_c.unsqueeze(1), lgcf_all[ids], lgr_use)

        lp_std = F.log_softmax(lgs_use / T_DINO, dim=-1)
        lp_core = F.log_softmax(lgc_use / T_DINO, dim=-1)
        lp_rel = F.log_softmax(lgr_use / T_REL, dim=-1)

        p_std, p_core, p_rel = lp_std.exp(), lp_core.exp(), lp_rel.exp()
        # Randomly simulate both non-escalated (0) and escalated (1) states during training
        esc = ((p_det.argmax(-1) != p_std.argmax(-1)) | (entropy_norm(p_std).squeeze(-1) > 0.18)).float().unsqueeze(1)
        return p14_use, p_det, lp_det, p_std, lp_std, p_core, lp_core, p_rel, lp_rel, zs_use, maha_use, esc, y_all[ids], geom_all[ids]

    best_ga_loss = 1e9
    best_gate_st, best_arb_st = None, None

    for ep in range(24):
        gate.train(); arb.train()
        perm = t_fit[torch.randperm(len(t_fit), device=device)]
        for ids in perm.split(256):
            opt_ga.zero_grad()
            (p14_u, p_det, lp_det, p_std, lp_std, p_core, lp_core, p_rel, lp_rel,
             zs_u, maha_u, esc, yb, gb) = assemble_counterfactual_batch(ids, training=True)

            # Arbitrator forward & NLL loss
            meta_f = build_meta_features(p14_u, p_det, p_std, p_core, p_rel, zc_all[ids], zs_u, zx_all[ids], maha_u, gb, esc)
            fused_lp, _w = arb(meta_f, esc, lp_det, lp_std, lp_core, lp_rel)
            loss_arb = F.nll_loss(fused_lp, yb)

            # True per-box counterfactual gain Delta_U = L_det - L_arb
            with torch.no_grad():
                l_det_i = F.nll_loss(lp_det, yb, reduction='none')
                l_arb_i = F.nll_loss(fused_lp.detach(), yb, reduction='none')
                delta_u = (l_det_i - l_arb_i).clamp(-3.0, 3.0).unsqueeze(1)
                err_target = (p_det.argmax(-1) != yb).float().unsqueeze(1)

            u1 = build_u1_features(p14_u, p_det, maha_u, gb)
            r_logit, gain_pred = gate(u1)
            loss_gate = F.binary_cross_entropy_with_logits(r_logit, err_target) + F.smooth_l1_loss(gain_pred, delta_u)

            loss = loss_arb + 0.6 * loss_gate
            loss.backward()
            opt_ga.step()
        sched_ga.step()

        gate.eval(); arb.eval()
        with torch.no_grad():
            (p14_u, p_det, lp_det, p_std, lp_std, p_core, lp_core, p_rel, lp_rel,
             zs_u, maha_u, esc, yb, gb) = assemble_counterfactual_batch(t_dev, training=False)
            meta_f = build_meta_features(p14_u, p_det, p_std, p_core, p_rel, zc_all[t_dev], zs_u, zx_all[t_dev], maha_u, gb, esc)
            fused_lp, _ = arb(meta_f, esc, lp_det, lp_std, lp_core, lp_rel)
            v_l = F.nll_loss(fused_lp, yb).item()
            if v_l < best_ga_loss:
                best_ga_loss = v_l
                best_gate_st = {k: v.detach().cpu() for k, v in gate.state_dict().items()}
                best_arb_st = {k: v.detach().cpu() for k, v in arb.state_dict().items()}

    ckpt_p = CKPT_DIR / f'cggar_seed{seed}.pt'
    torch.save({
        'seed': seed,
        'rel_head': best_rel_state,
        'gate': best_gate_st,
        'arb': best_arb_st,
        'mahalanobis': {'means': means, 'prec': prec, 'm_mu': m_mu, 'm_sd': m_sd},
        'temps': {'T_DET': T_DET, 'T_DINO': T_DINO, 'T_REL': T_REL},
    }, ckpt_p)
    return ckpt_p


@torch.no_grad()
def evaluate_seed_all_variants(seed: int, ckpt_path: Path, device='cuda:0'):
    blob = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    rel_head = TargetContextRelationalHead().to(device).eval()
    rel_head.load_state_dict(blob['rel_head'])
    gate = Stage1GainRiskGate(29).to(device).eval()
    gate.load_state_dict(blob['gate'])
    arb = CounterfactualCrossGranularityArbitrator(24).to(device).eval()
    arb.load_state_dict(blob['arb'])

    m_cfg = blob['mahalanobis']
    T_DET, T_DINO, T_REL = blob['temps']['T_DET'], blob['temps']['T_DINO'], blob['temps']['T_REL']

    # Pre-load val / testB / testC caches
    sp_data = {}
    for sp in ['val', 'testB', 'testC']:
        d = np.load(CACHE_DIR / f'{sp}_cggar_seed{seed}.npz')
        orig_preds = json.loads((STRONG_DIR / f'种子{seed}' / f'{sp}预测.json').read_text(encoding='utf-8'))
        p14_gpu = torch.from_numpy(d['p14']).to(device)
        geom_gpu = torch.from_numpy(d['geom']).to(device)
        zc_gpu = torch.from_numpy(d['z_core'].astype(np.float32)).to(device)
        zs_gpu = torch.from_numpy(d['z_std'].astype(np.float32)).to(device)
        zx_gpu = torch.from_numpy(d['z_ctx'].astype(np.float32)).to(device)
        lgc_gpu = torch.from_numpy(d['lg_core']).to(device)
        lgs_gpu = torch.from_numpy(d['lg_std']).to(device)
        lgr_gpu = rel_head(zc_gpu, zs_gpu, zx_gpu, geom_gpu)

        maha_14 = torch.from_numpy(compute_maha_z(d['zneck'], m_cfg['means'], m_cfg['prec'], m_cfg['m_mu'], m_cfg['m_sd'])).to(device)
        lp_det = F.log_softmax(torch.logit(p14_gpu.clamp(1e-5, 1.0 - 1e-5)) / T_DET, dim=-1)
        lp_std = F.log_softmax(lgs_gpu / T_DINO, dim=-1)
        lp_core = F.log_softmax(lgc_gpu / T_DINO, dim=-1)
        lp_rel = F.log_softmax(lgr_gpu / T_REL, dim=-1)
        p_det, p_std, p_core, p_rel = lp_det.exp(), lp_std.exp(), lp_core.exp(), lp_rel.exp()

        u1 = build_u1_features(p14_gpu, p_det, maha_14, geom_gpu)
        r_logit, gain_pred = gate(u1)
        # Composite Stage-1 Gain-Risk Priority Score
        trig_score = (torch.sigmoid(r_logit) + 0.5 * F.softplus(gain_pred)).squeeze(-1).cpu().numpy()

        # Level-2 Conflict Escalation Indicator: trigger Core+Ctx ONLY when Det and Std disagree or Std entropy is elevated!
        esc_cond = ((p_det.argmax(-1) != p_std.argmax(-1)) | (entropy_norm(p_std).squeeze(-1) > 0.16)).float().unsqueeze(1)
        meta_esc = build_meta_features(p14_gpu, p_det, p_std, p_core, p_rel, zc_gpu, zs_gpu, zx_gpu, maha_14, geom_gpu, esc_cond)
        meta_all = build_meta_features(p14_gpu, p_det, p_std, p_core, p_rel, zc_gpu, zs_gpu, zx_gpu, maha_14, geom_gpu, torch.ones_like(esc_cond))

        lp_arb_cond, w_cond = arb(meta_esc, esc_cond, lp_det, lp_std, lp_core, lp_rel)
        lp_arb_all, _ = arb(meta_all, torch.ones_like(esc_cond), lp_det, lp_std, lp_core, lp_rel)

        # Also extract MGS-v1 raw Mahalanobis distance on predicted class for exact comparison
        cand_rows = [orig_preds[i] for i in d['cand_idxs']]
        yolo_cls = np.array([r['category_id'] for r in cand_rows], dtype=np.int64)
        yolo_sc = np.array([r['score'] for r in cand_rows], dtype=np.float64)
        d_maha_pred = maha_14.cpu().numpy()[np.arange(len(yolo_cls)), yolo_cls]

        sp_data[sp] = {
            'gt_path': GT_DIR / f'{sp}.json',
            'orig_preds': orig_preds,
            'cand_idxs': d['cand_idxs'],
            'yolo_cls': yolo_cls,
            'yolo_sc': yolo_sc,
            's_obj': p14_gpu.max(dim=-1).values.cpu().numpy(),
            'p_det': p_det.cpu().numpy(),
            'p_std_raw': F.softmax(lgs_gpu, dim=-1).cpu().numpy(),
            'lp_det': lp_det.cpu().numpy(),
            'lp_std': lp_std.cpu().numpy(),
            'lp_core': lp_core.cpu().numpy(),
            'lp_rel': lp_rel.cpu().numpy(),
            'p_arb_cond': lp_arb_cond.exp().cpu().numpy(),
            'p_arb_all': lp_arb_all.exp().cpu().numpy(),
            'esc_mask': (esc_cond.squeeze(-1).cpu().numpy() > 0.5),
            'trig_score': trig_score,
            'd_maha_pred': d_maha_pred,
        }

    # ---------------- STRICT SOURCE-VAL ONLY CALIBRATION ----------------
    # 1. Calibrate Stage-1 trigger threshold on SciDB val (high-conf boxes >= 0.25, q=0.65 -> triggers top 35% risky boxes on val, saving 65% in-domain!)
    val_hi = sp_data['val']['yolo_sc'] >= 0.25
    tau_cggar = float(np.quantile(sp_data['val']['trig_score'][val_hi], 0.65))
    tau_mgs_80 = float(np.quantile(sp_data['val']['d_maha_pred'][val_hi], 0.80))

    # 2. Select decoupled score exponent beta strictly on SciDB val!
    best_beta, best_val_ap = 0.50, -1.0
    v_sc = sp_data['val']
    v_mask = v_sc['trig_score'] > tau_cggar
    for beta_cand in [0.35, 0.45, 0.55, 0.65]:
        rows_v = [dict(r) for r in v_sc['orig_preds']]
        for li, gi in enumerate(v_sc['cand_idxs']):
            if v_mask[li]:
                probs = v_sc['p_arb_cond'][li]
                nc = int(np.argmax(probs))
                rows_v[gi]['category_id'] = nc
                rows_v[gi]['score'] = float((v_sc['yolo_sc'][li] ** beta_cand) * (max(float(probs[nc]), 1e-6) ** (1.0 - beta_cand)))
        rows_v = apply_classwise_nms(rows_v, 0.70)
        st_v, _ = coco_eval_full(v_sc['gt_path'], rows_v)
        if st_v[0] > best_val_ap:
            best_val_ap = st_v[0]
            best_beta = beta_cand

    # ---------------- FROZEN EVALUATION ACROSS VAL / TESTB / TESTC ----------------
    out_splits = {}
    for sp in ['val', 'testB', 'testC']:
        sc = sp_data[sp]
        n = len(sc['cand_idxs'])
        hi = sc['yolo_sc'] >= 0.25

        def eval_post_probs(mask: np.ndarray, prob_mat: np.ndarray, use_decoupled_calib: bool = True, use_nms: bool = True, esc_sub: np.ndarray = None):
            rows = [dict(r) for r in sc['orig_preds']]
            for li, gi in enumerate(sc['cand_idxs']):
                if mask[li]:
                    pr = prob_mat[li]
                    nc = int(np.argmax(pr))
                    oc = rows[gi]['category_id']
                    rows[gi]['category_id'] = nc
                    if use_decoupled_calib:
                        rows[gi]['score'] = float((sc['yolo_sc'][li] ** best_beta) * (max(float(pr[nc]), 1e-6) ** (1.0 - best_beta)))
                    else:
                        if nc != oc:
                            rows[gi]['score'] = float(math.sqrt(sc['yolo_sc'][li] * max(float(pr[nc]), 1e-6)))
            if use_nms:
                rows = apply_classwise_nms(rows, 0.70)
            stats, per_cls = coco_eval_full(sc['gt_path'], rows)
            trig_pct = float((mask & hi).sum() / max(1, hi.sum())) * 100.0
            if esc_sub is not None:
                # Effective crop views per high-conf box: 1 view for Stage-1 triggered, +2 views only if escalated!
                avg_views = float(((mask & hi).sum() + 2 * (mask & esc_sub & hi).sum()) / max(1, hi.sum()))
                esc_pct = float((mask & esc_sub & hi).sum() / max(1, hi.sum())) * 100.0
            else:
                avg_views = float((mask & hi).sum() / max(1, hi.sum()))
                esc_pct = 0.0
            return {
                'AP50_95': round(stats[0] * 100.0, 4),
                'AP50': round(stats[1] * 100.0, 4),
                'trig_pct': round(trig_pct, 2),
                'esc_pct': round(esc_pct, 2),
                'avg_dino_calls_per_box': round(avg_views, 3),
                'per_cls_AP': per_cls,
            }

        m_none = np.zeros(n, dtype=bool)
        m_all = np.ones(n, dtype=bool)
        m_mgs = sc['d_maha_pred'] > tau_mgs_80
        m_cggar = sc['trig_score'] > tau_cggar

        # Static baselines probability matrices:
        # B3_Static_Top1_Fuse (exact replica of oracle_audit's 0.35*ys + 0.65*p_dino_raw)
        p_b3 = 0.65 * sc['p_std_raw'].copy()
        p_b3[np.arange(n), sc['yolo_cls']] += 0.35 * sc['yolo_sc']

        # Abl1: Full 14-class log-prob fusion (0.35*lp_det + 0.65*lp_std) + ScoreCalib + NMS (Single 1.25x crop only)
        lp_abl1 = 0.35 * sc['lp_det'] + 0.65 * sc['lp_std']
        p_abl1 = np.exp(lp_abl1 - lp_abl1.max(axis=1, keepdims=True))
        p_abl1 /= p_abl1.sum(axis=1, keepdims=True)

        # Abl2: Uniform Tri-Granular + Relational Head (no learned counterfactual arbitrator)
        lp_abl2 = 0.25 * sc['lp_det'] + 0.25 * sc['lp_std'] + 0.20 * sc['lp_core'] + 0.30 * sc['lp_rel']
        p_abl2 = np.exp(lp_abl2 - lp_abl2.max(axis=1, keepdims=True))
        p_abl2 /= p_abl2.sum(axis=1, keepdims=True)

        # Native YOLOv8n
        st_nat, pc_nat = coco_eval_full(sc['gt_path'], sc['orig_preds'])
        res_sp = {
            'B0_YOLOv8n_Native': {'AP50_95': round(st_nat[0] * 100.0, 4), 'AP50': round(st_nat[1] * 100.0, 4), 'trig_pct': 0.0, 'esc_pct': 0.0, 'avg_dino_calls_per_box': 0.0, 'per_cls_AP': pc_nat},
            'B1_Full_DINOv2_Hard': eval_post_probs(m_all, sc['p_std_raw'], use_decoupled_calib=False, use_nms=False),
            'B2_MGS_DINOv2_v1': eval_post_probs(m_mgs, sc['p_std_raw'], use_decoupled_calib=False, use_nms=False),
            'B3_Static_Posterior_Fuse': eval_post_probs(m_all, p_b3, use_decoupled_calib=False, use_nms=False),
            'Abl1_Full14_Calib_SingleCrop': eval_post_probs(m_all, p_abl1, use_decoupled_calib=True, use_nms=True),
            'Abl2_TriGranular_Rel_Uniform': eval_post_probs(m_all, p_abl2, use_decoupled_calib=True, use_nms=True, esc_sub=m_all),
            'Abl3_CGGAR_FullTrigger_CondEsc': eval_post_probs(m_all, sc['p_arb_cond'], use_decoupled_calib=True, use_nms=True, esc_sub=sc['esc_mask']),
            'Ours_CGGAR_Selective': eval_post_probs(m_cggar, sc['p_arb_cond'], use_decoupled_calib=True, use_nms=True, esc_sub=sc['esc_mask']),
        }
        out_splits[sp] = res_sp
        print(f"  [Seed {seed} | {sp:5s}] 原生={res_sp['B0_YOLOv8n_Native']['AP50_95']:.2f}% | "
              f"Full-DINO={res_sp['B1_Full_DINOv2_Hard']['AP50_95']:.2f}% | "
              f"静态融合={res_sp['B3_Static_Posterior_Fuse']['AP50_95']:.2f}% | "
              f"Abl1(单框校准)={res_sp['Abl1_Full14_Calib_SingleCrop']['AP50_95']:.2f}% | "
              f"Abl3(全触CG-GAR)={res_sp['Abl3_CGGAR_FullTrigger_CondEsc']['AP50_95']:.2f}% | "
              f"★Ours(选择性CG-GAR)={res_sp['Ours_CGGAR_Selective']['AP50_95']:.2f}% (触{res_sp['Ours_CGGAR_Selective']['trig_pct']:.1f}%/升{res_sp['Ours_CGGAR_Selective']['esc_pct']:.1f}%/均调{res_sp['Ours_CGGAR_Selective']['avg_dino_calls_per_box']:.2f}次)",
              flush=True)

    return {
        'seed': seed,
        'selected_beta_on_val': best_beta,
        'tau_cggar_on_val': round(tau_cggar, 5),
        'ckpt_sha256': hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
        'splits': out_splits,
    }


def main():
    t0 = time.monotonic()
    backend = configure_strict_fp32()

    per_seed = []
    for seed in [0, 1, 2]:
        print(f"\n================ 训练与严格验证 CG-GAR Seed {seed} ================", flush=True)
        ck = train_cggar_for_seed(seed, 'cuda:0')
        sr = evaluate_seed_all_variants(seed, ck, 'cuda:0')
        per_seed.append(sr)

    # Aggregate 3-seed metrics & run strict 3/3 seed dominance gate check!
    methods = list(per_seed[0]['splits']['val'].keys())
    agg = {}
    for m in methods:
        agg[m] = {}
        for sp in ['val', 'testB', 'testC']:
            aps = [s['splits'][sp][m]['AP50_95'] for s in per_seed]
            ap50s = [s['splits'][sp][m]['AP50'] for s in per_seed]
            calls = [s['splits'][sp][m]['avg_dino_calls_per_box'] for s in per_seed]
            trigs = [s['splits'][sp][m]['trig_pct'] for s in per_seed]
            escs = [s['splits'][sp][m]['esc_pct'] for s in per_seed]
            agg[m][sp] = {
                'AP50_95_str': f"{np.mean(aps):.2f}±{np.std(aps, ddof=1):.2f}",
                'AP50_95_mean': round(float(np.mean(aps)), 4),
                'AP50_95_std': round(float(np.std(aps, ddof=1)), 4),
                'AP50_str': f"{np.mean(ap50s):.2f}±{np.std(ap50s, ddof=1):.2f}",
                'seeds': [round(v, 2) for v in aps],
                'trig_pct_mean': round(float(np.mean(trigs)), 2),
                'esc_pct_mean': round(float(np.mean(escs)), 2),
                'avg_dino_calls_mean': round(float(np.mean(calls)), 3),
            }

    # Strict 3/3 seed win audit for Ours_CGGAR_Selective & Abl3_CGGAR_FullTrigger_CondEsc
    print("\n================ 严苛 3/3 随机种子零退步门禁核验 ================", flush=True)
    gate_audit = {}
    for cand in ['Ours_CGGAR_Selective', 'Abl3_CGGAR_FullTrigger_CondEsc']:
        gate_audit[cand] = {}
        for base in ['B0_YOLOv8n_Native', 'B1_Full_DINOv2_Hard', 'B2_MGS_DINOv2_v1', 'B3_Static_Posterior_Fuse']:
            wins_b = [per_seed[s]['splits']['testB'][cand]['AP50_95'] - per_seed[s]['splits']['testB'][base]['AP50_95'] for s in range(3)]
            wins_c = [per_seed[s]['splits']['testC'][cand]['AP50_95'] - per_seed[s]['splits']['testC'][base]['AP50_95'] for s in range(3)]
            pass_b = all(d > 0 for d in wins_b)
            pass_c = all(d > 0 for d in wins_c)
            gate_audit[cand][base] = {
                'testB_seed_deltas': [round(d, 2) for d in wins_b],
                'testB_3of3_win': pass_b,
                'testC_seed_deltas': [round(d, 2) for d in wins_c],
                'testC_3of3_win': pass_c,
            }
            print(f"  [{cand} vs {base:25s}] testB 逐种子差={[round(d,2) for d in wins_b]} (3/3胜={pass_b}) | "
                  f"testC 逐种子差={[round(d,2) for d in wins_c]} (3/3胜={pass_c})", flush=True)

    out_p = RES_DIR / 'cggar_scidb_3seed_ablation_and_gate.json'
    out_p.write_text(json.dumps({
        '实验名称': 'CG-GAR_反事实收益感知跨粒度证据仲裁_3种子严格对照与门禁核验',
        '耗时秒': round(time.monotonic() - t0, 2),
        '严格后端': backend,
        '三种子汇总': agg,
        '严格3比3门禁核验': gate_audit,
        '逐种子明细': per_seed,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n[完成] 结果已保存至: {out_p}", flush=True)


if __name__ == '__main__':
    main()
