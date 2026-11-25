# -*- coding: utf-8 -*-
"""
Step 2 (v2): 无污染反事实收益学习与流形-分歧双驱动按需三级跨粒度仲裁 (CG-GAR v2, 3 随机种子 0/1/2)
严格红线：
1. 全部神经网络参数仅在源训练集 SciDB train (85% train_fit / 15% train_dev) 上训练；
2. 升级阈值 tau_esc 与分数校准指数 beta 100% 仅在源域 SciDB val 上标定并冻结；
3. 严禁利用 testB / testC 调参；同时纳入原版灰底裁框 Full-DINOv2 (75.90% / 83.31%) 与原版静态融合 (77.53% / 84.60%)
   以及方形裁框版全部基线，执行最严苛的 3/3 种子全胜硬门禁检验！
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(r'D:\新版柑橘')
EXT = ROOT / '方法路线扩展_20261031'
sys.path.append(str(EXT))
from 共同评测后端 import configure_strict_fp32

WORK_DIR = ROOT / 'Antigravity_CGGAR_真创新攻关_20261125'
sys.path.append(str(WORK_DIR))
from step2_train_and_eval_cggar import (
    set_seed, coco_eval_full, apply_classwise_nms, fit_mahalanobis_stats, compute_maha_z,
    entropy_norm, top2_margin, js_divergence, TargetContextRelationalHead, build_meta_features
)

CACHE_DIR = WORK_DIR / 'cache'
CKPT_DIR = WORK_DIR / 'models'
RES_DIR = WORK_DIR / 'results'
GT_DIR = EXT / '检测数据'
STRONG_DIR = EXT / 'YOLOv8强基线重复_20261109'
E2_OLD_JSON = ROOT / '多路线创新筛选_20261124' / '小论文完整实验_20261124' / 'results' / 'e2_end2end_mgs_symtex_summary.json'


class GainAwareArbitratorV2(nn.Module):
    """
    反事实收益感知跨粒度证据仲裁器 A_phi (v2)：
    输入 24 维视角不变不确定性/分歧/跨尺度余弦一致性/流形偏移/框几何元特征，
    同时预测：(1) 按需升级三级视图的预期净收益 Delta_U_esc；(2) 四路对数后验自适应可靠性权重 [det, std, core, rel]。
    """
    def __init__(self, meta_dim: int = 24):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LayerNorm(meta_dim),
            nn.Linear(meta_dim, 64),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.GELU(),
        )
        self.head_weights = nn.Linear(32, 4)
        self.head_esc_gain = nn.Linear(32, 1)  # predicts E[L_2view - L_4view] from Level-1 state!
        nn.init.zeros_(self.head_weights.weight)
        nn.init.constant_(self.head_weights.bias, 0.0)

        # Level-1 Escalation Gate Network (uses ONLY Det + Std 1.25x + Neck Maha + Geom -> 15-d, BEFORE extracting Core/Ctx!)
        self.esc_gate = nn.Sequential(
            nn.LayerNorm(17),
            nn.Linear(17, 32),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(32, 1),
        )

    def predict_escalation_gain(self, esc_in17: torch.Tensor) -> torch.Tensor:
        return self.esc_gate(esc_in17)

    def forward(self, meta_f: torch.Tensor, escalated: torch.Tensor,
                lp_det: torch.Tensor, lp_std: torch.Tensor, lp_core: torch.Tensor, lp_rel: torch.Tensor):
        h = self.backbone(meta_f)
        raw = self.head_weights(h)
        # Balanced prior matching multi-granularity complementary strengths: [det=-0.25, std=0.05, core=-0.15, rel=0.35]
        prior = raw.new_tensor([-0.25, 0.05, -0.15, 0.35]).unsqueeze(0)
        logits_w = prior + 0.65 * torch.tanh(raw)

        mask_no_esc = (escalated.squeeze(-1) < 0.5)
        if mask_no_esc.any():
            logits_w = logits_w.clone()
            # In Level-1 (non-escalated, clean agreement), fuse Det (0.35) & Std (0.65) cleanly
            logits_w[mask_no_esc, 0] = -0.31 + 0.35 * torch.tanh(raw[mask_no_esc, 0])
            logits_w[mask_no_esc, 1] = 0.31 + 0.35 * torch.tanh(raw[mask_no_esc, 1])
            logits_w[mask_no_esc, 2] = -1e9
            logits_w[mask_no_esc, 3] = -1e9

        w = F.softmax(logits_w, dim=-1)
        fused_lp = (
            w[:, 0:1] * lp_det
            + w[:, 1:2] * lp_std
            + w[:, 2:3] * lp_core
            + w[:, 3:4] * lp_rel
        )
        return F.log_softmax(fused_lp, dim=-1), w


def build_level1_esc_features(p14_sig: torch.Tensor, p_det: torch.Tensor, p_std: torch.Tensor,
                              maha_14: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
    """仅利用整图检测与 1.25x 标准框（无需提取 Core/Ctx）构建 17 维升级收益预测特征。"""
    s_obj = p14_sig.max(dim=-1, keepdim=True).values
    h_det = entropy_norm(p_det)
    h_std = entropy_norm(p_std)
    m_det = top2_margin(p_det)
    m_std = top2_margin(p_std)
    dis = (p_det.argmax(-1) != p_std.argmax(-1)).float().unsqueeze(1)
    js = js_divergence(p_det, p_std)
    c_hat = p_det.argmax(dim=-1)
    d_pred = maha_14.gather(1, c_hat.unsqueeze(1)).clamp(-3.0, 15.0) / 4.0
    return torch.cat([s_obj, h_det, h_std, m_det, m_std, dis, js, d_pred, geom], dim=-1)  # 8 + 9 = 17


def train_cggar_v2_for_seed(seed: int, device='cuda:0'):
    set_seed(seed + 200)
    tr = np.load(CACHE_DIR / f'train_cggar_seed{seed}.npz')
    box_img_ids = tr['box_img_ids']
    unique_imgs = np.unique(box_img_ids)
    rng = np.random.RandomState(2026 + seed)
    rng.shuffle(unique_imgs)
    fit_img_set = set(unique_imgs[:int(len(unique_imgs) * 0.85)].tolist())
    is_fit = np.array([iid in fit_img_set for iid in box_img_ids], dtype=bool)
    fit_idx = np.where(is_fit)[0]
    dev_idx = np.where(~is_fit)[0]

    means, prec, m_mu, m_sd = fit_mahalanobis_stats(tr['zneck_clean'][fit_idx], tr['labels'][fit_idx])
    maha_clean = compute_maha_z(tr['zneck_clean'], means, prec, m_mu, m_sd)
    maha_deg = compute_maha_z(tr['zneck_deg'], means, prec, m_mu, m_sd)

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

    # 1. Reuse or train Target-Context Relational Head R_psi (strictly on train_fit)
    rel_head = TargetContextRelationalHead().to(device)
    opt_r = torch.optim.AdamW(rel_head.parameters(), lr=1.2e-3, weight_decay=2e-3)
    sched_r = torch.optim.lr_scheduler.CosineAnnealingLR(opt_r, T_max=20)
    best_rel_loss, best_rel_state = 1e9, None

    for _ep in range(20):
        rel_head.train()
        perm = t_fit[torch.randperm(len(t_fit), device=device)]
        for ids in perm.split(256):
            opt_r.zero_grad()
            zc_b, zs_b, zx_b = zc_all[ids], zs_all[ids], zx_all[ids]
            cm_noise = torch.randn((len(ids), 384), device=device) * 0.08
            zc_in = F.normalize(zc_b + cm_noise, dim=-1)
            zs_in = F.normalize(zs_b + cm_noise, dim=-1)
            zx_in = F.normalize(zx_b + cm_noise, dim=-1)
            corrupt_m = torch.rand(len(ids), device=device) < 0.25
            if corrupt_m.any():
                zs_in = torch.where(corrupt_m.unsqueeze(1), zcf_all[ids], zs_in)
            loss_r = F.cross_entropy(rel_head(zc_in, zs_in, zx_in, geom_all[ids]), y_all[ids], label_smoothing=0.05)
            loss_r.backward()
            opt_r.step()
        sched_r.step()

        rel_head.eval()
        with torch.no_grad():
            dl1 = F.cross_entropy(rel_head(zc_all[t_dev], zs_all[t_dev], zx_all[t_dev], geom_all[t_dev]), y_all[t_dev]).item()
            dl2 = F.cross_entropy(rel_head(zc_all[t_dev], zcf_all[t_dev], zx_all[t_dev], geom_all[t_dev]), y_all[t_dev]).item()
            if 0.5 * (dl1 + dl2) < best_rel_loss:
                best_rel_loss = 0.5 * (dl1 + dl2)
                best_rel_state = {k: v.detach().cpu() for k, v in rel_head.state_dict().items()}

    rel_head.load_state_dict(best_rel_state)
    rel_head.eval()
    with torch.no_grad():
        lgr_clean = rel_head(zc_all, zs_all, zx_all, geom_all)
        lgr_border_cf = rel_head(zc_all, zcf_all, zx_all, geom_all)   # border of std corrupted, core & ctx intact!
        lgr_core_cf = rel_head(zcf_all, zcf_all, zx_all, geom_all)     # severe target occlusion: both core & std corrupted, det intact!

    # 2. Train GainAwareArbitratorV2 with physically faithful counterfactual regimes!
    T_DET, T_DINO, T_REL = 1.35, 1.15, 1.20
    arb = GainAwareArbitratorV2(meta_dim=24).to(device)
    opt_a = torch.optim.AdamW(arb.parameters(), lr=1.2e-3, weight_decay=1e-3)
    sched_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=24)

    def make_faithful_cf_batch(ids: torch.Tensor, training: bool = True):
        """
        四种物理自洽的反事实退化模态：
        - Mode A (25%): 干净源域 (Det 与全部局部视图均正常)
        - Mode B (30%): 整图模糊/曝光退化 (Det 退化, 局部 Std/Core/Rel 清晰准确 -> 应信任局部多粒度!)
        - Mode C_occ (20%): 目标局部截断/重度遮挡 (Std 与 Core 均受损 `zcf`, 唯独整图 Det 靠全局上下文准确 -> 应信任 Det!)
        - Mode D_bg (25%): 边框背景强反光/异源底板偏移 (Det 受损 + Std 边框受损 `zcf`, 但内部叶肉 Core 与差分头 R_psi 保持清洁准确 -> 必须升级三级视图靠 Core + R_psi 攻克!)
        """
        B = len(ids)
        reg = torch.randint(0, 100, (B,), device=device) if training else ((torch.arange(B, device=device) * 31) % 100)
        m_b = (reg >= 25) & (reg < 55)
        m_c = (reg >= 55) & (reg < 75)
        m_d = (reg >= 75)

        p14_u = torch.where((m_b | m_d).unsqueeze(1), p14_d[ids], p14_c[ids])
        maha_u = torch.where((m_b | m_d).unsqueeze(1), md_gpu[ids], mc_gpu[ids])
        t_det_eff = torch.where((m_b | m_d).unsqueeze(1), torch.full((B, 1), T_DET, device=device), torch.full((B, 1), T_DET * 1.35, device=device))
        lp_det = F.log_softmax(torch.logit(p14_u.clamp(1e-5, 1.0 - 1e-5)) / t_det_eff, dim=-1)
        p_det = lp_det.exp()

        # Std 1.25x is corrupted in BOTH Mode C_occ and Mode D_bg
        lgs_u = torch.where((m_c | m_d).unsqueeze(1), lgcf_all[ids], lgs_all[ids] / 1.30)
        zs_u = torch.where((m_c | m_d).unsqueeze(1), zcf_all[ids], zs_all[ids])

        # Core 0.82x & Relational Head R_psi are corrupted ONLY in Mode C_occ (true leaf occlusion), NOT in Mode D_bg (border background glare)!
        lgc_u = torch.where(m_c.unsqueeze(1), lgcf_all[ids], lgc_all[ids] / 1.30)
        zc_u = torch.where(m_c.unsqueeze(1), zcf_all[ids], zc_all[ids])
        lgr_u = torch.where(m_c.unsqueeze(1), lgr_core_cf[ids],
                            torch.where(m_d.unsqueeze(1), lgr_border_cf[ids], lgr_clean[ids] / 1.20))

        lp_std = F.log_softmax(lgs_u / T_DINO, dim=-1)
        lp_core = F.log_softmax(lgc_u / T_DINO, dim=-1)
        lp_rel = F.log_softmax(lgr_u / T_REL, dim=-1)
        p_std, p_core, p_rel = lp_std.exp(), lp_core.exp(), lp_rel.exp()

        return p14_u, p_det, lp_det, p_std, lp_std, p_core, lp_core, p_rel, lp_rel, zc_u, zs_u, maha_u, y_all[ids], geom_all[ids]

    best_loss, best_arb_state = 1e9, None
    for _ep in range(24):
        arb.train()
        perm = t_fit[torch.randperm(len(t_fit), device=device)]
        for ids in perm.split(256):
            opt_a.zero_grad()
            (p14_u, p_det, lp_det, p_std, lp_std, p_core, lp_core, p_rel, lp_rel,
             zc_u, zs_u, maha_u, yb, gb) = make_faithful_cf_batch(ids, training=True)

            esc_ones = torch.ones((len(ids), 1), device=device)
            esc_zeros = torch.zeros((len(ids), 1), device=device)

            meta_full = build_meta_features(p14_u, p_det, p_std, p_core, p_rel, zc_u, zs_u, zx_all[ids], maha_u, gb, esc_ones)
            meta_lvl1 = build_meta_features(p14_u, p_det, p_std, p_core, p_rel, zc_u, zs_u, zx_all[ids], maha_u, gb, esc_zeros)

            lp_4view, _ = arb(meta_full, esc_ones, lp_det, lp_std, lp_core, lp_rel)
            lp_2view, _ = arb(meta_lvl1, esc_zeros, lp_det, lp_std, lp_core, lp_rel)

            loss_4v = F.nll_loss(lp_4view, yb)
            loss_2v = F.nll_loss(lp_2view, yb)

            # Explicit Escalation Gain Target: Delta_U_esc = L(2-view) - L(4-view) (how much Core + Context Differential reduces NLL!)
            with torch.no_grad():
                l2_i = F.nll_loss(lp_2view.detach(), yb, reduction='none')
                l4_i = F.nll_loss(lp_4view.detach(), yb, reduction='none')
                delta_esc = (l2_i - l4_i).clamp(-2.0, 2.0).unsqueeze(1)

            esc_in17 = build_level1_esc_features(p14_u, p_det, p_std, maha_u, gb)
            pred_esc_gain = arb.predict_escalation_gain(esc_in17)
            loss_esc_gate = F.smooth_l1_loss(pred_esc_gain, delta_esc)

            loss = loss_4v + 0.4 * loss_2v + 0.8 * loss_esc_gate
            loss.backward()
            opt_a.step()
        sched_a.step()

        arb.eval()
        with torch.no_grad():
            (p14_u, p_det, lp_det, p_std, lp_std, p_core, lp_core, p_rel, lp_rel,
             zc_u, zs_u, maha_u, yb, gb) = make_faithful_cf_batch(t_dev, training=False)
            esc_ones = torch.ones((len(t_dev), 1), device=device)
            meta_full = build_meta_features(p14_u, p_det, p_std, p_core, p_rel, zc_u, zs_u, zx_all[t_dev], maha_u, gb, esc_ones)
            lp_4view, _ = arb(meta_full, esc_ones, lp_det, lp_std, lp_core, lp_rel)
            vl = F.nll_loss(lp_4view, yb).item()
            if vl < best_loss:
                best_loss = vl
                best_arb_state = {k: v.detach().cpu() for k, v in arb.state_dict().items()}

    ckpt_p = CKPT_DIR / f'cggar_v2_seed{seed}.pt'
    torch.save({
        'seed': seed,
        'rel_head': best_rel_state,
        'arb_v2': best_arb_state,
        'mahalanobis': {'means': means, 'prec': prec, 'm_mu': m_mu, 'm_sd': m_sd},
        'temps': {'T_DET': T_DET, 'T_DINO': T_DINO, 'T_REL': T_REL},
    }, ckpt_p)
    return ckpt_p


@torch.no_grad()
def evaluate_seed_v2(seed: int, ckpt_path: Path, old_e2_seed: dict, device='cuda:0'):
    blob = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    rel_head = TargetContextRelationalHead().to(device).eval(); rel_head.load_state_dict(blob['rel_head'])
    arb = GainAwareArbitratorV2(24).to(device).eval(); arb.load_state_dict(blob['arb_v2'])
    m_cfg = blob['mahalanobis']
    T_DET, T_DINO, T_REL = blob['temps']['T_DET'], blob['temps']['T_DINO'], blob['temps']['T_REL']

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

        # Compute Level-1 Predicted Escalation Gain Score + Manifold/Disagreement Composite Escalation Priority!
        esc_in17 = build_level1_esc_features(p14_gpu, p_det, p_std, maha_14, geom_gpu)
        pred_gain = arb.predict_escalation_gain(esc_in17).squeeze(-1)
        c_hat = p_det.argmax(dim=-1)
        d_maha_z = maha_14.gather(1, c_hat.unsqueeze(1)).squeeze(-1)
        dis_flag = (p_det.argmax(-1) != p_std.argmax(-1)).float()
        # Unified Gain + Manifold Shift + Disagreement Escalation Score (calibrated ONLY on SciDB val!)
        esc_priority = (pred_gain + 0.25 * dis_flag + 0.08 * d_maha_z.clamp_min(0.0)).cpu().numpy()

        # Precompute both Level-1 (2-view) and Level-2 (4-view) arbitrated posteriors
        z_ones = torch.ones((len(p_det), 1), device=device)
        z_zeros = torch.zeros((len(p_det), 1), device=device)
        meta_4v = build_meta_features(p14_gpu, p_det, p_std, p_core, p_rel, zc_gpu, zs_gpu, zx_gpu, maha_14, geom_gpu, z_ones)
        meta_2v = build_meta_features(p14_gpu, p_det, p_std, p_core, p_rel, zc_gpu, zs_gpu, zx_gpu, maha_14, geom_gpu, z_zeros)
        lp_4v, w_4v = arb(meta_4v, z_ones, lp_det, lp_std, lp_core, lp_rel)
        lp_2v, _ = arb(meta_2v, z_zeros, lp_det, lp_std, lp_core, lp_rel)

        cand_rows = [orig_preds[i] for i in d['cand_idxs']]
        yolo_cls = np.array([r['category_id'] for r in cand_rows], dtype=np.int64)
        yolo_sc = np.array([r['score'] for r in cand_rows], dtype=np.float64)

        sp_data[sp] = {
            'gt_path': GT_DIR / f'{sp}.json',
            'orig_preds': orig_preds,
            'cand_idxs': d['cand_idxs'],
            'yolo_cls': yolo_cls,
            'yolo_sc': yolo_sc,
            'p_std_raw': F.softmax(lgs_gpu, dim=-1).cpu().numpy(),
            'lp_det': lp_det.cpu().numpy(),
            'lp_std': lp_std.cpu().numpy(),
            'lp_core': lp_core.cpu().numpy(),
            'lp_rel': lp_rel.cpu().numpy(),
            'p_4v': lp_4v.exp().cpu().numpy(),
            'p_2v': lp_2v.exp().cpu().numpy(),
            'w_4v_mean': w_4v.mean(dim=0).cpu().numpy().tolist(),
            'esc_priority': esc_priority,
        }

    # ---------------- STRICT SOURCE-VAL ONLY CALIBRATION ----------------
    # 1. Calibrate Level-2 escalation threshold tau_esc on SciDB val (q=0.75 -> only top 25% most conflicted/shifted boxes on val escalate to Core+Ctx!)
    val_hi = sp_data['val']['yolo_sc'] >= 0.25
    tau_esc = float(np.quantile(sp_data['val']['esc_priority'][val_hi], 0.75))
    # Foreground box threshold for running Stage-1 crop: score >= 0.12 (skips tail noise boxes in 0.05..0.12!)
    FG_SCORE_THR = 0.12

    # 2. Select score calibration exponent beta strictly on SciDB val!
    best_beta, best_val_ap = 0.45, -1.0
    v_sc = sp_data['val']
    v_fg = v_sc['yolo_sc'] >= FG_SCORE_THR
    v_esc = v_fg & (v_sc['esc_priority'] > tau_esc)
    for beta_cand in [0.35, 0.45, 0.55]:
        rows_v = [dict(r) for r in v_sc['orig_preds']]
        for li, gi in enumerate(v_sc['cand_idxs']):
            if v_fg[li]:
                pr = v_sc['p_4v'][li] if v_esc[li] else v_sc['p_2v'][li]
                nc = int(np.argmax(pr))
                rows_v[gi]['category_id'] = nc
                rows_v[gi]['score'] = float((v_sc['yolo_sc'][li] ** beta_cand) * (max(float(pr[nc]), 1e-6) ** (1.0 - beta_cand)))
        rows_v = apply_classwise_nms(rows_v, 0.70)
        st_v, _ = coco_eval_full(v_sc['gt_path'], rows_v)
        if st_v[0] > best_val_ap:
            best_val_ap = st_v[0]
            best_beta = beta_cand

    out_splits = {}
    for sp in ['val', 'testB', 'testC']:
        sc = sp_data[sp]
        n = len(sc['cand_idxs'])
        hi = sc['yolo_sc'] >= 0.25
        fg = sc['yolo_sc'] >= FG_SCORE_THR
        esc_m = fg & (sc['esc_priority'] > tau_esc)

        def eval_prob_cfg(active_mask: np.ndarray, prob_mat: np.ndarray, calib: bool, use_nms: bool, esc_mask_opt: np.ndarray = None):
            rows = [dict(r) for r in sc['orig_preds']]
            for li, gi in enumerate(sc['cand_idxs']):
                if active_mask[li]:
                    pr = prob_mat[li]
                    nc = int(np.argmax(pr))
                    oc = rows[gi]['category_id']
                    rows[gi]['category_id'] = nc
                    if calib:
                        rows[gi]['score'] = float((sc['yolo_sc'][li] ** best_beta) * (max(float(pr[nc]), 1e-6) ** (1.0 - best_beta)))
                    elif nc != oc:
                        rows[gi]['score'] = float(math.sqrt(sc['yolo_sc'][li] * max(float(pr[nc]), 1e-6)))
            if use_nms:
                rows = apply_classwise_nms(rows, 0.70)
            stats, per_cls = coco_eval_full(sc['gt_path'], rows)
            if esc_mask_opt is not None:
                # Total crop forward views across ALL candidates (>=0.05) relative to Full-DINO's N_cand calls!
                crop_ratio_vs_fulldino = float((active_mask.sum() + 2 * esc_mask_opt.sum()) / max(1, n))
                esc_hi_pct = float((esc_mask_opt & hi).sum() / max(1, hi.sum())) * 100.0
            else:
                crop_ratio_vs_fulldino = float(active_mask.sum() / max(1, n))
                esc_hi_pct = 0.0
            return {
                'AP50_95': round(stats[0] * 100.0, 4),
                'AP50': round(stats[1] * 100.0, 4),
                'esc_hi_pct': round(esc_hi_pct, 2),
                'crop_cost_vs_fulldino': round(crop_ratio_vs_fulldino, 3),
                'per_cls_AP': per_cls,
            }

        m_all = np.ones(n, dtype=bool)
        p_b3 = 0.65 * sc['p_std_raw'].copy()
        p_b3[np.arange(n), sc['yolo_cls']] += 0.35 * sc['yolo_sc']

        lp_abl1 = 0.35 * sc['lp_det'] + 0.65 * sc['lp_std']
        p_abl1 = np.exp(lp_abl1 - lp_abl1.max(axis=1, keepdims=True)); p_abl1 /= p_abl1.sum(axis=1, keepdims=True)

        lp_abl2 = 0.25 * sc['lp_det'] + 0.25 * sc['lp_std'] + 0.20 * sc['lp_core'] + 0.30 * sc['lp_rel']
        p_abl2 = np.exp(lp_abl2 - lp_abl2.max(axis=1, keepdims=True)); p_abl2 /= p_abl2.sum(axis=1, keepdims=True)

        # Construct selective escalated probability matrix: 4-view if escalated else 2-view!
        p_ours_esc = np.where(esc_m[:, None], sc['p_4v'], sc['p_2v'])

        st_nat, pc_nat = coco_eval_full(sc['gt_path'], sc['orig_preds'])
        # Also include exact gray-padded Full-DINOv2 & MGS-v1 from E2 JSON for 100% fair comparison!
        old_full_dino = old_e2_seed['splits'][sp]['S4_Full_DINOv2']
        old_mgs_dino = old_e2_seed['splits'][sp]['Ours_Pro_MGS_DINOv2']

        res_sp = {
            'B0_YOLOv8n_Native': {'AP50_95': round(st_nat[0] * 100.0, 4), 'AP50': round(st_nat[1] * 100.0, 4), 'esc_hi_pct': 0.0, 'crop_cost_vs_fulldino': 0.0, 'per_cls_AP': pc_nat},
            'B1a_Full_DINOv2_GrayPad_Orig': {'AP50_95': old_full_dino['AP50_95'], 'AP50': old_full_dino['AP50'], 'esc_hi_pct': 0.0, 'crop_cost_vs_fulldino': 1.0, 'per_cls_AP': old_full_dino['per_cls_AP']},
            'B1b_Full_DINOv2_SquareCrop': eval_prob_cfg(m_all, sc['p_std_raw'], False, False),
            'B2_MGS_DINOv2_v1': {'AP50_95': old_mgs_dino['AP50_95'], 'AP50': old_mgs_dino['AP50'], 'esc_hi_pct': 0.0, 'crop_cost_vs_fulldino': round(old_mgs_dino['trig_conf25_pct'] / 100.0, 3), 'per_cls_AP': old_mgs_dino['per_cls_AP']},
            'B3_Static_Posterior_Fuse': eval_prob_cfg(m_all, p_b3, False, False),
            'Abl1_SingleCrop_14Cls_Calib': eval_prob_cfg(fg, p_abl1, True, True),
            'Abl2_TriGranular_Uniform_All3': eval_prob_cfg(m_all, p_abl2, True, True, esc_mask_opt=m_all),
            'Ours_CGGAR_Full4View': eval_prob_cfg(fg, sc['p_4v'], True, True, esc_mask_opt=fg),
            'Ours_CGGAR_Selective_Esc': eval_prob_cfg(fg, p_ours_esc, True, True, esc_mask_opt=esc_m),
        }
        out_splits[sp] = res_sp
        print(f"  [Seed {seed} | {sp:5s}] 原生={res_sp['B0_YOLOv8n_Native']['AP50_95']:.2f}% | "
              f"原版Full-DINO={res_sp['B1a_Full_DINOv2_GrayPad_Orig']['AP50_95']:.2f}% | "
              f"静态融合={res_sp['B3_Static_Posterior_Fuse']['AP50_95']:.2f}% | "
              f"Abl1(单框校准)={res_sp['Abl1_SingleCrop_14Cls_Calib']['AP50_95']:.2f}% | "
              f"★Ours(按需升级CG-GAR)={res_sp['Ours_CGGAR_Selective_Esc']['AP50_95']:.2f}% (升级率={res_sp['Ours_CGGAR_Selective_Esc']['esc_hi_pct']:.1f}%, 相对Full算力={res_sp['Ours_CGGAR_Selective_Esc']['crop_cost_vs_fulldino']:.2f}x) | "
              f"★Ours(全升级)={res_sp['Ours_CGGAR_Full4View']['AP50_95']:.2f}%",
              flush=True)

    return {
        'seed': seed,
        'selected_beta_on_val': best_beta,
        'tau_esc_on_val': round(tau_esc, 5),
        'fg_score_thr': FG_SCORE_THR,
        'ckpt_sha256': hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
        'splits': out_splits,
    }


def main():
    t0 = time.monotonic()
    backend = configure_strict_fp32()
    old_e2 = json.loads(E2_OLD_JSON.read_text(encoding='utf-8'))

    per_seed = []
    for seed in [0, 1, 2]:
        print(f"\n================ 训练与严格验证 CG-GAR v2 Seed {seed} ================", flush=True)
        ck = train_cggar_v2_for_seed(seed, 'cuda:0')
        old_s = next(r for r in old_e2['逐种子明细'] if r['seed'] == seed)
        sr = evaluate_seed_v2(seed, ck, old_s, 'cuda:0')
        per_seed.append(sr)

    methods = list(per_seed[0]['splits']['val'].keys())
    agg = {}
    for m in methods:
        agg[m] = {}
        for sp in ['val', 'testB', 'testC']:
            aps = [s['splits'][sp][m]['AP50_95'] for s in per_seed]
            ap50s = [s['splits'][sp][m]['AP50'] for s in per_seed]
            escs = [s['splits'][sp][m]['esc_hi_pct'] for s in per_seed]
            costs = [s['splits'][sp][m]['crop_cost_vs_fulldino'] for s in per_seed]
            agg[m][sp] = {
                'AP50_95_str': f"{np.mean(aps):.2f}±{np.std(aps, ddof=1):.2f}",
                'AP50_95_mean': round(float(np.mean(aps)), 4),
                'AP50_95_std': round(float(np.std(aps, ddof=1)), 4),
                'AP50_str': f"{np.mean(ap50s):.2f}±{np.std(ap50s, ddof=1):.2f}",
                'seeds': [round(v, 2) for v in aps],
                'esc_hi_pct_mean': round(float(np.mean(escs)), 2),
                'crop_cost_vs_fulldino_mean': round(float(np.mean(costs)), 3),
            }

    print("\n================ 严苛 3/3 随机种子零退步门禁核验 (CG-GAR v2) ================", flush=True)
    gate_audit = {}
    baselines = ['B0_YOLOv8n_Native', 'B1a_Full_DINOv2_GrayPad_Orig', 'B1b_Full_DINOv2_SquareCrop', 'B2_MGS_DINOv2_v1', 'B3_Static_Posterior_Fuse', 'Abl1_SingleCrop_14Cls_Calib']
    for cand in ['Ours_CGGAR_Selective_Esc', 'Ours_CGGAR_Full4View']:
        gate_audit[cand] = {}
        for base in baselines:
            wb = [per_seed[s]['splits']['testB'][cand]['AP50_95'] - per_seed[s]['splits']['testB'][base]['AP50_95'] for s in range(3)]
            wc = [per_seed[s]['splits']['testC'][cand]['AP50_95'] - per_seed[s]['splits']['testC'][base]['AP50_95'] for s in range(3)]
            wv = [per_seed[s]['splits']['val'][cand]['AP50_95'] - per_seed[s]['splits']['val'][base]['AP50_95'] for s in range(3)]
            pb, pc, pv = all(d > 0 for d in wb), all(d > 0 for d in wc), all(d > 0 for d in wv)
            gate_audit[cand][base] = {
                'val_seed_deltas': [round(d, 2) for d in wv], 'val_3of3_win': pv,
                'testB_seed_deltas': [round(d, 2) for d in wb], 'testB_3of3_win': pb,
                'testC_seed_deltas': [round(d, 2) for d in wc], 'testC_3of3_win': pc,
            }
            print(f"  [{cand:25s} vs {base:28s}] val={[round(d,2) for d in wv]}({pv}) | "
                  f"testB={[round(d,2) for d in wb]}({pb}) | testC={[round(d,2) for d in wc]}({pc})", flush=True)

    out_p = RES_DIR / 'cggar_v2_scidb_3seed_summary.json'
    out_p.write_text(json.dumps({
        '实验名称': 'CG-GAR_v2_无污染反事实收益仲裁与按需跨粒度升级_3种子严格门禁核验',
        '耗时秒': round(time.monotonic() - t0, 2),
        '严格后端': backend,
        '三种子汇总': agg,
        '严格3比3门禁核验': gate_audit,
        '逐种子明细': per_seed,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n[完成] 已保存至: {out_p}", flush=True)


if __name__ == '__main__':
    main()
