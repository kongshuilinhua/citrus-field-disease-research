# -*- coding: utf-8 -*-
"""
Antigravity v4: 面向真实果园落地的两层八候选预注册方法库与有限样本保形风险控制实测套件
(Pre-Registered Two-Layer Orchard Candidate Arsenal & Finite-Sample Conformal Risk Control Suite)

严格遵守零作假、零测试集调参、零基座重训铁律：
1. 冻结权重完整性校验：强制核验 cggar_v2_seed{0,1,2}.pt 与 bgopta_frozen_seed{0,1,2}.pt 共 6 份权重 SHA-256 100% 未改动。
2. 零目标域泄漏选参：第一层（路线 1）全部 5 个检测校准候选 C1..C5 的连续超参数 100% 仅在源域 SciDB val 上网格锁定或为零学习参数解析公式，
   绝不接触 testB / testC / Zenodo / CitrusUAT / PCR 选参！
3. 第二层（路线 2）全部校准与保形风险控制候选 R1..R3 严格采用“校准集 (Calibration Split) 与评测集 (Held-Out Eval Split) 100% 图像互斥”协议
   （重复 200 次独立随机分层/互斥划分，汇报均值与 95% 置信区间），严禁在评估集自身上拟合先验或阈值。
4. 完整覆盖用户需求场景：
   - 检测主指标：AP50-95、AP50、B=1000 逐图配对 Bootstrap (相对 Native YOLOv8n 与 Full DINOv2)；
   - App 交互确认指标：GT 匹配框 Top-1 / Top-2 / Top-3 症状候选命中率（面向胡柚手机端“提供可确认的症状候选结果”）；
   - 病虫害 vs 缺素非生物胁迫 OOD 区分度（面向“病害为主、缺素作为未知/OOD干扰测试”）；
   - 跨域行为全景：SciDB (val/testB/testC) + Zenodo 11148332 (519图果园外部集) + CitrusUAT (500图室内近景单叶 OOD 集)；
   - 有限样本保形风险控制 (CRC, Angelopoulos et al., ICLR 2024)：验证有限样本数学风险上界 E[L_eval(λ_hat)] <= α_risk 与自动放行覆盖率增益；
   - 597 样本 PCR 疑似症状风险分层送检 (G1 与 G2 双口径，严格遵守“黄化不等于确诊黄龙病，仅作疑似症状送检优先级排序”)。
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'

import sys
import io
import csv
import json
import math
import time
import hashlib
import contextlib
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

for _extra_sp in [r'D:\新版柑橘\方法路线扩展_20261031\检测环境\Lib\site-packages', r'D:\新版柑橘\检测环境\Lib\site-packages']:
    if Path(_extra_sp).exists() and _extra_sp not in sys.path:
        sys.path.append(_extra_sp)

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from faster_coco_eval import COCO, COCOeval_faster

ROOT = Path(r'D:\新版柑橘')
EXT = ROOT / '方法路线扩展_20261031'
WORK_BG = ROOT / 'Antigravity_BGOPTA_单趟正交Token攻关_20261126'
PREV_TCD = ROOT / 'Antigravity_CGGAR_真创新攻关_20261125'
PCR_DIR = ROOT / 'DS执行_病原学层面评估_20261123'
REPO_AG = ROOT / 'citrus-field-disease-research' / 'antigravity'
V4_DIR = REPO_AG / 'v4_orchard_candidate_arsenal'
RES_DIR = V4_DIR / 'results'
FIG_DIR = V4_DIR / 'figures'
DOC_DIR = V4_DIR / 'docs'
for _d in [RES_DIR, FIG_DIR, DOC_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

sys.path.append(str(EXT))
sys.path.append(str(WORK_BG))
sys.path.append(str(PREV_TCD))
sys.path.append(str(REPO_AG / 'v3_bgopta_and_dtemop' / 'scripts'))

from 共同评测后端 import configure_strict_fp32
from phase1_source_only_train_and_select import set_seed, apply_classwise_nms, compute_maha_z
from phase2_frozen_oneshot_eval import (
    verify_phase1_lock, coco_eval_metrics, YoloFullStateExtractor, CITRUSUAT_DIR
)
from step1_extract_multigranular_cache import DinoExtractor, crop_square_scale, compute_box_geom
from step2_train_and_eval_cggar import TargetContextRelationalHead, entropy_norm
from step3_external_blind_and_latency import build_external_zenodo_gt
from run_rigorous_audit_suite import PlainConcat3VecHead, paired_image_bootstrap_coco

CKPT_BG = WORK_BG / 'frozen_models'
CKPT_TCD = PREV_TCD / 'models'
CACHE_DIR = PREV_TCD / 'cache'
GT_DIR = EXT / '检测数据'
STRONG_DIR = EXT / 'YOLOv8强基线重复_20261109'

# 0-based SciDB category definitions:
# 0:S1 Healthy fruit, 1:S2 Healthy leaf, 2:S3 Blotchy mottling, 3:S4 Red nose fruit,
# 4:S5 Zinc deficiency, 5:S6 Vein yellowing, 6:S7 Uniform yellowing, 7:S8 Magnesium deficiency,
# 8:S9 Boron deficiency, 9:S10 Anthracnose, 10:S11 Citrus greasy spot, 11:S14 Canker,
# 12:S13 Sooty mould, 13:S12 Algal spot and moss
HEALTHY_SET = {0, 1}
ABIOTIC_DEFICIENCY_SET = {4, 7, 8}          # S5 Zn, S8 Mg, S9 B (非生物营养缺素胁迫)
BIOTIC_DISEASE_SET = {2, 3, 5, 6, 9, 10, 11, 12, 13}  # 传染性/寄生性病害症状
G1_HLB_SUSPECT_0B = {2, 3, 5, 6}            # 严格疑似黄化四症状 (S3/S4/S6/S7，不等于确诊黄龙病)
G2_HLB_SUSPECT_0B = {2, 3, 4, 5, 6}         # 宽口径疑似五症状 (含 S5 缺锌样黄化)


def verify_all_6_frozen_checkpoints():
    p1 = verify_phase1_lock()
    expected_tcd = {
        0: 'b277920fab8911f2cbee33bbd2ec4044d3f2fcf60d68584db9b751c5b264e49a',
        1: '47842e4831017911daa4197b1b4442a16bb2f8d28478e747bd1e00d84c2e962b',
        2: '67557b6768837380474b93249e81fc0e7e8870fb8b0530d9225c0edde770c7ae',
    }
    for s, sh in expected_tcd.items():
        actual = hashlib.sha256((CKPT_TCD / f'cggar_v2_seed{s}.pt').read_bytes()).hexdigest()
        assert actual == sh, f"cggar_v2_seed{s}.pt SHA mismatch!"
    print("[零作假门禁] 全部 6 份冻结模型权重 SHA-256 核验 100% 一致，零改动！", flush=True)
    return p1


def greedy_match_gt_1to1(gt_dict: dict, rows: list, iou_thr: float = 0.50):
    anns_by_img = defaultdict(list)
    for a in gt_dict['annotations']:
        anns_by_img[a['image_id']].append(a)
    n = len(rows)
    gt_cls = np.full(n, -1, dtype=np.int64)
    gt_iou = np.zeros(n, dtype=np.float32)
    by_img = defaultdict(list)
    for idx, r in enumerate(rows):
        by_img[r['image_id']].append((r['score'], idx, r['bbox']))
    for iid, rlist in by_img.items():
        rlist.sort(key=lambda x: -x[0])
        glist = anns_by_img.get(iid, [])
        used = set()
        for _sc, idx, (bx, by, bw, bh) in rlist:
            best_iou, best_j = 0.0, -1
            for j, a in enumerate(glist):
                if j in used:
                    continue
                gx, gy, gw, gh = a['bbox']
                ix1, iy1 = max(bx, gx), max(by, gy)
                ix2, iy2 = min(bx + bw, gx + gw), min(by + bh, gy + gh)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                iou = inter / (bw * bh + gw * gh - inter + 1e-9)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thr and best_j >= 0:
                used.add(best_j)
                gt_cls[idx] = int(glist[best_j]['category_id'])
                gt_iou[idx] = float(best_iou)
    return gt_cls, gt_iou


def train_source_plain_concat_head(seed: int, device: str = 'cuda:0') -> PlainConcat3VecHead:
    """严格仅在源域 SciDB train 缓存上训练三尺度普通拼接头 (零测试集接触)。"""
    tr_npz = np.load(CACHE_DIR / f'train_cggar_seed{seed}.npz')
    set_seed(seed + 100)
    u_imgs = np.unique(tr_npz['box_img_ids'])
    rng = np.random.RandomState(2026 + seed)
    rng.shuffle(u_imgs)
    fit_set = set(u_imgs[:int(len(u_imgs) * 0.85)].tolist())
    is_fit = np.array([i in fit_set for i in tr_npz['box_img_ids']], bool)
    t_fit = torch.from_numpy(np.where(is_fit)[0]).to(device)
    t_dev = torch.from_numpy(np.where(~is_fit)[0]).to(device)

    y_all = torch.from_numpy(tr_npz['labels']).to(device)
    g_all = torch.from_numpy(tr_npz['geom']).to(device)
    zc_all = torch.from_numpy(tr_npz['z_core'].astype(np.float32)).to(device)
    zs_all = torch.from_numpy(tr_npz['z_std'].astype(np.float32)).to(device)
    zx_all = torch.from_numpy(tr_npz['z_ctx'].astype(np.float32)).to(device)
    zcf_all = torch.from_numpy(tr_npz['z_cf_bad'].astype(np.float32)).to(device)

    hd = PlainConcat3VecHead().to(device)
    opt = torch.optim.AdamW(hd.parameters(), lr=1.2e-3, weight_decay=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
    best_l, best_st = 1e9, None
    for _ep in range(20):
        hd.train()
        perm = t_fit[torch.randperm(len(t_fit), device=device)]
        for ids in perm.split(256):
            opt.zero_grad()
            cm = torch.randn((len(ids), 384), device=device) * 0.08
            zci = F.normalize(zc_all[ids] + cm, dim=-1)
            zsi = F.normalize(zs_all[ids] + cm, dim=-1)
            zxi = F.normalize(zx_all[ids] + cm, dim=-1)
            cm_m = torch.rand(len(ids), device=device) < 0.25
            if cm_m.any():
                zsi = torch.where(cm_m.unsqueeze(1), zcf_all[ids], zsi)
            loss = F.cross_entropy(hd(zci, zsi, zxi, g_all[ids]), y_all[ids], label_smoothing=0.05)
            loss.backward()
            opt.step()
        sched.step()
        hd.eval()
        with torch.no_grad():
            dl = 0.5 * F.cross_entropy(hd(zc_all[t_dev], zs_all[t_dev], zx_all[t_dev], g_all[t_dev]), y_all[t_dev]).item() + \
                 0.5 * F.cross_entropy(hd(zc_all[t_dev], zcf_all[t_dev], zx_all[t_dev], g_all[t_dev]), y_all[t_dev]).item()
            if dl < best_l:
                best_l = dl
                best_st = {k: v.detach().cpu() for k, v in hd.state_dict().items()}
    hd.load_state_dict(best_st)
    hd.eval()
    return hd


def norm_entropy_np(p: np.ndarray) -> np.ndarray:
    p_c = np.clip(p, 1e-7, 1.0)
    return -(p_c * np.log(p_c)).sum(axis=1) / math.log(p.shape[1])


def js_divergence_np(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p_c = np.clip(p, 1e-7, 1.0)
    q_c = np.clip(q, 1e-7, 1.0)
    m = 0.5 * (p_c + q_c)
    return 0.5 * ((p_c * (np.log(p_c) - np.log(m))).sum(axis=1) + (q_c * (np.log(q_c) - np.log(m))).sum(axis=1)) / math.log(2.0)


def top1_margin_np(p: np.ndarray) -> np.ndarray:
    s = np.sort(p, axis=1)
    return s[:, -1] - s[:, -2]


def softmax_from_logp(lp: np.ndarray) -> np.ndarray:
    z = lp - lp.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def calibrate_source_val_params(val_pack: dict, gt_val_path: Path):
    """
    严格仅在源域 SciDB val 上标定并锁定各候选方法的标量超参数 (零目标域接触！)：
    - C2: gamma_ood ∈ {0.10, 0.20, 0.30, 0.40} (流形衰减强度，选 val AP50-95 最优)
    - C3: Zero-MLP 双裁剪 (0.82x Core + 1.25x Std) 融合比例 (w_det, w_core, w_std) 选自固定小网格
    """
    orig_preds = val_pack['orig_preds']
    cand_idxs = val_pack['cand_idxs']
    yolo_sc = val_pack['yolo_sc']
    fg = val_pack['fg']
    lp_det = val_pack['lp_det']
    lp_std = val_pack['lp_std']
    lp_core = val_pack['lp_core']
    h_det = val_pack['h_det']
    h_std = val_pack['h_std']
    d_maha_z = val_pack['d_maha_z']
    beta = val_pack['beta_lock']

    def quick_val_ap(p_mat: np.ndarray) -> float:
        rows = [dict(r) for r in orig_preds]
        for li, gi in enumerate(cand_idxs):
            if fg[li]:
                pr = p_mat[li]
                nc = int(np.argmax(pr))
                rows[gi]['category_id'] = nc
                rows[gi]['score'] = float((yolo_sc[li] ** beta) * (max(float(pr[nc]), 1e-6) ** (1.0 - beta)))
        return coco_eval_metrics(gt_val_path, apply_classwise_nms(rows, 0.70))['AP50_95']

    # Lock gamma_ood for C2 on SciDB val (tie-breaking toward moderate 0.25 OOD safeguard)
    best_gamma, best_c2_ap = 0.25, -1.0
    c_det_base = np.clip(1.0 - h_det, 0.05, 1.0)
    c_std_base = np.clip(1.0 - h_std, 0.05, 1.0)
    for gm in [0.15, 0.25, 0.35]:
        u_d = 0.35 * c_det_base * np.exp(-gm * np.maximum(0.0, d_maha_z))
        u_s = 0.65 * c_std_base
        lp_m = (u_d[:, None] * lp_det + u_s[:, None] * lp_std) / (u_d + u_s + 1e-8)[:, None]
        ap = quick_val_ap(softmax_from_logp(lp_m))
        if ap > best_c2_ap + 1e-4:
            best_c2_ap = ap
            best_gamma = gm

    # Lock (w_det, w_core, w_std) for C3 (Zero-MLP Dual-Crop Core+Std) on SciDB val
    best_w3, best_c3_ap = (0.30, 0.25, 0.45), -1.0
    for wd, wc, ws in [(0.30, 0.20, 0.50), (0.30, 0.25, 0.45), (0.28, 0.27, 0.45), (0.25, 0.30, 0.45)]:
        lp_m = wd * lp_det + wc * lp_core + ws * lp_std
        ap = quick_val_ap(softmax_from_logp(lp_m))
        if ap > best_c3_ap + 1e-4:
            best_c3_ap = ap
            best_w3 = (wd, wc, ws)

    return {
        'gamma_ood_C2': best_gamma,
        'val_ap_C2': round(best_c2_ap, 3),
        'weights_C3_det_core_std': list(best_w3),
        'val_ap_C3': round(best_c3_ap, 3),
    }


def compute_all_candidate_posteriors(pack: dict, locked_cfg: dict):
    """
    给定单划分特征包与源域锁定的超参数，一次性计算 3 个基线与 5 个零样本检测候选 (C1..C5)：
    返回每种方法的：(pred_rows, cand_prob_14, avg_dino_crops_per_fg_box)
    """
    orig_preds = pack['orig_preds']
    cand_idxs = pack['cand_idxs']
    n_c = len(cand_idxs)
    yolo_sc = pack['yolo_sc']
    yolo_cls = pack['yolo_cls']
    fg = pack['fg']
    beta = pack['beta_lock']

    p_det = pack['p_det']
    p_std_unscaled = pack['p_std_unscaled']
    lp_det = pack['lp_det']
    lp_std = pack['lp_std']
    lp_core = pack['lp_core']
    lp_cat3 = pack['lp_cat3']
    lp_diff = pack['lp_diff']
    h_det = pack['h_det']
    h_std = pack['h_std']
    d_maha_z = pack['d_maha_z']

    gm = locked_cfg['gamma_ood_C2']
    wd3, wc3, ws3 = locked_cfg['weights_C3_det_core_std']

    def build_rows_calibrated(p_mat: np.ndarray):
        rows = [dict(r) for r in orig_preds]
        for li, gi in enumerate(cand_idxs):
            if fg[li]:
                pr = p_mat[li]
                nc = int(np.argmax(pr))
                rows[gi]['category_id'] = nc
                rows[gi]['score'] = float((yolo_sc[li] ** beta) * (max(float(pr[nc]), 1e-6) ** (1.0 - beta)))
        return apply_classwise_nms(rows, 0.70)

    out = {}

    # B0: Native YOLOv8n (0 ViT crops)
    out['B0_Native_YOLOv8n'] = ([dict(r) for r in orig_preds], pack['p14_raw'], 0.0)

    # B1: Full DINOv2 (1.25x Hard Top-1 Overwrite, 1.0 ViT crop/box)
    rows_b1 = [dict(r) for r in orig_preds]
    for li, gi in enumerate(cand_idxs):
        nc = int(np.argmax(p_std_unscaled[li]))
        if nc != rows_b1[gi]['category_id']:
            rows_b1[gi]['category_id'] = nc
            rows_b1[gi]['score'] = float(math.sqrt(yolo_sc[li] * max(float(p_std_unscaled[li, nc]), 1e-6)))
    out['B1_Full_DINOv2_1Crop'] = (rows_b1, p_std_unscaled, 1.0)

    # B2: Static Linear Prob Fuse (legacy conditional score, no post-NMS)
    p_b2 = 0.65 * p_std_unscaled.copy()
    p_b2[np.arange(n_c), yolo_cls] += 0.35 * yolo_sc
    p_b2_norm = p_b2 / np.maximum(p_b2.sum(axis=1, keepdims=True), 1e-8)
    rows_b2 = [dict(r) for r in orig_preds]
    for li, gi in enumerate(cand_idxs):
        nc = int(np.argmax(p_b2[li]))
        if nc != rows_b2[gi]['category_id']:
            rows_b2[gi]['category_id'] = nc
            rows_b2[gi]['score'] = float(math.sqrt(yolo_sc[li] * max(float(p_b2[li, nc]), 1e-6)))
    out['B2_Static_Prob_Fuse'] = (rows_b2, p_b2_norm, 1.0)

    # C1: Fast 1-Crop Static Log-Linear Calibration (Abl1: 0.35/0.65 + ScoreRew + PostNMS, 1.0 ViT crop/box)
    p_c1 = softmax_from_logp(0.35 * lp_det + 0.65 * lp_std)
    out['C1_Fast1Crop_StaticCalib'] = (build_rows_calibrated(p_c1), p_c1, 1.0)

    # C2: Fast 1-Crop OOD-Attenuated Analytic Certainty Pool (Zero learned gate params, 1.0 ViT crop/box)
    c_det = np.clip(1.0 - h_det, 0.05, 1.0) * np.exp(-gm * np.maximum(0.0, d_maha_z))
    c_std = np.clip(1.0 - h_std, 0.05, 1.0)
    u0 = 0.35 * c_det
    u1 = 0.65 * c_std
    p_c2 = softmax_from_logp((u0[:, None] * lp_det + u1[:, None] * lp_std) / (u0 + u1 + 1e-8)[:, None])
    out['C2_Fast1Crop_OOD_CertaintyPool'] = (build_rows_calibrated(p_c2), p_c2, 1.0)

    # C3: Zero-MLP Dual-Crop Pure LP-FT Pool (0.82x Core + 1.25x Std, zero extra MLP head, 2.0 ViT crops/box)
    p_c3 = softmax_from_logp(wd3 * lp_det + wc3 * lp_core + ws3 * lp_std)
    out['C3_ZeroMLP_DualCrop_CoreStd'] = (build_rows_calibrated(p_c3), p_c3, 2.0)

    # C4: Hard-25% Compute-Budgeted Adaptive Escalation (75% boxes stay 1-crop C2, exact top 25% uncertain escalate to 3-crop, 1.50 ViT crops/box!)
    p_3crop_cat = softmax_from_logp(0.24 * lp_det + 0.24 * lp_std + 0.20 * lp_core + 0.32 * lp_cat3)
    disc_1pass = js_divergence_np(p_det, softmax_from_logp(lp_std)) + 0.5 * h_std
    fg_idx = np.where(fg)[0]
    esc_mask = np.zeros(n_c, dtype=bool)
    if len(fg_idx) > 0:
        k_esc = max(1, int(round(0.25 * len(fg_idx))))
        top_fg = fg_idx[np.argsort(-disc_1pass[fg_idx])[:k_esc]]
        esc_mask[top_fg] = True
    p_c4 = np.where(esc_mask[:, None], p_3crop_cat, p_c2)
    avg_crops_c4 = 1.0 + 2.0 * float(esc_mask[fg].mean()) if fg.sum() > 0 else 1.0
    out['C4_HardBudget25_MultiScale'] = (build_rows_calibrated(p_c4), p_c4, round(avg_crops_c4, 3))

    # C5a: Server 3-Crop Plain Concatenation Ensemble (PlainConcat_3Vec, 3.0 ViT crops/box)
    out['C5a_Server_3Crop_PlainConcat'] = (build_rows_calibrated(p_3crop_cat), p_3crop_cat, 3.0)

    # C5b: Server 3-Crop Differential Ensemble (Abl3 historical reference, 3.0 ViT crops/box)
    p_3crop_diff = softmax_from_logp(0.24 * lp_det + 0.24 * lp_std + 0.20 * lp_core + 0.32 * lp_diff)
    out['C5b_Server_3Crop_DiffAbl3'] = (build_rows_calibrated(p_3crop_diff), p_3crop_diff, 3.0)

    return out


def compute_interactive_topk_and_abiotic_ood(gt_dict: dict, cand_rows: list, fg: np.ndarray, p_mat: np.ndarray):
    """
    计算两项真实果园 App 核心指标：
    1) GT 匹配前景框的 Top-1 / Top-2 / Top-3 症状候选命中率（对应 App 向农户/农技员展示 Top-K 候选确认卡片）；
    2) 传染性病害 (Biotic Disease) vs 非生物营养缺素 (Abiotic Nutrient Deficiency: Zn/Mg/B) 的后验质量区分 ROC-AUC。
    """
    gt_cls, _ = greedy_match_gt_1to1(gt_dict, cand_rows, iou_thr=0.50)
    m = fg & (gt_cls >= 0)
    y = gt_cls[m]
    pm = p_mat[m]
    order = np.argsort(-pm, axis=1)
    top1 = float((order[:, 0] == y).mean() * 100.0)
    top2 = float(np.any(order[:, :2] == y[:, None], axis=1).mean() * 100.0)
    top3 = float(np.any(order[:, :3] == y[:, None], axis=1).mean() * 100.0)

    # Biotic vs Abiotic OOD separation among symptomatic leaves (excluding Healthy {0,1}):
    sym_mask = np.array([c not in HEALTHY_SET for c in y], dtype=bool)
    y_sym = y[sym_mask]
    pm_sym = pm[sym_mask]
    is_abiotic_gt = np.array([1 if c in ABIOTIC_DEFICIENCY_SET else 0 for c in y_sym], dtype=np.int64)
    abiotic_cols = sorted(list(ABIOTIC_DEFICIENCY_SET))
    p_abiotic_mass = pm_sym[:, abiotic_cols].sum(axis=1)
    ood_auc = float(roc_auc_score(is_abiotic_gt, p_abiotic_mass)) if (is_abiotic_gt.sum() >= 3 and (1 - is_abiotic_gt).sum() >= 3) else None

    return {
        'Top1_Hit_pct': round(top1, 2),
        'Top2_Hit_pct': round(top2, 2),
        'Top3_Hit_pct': round(top3, 2),
        'Abiotic_vs_Biotic_ROC_AUC': round(ood_auc, 4) if ood_auc is not None else None,
    }


# ==============================================================================
# 第二层 (路线 2): R1 零标签分歧复核、R2 有限样本保形风险控制 (CRC)、R3 后验经验贝叶斯 PCR/症状分层送检
# ==============================================================================
def run_conformal_risk_control_simulation(gt_dict: dict, post_rows: list, n_cal_list=(40, 80, 120),
                                          alpha_targets=(0.10, 0.15, 0.20), n_repeats: int = 200, seed: int = 20261126):
    """
    有限样本保形风险控制 (Conformal Risk Control, Angelopoulos et al., ICLR 2024)：
    以图像为交换单元 (i = 1..M)，定义单调有界图像级自动放行损失 L_i(λ) ∈ [0, 1]：
      对于置信度阈值 λ ∈ [0, 1]，记 S_i(λ) = {b ∈ D_i : score(b) >= λ} 为自动放行检测框集合。
      若 |S_i(λ)| == 0（全部转人工/拒识放行），自动误判损失 L_i(λ) = 0；
      若 |S_i(λ)| > 0，L_i(λ) = (# S_i(λ) 中假阳性或分类错误框) / |S_i(λ)| （图像级自动放行错误发现率 FDR_i(λ)）。
      为保证 L_i(λ) 对 λ 严格单调非增（CRC 定理充分条件），对每个 λ 取前缀包络 L_mon_i(λ) = sup_{λ' >= λ} L_raw_i(λ')...
      注：更标准的做法是对图像最高置信度主诊断框 b*_i 定义单框选择性误判损失：
        L_i(λ) = 1{ score(b*_i) >= λ 且 b*_i 预测错误(IoU<0.5 或 类别错) } ∈ {0, 1}！
      检验单调性：当 λ 增大时，1{ score(b*_i) >= λ } 单调非增，故 L_i(λ) 对 λ 100% 严格单调非增且有界 B = 1！
      直接满足 Angelopoulos et al. (ICLR 2024) 定理 1 的全部精确有限样本条件：
        λ_hat = inf { λ ∈ Λ : (n / (n + 1)) * R_hat_n(λ) + 1 / (n + 1) <= α_risk }
      保证在任意分布下：E[ L_{n+1}(λ_hat) ] <= α_risk！
      同时报告：(1) 留出集真实期望风险 E[L_eval(λ_hat)]（验证 <= α_risk）；
               (2) 自动放行率 Coverage = P(score(b*_i) >= λ_hat)（越高代表省下越多人工复核工作量！）；
               (3) 自动放行子集内的条件精准率 Selective Accuracy = P(正确 | 自动放行)。
    """
    img_ids = sorted([im['id'] for im in gt_dict['images']])
    M = len(img_ids)
    gt_cls_all, _ = greedy_match_gt_1to1(gt_dict, post_rows, iou_thr=0.50)

    best_by_img = {}
    for idx, r in enumerate(post_rows):
        iid = r['image_id']
        if iid not in best_by_img or r['score'] > best_by_img[iid][0]:
            is_correct = (gt_cls_all[idx] == r['category_id'])
            best_by_img[iid] = (float(r['score']), bool(is_correct))

    scores = np.zeros(M, dtype=np.float64)
    errors = np.zeros(M, dtype=np.float64)  # 1 if top box wrong (when emitted), 0 if right
    has_det = np.zeros(M, dtype=bool)
    for m, iid in enumerate(img_ids):
        if iid in best_by_img:
            sc, ok = best_by_img[iid]
            scores[m] = sc
            errors[m] = 0.0 if ok else 1.0
            has_det[m] = True
        else:
            scores[m] = -1.0
            errors[m] = 0.0
            has_det[m] = False

    lambdas = np.linspace(0.0, 1.0001, 501)  # λ = 1.0001 -> abstain all -> L_i = 0
    # Precompute monotone bounded loss matrix L_mat: shape (M, G), L_mat[m, g] = 1{has_det[m] & scores[m] >= lambdas[g] & errors[m]==1}
    accept_mat = has_det[:, None] & (scores[:, None] >= lambdas[None, :])  # (M, G), monotone non-increasing in g!
    loss_mat = accept_mat.astype(np.float64) * errors[:, None]             # (M, G), strictly monotone non-increasing in g, bounded in [0, 1]!

    rng = np.random.RandomState(seed)
    res_crc = {}
    for n_cal in n_cal_list:
        res_crc[f'n_cal_{n_cal}'] = {}
        for alpha in alpha_targets:
            emp_risks, coverages, sel_accs, naive_risks = [], [], [], []
            for _ in range(n_repeats):
                perm = rng.permutation(M)
                cal_idx = perm[:n_cal]
                ev_idx = perm[n_cal:]

                # CRC finite-sample upper-adjusted risk on calibration split:
                r_hat_cal = loss_mat[cal_idx].mean(axis=0)  # (G,)
                crc_bound = (n_cal / (n_cal + 1.0)) * r_hat_cal + (1.0 / (n_cal + 1.0))
                valid_g = np.where(crc_bound <= alpha)[0]
                g_crc = int(valid_g[0]) if len(valid_g) > 0 else (len(lambdas) - 1)

                # Contrast with Naive uncalibrated empirical quantile (no +1/(n+1) finite-sample correction, or fixed λ=0.50)
                valid_naive = np.where(r_hat_cal <= alpha)[0]
                g_naive = int(valid_naive[0]) if len(valid_naive) > 0 else (len(lambdas) - 1)

                ev_loss = float(loss_mat[ev_idx, g_crc].mean())
                ev_cov = float(accept_mat[ev_idx, g_crc].mean() * 100.0)
                acc_cnt = accept_mat[ev_idx, g_crc].sum()
                ev_sel_acc = float((1.0 - loss_mat[ev_idx, g_crc].sum() / acc_cnt) * 100.0) if acc_cnt > 0 else 100.0

                emp_risks.append(ev_loss)
                coverages.append(ev_cov)
                sel_accs.append(ev_sel_acc)
                naive_risks.append(float(loss_mat[ev_idx, g_naive].mean()))

            res_crc[f'n_cal_{n_cal}'][f'alpha_{alpha:.2f}'] = {
                'target_risk_ceiling': alpha,
                'empirical_heldout_risk_mean': round(float(np.mean(emp_risks)), 4),
                'empirical_heldout_risk_ci95': [round(float(np.percentile(emp_risks, 2.5)), 4), round(float(np.percentile(emp_risks, 97.5)), 4)],
                'bound_satisfied': bool(float(np.mean(emp_risks)) <= alpha + 1e-4),
                'auto_coverage_pct_mean': round(float(np.mean(coverages)), 2),
                'auto_coverage_pct_ci95': [round(float(np.percentile(coverages, 2.5)), 2), round(float(np.percentile(coverages, 97.5)), 2)],
                'selective_accuracy_on_accepted_pct': round(float(np.mean(sel_accs)), 2),
                'naive_uncorrected_risk_mean': round(float(np.mean(naive_risks)), 4),
            }
    return res_crc


def run_visual_and_pcr_triage_suite(gt_testB: dict, gt_testC: dict, seed_packs_testB: dict, seed_packs_testC: dict,
                                    cand_outputs_testB: dict, cand_outputs_testC: dict):
    """
    评测 R1 (零标签视觉分歧送审) 与 R3 (后验经验贝叶斯期望风险送审 S_PEB = sum_c p(c|x) * r_EB(c))：
    1) 视觉症状错判捕获率 (testB & testC, 预算 5% / 10% / 20% / 30%)：对比零标签分歧 R1 vs 单纯最小置信度 vs 后验经验贝叶斯 R3；
    2) 597 样本 PCR 疑似症状风险分层送检 (严格保持仓库 597 样本口径，G1 与 G2 双报告，200 次独立互斥校准/评测划分)：
       对比 Naive 置信度 vs DS 硬类别先验 (YOLOv8n Top-1) vs 我们提出的级联软后验经验贝叶斯期望风险 R3 (C1 / C2 / C5a)！
    """
    budgets = [0.05, 0.10, 0.20, 0.30]

    # --- Part A: Visual Symptom Error Capture on Matched Boxes (testB & testC, 3 seeds) ---
    vis_triage = {}
    for sp_name, gt_d, s_packs, c_outs in [
        ('testB', gt_testB, seed_packs_testB, cand_outputs_testB),
        ('testC', gt_testC, seed_packs_testC, cand_outputs_testC),
    ]:
        vis_triage[sp_name] = defaultdict(lambda: {f'{int(b*100)}%': [] for b in budgets})
        for seed in [0, 1, 2]:
            pk = s_packs[seed]
            fg = pk['fg']
            gt_cls, _ = greedy_match_gt_1to1(gt_d, pk['cand_rows'], iou_thr=0.50)
            m = fg & (gt_cls >= 0)
            y_true = gt_cls[m]
            p_det = pk['p_det'][m]
            p_std = softmax_from_logp(pk['lp_std'][m])
            p_c2 = c_outs[seed]['C2_Fast1Crop_OOD_CertaintyPool'][1][m]

            pred_c2 = p_c2.argmax(axis=1)
            vis_err = (pred_c2 != y_true).astype(np.int64)
            n_tot = len(vis_err)
            tot_err = int(vis_err.sum())

            # 1. Baseline: 1 - max_prob(p_c2)
            sc_minconf = 1.0 - p_c2.max(axis=1)
            # 2. R1 Zero-Label Hybrid Discordance: JS(p_det || p_std) + 0.5 * (1 - margin(p_c2))
            sc_r1_disc = js_divergence_np(p_det, p_std) + 0.5 * (1.0 - top1_margin_np(p_c2))

            for tag, sc_arr in [('Baseline_1MinusMaxProb', sc_minconf), ('R1_ZeroLabel_JS_Margin_Discordance', sc_r1_disc)]:
                order = np.argsort(-sc_arr, kind='mergesort')
                for b in budgets:
                    k = max(1, int(round(b * n_tot)))
                    cap = float(vis_err[order[:k]].sum() / max(1, tot_err) * 100.0)
                    vis_triage[sp_name][tag][f'{int(b*100)}%'].append(cap)

            # 3. R3 Few-Shot Calibration (n_cal = 100 boxes disjoint, 200 splits): S_PEB + 0.25 * R1 tie-break
            rng = np.random.RandomState(2026 + seed)
            for b in budgets:
                caps_b = []
                for _ in range(200):
                    perm = rng.permutation(n_tot)
                    cal_i, ev_i = perm[:100], perm[100:]
                    ev_err = vis_err[ev_i]
                    if ev_err.sum() == 0:
                        continue
                    g_bar = float(vis_err[cal_i].mean())
                    r_eb = np.zeros(14, dtype=np.float64)
                    c_cal = pred_c2[cal_i]
                    for c in range(14):
                         idx_c = (c_cal == c)
                         nc = int(idx_c.sum())
                         ec = float(vis_err[cal_i][idx_c].sum())
                         r_eb[c] = (ec + 5.0 * g_bar) / (nc + 5.0)
                    s_peb = p_c2[ev_i] @ r_eb + 0.15 * sc_r1_disc[ev_i]
                    k = max(1, int(round(b * len(ev_i))))
                    ord_ev = np.argsort(-s_peb, kind='mergesort')
                    caps_b.append(float(ev_err[ord_ev[:k]].sum() / ev_err.sum() * 100.0))
                vis_triage[sp_name]['R3_Posterior_EB_Plus_Discordance_n100'][f'{int(b*100)}%'].append(float(np.mean(caps_b)))

    vis_summary = {}
    for sp_name in ['testB', 'testC']:
        vis_summary[sp_name] = {}
        for tag, bdict in vis_triage[sp_name].items():
            vis_summary[sp_name][tag] = {
                bk: {
                    'capture_pct_mean': round(float(np.mean(v)), 2),
                    'capture_pct_std': round(float(np.std(v, ddof=1)), 2),
                    'gain_over_random_x': round(float(np.mean(v)) / float(bk.replace('%', '')), 2),
                    'seeds': [round(x, 2) for x in v],
                }
                for bk, v in bdict.items()
            }

    # --- Part B: Exact 597-Sample PCR Pathogen Alignment & Triage Suite (G1 & G2, 3 Seeds × 200 Disjoint Splits) ---
    pcr_rows = [r for r in csv.DictReader((PCR_DIR / 'PCR映射_逐图.csv').open(encoding='utf-8-sig'))
                if r['PCR'] in ('PCR+', 'PCR-')]
    assert len(pcr_rows) == 597, f"Expected exact 597 PCR samples, got {len(pcr_rows)}"
    fname2iid = {im['file_name']: im['id'] for im in gt_testB['images']}

    pcr_summary = {}
    for grp_name, suspect_set in [('G1_Strict4Symptoms', G1_HLB_SUSPECT_0B), ('G2_WithZn5Symptoms', G2_HLB_SUSPECT_0B)]:
        # Build per-seed image-level predictions for B0 (YOLOv8n), C2 (1-crop OOD Pool), C5a (3-crop Ensemble)
        model_img_records = {mk: [] for mk in ['B0_Native_YOLOv8n', 'C1_Fast1Crop_StaticCalib', 'C2_Fast1Crop_OOD_CertaintyPool', 'C5a_Server_3Crop_PlainConcat']}
        sens_spec_stats = defaultdict(lambda: defaultdict(list))

        for seed in [0, 1, 2]:
            pk = seed_packs_testB[seed]
            iid_to_local = defaultdict(list)
            for li, r in enumerate(pk['cand_rows']):
                iid_to_local[r['image_id']].append((r['score'], li))

            for mk in model_img_records.keys():
                post_rows, p_mat, _ = cand_outputs_testB[seed][mk]
                # Map post-NMS top-scoring box per image (conf >= 0.25 as standard diagnostic call threshold, matching DS 10_病原学评估.py!)
                top_post = {}
                for r in post_rows:
                    if r['score'] >= 0.25:
                        iid = r['image_id']
                        if iid not in top_post or r['score'] > top_post[iid]['score']:
                            top_post[iid] = r

                seed_recs = []
                tp = fn = fp = tn = no_call = 0
                for pr in pcr_rows:
                    iid = fname2iid[pr['文件']]
                    y_pcr = 1 if pr['PCR'] == 'PCR+' else 0
                    if iid in top_post:
                        pred_c = int(top_post[iid]['category_id'])
                        conf_sc = float(top_post[iid]['score'])
                        # Find matching candidate posterior vector for soft expected risk
                        loc_cands = iid_to_local.get(iid, [])
                        if loc_cands:
                            loc_cands.sort(key=lambda x: -x[0])
                            p_vec = p_mat[loc_cands[0][1]].astype(np.float64)
                        else:
                            p_vec = np.full(14, 1.0 / 14.0)
                    else:
                        pred_c = -1  # No diagnosis call
                        conf_sc = 0.0
                        no_call += 1
                        p_vec = np.zeros(14, dtype=np.float64)

                    pred_suspect = 1 if (pred_c in suspect_set) else 0
                    pcr_err = 1 if (pred_suspect != y_pcr) else 0
                    if y_pcr == 1 and pred_suspect == 1:
                        tp += 1
                    elif y_pcr == 1 and pred_suspect == 0:
                        fn += 1
                    elif y_pcr == 0 and pred_suspect == 1:
                        fp += 1
                    else:
                        tn += 1

                    seed_recs.append({
                        'pred_c': pred_c,
                        'conf': conf_sc,
                        'p_vec': p_vec,
                        'y_pcr': y_pcr,
                        'pcr_err': pcr_err,
                    })
                model_img_records[mk].append(seed_recs)
                sens_spec_stats[mk]['sensitivity_pct'].append(round(tp / max(1, tp + fn) * 100.0, 2))
                sens_spec_stats[mk]['specificity_pct'].append(round(tn / max(1, tn + fp) * 100.0, 2))
                sens_spec_stats[mk]['accuracy_pct'].append(round((tp + tn) / 597.0 * 100.0, 2))
                sens_spec_stats[mk]['total_pcr_errors'].append(int(fn + fp))
                sens_spec_stats[mk]['no_call_count'].append(int(no_call))

        # Evaluate disjoint calibration sample size curve n_cal ∈ {0 (zero-label conf), 50, 100, 200} across 200 splits!
        triage_curves = {}
        # Strategy 1: Zero-Label Min-Confidence on C2 (verifies negative finding: confidence fails on PCR!)
        zero_conf_caps = {f'{int(b*100)}%': [] for b in budgets}
        for seed in [0, 1, 2]:
            recs = model_img_records['C2_Fast1Crop_OOD_CertaintyPool'][seed]
            errs = np.array([r['pcr_err'] for r in recs], dtype=np.int64)
            confs = np.array([r['conf'] if r['pred_c'] >= 0 else 1.0 for r in recs], dtype=np.float64)
            ord_c = np.argsort(confs, kind='mergesort')
            for b in budgets:
                k = max(1, int(round(b * len(errs))))
                zero_conf_caps[f'{int(b*100)}%'].append(float(errs[ord_c[:k]].sum() / max(1, errs.sum()) * 100.0))
        triage_curves['ZeroCalib_MinConfidence_NegativeCtrl'] = {
            bk: {'capture_pct_mean': round(float(np.mean(v)), 2), 'gain_over_random_x': round(float(np.mean(v)) / float(bk.replace('%', '')), 2)}
            for bk, v in zero_conf_caps.items()
        }

        for n_cal in [50, 100, 200]:
            for strat_name, mk, use_soft_posterior in [
                (f'HardClassPrior_B0_YOLOv8n_n{n_cal}', 'B0_Native_YOLOv8n', False),
                (f'R3_SoftPosteriorEB_C2_1Crop_n{n_cal}', 'C2_Fast1Crop_OOD_CertaintyPool', True),
                (f'R3_SoftPosteriorEB_C5a_3Crop_n{n_cal}', 'C5a_Server_3Crop_PlainConcat', True),
            ]:
                b_seed_means = {f'{int(b*100)}%': [] for b in budgets}
                b_seed_ci_lo = {f'{int(b*100)}%': [] for b in budgets}
                b_seed_ci_hi = {f'{int(b*100)}%': [] for b in budgets}
                for seed in [0, 1, 2]:
                    recs = model_img_records[mk][seed]
                    N = len(recs)
                    pred_cs = np.array([r['pred_c'] for r in recs], dtype=np.int64)
                    errs = np.array([r['pcr_err'] for r in recs], dtype=np.int64)
                    p_vecs = np.stack([r['p_vec'] for r in recs], axis=0)  # (597, 14)
                    confs = np.array([r['conf'] for r in recs], dtype=np.float64)

                    rng = np.random.RandomState(20261126 + seed * 17 + n_cal)
                    split_caps = {b: [] for b in budgets}
                    for _ in range(200):
                        perm = rng.permutation(N)
                        cal_i, ev_i = perm[:n_cal], perm[n_cal:]
                        ev_err = errs[ev_i]
                        if ev_err.sum() == 0:
                            continue
                        g_bar = float(errs[cal_i].mean())

                        if not use_soft_posterior:
                            # DS Hard Class Prior (MLE on predicted discrete class, random/conf tie-break)
                            cls_rate = {}
                            for c in range(-1, 14):
                                m_c = (pred_cs[cal_i] == c)
                                if m_c.sum() > 0:
                                    cls_rate[c] = float(errs[cal_i][m_c].mean())
                                else:
                                    cls_rate[c] = g_bar
                            pri_ev = np.array([cls_rate[pred_cs[j]] for j in ev_i], dtype=np.float64)
                            ord_ev = np.lexsort((confs[ev_i], -pri_ev))
                        else:
                            # R3: Empirical-Bayes Shrunk Class Risk (m=5) + Soft Calibrated Cascade Posterior Expectation!
                            r_eb = np.zeros(14, dtype=np.float64)
                            for c in range(14):
                                m_c = (pred_cs[cal_i] == c)
                                nc = int(m_c.sum())
                                ec = float(errs[cal_i][m_c].sum())
                                r_eb[c] = (ec + 5.0 * g_bar) / (nc + 5.0)
                            m_nc = (pred_cs[cal_i] == -1)
                            r_nocall = (float(errs[cal_i][m_nc].sum()) + 5.0 * g_bar) / (int(m_nc.sum()) + 5.0)
                            # Expected risk sum_c p(c|x) * r_eb[c] (or r_nocall if no detection call)
                            s_peb = p_vecs[ev_i] @ r_eb
                            is_nc_ev = (pred_cs[ev_i] == -1)
                            s_peb = np.where(is_nc_ev, r_nocall, s_peb)
                            ord_ev = np.argsort(-s_peb, kind='mergesort')

                        for b in budgets:
                            k = max(1, int(round(b * len(ev_i))))
                            split_caps[b].append(float(ev_err[ord_ev[:k]].sum() / ev_err.sum() * 100.0))

                    for b in budgets:
                        bk = f'{int(b*100)}%'
                        b_seed_means[bk].append(float(np.mean(split_caps[b])))
                        b_seed_ci_lo[bk].append(float(np.percentile(split_caps[b], 2.5)))
                        b_seed_ci_hi[bk].append(float(np.percentile(split_caps[b], 97.5)))

                triage_curves[strat_name] = {
                    bk: {
                        'capture_pct_3seed_mean': round(float(np.mean(b_seed_means[bk])), 2),
                        'capture_pct_seeds': [round(x, 2) for x in b_seed_means[bk]],
                        'avg_split_ci95': [round(float(np.mean(b_seed_ci_lo[bk])), 2), round(float(np.mean(b_seed_ci_hi[bk])), 2)],
                        'gain_over_random_x': round(float(np.mean(b_seed_means[bk])) / float(bk.replace('%', '')), 2),
                    }
                    for bk in [f'{int(b*100)}%' for b in budgets]
                }

        pcr_summary[grp_name] = {
            'unreviewed_symptom_vs_pcr_baseline': {
                mk: {k: {'mean': round(float(np.mean(v)), 2), 'seeds': v} for k, v in st_d.items()}
                for mk, st_d in sens_spec_stats.items()
            },
            'disjoint_calibration_triage_curves': triage_curves,
        }

    return {'visual_symptom_triage': vis_summary, 'pcr_597_suspect_triage': pcr_summary}


# ==============================================================================
# 跨域补充验证：测试 C2 (OOD 流形衰减池) 在 Zenodo 果园集与 CitrusUAT 室内单叶集上的行为
# ==============================================================================
@torch.no_grad()
def eval_cross_domain_zenodo_and_citrusuat(locked_per_seed: dict, cat3_heads: dict, device: str = 'cuda:0'):
    print("\n================ [模块 3] 跨域全景检验：Zenodo (519图果园) & CitrusUAT (500图近景单叶 OOD) ================", flush=True)
    # --- 3A. Zenodo 11148332 (519 images, 1765 GT boxes, zero-drop 2cls & 3cls) ---
    coco_images, img_paths, gt2_p, gt3_p = build_external_zenodo_gt()
    scidb_to_2cls = {0: 0, 1: 0, **{c: 1 for c in range(2, 14)}}
    scidb_to_3cls = {0: 0, 1: 0, 2: 1, 3: 1, **{c: 2 for c in range(4, 14)}}

    zen_methods = ['B0_Native_YOLOv8n', 'B1_Full_DINOv2_1Crop', 'C1_Fast1Crop_StaticCalib',
                   'C2_Fast1Crop_OOD_CertaintyPool', 'C3_ZeroMLP_DualCrop_CoreStd', 'C5a_Server_3Crop_PlainConcat']
    zen_records = {m: {'2cls_AP50': [], '2cls_AP50_95': [], '3cls_AP50': [], '3cls_AP50_95': []} for m in zen_methods}

    # --- 3B. CitrusUAT (500 images, 5 aligned classes — testing OOD detector attenuation!) ---
    uat_classes = ['Greasy_spot', 'Healthy', 'HLB', 'Mg', 'Zn']
    uat_groups = [[10], [0, 1], [2, 3], [7], [4]]
    uat_samples = []
    for ci, cname in enumerate(uat_classes):
        for p in sorted((CITRUSUAT_DIR / cname).glob('*.*')):
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                uat_samples.append((ci, p))

    def dec_uat(item):
        ci, p = item
        bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), 1)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if max(h, w) > 1280:
            f = 1280.0 / max(h, w)
            bgr = cv2.resize(bgr, (max(1, int(round(w * f))), max(1, int(round(h * f)))), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return ci, bgr, rgb

    with ThreadPoolExecutor(max_workers=6) as ex:
        uat_decoded = list(ex.map(dec_uat, uat_samples))
    y_uat = np.array([x[0] for x in uat_decoded], dtype=np.int64)
    uat_records = {m: {'MacroF1_5cls': [], 'Acc_5cls': []} for m in zen_methods}
    uat_maha_diag = []

    for seed in [0, 1, 2]:
        blob = torch.load(CKPT_TCD / f'cggar_v2_seed{seed}.pt', map_location='cpu', weights_only=False)
        m_cfg = blob['mahalanobis']
        T_DET, T_DINO, T_REL = blob['temps']['T_DET'], blob['temps']['T_DINO'], blob['temps']['T_REL']
        ckpt_bg = torch.load(CKPT_BG / f'bgopta_frozen_seed{seed}.pt', map_location='cpu', weights_only=False)
        beta = ckpt_bg['val_frozen_hyperparams']['beta_score']
        gm = locked_per_seed[seed]['gamma_ood_C2']
        wd3, wc3, ws3 = locked_per_seed[seed]['weights_C3_det_core_std']

        yolo_ext = YoloFullStateExtractor(Path(r'D:\citrus_runs') / f'yolov8n_unified_seed{seed}' / 'weights' / 'best.pt', device)
        dino_ext = DinoExtractor(seed, device)
        cat3_hd = cat3_heads[seed]

        # ---- Run Zenodo for this seed ----
        raw_preds = []
        for start in range(0, len(coco_images), 16):
            grp = coco_images[start:start + 16]
            bgrs = [cv2.imdecode(np.fromfile(str(img_paths[it['id']]), dtype=np.uint8), 1) for it in grp]
            res_b = yolo_ext.yolo.predict(bgrs, imgsz=640, conf=0.05, iou=0.7, max_det=300, device=0, half=False, rect=False, verbose=False)
            for it, r in zip(grp, res_b):
                for bb, sc, cl in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.conf.cpu().tolist(), r.boxes.cls.cpu().tolist()):
                    x1, y1, x2, y2 = bb
                    raw_preds.append({'image_id': it['id'], 'category_id': int(cl), 'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': sc})

        n_z = len(raw_preds)
        g_pos = defaultdict(list)
        for pos, r in enumerate(raw_preds):
            g_pos[r['image_id']].append(pos)
        p14_z = np.zeros((n_z, 14), dtype=np.float32)
        znk_z = np.zeros((n_z, 192), dtype=np.float32)
        geo_z = np.zeros((n_z, 9), dtype=np.float32)
        c082_z = np.zeros((n_z, 3, 224, 224), dtype=np.uint8)
        c125_z = np.zeros((n_z, 3, 224, 224), dtype=np.uint8)
        c250_z = np.zeros((n_z, 3, 224, 224), dtype=np.uint8)

        ord_iids = list(g_pos.keys())
        for start in range(0, len(ord_iids), 16):
            sub_iids = ord_iids[start:start + 16]
            bgrs, bxs_b, tcls_b, pos_b = [], [], [], []
            for iid in sub_iids:
                bgr = cv2.imdecode(np.fromfile(str(img_paths[iid]), dtype=np.uint8), 1)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                plist = g_pos[iid]
                bl = [raw_preds[p]['bbox'] for p in plist]
                cl = [raw_preds[p]['category_id'] for p in plist]
                for p, bb in zip(plist, bl):
                    geo_z[p] = compute_box_geom(bb, w, h, bl)
                    c082_z[p] = crop_square_scale(rgb, bb, 0.82)
                    c125_z[p] = crop_square_scale(rgb, bb, 1.25)
                    c250_z[p] = crop_square_scale(rgb, bb, 2.50)
                bgrs.append(bgr); bxs_b.append(bl); tcls_b.append(cl); pos_b.extend(plist)
            ps, zs = yolo_ext.extract_batch(bgrs, bxs_b, tcls_b)
            for k, p_idx in enumerate(pos_b):
                p14_z[p_idx] = ps[k]; znk_z[p_idx] = zs[k]

        zc_z, lgc_z = dino_ext.encode_crops(c082_z)
        zs_z, lgs_z = dino_ext.encode_crops(c125_z)
        zx_z, _ = dino_ext.encode_crops(c250_z)
        lg_cat3_z = cat3_hd(
            torch.from_numpy(zc_z.astype(np.float32)).to(device),
            torch.from_numpy(zs_z.astype(np.float32)).to(device),
            torch.from_numpy(zx_z.astype(np.float32)).to(device),
            torch.from_numpy(geo_z).to(device),
        ).cpu().numpy()

        yolo_sc_z = np.array([r['score'] for r in raw_preds], dtype=np.float64)
        fg_z = yolo_sc_z >= 0.08
        lp_det_z = F.log_softmax(torch.logit(torch.from_numpy(p14_z).clamp(1e-5, 1.0 - 1e-5)) / T_DET, dim=-1).numpy()
        lp_std_z = F.log_softmax(torch.from_numpy(lgs_z) / T_DINO, dim=-1).numpy()
        lp_core_z = F.log_softmax(torch.from_numpy(lgc_z) / T_DINO, dim=-1).numpy()
        lp_cat3_z = F.log_softmax(torch.from_numpy(lg_cat3_z) / T_REL, dim=-1).numpy()
        p_std_raw_z = F.softmax(torch.from_numpy(lgs_z), dim=-1).numpy()

        h_det_z = norm_entropy_np(np.exp(lp_det_z))
        h_std_z = norm_entropy_np(np.exp(lp_std_z))
        maha_z14 = compute_maha_z(znk_z, m_cfg['means'], m_cfg['prec'], m_cfg['m_mu'], m_cfg['m_sd'])
        dm_z = maha_z14[np.arange(n_z), lp_det_z.argmax(axis=1)]

        p_c1_z = softmax_from_logp(0.35 * lp_det_z + 0.65 * lp_std_z)
        u0_z = 0.35 * np.clip(1.0 - h_det_z, 0.05, 1.0) * np.exp(-gm * np.maximum(0.0, dm_z))
        u1_z = 0.65 * np.clip(1.0 - h_std_z, 0.05, 1.0)
        p_c2_z = softmax_from_logp((u0_z[:, None] * lp_det_z + u1_z[:, None] * lp_std_z) / (u0_z + u1_z + 1e-8)[:, None])
        p_c3_z = softmax_from_logp(wd3 * lp_det_z + wc3 * lp_core_z + ws3 * lp_std_z)
        p_c5_z = softmax_from_logp(0.24 * lp_det_z + 0.24 * lp_std_z + 0.20 * lp_core_z + 0.32 * lp_cat3_z)

        def eval_z_rows(rows_14):
            m2 = [{'image_id': r['image_id'], 'category_id': scidb_to_2cls[r['category_id']], 'bbox': r['bbox'], 'score': r['score']} for r in rows_14]
            m3 = [{'image_id': r['image_id'], 'category_id': scidb_to_3cls[r['category_id']], 'bbox': r['bbox'], 'score': r['score']} for r in rows_14]
            r2 = coco_eval_metrics(gt2_p, m2)
            r3 = coco_eval_metrics(gt3_p, m3)
            return r2['AP50'], r2['AP50_95'], r3['AP50'], r3['AP50_95']

        def build_z_cal(pm):
            rw = [dict(r) for r in raw_preds]
            for i in range(n_z):
                if fg_z[i]:
                    nc = int(np.argmax(pm[i]))
                    rw[i]['category_id'] = nc
                    rw[i]['score'] = float((yolo_sc_z[i] ** beta) * (max(float(pm[i, nc]), 1e-6) ** (1.0 - beta)))
            return apply_classwise_nms(rw, 0.70)

        rw_b1 = [dict(r) for r in raw_preds]
        for i in range(n_z):
            nc = int(np.argmax(p_std_raw_z[i]))
            if nc != rw_b1[i]['category_id']:
                rw_b1[i]['category_id'] = nc
                rw_b1[i]['score'] = float(math.sqrt(yolo_sc_z[i] * max(float(p_std_raw_z[i, nc]), 1e-6)))

        for m_tag, rw_m in [
            ('B0_Native_YOLOv8n', [dict(r) for r in raw_preds]),
            ('B1_Full_DINOv2_1Crop', rw_b1),
            ('C1_Fast1Crop_StaticCalib', build_z_cal(p_c1_z)),
            ('C2_Fast1Crop_OOD_CertaintyPool', build_z_cal(p_c2_z)),
            ('C3_ZeroMLP_DualCrop_CoreStd', build_z_cal(p_c3_z)),
            ('C5a_Server_3Crop_PlainConcat', build_z_cal(p_c5_z)),
        ]:
            a2_50, a2_95, a3_50, a3_95 = eval_z_rows(rw_m)
            zen_records[m_tag]['2cls_AP50'].append(round(a2_50, 2))
            zen_records[m_tag]['2cls_AP50_95'].append(round(a2_95, 2))
            zen_records[m_tag]['3cls_AP50'].append(round(a3_50, 2))
            zen_records[m_tag]['3cls_AP50_95'].append(round(a3_95, 2))

        # ---- Run CitrusUAT (500 indoor single-leaf OOD images) for this seed ----
        n_u = len(uat_decoded)
        p14_u = np.zeros((n_u, 14), dtype=np.float32)
        znk_u = np.zeros((n_u, 192), dtype=np.float32)
        geo_u = np.zeros((n_u, 9), dtype=np.float32)
        c082_u = np.zeros((n_u, 3, 224, 224), dtype=np.uint8)
        c125_u = np.zeros((n_u, 3, 224, 224), dtype=np.uint8)
        c250_u = np.zeros((n_u, 3, 224, 224), dtype=np.uint8)

        for start in range(0, n_u, 16):
            sub = uat_decoded[start:start + 16]
            bgrs = [x[1] for x in sub]
            res_b = yolo_ext.yolo.predict(bgrs, imgsz=640, conf=0.05, iou=0.7, max_det=20, device=0, half=False, rect=False, verbose=False)
            bxs_b, tcls_b = [], []
            for k, ((_, bgr_s, rgb_s), r) in enumerate(zip(sub, res_b)):
                idx = start + k
                h, w = rgb_s.shape[:2]
                if len(r.boxes) > 0:
                    x1, y1, x2, y2 = r.boxes.xyxy[0].cpu().tolist()
                    bb = [x1, y1, x2 - x1, y2 - y1]
                    tc = int(r.boxes.cls[0].item())
                else:
                    bb = [w * 0.08, h * 0.08, w * 0.84, h * 0.84]
                    tc = None
                bxs_b.append([bb])
                tcls_b.append([tc])
                geo_u[idx] = compute_box_geom(bb, w, h, [bb])
                c082_u[idx] = crop_square_scale(rgb_s, bb, 0.82)
                c125_u[idx] = crop_square_scale(rgb_s, bb, 1.25)
                c250_u[idx] = crop_square_scale(rgb_s, bb, 2.50)
            ps, zs = yolo_ext.extract_batch(bgrs, bxs_b, tcls_b)
            for k in range(len(sub)):
                p14_u[start + k] = ps[k]
                znk_u[start + k] = zs[k]

        zc_u, lgc_u = dino_ext.encode_crops(c082_u)
        zs_u, lgs_u = dino_ext.encode_crops(c125_u)
        zx_u, _ = dino_ext.encode_crops(c250_u)
        lg_cat3_u = cat3_hd(
            torch.from_numpy(zc_u.astype(np.float32)).to(device),
            torch.from_numpy(zs_u.astype(np.float32)).to(device),
            torch.from_numpy(zx_u.astype(np.float32)).to(device),
            torch.from_numpy(geo_u).to(device),
        ).cpu().numpy()

        lp_det_u = F.log_softmax(torch.logit(torch.from_numpy(p14_u).clamp(1e-5, 1.0 - 1e-5)) / T_DET, dim=-1).numpy()
        lp_std_u = F.log_softmax(torch.from_numpy(lgs_u) / T_DINO, dim=-1).numpy()
        lp_core_u = F.log_softmax(torch.from_numpy(lgc_u) / T_DINO, dim=-1).numpy()
        lp_cat3_u = F.log_softmax(torch.from_numpy(lg_cat3_u) / T_REL, dim=-1).numpy()

        h_det_u = norm_entropy_np(np.exp(lp_det_u))
        h_std_u = norm_entropy_np(np.exp(lp_std_u))
        maha_u14 = compute_maha_z(znk_u, m_cfg['means'], m_cfg['prec'], m_cfg['m_mu'], m_cfg['m_sd'])
        dm_u = maha_u14[np.arange(n_u), lp_det_u.argmax(axis=1)]
        u0_u = 0.35 * np.clip(1.0 - h_det_u, 0.05, 1.0) * np.exp(-gm * np.maximum(0.0, dm_u))
        u1_u = 0.65 * np.clip(1.0 - h_std_u, 0.05, 1.0)
        eff_yolo_w_u = float((u0_u / (u0_u + u1_u + 1e-8)).mean())
        uat_maha_diag.append({'seed': seed, 'mean_maha_z_on_uat': round(float(dm_u.mean()), 2), 'mean_effective_yolo_weight_C2': round(eff_yolo_w_u, 3)})

        p_u_map = {
            'B0_Native_YOLOv8n': p14_u / np.maximum(p14_u.sum(axis=1, keepdims=True), 1e-8),
            'B1_Full_DINOv2_1Crop': F.softmax(torch.from_numpy(lgs_u), dim=-1).numpy(),
            'C1_Fast1Crop_StaticCalib': softmax_from_logp(0.35 * lp_det_u + 0.65 * lp_std_u),
            'C2_Fast1Crop_OOD_CertaintyPool': softmax_from_logp((u0_u[:, None] * lp_det_u + u1_u[:, None] * lp_std_u) / (u0_u + u1_u + 1e-8)[:, None]),
            'C3_ZeroMLP_DualCrop_CoreStd': softmax_from_logp(wd3 * lp_det_u + wc3 * lp_core_u + ws3 * lp_std_u),
            'C5a_Server_3Crop_PlainConcat': softmax_from_logp(0.24 * lp_det_u + 0.24 * lp_std_u + 0.20 * lp_core_u + 0.32 * lp_cat3_u),
        }
        for m_tag, pm in p_u_map.items():
            # Aligned 5-class projection
            p5 = np.stack([pm[:, grp].sum(axis=1) for grp in uat_groups], axis=1)
            yp5 = p5.argmax(axis=1)
            uat_records[m_tag]['Acc_5cls'].append(round(float((yp5 == y_uat).mean() * 100.0), 2))
            uat_records[m_tag]['MacroF1_5cls'].append(round(float(f1_score(y_uat, yp5, average='macro') * 100.0), 2))

        yolo_ext.close()
        del yolo_ext, dino_ext
        torch.cuda.empty_cache()
        print(f"  [跨域检验 Seed {seed} 完成] Zenodo 3cls AP50: B0={zen_records['B0_Native_YOLOv8n']['3cls_AP50'][-1]:.2f}, "
              f"B1={zen_records['B1_Full_DINOv2_1Crop']['3cls_AP50'][-1]:.2f}, C2={zen_records['C2_Fast1Crop_OOD_CertaintyPool']['3cls_AP50'][-1]:.2f}, "
              f"C3={zen_records['C3_ZeroMLP_DualCrop_CoreStd']['3cls_AP50'][-1]:.2f} | "
              f"CitrusUAT 5cls F1: C1={uat_records['C1_Fast1Crop_StaticCalib']['MacroF1_5cls'][-1]:.2f} -> "
              f"C2(OOD衰减后)={uat_records['C2_Fast1Crop_OOD_CertaintyPool']['MacroF1_5cls'][-1]:.2f} (B1={uat_records['B1_Full_DINOv2_1Crop']['MacroF1_5cls'][-1]:.2f})",
              flush=True)

    def summarize_dict(rec_d):
        out = {}
        for m, md in rec_d.items():
            out[m] = {}
            for k, vals in md.items():
                out[m][k] = {
                    'mean': round(float(np.mean(vals)), 2),
                    'std': round(float(np.std(vals, ddof=1)), 2),
                    'seeds': vals,
                }
        return out

    return {
        'Zenodo_11148332_519imgs': summarize_dict(zen_records),
        'CitrusUAT_500imgs_IndoorSingleLeaf_OOD': summarize_dict(uat_records),
        'CitrusUAT_OOD_Attenuation_Diagnostics': uat_maha_diag,
    }


def main():
    t0 = time.time()
    env_meta = configure_strict_fp32()
    verify_all_6_frozen_checkpoints()
    device = 'cuda:0'

    gt_dicts = {sp: json.loads((GT_DIR / f'{sp}.json').read_text(encoding='utf-8')) for sp in ['val', 'testB', 'testC']}

    # Step 1: Load caches, train strictly source-only PlainConcat3VecHead (on SciDB train), and lock scalar params strictly on SciDB val!
    print("\n================ [模块 1] 第一层：5 个零样本检测候选方法库 (SciDB 3-Seed + B=1000 Bootstrap) ================", flush=True)
    cat3_heads = {}
    locked_per_seed = {}
    seed_packs = {sp: {} for sp in ['val', 'testB', 'testC']}
    cand_outputs = {sp: {} for sp in ['val', 'testB', 'testC']}

    for seed in [0, 1, 2]:
        cat3_hd = train_source_plain_concat_head(seed, device)
        cat3_heads[seed] = cat3_hd

        blob = torch.load(CKPT_TCD / f'cggar_v2_seed{seed}.pt', map_location='cpu', weights_only=False)
        diff_hd = TargetContextRelationalHead().to(device).eval()
        diff_hd.load_state_dict(blob['rel_head'])
        m_cfg = blob['mahalanobis']
        T_DET, T_DINO, T_REL = blob['temps']['T_DET'], blob['temps']['T_DINO'], blob['temps']['T_REL']

        ckpt_bg = torch.load(CKPT_BG / f'bgopta_frozen_seed{seed}.pt', map_location='cpu', weights_only=False)
        beta_lock = ckpt_bg['val_frozen_hyperparams']['beta_score']

        for sp in ['val', 'testB', 'testC']:
            d = np.load(CACHE_DIR / f'{sp}_cggar_seed{seed}.npz')
            orig_preds = json.loads((STRONG_DIR / f'种子{seed}' / f'{sp}预测.json').read_text(encoding='utf-8'))
            cand_idxs = d['cand_idxs']
            cand_rows = [orig_preds[i] for i in cand_idxs]
            n_c = len(cand_idxs)
            yolo_sc = np.array([r['score'] for r in cand_rows], dtype=np.float64)
            yolo_cls = np.array([r['category_id'] for r in cand_rows], dtype=np.int64)
            fg = yolo_sc >= 0.08

            with torch.no_grad():
                zc_g = torch.from_numpy(d['z_core'].astype(np.float32)).to(device)
                zs_g = torch.from_numpy(d['z_std'].astype(np.float32)).to(device)
                zx_g = torch.from_numpy(d['z_ctx'].astype(np.float32)).to(device)
                gm_g = torch.from_numpy(d['geom']).to(device)
                lg_cat3 = cat3_hd(zc_g, zs_g, zx_g, gm_g).cpu().numpy()
                lg_diff = diff_hd(zc_g, zs_g, zx_g, gm_g).cpu().numpy()

            lp_det = F.log_softmax(torch.logit(torch.from_numpy(d['p14']).clamp(1e-5, 1.0 - 1e-5)) / T_DET, dim=-1).numpy()
            lp_std = F.log_softmax(torch.from_numpy(d['lg_std']) / T_DINO, dim=-1).numpy()
            lp_core = F.log_softmax(torch.from_numpy(d['lg_core']) / T_DINO, dim=-1).numpy()
            lp_cat3_n = F.log_softmax(torch.from_numpy(lg_cat3) / T_REL, dim=-1).numpy()
            lp_diff_n = F.log_softmax(torch.from_numpy(lg_diff) / T_REL, dim=-1).numpy()

            p_det = np.exp(lp_det)
            p_std_unscaled = F.softmax(torch.from_numpy(d['lg_std']), dim=-1).numpy()
            maha_14 = compute_maha_z(d['zneck'], m_cfg['means'], m_cfg['prec'], m_cfg['m_mu'], m_cfg['m_sd'])
            d_maha_z = maha_14[np.arange(n_c), p_det.argmax(axis=1)]

            seed_packs[sp][seed] = {
                'orig_preds': orig_preds,
                'cand_idxs': cand_idxs,
                'cand_rows': cand_rows,
                'yolo_sc': yolo_sc,
                'yolo_cls': yolo_cls,
                'fg': fg,
                'beta_lock': beta_lock,
                'p14_raw': d['p14'] / np.maximum(d['p14'].sum(axis=1, keepdims=True), 1e-8),
                'p_det': p_det,
                'p_std_unscaled': p_std_unscaled,
                'lp_det': lp_det,
                'lp_std': lp_std,
                'lp_core': lp_core,
                'lp_cat3': lp_cat3_n,
                'lp_diff': lp_diff_n,
                'h_det': norm_entropy_np(p_det),
                'h_std': norm_entropy_np(np.exp(lp_std)),
                'd_maha_z': d_maha_z,
            }

        # Lock hyperparameters STRICTLY on SciDB val BEFORE touching testB/testC!
        locked_cfg = calibrate_source_val_params(seed_packs['val'][seed], GT_DIR / 'val.json')
        locked_per_seed[seed] = locked_cfg
        for sp in ['val', 'testB', 'testC']:
            cand_outputs[sp][seed] = compute_all_candidate_posteriors(seed_packs[sp][seed], locked_cfg)

    all_methods = list(cand_outputs['val'][0].keys())
    scidb_table = {m: {sp: {'AP50_95': [], 'AP50': [], 'Top1': [], 'Top2': [], 'Top3': [], 'AbioticOOD_AUC': [], 'AvgViTCrops': []}
                       for sp in ['val', 'testB', 'testC']} for m in all_methods}

    for sp in ['val', 'testB', 'testC']:
        gt_p = GT_DIR / f'{sp}.json'
        gt_d = gt_dicts[sp]
        for seed in [0, 1, 2]:
            pk = seed_packs[sp][seed]
            for m in all_methods:
                rows_m, p_m, avg_cr = cand_outputs[sp][seed][m]
                met = coco_eval_metrics(gt_p, rows_m)
                tk = compute_interactive_topk_and_abiotic_ood(gt_d, pk['cand_rows'], pk['fg'], p_m)
                scidb_table[m][sp]['AP50_95'].append(round(met['AP50_95'], 2))
                scidb_table[m][sp]['AP50'].append(round(met['AP50'], 2))
                scidb_table[m][sp]['Top1'].append(tk['Top1_Hit_pct'])
                scidb_table[m][sp]['Top2'].append(tk['Top2_Hit_pct'])
                scidb_table[m][sp]['Top3'].append(tk['Top3_Hit_pct'])
                scidb_table[m][sp]['AbioticOOD_AUC'].append(tk['Abiotic_vs_Biotic_ROC_AUC'])
                scidb_table[m][sp]['AvgViTCrops'].append(avg_cr)

    scidb_summary = {}
    for m in all_methods:
        scidb_summary[m] = {}
        for sp in ['val', 'testB', 'testC']:
            d_sp = scidb_table[m][sp]
            scidb_summary[m][sp] = {
                'AP50_95_mean': round(float(np.mean(d_sp['AP50_95'])), 2),
                'AP50_95_std': round(float(np.std(d_sp['AP50_95'], ddof=1)), 2),
                'AP50_95_seeds': d_sp['AP50_95'],
                'AP50_mean': round(float(np.mean(d_sp['AP50'])), 2),
                'AP50_seeds': d_sp['AP50'],
                'Top1_Hit_mean': round(float(np.mean(d_sp['Top1'])), 2),
                'Top2_Hit_mean': round(float(np.mean(d_sp['Top2'])), 2),
                'Top3_Hit_mean': round(float(np.mean(d_sp['Top3'])), 2),
                'AbioticOOD_ROC_AUC_mean': round(float(np.mean([x for x in d_sp['AbioticOOD_AUC'] if x is not None])), 4),
                'Avg_ViT_Crops_Per_Box': round(float(np.mean(d_sp['AvgViTCrops'])), 2),
            }
        print(f"  [{m:32s}] val={scidb_summary[m]['val']['AP50_95_mean']:.2f} | "
              f"testB={scidb_summary[m]['testB']['AP50_95_mean']:.2f}±{scidb_summary[m]['testB']['AP50_95_std']:.2f} {scidb_summary[m]['testB']['AP50_95_seeds']} | "
              f"testC={scidb_summary[m]['testC']['AP50_95_mean']:.2f}±{scidb_summary[m]['testC']['AP50_95_std']:.2f} {scidb_summary[m]['testC']['AP50_95_seeds']} | "
              f"testB Top1/2/3={scidb_summary[m]['testB']['Top1_Hit_mean']:.1f}/{scidb_summary[m]['testB']['Top2_Hit_mean']:.1f}/{scidb_summary[m]['testB']['Top3_Hit_mean']:.1f}%",
              flush=True)

    # Paired B=1000 Image-Level Bootstrap (3-Seed Synchronized Mean AP50-95 Difference vs B0 & B1)
    pairs_boot = [
        ('C1_Fast1Crop_StaticCalib', 'B1_Full_DINOv2_1Crop'),
        ('C2_Fast1Crop_OOD_CertaintyPool', 'B1_Full_DINOv2_1Crop'),
        ('C3_ZeroMLP_DualCrop_CoreStd', 'B1_Full_DINOv2_1Crop'),
        ('C4_HardBudget25_MultiScale', 'B1_Full_DINOv2_1Crop'),
        ('C5a_Server_3Crop_PlainConcat', 'B1_Full_DINOv2_1Crop'),
        ('C2_Fast1Crop_OOD_CertaintyPool', 'C1_Fast1Crop_StaticCalib'),
        ('C3_ZeroMLP_DualCrop_CoreStd', 'C1_Fast1Crop_StaticCalib'),
        ('C5a_Server_3Crop_PlainConcat', 'C3_ZeroMLP_DualCrop_CoreStd'),
    ]
    bootstrap_3seed_mean = {}
    for sp in ['testB', 'testC']:
        boot_by_seed = []
        for seed in [0, 1, 2]:
            pm_rows = {m: cand_outputs[sp][seed][m][0] for m in all_methods}
            b_arrs, _ = paired_image_bootstrap_coco(gt_dicts[sp], pm_rows, pairs_boot, n_boot=1000, rng_seed=20261126)
            boot_by_seed.append(b_arrs)
        bootstrap_3seed_mean[sp] = {}
        for m_a, m_b in pairs_boot:
            diffs_3s = np.mean([boot_by_seed[s][m_a][:, 0] - boot_by_seed[s][m_b][:, 0] for s in [0, 1, 2]], axis=0)
            ci_lo, ci_hi = float(np.percentile(diffs_3s, 2.5)), float(np.percentile(diffs_3s, 97.5))
            p_two = min(1.0, 2.0 * min(float(np.mean(diffs_3s <= 0.0)), float(np.mean(diffs_3s >= 0.0))))
            bootstrap_3seed_mean[sp][f'{m_a}__minus__{m_b}'] = {
                'mean_diff_AP50_95': round(float(diffs_3s.mean()), 3),
                'ci95': [round(ci_lo, 3), round(ci_hi, 3)],
                'p_value_two_sided': round(p_two, 4),
                'significant_95': bool(ci_lo > 0.0 or ci_hi < 0.0),
            }

    # Step 2: Conformal Risk Control (R2) + Visual Discordance Triage (R1) + 597-Sample PCR Soft Posterior-EB Triage (R3)
    print("\n================ [模块 2] 第二层：保形风险控制 (R2)、视觉分歧送审 (R1) 与 597例 PCR 期望风险分层 (R3) ================", flush=True)
    crc_methods = ['B0_Native_YOLOv8n', 'B1_Full_DINOv2_1Crop', 'C1_Fast1Crop_StaticCalib',
                   'C2_Fast1Crop_OOD_CertaintyPool', 'C3_ZeroMLP_DualCrop_CoreStd', 'C5a_Server_3Crop_PlainConcat']
    crc_summary = {sp: {} for sp in ['testB', 'testC']}
    for sp in ['testB', 'testC']:
        for m in crc_methods:
            # Aggregate across 3 seeds
            per_s_crc = [run_conformal_risk_control_simulation(gt_dicts[sp], cand_outputs[sp][s][m][0],
                                                               n_cal_list=(40, 80, 120), alpha_targets=(0.10, 0.15, 0.20),
                                                               n_repeats=200, seed=20261126 + s) for s in [0, 1, 2]]
            crc_summary[sp][m] = {}
            for nc_k in per_s_crc[0].keys():
                crc_summary[sp][m][nc_k] = {}
                for al_k in per_s_crc[0][nc_k].keys():
                    r_means = [per_s_crc[s][nc_k][al_k]['empirical_heldout_risk_mean'] for s in [0, 1, 2]]
                    c_means = [per_s_crc[s][nc_k][al_k]['auto_coverage_pct_mean'] for s in [0, 1, 2]]
                    sa_means = [per_s_crc[s][nc_k][al_k]['selective_accuracy_on_accepted_pct'] for s in [0, 1, 2]]
                    nv_means = [per_s_crc[s][nc_k][al_k]['naive_uncorrected_risk_mean'] for s in [0, 1, 2]]
                    al_num = per_s_crc[0][nc_k][al_k]['target_risk_ceiling']
                    crc_summary[sp][m][nc_k][al_k] = {
                        'target_risk_ceiling': al_num,
                        'empirical_risk_3seed_mean': round(float(np.mean(r_means)), 4),
                        'bound_strictly_satisfied': bool(float(np.mean(r_means)) <= al_num + 1e-4),
                        'auto_coverage_pct_3seed_mean': round(float(np.mean(c_means)), 2),
                        'auto_coverage_seeds': [round(x, 2) for x in c_means],
                        'selective_acc_pct_3seed_mean': round(float(np.mean(sa_means)), 2),
                        'naive_uncorrected_risk_3seed_mean': round(float(np.mean(nv_means)), 4),
                    }
        print(f"  [CRC {sp} | n_cal=80, α=0.15 自动放行率 & 实测风险]: " + " | ".join(
            f"{m.split('_')[0]}: cov={crc_summary[sp][m]['n_cal_80']['alpha_0.15']['auto_coverage_pct_3seed_mean']:.1f}% "
            f"(risk={crc_summary[sp][m]['n_cal_80']['alpha_0.15']['empirical_risk_3seed_mean']:.3f})"
            for m in crc_methods
        ), flush=True)

    triage_res = run_visual_and_pcr_triage_suite(
        gt_dicts['testB'], gt_dicts['testC'],
        seed_packs['testB'], seed_packs['testC'],
        cand_outputs['testB'], cand_outputs['testC'],
    )
    g1_c = triage_res['pcr_597_suspect_triage']['G1_Strict4Symptoms']['disjoint_calibration_triage_curves']
    print(f"  [PCR 597 G1 | n_cal=100, 5%/10%/20% 预算捕获率]: "
          f"MinConf={triage_res['pcr_597_suspect_triage']['G1_Strict4Symptoms']['disjoint_calibration_triage_curves']['ZeroCalib_MinConfidence_NegativeCtrl']['5%']['capture_pct_mean']:.1f}%/"
          f"{triage_res['pcr_597_suspect_triage']['G1_Strict4Symptoms']['disjoint_calibration_triage_curves']['ZeroCalib_MinConfidence_NegativeCtrl']['10%']['capture_pct_mean']:.1f}% | "
          f"DS硬先验(B0)={g1_c['HardClassPrior_B0_YOLOv8n_n100']['5%']['capture_pct_3seed_mean']:.1f}%/{g1_c['HardClassPrior_B0_YOLOv8n_n100']['10%']['capture_pct_3seed_mean']:.1f}%/{g1_c['HardClassPrior_B0_YOLOv8n_n100']['20%']['capture_pct_3seed_mean']:.1f}% | "
          f"R3软后验EB(C2)={g1_c['R3_SoftPosteriorEB_C2_1Crop_n100']['5%']['capture_pct_3seed_mean']:.1f}%/{g1_c['R3_SoftPosteriorEB_C2_1Crop_n100']['10%']['capture_pct_3seed_mean']:.1f}%/{g1_c['R3_SoftPosteriorEB_C2_1Crop_n100']['20%']['capture_pct_3seed_mean']:.1f}% | "
          f"R3软后验EB(C5a)={g1_c['R3_SoftPosteriorEB_C5a_3Crop_n100']['5%']['capture_pct_3seed_mean']:.1f}%/{g1_c['R3_SoftPosteriorEB_C5a_3Crop_n100']['10%']['capture_pct_3seed_mean']:.1f}%/{g1_c['R3_SoftPosteriorEB_C5a_3Crop_n100']['20%']['capture_pct_3seed_mean']:.1f}%",
          flush=True)

    # Step 3: Cross-domain Zenodo + CitrusUAT OOD verification
    cross_res = eval_cross_domain_zenodo_and_citrusuat(locked_per_seed, cat3_heads, device)

    final_payload = {
        'protocol': 'Antigravity_v4_PreRegistered_Orchard_Candidate_Arsenal_and_Conformal_Suite',
        'elapsed_seconds': round(time.time() - t0, 2),
        'strict_fp32_env': env_meta,
        'source_val_locked_hyperparams': locked_per_seed,
        'layer1_zero_shot_detection_candidates_scidb': scidb_summary,
        'layer1_paired_bootstrap_1000_3seed_mean': bootstrap_3seed_mean,
        'layer1_cross_domain_zenodo_and_citrusuat': cross_res,
        'layer2_R2_conformal_risk_control': crc_summary,
        'layer2_R1_visual_and_R3_pcr_triage': triage_res,
    }
    out_json = RES_DIR / 'orchard_candidate_arsenal_complete.json'
    out_json.write_text(json.dumps(final_payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n[全部完成] 耗时 {final_payload['elapsed_seconds']}s，完整结果已保存至: {out_json}", flush=True)


if __name__ == '__main__':
    main()
