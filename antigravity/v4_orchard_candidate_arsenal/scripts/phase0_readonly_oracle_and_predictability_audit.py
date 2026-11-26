# -*- coding: utf-8 -*-
"""
只读可行性审计脚本 (Phase 1 Read-Only Feasibility Audit: Oracle Headroom, Disagreement Predictability & Class-Level Shrinkage)
严格遵守全部 7 项铁律：
1. 零权重训练（不训练任何骨干或深度网络权重）；
2. 零仓库修改（本脚本与结果独立存放于 D:\新版柑橘\Antigravity_只读可行性审计_20261126，不触碰 git 仓库内任何文件）；
3. 零测试集调参（以源域 SciDB val 5 折图像分组交叉拟合作为主决策门禁；testB/testC 仅作事后探索性诊断对照）；
4. 逐种子输出 Seed 0 / 1 / 2 样本计数、分歧四象限、14 类冲突分布、分解式 Oracle、AUC/PR-AUC 95% CI 与交叉验证实际 ΔAP；
5. 严格分离：(a) 纯类别选择变化、(b) 纯候选框置信度排序变化、(c) 重判后同类别 NMS 影响、以及 (d) 门控参数空间 α_i∈[0,1] 真实上限；
6. 纳入全局固定权重、温度校准、无正则 14 类独立权重、向全局收缩 (L2 Empirical-Bayes Shrinkage) 14 类权重对比；
7. 量化核查 train_cggar_seed*.npz 中样本内记忆效应与 GT 匹配候选框偏差。
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '2'

import sys
import io
import json
import math
import time
import contextlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, average_precision_score
from torchvision.ops import batched_nms
from faster_coco_eval import COCO, COCOeval_faster

ROOT = Path(r'D:\新版柑橘')
EXT = ROOT / '方法路线扩展_20261031'
CACHE_DIR = ROOT / 'Antigravity_CGGAR_真创新攻关_20261125' / 'cache'
CKPT_TCD_DIR = ROOT / 'Antigravity_CGGAR_真创新攻关_20261125' / 'models'
GT_DIR = EXT / '检测数据'
STRONG_DIR = EXT / 'YOLOv8强基线重复_20261109'

OUT_DIR = ROOT / 'Antigravity_只读可行性审计_20261126'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def coco_ap(gt_path: Path, pred_rows: list):
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


def greedy_match_candidates_to_gt(gt_dict: dict, cand_rows: list, iou_thr: float = 0.50):
    """
    将候选框按置信度降序一对一贪心匹配至同图真值框 (IoU >= iou_thr)：
    返回：
    - gt_cls: (N,) 匹配到的真值类别 (未匹配假阳性框为 -1)
    - gt_iou: (N,) 与匹配真值框的 IoU (未匹配为 0.0)
    - any_gt_cls: (N,) 若不要求一对一排他、与任意真值框最大 IoU >= iou_thr 时的真值类别 (用于区分同叶重复框 vs 纯背景框)
    """
    anns_by_img = defaultdict(list)
    for a in gt_dict['annotations']:
        anns_by_img[a['image_id']].append(a)

    n = len(cand_rows)
    gt_cls = np.full(n, -1, dtype=np.int64)
    gt_iou = np.zeros(n, dtype=np.float32)
    any_gt_cls = np.full(n, -1, dtype=np.int64)

    by_img = defaultdict(list)
    for idx, r in enumerate(cand_rows):
        by_img[r['image_id']].append((r['score'], idx, r['bbox']))

    for iid, rlist in by_img.items():
        rlist.sort(key=lambda x: -x[0])
        glist = anns_by_img.get(iid, [])
        used = set()
        for _sc, idx, (bx, by, bw, bh) in rlist:
            best_any_iou, best_any_c = 0.0, -1
            best_ex_iou, best_ex_j = 0.0, -1
            for j, a in enumerate(glist):
                gx, gy, gw, gh = a['bbox']
                ix1, iy1 = max(bx, gx), max(by, gy)
                ix2, iy2 = min(bx + bw, gx + gw), min(by + bh, gy + gh)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                iou = inter / (bw * bh + gw * gh - inter + 1e-9)
                if iou > best_any_iou:
                    best_any_iou, best_any_c = iou, int(a['category_id'])
                if j not in used and iou > best_ex_iou:
                    best_ex_iou, best_ex_j = iou, j
            if best_any_iou >= iou_thr:
                any_gt_cls[idx] = best_any_c
            if best_ex_iou >= iou_thr and best_ex_j >= 0:
                used.add(best_ex_j)
                gt_cls[idx] = int(glist[best_ex_j]['category_id'])
                gt_iou[idx] = float(best_ex_iou)
    return gt_cls, gt_iou, any_gt_cls


def compute_maha_z(feats: np.ndarray, means: np.ndarray, prec: np.ndarray, m_mu: float, m_sd: float):
    N = len(feats)
    out = np.zeros((N, 14), dtype=np.float32)
    X = feats.astype(np.float64)
    for c in range(14):
        df = X - means[c]
        dc = np.sqrt(np.maximum(0.0, np.sum((df @ prec) * df, axis=1)))
        out[:, c] = ((dc - m_mu) / m_sd).astype(np.float32)
    return out


def build_single_pass_reliability_features(yolo_sc: np.ndarray, p_det: np.ndarray, p_std: np.ndarray, d_maha_z: np.ndarray, geom: np.ndarray) -> np.ndarray:
    """
    构造零额外前向开销的 14 维逐框冲突/可靠性预测特征：
    [yolo_sc, h_det, mar_det, max_p_det, h_std, mar_std, max_p_std, js_div, cos_sim, l1_dist, disagree_ind, d_maha_z, log_area, border_touch]
    """
    eps = 1e-7
    pd = np.clip(p_det, eps, 1.0)
    ps = np.clip(p_std, eps, 1.0)
    h_det = -(pd * np.log(pd)).sum(axis=1) / math.log(14.0)
    h_std = -(ps * np.log(ps)).sum(axis=1) / math.log(14.0)
    s_det = np.sort(pd, axis=1)
    s_std = np.sort(ps, axis=1)
    mar_det = s_det[:, -1] - s_det[:, -2]
    mar_std = s_std[:, -1] - s_std[:, -2]
    m = 0.5 * (pd + ps)
    js = 0.5 * ((pd * (np.log(pd) - np.log(m))).sum(axis=1) + (ps * (np.log(ps) - np.log(m))).sum(axis=1)) / math.log(2.0)
    cos = (pd * ps).sum(axis=1) / (np.linalg.norm(pd, axis=1) * np.linalg.norm(ps, axis=1) + eps)
    l1 = np.abs(pd - ps).sum(axis=1)
    dis = (pd.argmax(axis=1) != ps.argmax(axis=1)).astype(np.float64)
    # geom columns from step1: [bw/w, bh/h, area_ratio, log_ar, cx/w, cy/h, touch_border, n_boxes_norm, rel_size]
    area_r = geom[:, 2].astype(np.float64)
    touch_b = geom[:, 6].astype(np.float64)
    return np.column_stack([
        yolo_sc, h_det, mar_det, s_det[:, -1],
        h_std, mar_std, s_std[:, -1],
        js, cos, l1, dis, d_maha_z.astype(np.float64),
        np.log(np.clip(area_r, 1e-5, 1.0)), touch_b
    ])


def stratified_auc_bootstrap(y_bin: np.ndarray, scores: np.ndarray, n_boot: int = 1000, seed: int = 42):
    """计算二分类 ROC-AUC 与 PR-AUC 及其分层 Bootstrap 95% 置信区间（小样本安全）。"""
    pos = np.where(y_bin == 1)[0]
    neg = np.where(y_bin == 0)[0]
    if len(pos) < 2 or len(neg) < 2:
        return {'n_total': int(len(y_bin)), 'n_pos': int(len(pos)), 'n_neg': int(len(neg)), 'roc_auc': None, 'pr_auc': None}
    roc_pt = float(roc_auc_score(y_bin, scores))
    pr_pt = float(average_precision_score(y_bin, scores))
    rng = np.random.RandomState(seed)
    rocs, prs = [], []
    for _ in range(n_boot):
        bp = rng.choice(pos, size=len(pos), replace=True)
        bn = rng.choice(neg, size=len(neg), replace=True)
        idx = np.concatenate([bp, bn])
        rocs.append(roc_auc_score(y_bin[idx], scores[idx]))
        prs.append(average_precision_score(y_bin[idx], scores[idx]))
    return {
        'n_total': int(len(y_bin)),
        'n_pos': int(len(pos)),
        'n_neg': int(len(neg)),
        'pos_rate_pct': round(len(pos) / len(y_bin) * 100.0, 2),
        'roc_auc': round(roc_pt, 4),
        'roc_ci95': [round(float(np.percentile(rocs, 2.5)), 4), round(float(np.percentile(rocs, 97.5)), 4)],
        'pr_auc': round(pr_pt, 4),
        'pr_ci95': [round(float(np.percentile(prs, 2.5)), 4), round(float(np.percentile(prs, 97.5)), 4)],
    }


def fit_classwise_weights_nll(lp_det: np.ndarray, lp_std: np.ndarray, y_true: np.ndarray, alpha0: float = 0.35, l2_shrink: float = 0.0):
    """
    在匹配前景框上拟合 14 类融合权重 w_c ∈ [0, 1] (按 DINO 预测主类 c_std 或双专家联合类索引)：
    这里对全部 14 个类别 logits 分别赋予类级混合权重向量 w ∈ [0,1]^{14}:
      logits_fused[:, c] = w[c] * lp_det[:, c] + (1 - w[c]) * lp_std[:, c]
    当 l2_shrink == 0 时为无正则化 14 类独立权重；
    当 l2_shrink > 0 时为向全局权重 alpha0 收缩的正则化经验贝叶斯权重：
      L(w) = Mean_CE(softmax(logits_fused), y_true) + l2_shrink * sum_c (w[c] - alpha0)^2
    """
    N = len(y_true)
    def obj(w):
        w_clip = np.clip(w, 0.0, 1.0)
        lg = lp_det * w_clip[None, :] + lp_std * (1.0 - w_clip[None, :])
        lg = lg - lg.max(axis=1, keepdims=True)
        log_p = lg - np.log(np.exp(lg).sum(axis=1, keepdims=True))
        nll = -np.mean(log_p[np.arange(N), y_true])
        reg = l2_shrink * float(np.sum((w_clip - alpha0) ** 2))
        return nll + reg

    res = minimize(obj, x0=np.full(14, alpha0, dtype=np.float64), bounds=[(0.0, 1.0)] * 14, method='L-BFGS-B')
    return np.clip(res.x, 0.0, 1.0)


def fit_temperature_and_global_alpha(logits_det: np.ndarray, logits_std: np.ndarray, y_true: np.ndarray):
    """源域 NLL 联合校准 (T_det, T_dino, alpha_global)。"""
    N = len(y_true)
    def obj(params):
        td, ts, al = params
        lp_d = logits_det / td
        lp_d = lp_d - np.log(np.exp(lp_d - lp_d.max(1, keepdims=True)).sum(1, keepdims=True)) - lp_d.max(1, keepdims=True)
        lp_s = logits_std / ts
        lp_s = lp_s - np.log(np.exp(lp_s - lp_s.max(1, keepdims=True)).sum(1, keepdims=True)) - lp_s.max(1, keepdims=True)
        lg = al * lp_d + (1.0 - al) * lp_s
        lg = lg - lg.max(1, keepdims=True)
        log_p = lg - np.log(np.exp(lg).sum(1, keepdims=True))
        return -float(np.mean(log_p[np.arange(N), y_true]))

    res = minimize(obj, x0=[1.35, 1.15, 0.35], bounds=[(0.5, 5.0), (0.5, 5.0), (0.0, 1.0)], method='L-BFGS-B')
    return float(res.x[0]), float(res.x[1]), float(res.x[2])


def audit_train_cache_selection_bias():
    """定量核验 GPT 第 II.4 点：检查 train_cggar_seed*.npz 中样本内记忆效应与 GT 匹配框偏差。"""
    out = []
    for seed in [0, 1, 2]:
        tr = np.load(CACHE_DIR / f'train_cggar_seed{seed}.npz')
        y = tr['labels']
        p_det = tr['p14_clean']
        lg_std = tr['lg_std']
        c_det = p_det.argmax(axis=1)
        c_std = lg_std.argmax(axis=1)
        dis = (c_det != c_std)
        out.append({
            'seed': seed,
            'n_gt_matched_train_boxes': int(len(y)),
            'yolo_in_sample_acc_pct': round(float((c_det == y).mean() * 100.0), 3),
            'dino_in_sample_acc_pct': round(float((c_std == y).mean() * 100.0), 3),
            'disagreement_count': int(dis.sum()),
            'disagreement_pct': round(float(dis.mean() * 100.0), 3),
            'caseC_yolo_right_dino_wrong': int(((c_det == y) & (c_std != y)).sum()),
            'caseB_dino_right_yolo_wrong': int(((c_std == y) & (c_det != y)).sum()),
        })
    return out


def evaluate_split_oracles_and_counts(seed: int, sp: str, T_DET: float, T_DINO: float, beta: float, m_cfg: dict):
    gt_path = GT_DIR / f'{sp}.json'
    gt_dict = json.loads(gt_path.read_text(encoding='utf-8'))
    cat_names = {c['id']: c['name'] for c in gt_dict['categories']}
    d = np.load(CACHE_DIR / f'{sp}_cggar_seed{seed}.npz')
    orig_preds = json.loads((STRONG_DIR / f'种子{seed}' / f'{sp}预测.json').read_text(encoding='utf-8'))
    cand_idxs = d['cand_idxs']
    cand_rows = [orig_preds[i] for i in cand_idxs]
    n_cand = len(cand_rows)

    yolo_sc = np.array([r['score'] for r in cand_rows], dtype=np.float64)
    yolo_cls = np.array([r['category_id'] for r in cand_rows], dtype=np.int64)
    img_ids_cand = np.array([r['image_id'] for r in cand_rows], dtype=np.int64)
    fg = yolo_sc >= 0.08

    logits_det_raw = torch.logit(torch.from_numpy(d['p14']).clamp(1e-5, 1.0 - 1e-5)).numpy()
    logits_std_raw = d['lg_std'].astype(np.float64)

    lp_det = F.log_softmax(torch.from_numpy(logits_det_raw) / T_DET, dim=-1).numpy()
    lp_std = F.log_softmax(torch.from_numpy(logits_std_raw) / T_DINO, dim=-1).numpy()
    p_det = np.exp(lp_det)
    p_std = np.exp(lp_std)
    p_std_unscaled = F.softmax(torch.from_numpy(logits_std_raw), dim=-1).numpy()

    c_det = p_det.argmax(axis=1)
    c_std = p_std.argmax(axis=1)
    gt_cls, gt_iou, any_gt_cls = greedy_match_candidates_to_gt(gt_dict, cand_rows, iou_thr=0.50)

    # --- 1. Census & Four-Quadrant Counts (on fg >= 0.08 AND on 1-to-1 GT-matched boxes) ---
    fg_dis = fg & (c_det != c_std)
    m_mask = fg & (gt_cls >= 0)  # 1-to-1 GT-matched foreground boxes
    fp_dup_mask = fg & (gt_cls < 0) & (any_gt_cls >= 0)  # duplicate boxes on a GT leaf
    fp_bg_mask = fg & (gt_cls < 0) & (any_gt_cls < 0)    # pure background false positives

    m_A = m_mask & (c_det == gt_cls) & (c_std == gt_cls)
    m_B = m_mask & (c_std == gt_cls) & (c_det != gt_cls)  # DINO right, YOLO wrong
    m_C = m_mask & (c_det == gt_cls) & (c_std != gt_cls)  # YOLO right, DINO wrong
    m_D = m_mask & (c_det != gt_cls) & (c_std != gt_cls)  # Both wrong
    m_D_same = m_D & (c_det == c_std)
    m_D_diff = m_D & (c_det != c_std)

    # Check how many Case D boxes have true label in union of Top-2 of YOLO or DINO
    top2_det = np.argsort(p_det, axis=1)[:, -2:]
    top2_std = np.argsort(p_std, axis=1)[:, -2:]
    in_top2_union = np.array([gt_cls[i] in top2_det[i] or gt_cls[i] in top2_std[i] for i in range(n_cand)], dtype=bool)
    m_D_top2_rescuable = m_D & in_top2_union

    per_cls_conflict = {}
    for c in range(14):
        mc = m_mask & (gt_cls == c)
        if mc.sum() > 0:
            per_cls_conflict[cat_names.get(c, str(c))] = {
                'n_gt_matched': int(mc.sum()),
                'caseB_dino_right_yolo_wrong': int((m_B & (gt_cls == c)).sum()),
                'caseC_yolo_right_dino_wrong': int((m_C & (gt_cls == c)).sum()),
                'caseD_both_wrong': int((m_D & (gt_cls == c)).sum()),
            }

    census = {
        'n_gt_boxes_total': len(gt_dict['annotations']),
        'n_cand_conf05': int(n_cand),
        'n_cand_fg08': int(fg.sum()),
        'n_fg08_disagree_all': int(fg_dis.sum()),
        'fg08_disagree_pct': round(float(fg_dis.sum() / max(1, fg.sum()) * 100.0), 2),
        'n_matched_tp_candidates': int(m_mask.sum()),
        'n_duplicate_box_candidates': int(fp_dup_mask.sum()),
        'n_pure_bg_fp_candidates': int(fp_bg_mask.sum()),
        'matched_disagree_count': int((m_mask & (c_det != c_std)).sum()),
        'matched_disagree_pct': round(float((m_mask & (c_det != c_std)).sum() / max(1, m_mask.sum()) * 100.0), 2),
        'quadrant_A_both_right': int(m_A.sum()),
        'quadrant_B_dino_right_yolo_wrong': int(m_B.sum()),
        'quadrant_C_yolo_right_dino_wrong': int(m_C.sum()),
        'quadrant_D_both_wrong': int(m_D.sum()),
        'quadrant_D_same_wrong_cls': int(m_D_same.sum()),
        'quadrant_D_diff_wrong_cls': int(m_D_diff.sum()),
        'quadrant_D_in_top2_union': int(m_D_top2_rescuable.sum()),
        'per_class_conflict_distribution': per_cls_conflict,
    }

    # --- 2. Stepwise Baseline & Decomposed Oracles (Separating Class, Score Ranking, NMS, and Gate Space α∈[0,1]) ---
    lp_abl1 = 0.35 * lp_det + 0.65 * lp_std
    p_abl1 = np.exp(lp_abl1 - lp_abl1.max(axis=1, keepdims=True)); p_abl1 /= p_abl1.sum(axis=1, keepdims=True)
    c_abl1 = p_abl1.argmax(axis=1)

    def eval_custom(cls_vec, sc_vec, do_nms: bool):
        rows = [dict(r) for r in orig_preds]
        for li, gi in enumerate(cand_idxs):
            if fg[li]:
                rows[gi]['category_id'] = int(cls_vec[li])
                rows[gi]['score'] = float(sc_vec[li])
        if do_nms:
            rows = apply_classwise_nms(rows, 0.70)
        return round(coco_ap(gt_path, rows)[0], 3)

    # Baseline 0: Native YOLOv8n
    ap_native = round(coco_ap(gt_path, orig_preds)[0], 3)

    # Baseline 1: Square Full DINOv2 Hard Overwrite (legacy score, no NMS)
    sc_b1 = yolo_sc.copy()
    cls_b1 = yolo_cls.copy()
    for i in range(n_cand):
        nc = int(p_std_unscaled[i].argmax())
        if nc != cls_b1[i]:
            cls_b1[i] = nc
            sc_b1[i] = math.sqrt(yolo_sc[i] * max(float(p_std_unscaled[i, nc]), 1e-6))
    rows_b1 = [dict(r) for r in orig_preds]
    for li, gi in enumerate(cand_idxs):
        rows_b1[gi]['category_id'] = int(cls_b1[li])
        rows_b1[gi]['score'] = float(sc_b1[li])
    ap_b1_hard = round(coco_ap(gt_path, rows_b1)[0], 3)

    # Global Fixed Decomposed Chain (0.35 / 0.65):
    # (i) Class change ONLY (keep original yolo_sc untouched, NO post-NMS)
    ap_fixed_cls_only_no_nms = eval_custom(c_abl1, yolo_sc, do_nms=False)
    # (ii) Class change + Unified Score Reweight (NO post-NMS)
    sc_abl1 = (yolo_sc ** beta) * (np.clip(p_abl1[np.arange(n_cand), c_abl1], 1e-6, 1.0) ** (1.0 - beta))
    ap_fixed_cls_plus_score_no_nms = eval_custom(c_abl1, sc_abl1, do_nms=False)
    # (iii) Full Abl1: Class change + Unified Score Reweight + Class-wise NMS
    ap_abl1_full = eval_custom(c_abl1, sc_abl1, do_nms=True)

    # --- Decomposed Oracle A: Pure Class-Selection Oracle (Pick right expert between YOLO & DINO when one is right) ---
    # Rule: check any_gt_cls (or 1-to-1 gt_cls): strictly use 1-to-1 gt_cls on matched boxes so duplicate boxes are NOT artificially helped!
    c_oracle_sel = c_abl1.copy()
    p_oracle_sel = p_abl1[np.arange(n_cand), c_abl1].copy()
    for i in range(n_cand):
        if m_B[i]:
            c_oracle_sel[i] = c_std[i]
            p_oracle_sel[i] = p_std[i, c_std[i]]
        elif m_C[i]:
            c_oracle_sel[i] = c_det[i]
            p_oracle_sel[i] = p_det[i, c_det[i]]
    sc_oracle_sel = (yolo_sc ** beta) * (np.clip(p_oracle_sel, 1e-6, 1.0) ** (1.0 - beta))

    # A1: Pure Class-Selection Oracle keeping original yolo_sc & NO NMS (isolates pure label correction vs ap_fixed_cls_only_no_nms!)
    ap_orc_cls_keep_yolosc_no_nms = eval_custom(c_oracle_sel, yolo_sc, do_nms=False)
    # A2: Class-Selection Oracle keeping EXACT Abl1 scalar scores sc_abl1 & NO NMS (zero score change!)
    ap_orc_cls_keep_abl1sc_no_nms = eval_custom(c_oracle_sel, sc_abl1, do_nms=False)
    # A3: Class-Selection Oracle + Abl1 score formula + NMS (direct headroom over ap_abl1_full)
    ap_orc_cls_with_nms = eval_custom(c_oracle_sel, sc_oracle_sel, do_nms=True)

    # --- Decomposed Oracle B: Pure Score-Ranking Oracle (ZERO class change! Keep c_abl1 100% fixed!) ---
    # B1 (Achievable by α_i ∈ [0,1] Score-Only Gate): keep c_abl1 fixed, pick α_i ∈ [0,1] per box to maximize p_α(c_abl1) on true positives & minimize p_α(c_abl1) on false positives!
    alphas_grid = np.linspace(0.0, 1.0, 21)  # (21,)
    # shape (21, N, 14)
    lp_grid = alphas_grid[:, None, None] * lp_det[None, :, :] + (1.0 - alphas_grid[:, None, None]) * lp_std[None, :, :]
    lp_grid = lp_grid - lp_grid.max(axis=2, keepdims=True)
    p_grid = np.exp(lp_grid); p_grid /= p_grid.sum(axis=2, keepdims=True)  # (21, N, 14)

    p_c_abl1_grid = p_grid[:, np.arange(n_cand), c_abl1]  # (21, N)
    is_tp_abl1 = m_mask & (c_abl1 == gt_cls)
    p_score_gate_opt = np.where(is_tp_abl1, p_c_abl1_grid.max(axis=0), p_c_abl1_grid.min(axis=0))
    sc_score_gate_opt = (yolo_sc ** beta) * (np.clip(p_score_gate_opt, 1e-6, 1.0) ** (1.0 - beta))
    ap_orc_score_gate_no_nms = eval_custom(c_abl1, sc_score_gate_opt, do_nms=False)
    ap_orc_score_gate_with_nms = eval_custom(c_abl1, sc_score_gate_opt, do_nms=True)

    # B2 (Unconstrained Ideal Score-Ranking Oracle under Fixed Abl1 Classes & Post-NMS Boxes):
    # Take the EXACT post-NMS output of Abl1 (zero box change, zero class change!), rank all TPs above all FPs!
    rows_abl1_post = [dict(r) for r in orig_preds]
    for li, gi in enumerate(cand_idxs):
        if fg[li]:
            rows_abl1_post[gi]['category_id'] = int(c_abl1[li])
            rows_abl1_post[gi]['score'] = float(sc_abl1[li])
    rows_abl1_post = apply_classwise_nms(rows_abl1_post, 0.70)
    post_gt_cls, post_gt_iou, _ = greedy_match_candidates_to_gt(gt_dict, rows_abl1_post, iou_thr=0.50)
    rows_ideal_rank = []
    for idx_r, r in enumerate(rows_abl1_post):
        nr = dict(r)
        if post_gt_cls[idx_r] == r['category_id']:
            nr['score'] = float(0.90 + 0.099 * post_gt_iou[idx_r])
        else:
            nr['score'] = float(0.01 * r['score'])
        rows_ideal_rank.append(nr)
    ap_orc_unconstrained_ideal_rank = round(coco_ap(gt_path, rows_ideal_rank)[0], 3)

    # --- Decomposed Oracle C: TRUE Per-Box Gate Oracle α_i* ∈ [0, 1] (Joint Class + Score achievable by ANY gate α_i ∈ [0,1]!) ---
    # For each box i: if matched (gt_cls[i] >= 0), pick α_k ∈ [0,1] that first achieves argmax == gt_cls[i] (if any α_k can!) and maximizes p(gt_cls[i]);
    # if unmatched FP (gt_cls[i] < 0), pick α_k ∈ [0,1] that minimizes max_c p_α(c) so FP score is suppressed!
    best_k = np.full(n_cand, 7, dtype=np.int64)  # index 7 -> alpha=0.35 default
    pred_grid = p_grid.argmax(axis=2)  # (21, N)
    maxp_grid = p_grid.max(axis=2)     # (21, N)
    for i in range(n_cand):
        if not fg[i]:
            continue
        if m_mask[i]:
            y_i = gt_cls[i]
            hits = np.where(pred_grid[:, i] == y_i)[0]
            if len(hits) > 0:
                best_k[i] = hits[int(np.argmax(p_grid[hits, i, y_i]))]
            else:
                best_k[i] = int(np.argmax(p_grid[:, i, y_i]))
        else:
            best_k[i] = int(np.argmin(maxp_grid[:, i]))

    c_gate_orc = pred_grid[best_k, np.arange(n_cand)]
    p_gate_orc = maxp_grid[best_k, np.arange(n_cand)]
    sc_gate_orc = (yolo_sc ** beta) * (np.clip(p_gate_orc, 1e-6, 1.0) ** (1.0 - beta))
    ap_orc_alpha_gate_cls_only_no_nms = eval_custom(c_gate_orc, sc_abl1, do_nms=False)
    ap_orc_alpha_gate_full_with_nms = eval_custom(c_gate_orc, sc_gate_orc, do_nms=True)

    oracles = {
        'Ref0_Native_YOLOv8n': ap_native,
        'Ref1_FullDINO_Square_Hard': ap_b1_hard,
        'Ref2_GlobalFixed_ClsOnly_NoNMS': ap_fixed_cls_only_no_nms,
        'Ref3_GlobalFixed_ClsPlusScore_NoNMS': ap_fixed_cls_plus_score_no_nms,
        'Ref4_Abl1_GlobalFixed_Full_WithNMS': ap_abl1_full,
        'StepEffect_ScoreReweight_NoNMS': round(ap_fixed_cls_plus_score_no_nms - ap_fixed_cls_only_no_nms, 3),
        'StepEffect_PostReclass_NMS': round(ap_abl1_full - ap_fixed_cls_plus_score_no_nms, 3),
        'Oracle_A1_ClassChoice_KeepYoloSc_NoNMS': ap_orc_cls_keep_yolosc_no_nms,
        'Oracle_A2_ClassChoice_KeepAbl1Sc_NoNMS': ap_orc_cls_keep_abl1sc_no_nms,
        'Oracle_A3_ClassChoice_WithScoreAndNMS': ap_orc_cls_with_nms,
        'Delta_PureClassChoice_Over_Abl1_WithNMS': round(ap_orc_cls_with_nms - ap_abl1_full, 3),
        'Delta_PureClassChoice_FixedScore_NoNMS': round(ap_orc_cls_keep_abl1sc_no_nms - ap_fixed_cls_plus_score_no_nms, 3),
        'Oracle_B1_AlphaScoreOnly_FixedClass_NoNMS': ap_orc_score_gate_no_nms,
        'Oracle_B2_AlphaScoreOnly_FixedClass_WithNMS': ap_orc_score_gate_with_nms,
        'Delta_AlphaScoreOnly_Over_Abl1': round(ap_orc_score_gate_with_nms - ap_abl1_full, 3),
        'Oracle_B3_UnconstrainedIdealRank_FixedBoxesAndClass': ap_orc_unconstrained_ideal_rank,
        'Delta_UnconstrainedIdealRank_Over_Abl1': round(ap_orc_unconstrained_ideal_rank - ap_abl1_full, 3),
        'Oracle_C_TrueGateSpace_AlphaIn01_Full_WithNMS': ap_orc_alpha_gate_full_with_nms,
        'Delta_TrueGateSpace_AlphaIn01_Over_Abl1': round(ap_orc_alpha_gate_full_with_nms - ap_abl1_full, 3),
    }

    # Pack raw arrays needed for Predictability & Class-Level / Per-Box CV
    maha_14 = compute_maha_z(d['zneck'], m_cfg['means'], m_cfg['prec'], m_cfg['m_mu'], m_cfg['m_sd'])
    d_maha_z = maha_14[np.arange(n_cand), c_det]
    feats_14d = build_single_pass_reliability_features(yolo_sc, p_det, p_std, d_maha_z, d['geom'])

    arrays = {
        'gt_path': gt_path,
        'orig_preds': orig_preds,
        'cand_idxs': cand_idxs,
        'img_ids': img_ids_cand,
        'fg': fg,
        'm_mask': m_mask,
        'm_B': m_B,
        'm_C': m_C,
        'gt_cls': gt_cls,
        'yolo_sc': yolo_sc,
        'logits_det_raw': logits_det_raw,
        'logits_std_raw': logits_std_raw,
        'lp_det': lp_det,
        'lp_std': lp_std,
        'p_det': p_det,
        'p_std': p_std,
        'c_det': c_det,
        'c_std': c_std,
        'best_alpha_target': alphas_grid[best_k],
        'feats_14d': feats_14d,
        'ap_abl1_full': ap_abl1_full,
    }
    return census, oracles, arrays


def evaluate_predictability_and_calibration(seed: int, val_arr: dict, tb_arr: dict, tc_arr: dict, beta: float):
    """
    严格遵守不碰 testB/testC 选参：
    1. 在源域 SciDB val 上执行 5 折按图像分组交叉拟合 (5-Fold GroupKFold OOF)：
       - 评估冲突可预测性 ROC-AUC / PR-AUC 及分层 Bootstrap 95% CI；
       - 对比：(a) 全局固定权重 0.35、(b) NLL 温度+全局权重校准、(c) 无正则 14 类独立权重、(d) L2 向全局收缩 14 类权重 (λ=0.5, 2.0)、
               (e) 逐框逻辑回归/岭回归门控 α_hat(x) 的真实 5 折 OOF AP50-95 净收益！
    2. 将仅在 SciDB val 全集上拟合出的参数冻结，事后诊断其在探索集 testB / testC 上的表现与可预测性 AUC。
    """
    gkf = GroupKFold(n_splits=5)

    def eval_lp_on_arr(arr: dict, lp_comb: np.ndarray):
        p = np.exp(lp_comb - lp_comb.max(axis=1, keepdims=True)); p /= p.sum(axis=1, keepdims=True)
        rows = [dict(r) for r in arr['orig_preds']]
        fg = arr['fg']
        ysc = arr['yolo_sc']
        for li, gi in enumerate(arr['cand_idxs']):
            if fg[li]:
                nc = int(np.argmax(p[li]))
                rows[gi]['category_id'] = nc
                rows[gi]['score'] = float((ysc[li] ** beta) * (max(float(p[li, nc]), 1e-6) ** (1.0 - beta)))
        rows = apply_classwise_nms(rows, 0.70)
        return round(coco_ap(arr['gt_path'], rows)[0], 3)

    # --- A. 5-Fold Out-of-Fold Cross-Validation STRICTLY on Source SciDB val ---
    N_v = len(val_arr['cand_idxs'])
    oof_lp_temp = np.zeros_like(val_arr['lp_det'])
    oof_lp_cls_unreg = np.zeros_like(val_arr['lp_det'])
    oof_lp_cls_shrink05 = np.zeros_like(val_arr['lp_det'])
    oof_lp_cls_shrink20 = np.zeros_like(val_arr['lp_det'])
    oof_alpha_ridge = np.full(N_v, 0.35, dtype=np.float64)
    oof_prob_dino_better = np.full(N_v, 0.5, dtype=np.float64)
    oof_prob_dino_err = np.full(N_v, 0.5, dtype=np.float64)

    for tr_i, te_i in gkf.split(np.arange(N_v), groups=val_arr['img_ids']):
        m_tr = tr_i[val_arr['m_mask'][tr_i]]
        y_tr = val_arr['gt_cls'][m_tr]

        # 1) Temp + Global alpha
        td, ts, al_g = fit_temperature_and_global_alpha(val_arr['logits_det_raw'][m_tr], val_arr['logits_std_raw'][m_tr], y_tr)
        lpd_te = F.log_softmax(torch.from_numpy(val_arr['logits_det_raw'][te_i]) / td, dim=-1).numpy()
        lps_te = F.log_softmax(torch.from_numpy(val_arr['logits_std_raw'][te_i]) / ts, dim=-1).numpy()
        oof_lp_temp[te_i] = al_g * lpd_te + (1.0 - al_g) * lps_te

        # 2) Per-class weights: Unregularized vs Shrinkage (λ=0.02, λ=0.10 on mean NLL)
        w_unreg = fit_classwise_weights_nll(val_arr['lp_det'][m_tr], val_arr['lp_std'][m_tr], y_tr, alpha0=0.35, l2_shrink=0.0)
        w_sh05 = fit_classwise_weights_nll(val_arr['lp_det'][m_tr], val_arr['lp_std'][m_tr], y_tr, alpha0=0.35, l2_shrink=0.015)
        w_sh20 = fit_classwise_weights_nll(val_arr['lp_det'][m_tr], val_arr['lp_std'][m_tr], y_tr, alpha0=0.35, l2_shrink=0.06)

        oof_lp_cls_unreg[te_i] = val_arr['lp_det'][te_i] * w_unreg[None, :] + val_arr['lp_std'][te_i] * (1.0 - w_unreg[None, :])
        oof_lp_cls_shrink05[te_i] = val_arr['lp_det'][te_i] * w_sh05[None, :] + val_arr['lp_std'][te_i] * (1.0 - w_sh05[None, :])
        oof_lp_cls_shrink20[te_i] = val_arr['lp_det'][te_i] * w_sh20[None, :] + val_arr['lp_std'][te_i] * (1.0 - w_sh20[None, :])

        # 3) Per-box Ridge Gate regressing oracle best_alpha_target on fg boxes
        fg_tr = tr_i[val_arr['fg'][tr_i]]
        scaler = StandardScaler().fit(val_arr['feats_14d'][fg_tr])
        X_tr = scaler.transform(val_arr['feats_14d'][fg_tr])
        X_te = scaler.transform(val_arr['feats_14d'][te_i])
        reg_gate = Ridge(alpha=50.0).fit(X_tr, val_arr['best_alpha_target'][fg_tr])
        oof_alpha_ridge[te_i] = np.clip(reg_gate.predict(X_te), 0.0, 1.0)

        # 4) Binary Conflict Classifier on Case B (1: DINO right) vs Case C (0: YOLO right)
        bc_tr = tr_i[val_arr['m_B'][tr_i] | val_arr['m_C'][tr_i]]
        y_bc_tr = val_arr['m_B'][bc_tr].astype(int)
        if len(np.unique(y_bc_tr)) == 2 and len(bc_tr) >= 6:
            sc_bc = StandardScaler().fit(val_arr['feats_14d'][bc_tr])
            clf_bc = LogisticRegression(C=0.2, class_weight='balanced', max_iter=500).fit(sc_bc.transform(val_arr['feats_14d'][bc_tr]), y_bc_tr)
            oof_prob_dino_better[te_i] = clf_bc.predict_proba(sc_bc.transform(val_arr['feats_14d'][te_i]))[:, 1]

        # 5) Binary DINO Error Predictor across all matched boxes (1: DINO wrong, 0: DINO right)
        y_derr_tr = (val_arr['c_std'][m_tr] != y_tr).astype(int)
        if len(np.unique(y_derr_tr)) == 2:
            sc_de = StandardScaler().fit(val_arr['feats_14d'][m_tr])
            clf_de = LogisticRegression(C=0.2, class_weight='balanced', max_iter=500).fit(sc_de.transform(val_arr['feats_14d'][m_tr]), y_derr_tr)
            oof_prob_dino_err[te_i] = clf_de.predict_proba(sc_de.transform(val_arr['feats_14d'][te_i]))[:, 1]

    oof_lp_ridge = oof_alpha_ridge[:, None] * val_arr['lp_det'] + (1.0 - oof_alpha_ridge[:, None]) * val_arr['lp_std']

    # Evaluate OOF Predictability AUCs on SciDB val:
    bc_val = val_arr['m_B'] | val_arr['m_C']
    # Also evaluate zero-training heuristic margin difference (mar_std - mar_det) as a parameter-free conflict predictor!
    heur_dino_adv_val = val_arr['feats_14d'][:, 5] - val_arr['feats_14d'][:, 2]  # mar_std - mar_det

    val_oof_auc_learned_bc = stratified_auc_bootstrap(val_arr['m_B'][bc_val].astype(int), oof_prob_dino_better[bc_val], seed=100 + seed)
    val_oof_auc_heur_bc = stratified_auc_bootstrap(val_arr['m_B'][bc_val].astype(int), heur_dino_adv_val[bc_val], seed=200 + seed)
    val_oof_auc_dino_err = stratified_auc_bootstrap((val_arr['c_std'][val_arr['m_mask']] != val_arr['gt_cls'][val_arr['m_mask']]).astype(int),
                                                    oof_prob_dino_err[val_arr['m_mask']], seed=300 + seed)

    val_oof_aps = {
        'Global_Fixed_035_Abl1': val_arr['ap_abl1_full'],
        'OOF_TempScaled_Global': eval_lp_on_arr(val_arr, oof_lp_temp),
        'OOF_PerClass_Unregularized': eval_lp_on_arr(val_arr, oof_lp_cls_unreg),
        'OOF_PerClass_Shrinkage_Mild': eval_lp_on_arr(val_arr, oof_lp_cls_shrink05),
        'OOF_PerClass_Shrinkage_Strong': eval_lp_on_arr(val_arr, oof_lp_cls_shrink20),
        'OOF_PerBox_RidgeGate_14d': eval_lp_on_arr(val_arr, oof_lp_ridge),
    }

    # --- B. Fit ONCE on Full Source SciDB val -> Post-Hoc Diagnostic Evaluation on Exploratory testB / testC ---
    m_v_all = np.where(val_arr['m_mask'])[0]
    yv_all = val_arr['gt_cls'][m_v_all]
    td_full, ts_full, al_full = fit_temperature_and_global_alpha(val_arr['logits_det_raw'][m_v_all], val_arr['logits_std_raw'][m_v_all], yv_all)
    w_unreg_full = fit_classwise_weights_nll(val_arr['lp_det'][m_v_all], val_arr['lp_std'][m_v_all], yv_all, 0.35, 0.0)
    w_sh05_full = fit_classwise_weights_nll(val_arr['lp_det'][m_v_all], val_arr['lp_std'][m_v_all], yv_all, 0.35, 0.015)
    w_sh20_full = fit_classwise_weights_nll(val_arr['lp_det'][m_v_all], val_arr['lp_std'][m_v_all], yv_all, 0.35, 0.06)

    fg_v_all = np.where(val_arr['fg'])[0]
    sc_v_fg = StandardScaler().fit(val_arr['feats_14d'][fg_v_all])
    reg_v_full = Ridge(alpha=50.0).fit(sc_v_fg.transform(val_arr['feats_14d'][fg_v_all]), val_arr['best_alpha_target'][fg_v_all])

    bc_v_all = np.where(bc_val)[0]
    sc_v_bc = StandardScaler().fit(val_arr['feats_14d'][bc_v_all])
    clf_v_bc = LogisticRegression(C=0.2, class_weight='balanced', max_iter=500).fit(
        sc_v_bc.transform(val_arr['feats_14d'][bc_v_all]), val_arr['m_B'][bc_v_all].astype(int)
    )

    diag_splits = {}
    for sp_tag, t_arr in [('testB_exploratory', tb_arr), ('testC_exploratory', tc_arr)]:
        lpd_t = F.log_softmax(torch.from_numpy(t_arr['logits_det_raw']) / td_full, dim=-1).numpy()
        lps_t = F.log_softmax(torch.from_numpy(t_arr['logits_std_raw']) / ts_full, dim=-1).numpy()
        al_ridge_t = np.clip(reg_v_full.predict(sc_v_fg.transform(t_arr['feats_14d'])), 0.0, 1.0)

        bc_t = t_arr['m_B'] | t_arr['m_C']
        prob_bc_from_val = clf_v_bc.predict_proba(sc_v_bc.transform(t_arr['feats_14d'][bc_t]))[:, 1]
        heur_bc_t = t_arr['feats_14d'][bc_t, 5] - t_arr['feats_14d'][bc_t, 2]

        diag_splits[sp_tag] = {
            'predictability_CaseB_vs_CaseC_ValTrainedLogReg': stratified_auc_bootstrap(t_arr['m_B'][bc_t].astype(int), prob_bc_from_val, seed=400 + seed),
            'predictability_CaseB_vs_CaseC_ZeroParamMarginDiff': stratified_auc_bootstrap(t_arr['m_B'][bc_t].astype(int), heur_bc_t, seed=500 + seed),
            'frozen_from_val_AP50_95': {
                'Global_Fixed_035_Abl1': t_arr['ap_abl1_full'],
                'ValFit_TempScaled_Global': eval_lp_on_arr(t_arr, al_full * lpd_t + (1.0 - al_full) * lps_t),
                'ValFit_PerClass_Unregularized': eval_lp_on_arr(t_arr, t_arr['lp_det'] * w_unreg_full[None, :] + t_arr['lp_std'] * (1.0 - w_unreg_full[None, :])),
                'ValFit_PerClass_Shrinkage_Mild': eval_lp_on_arr(t_arr, t_arr['lp_det'] * w_sh05_full[None, :] + t_arr['lp_std'] * (1.0 - w_sh05_full[None, :])),
                'ValFit_PerClass_Shrinkage_Strong': eval_lp_on_arr(t_arr, t_arr['lp_det'] * w_sh20_full[None, :] + t_arr['lp_std'] * (1.0 - w_sh20_full[None, :])),
                'ValFit_PerBox_RidgeGate_14d': eval_lp_on_arr(t_arr, al_ridge_t[:, None] * t_arr['lp_det'] + (1.0 - al_ridge_t[:, None]) * t_arr['lp_std']),
            },
        }

    return {
        'learned_val_weights': {
            'temp_det': round(td_full, 3), 'temp_dino': round(ts_full, 3), 'alpha_global': round(al_full, 3),
            'w_unregularized_14cls': [round(float(x), 3) for x in w_unreg_full],
            'w_shrinkage_mild_14cls': [round(float(x), 3) for x in w_sh05_full],
            'w_shrinkage_strong_14cls': [round(float(x), 3) for x in w_sh20_full],
        },
        'source_val_5fold_OOF': {
            'predictability_CaseB_vs_CaseC_OOF_LogReg': val_oof_auc_learned_bc,
            'predictability_CaseB_vs_CaseC_ZeroParamMarginDiff': val_oof_auc_heur_bc,
            'predictability_DINO_Error_AllMatched_OOF': val_oof_auc_dino_err,
            'OOF_AP50_95_comparison': val_oof_aps,
        },
        'exploratory_posthoc_transfer': diag_splits,
    }


def main():
    t0 = time.perf_counter()
    s2_fin = json.loads((ROOT / 'Antigravity_CGGAR_真创新攻关_20261125' / 'results' / 'tcd_mge_scidb_3seed_final.json').read_text(encoding='utf-8'))

    train_bias_audit = audit_train_cache_selection_bias()
    print("================ [0] SciDB train 缓存样本内记忆与 GT 选框偏差核查 ================", flush=True)
    for r in train_bias_audit:
        print(f"  Seed {r['seed']}: GT框={r['n_gt_matched_train_boxes']} | YOLO样本内准确率={r['yolo_in_sample_acc_pct']:.2f}% vs DINO={r['dino_in_sample_acc_pct']:.2f}% | "
              f"分歧={r['disagreement_count']} ({r['disagreement_pct']:.2f}%) [YOLO对DINO错={r['caseC_yolo_right_dino_wrong']}, DINO对YOLO错={r['caseB_dino_right_yolo_wrong']}]", flush=True)

    per_seed_results = []
    for seed in [0, 1, 2]:
        blob = torch.load(CKPT_TCD_DIR / f'cggar_v2_seed{seed}.pt', map_location='cpu', weights_only=False)
        T_DET, T_DINO = blob['temps']['T_DET'], blob['temps']['T_DINO']
        m_cfg = blob['mahalanobis']
        beta = next(x for x in s2_fin['逐种子明细'] if x['seed'] == seed)['val_calibrated_params']['beta']

        c_val, o_val, arr_val = evaluate_split_oracles_and_counts(seed, 'val', T_DET, T_DINO, beta, m_cfg)
        c_tb, o_tb, arr_tb = evaluate_split_oracles_and_counts(seed, 'testB', T_DET, T_DINO, beta, m_cfg)
        c_tc, o_tc, arr_tc = evaluate_split_oracles_and_counts(seed, 'testC', T_DET, T_DINO, beta, m_cfg)

        pred_and_cal = evaluate_predictability_and_calibration(seed, arr_val, arr_tb, arr_tc, beta)

        per_seed_results.append({
            'seed': seed,
            'census': {'val_source': c_val, 'testB_exploratory': c_tb, 'testC_exploratory': c_tc},
            'decomposed_oracles': {'val_source': o_val, 'testB_exploratory': o_tb, 'testC_exploratory': o_tc},
            'predictability_and_calibration': pred_and_cal,
        })

        print(f"\n================ [Seed {seed} 只读可行性审计摘要] ================", flush=True)
        for sp_k, lbl in [('val_source', 'SciDB val (源域决策门)'), ('testB_exploratory', 'testB (事后探索诊断)'), ('testC_exploratory', 'testC (事后探索诊断)')]:
            cs = per_seed_results[-1]['census'][sp_k]
            oc = per_seed_results[-1]['decomposed_oracles'][sp_k]
            print(f"  [{lbl}] fg框={cs['n_cand_fg08']} (不一致={cs['n_fg08_disagree_all']}, {cs['fg08_disagree_pct']}%) | "
                  f"匹配框={cs['n_matched_tp_candidates']}: A(全对)={cs['quadrant_A_both_right']}, B(DINO对)={cs['quadrant_B_dino_right_yolo_wrong']}, "
                  f"C(YOLO对)={cs['quadrant_C_yolo_right_dino_wrong']}, D(全错)={cs['quadrant_D_both_wrong']}(Top2可救={cs['quadrant_D_in_top2_union']})", flush=True)
            print(f"    基线 Abl1={oc['Ref4_Abl1_GlobalFixed_Full_WithNMS']:.2f}% (纯类别无NMS={oc['Ref2_GlobalFixed_ClsOnly_NoNMS']:.2f}%, +重算分={oc['StepEffect_ScoreReweight_NoNMS']:+.2f}, +NMS={oc['StepEffect_PostReclass_NMS']:+.2f}) | "
                  f"Δ纯类别Oracle={oc['Delta_PureClassChoice_Over_Abl1_WithNMS']:+.2f}% | Δα排序Oracle={oc['Delta_AlphaScoreOnly_Over_Abl1']:+.2f}% "
                  f"(无约束完美排序Δ={oc['Delta_UnconstrainedIdealRank_Over_Abl1']:+.2f}%) | ★Δ门控空间α∈[0,1]总上限={oc['Delta_TrueGateSpace_AlphaIn01_Over_Abl1']:+.2f}%", flush=True)

        sv = pred_and_cal['source_val_5fold_OOF']
        print(f"  [Seed {seed} | 源域 val 5折OOF] 冲突可预测性 ROC-AUC: 学习LogReg={sv['predictability_CaseB_vs_CaseC_OOF_LogReg']['roc_auc']} "
              f"CI={sv['predictability_CaseB_vs_CaseC_OOF_LogReg']['roc_ci95']} | 零参Margin差={sv['predictability_CaseB_vs_CaseC_ZeroParamMarginDiff']['roc_auc']} "
              f"CI={sv['predictability_CaseB_vs_CaseC_ZeroParamMarginDiff']['roc_ci95']}", flush=True)
        print(f"    val OOF AP50-95: 固定Abl1={sv['OOF_AP50_95_comparison']['Global_Fixed_035_Abl1']:.2f}% | 温度校准={sv['OOF_AP50_95_comparison']['OOF_TempScaled_Global']:.2f}% | "
              f"无正则14类={sv['OOF_AP50_95_comparison']['OOF_PerClass_Unregularized']:.2f}% | 收缩14类(强)={sv['OOF_AP50_95_comparison']['OOF_PerClass_Shrinkage_Strong']:.2f}% | "
              f"逐框Ridge门控={sv['OOF_AP50_95_comparison']['OOF_PerBox_RidgeGate_14d']:.2f}%", flush=True)
        for tk in ['testB_exploratory', 'testC_exploratory']:
            dt = pred_and_cal['exploratory_posthoc_transfer'][tk]
            fa = dt['frozen_from_val_AP50_95']
            print(f"    [{tk}] 冲突AUC(源LogReg)={dt['predictability_CaseB_vs_CaseC_ValTrainedLogReg']['roc_auc']} CI={dt['predictability_CaseB_vs_CaseC_ValTrainedLogReg']['roc_ci95']} "
                  f"(零参Margin差={dt['predictability_CaseB_vs_CaseC_ZeroParamMarginDiff']['roc_auc']}) || "
                  f"固定Abl1={fa['Global_Fixed_035_Abl1']:.2f}% | 温度={fa['ValFit_TempScaled_Global']:.2f}% | 无正则14类={fa['ValFit_PerClass_Unregularized']:.2f}% | "
                  f"收缩14类(强)={fa['ValFit_PerClass_Shrinkage_Strong']:.2f}% | 逐框门控={fa['ValFit_PerBox_RidgeGate_14d']:.2f}%", flush=True)

    # Compute 3-seed mean±std summary tables
    summary_3seed = {'oracles': {}, 'val_oof_calibration_and_gate': {}, 'exploratory_transfer': {'testB_exploratory': {}, 'testC_exploratory': {}}}
    for sp_k in ['val_source', 'testB_exploratory', 'testC_exploratory']:
        summary_3seed['oracles'][sp_k] = {}
        for ok in per_seed_results[0]['decomposed_oracles'][sp_k].keys():
            vals = [s['decomposed_oracles'][sp_k][ok] for s in per_seed_results]
            summary_3seed['oracles'][sp_k][ok] = {
                'mean': round(float(np.mean(vals)), 3),
                'std': round(float(np.std(vals, ddof=1)), 3),
                'seeds': vals,
            }
    for mk in per_seed_results[0]['predictability_and_calibration']['source_val_5fold_OOF']['OOF_AP50_95_comparison'].keys():
        vals = [s['predictability_and_calibration']['source_val_5fold_OOF']['OOF_AP50_95_comparison'][mk] for s in per_seed_results]
        summary_3seed['val_oof_calibration_and_gate'][mk] = {'mean': round(float(np.mean(vals)), 3), 'std': round(float(np.std(vals, ddof=1)), 3), 'seeds': vals}
    for tk in ['testB_exploratory', 'testC_exploratory']:
        for mk in per_seed_results[0]['predictability_and_calibration']['exploratory_posthoc_transfer'][tk]['frozen_from_val_AP50_95'].keys():
            vals = [s['predictability_and_calibration']['exploratory_posthoc_transfer'][tk]['frozen_from_val_AP50_95'][mk] for s in per_seed_results]
            summary_3seed['exploratory_transfer'][tk][mk] = {'mean': round(float(np.mean(vals)), 3), 'std': round(float(np.std(vals, ddof=1)), 3), 'seeds': vals}

    out_json = OUT_DIR / 'phase1_readonly_oracle_and_predictability_results.json'
    out_json.write_text(json.dumps({
        '审计名称': '阶段一只读可行性审计_Oracle分解与冲突可预测性及类级收缩校准',
        '耗时秒': round(time.perf_counter() - t0, 2),
        '仓库零修改保证': True,
        '训练集GT选框与样本内记忆核查': train_bias_audit,
        '三种子汇总': summary_3seed,
        '逐种子完整明细': per_seed_results,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n[只读可行性审计完成] 耗时 {time.perf_counter()-t0:.1f}s，结果已保存至: {out_json}", flush=True)


if __name__ == '__main__':
    main()
