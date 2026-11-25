# -*- coding: utf-8 -*-
"""
Step 2 (Final - TCD-MGE): 目标-环境相对差分表征与流形衰减确信度对数池化按需升级检测纠错方法
(Target-Context Differential Multi-Granularity Escalation & Manifold-Attenuated Logarithmic Opinion Pooling)

核心设计（彻底消除源训练集内检测器 99.8% 记忆效应对 MLP 仲裁权重的偏置）：
1. 直接复用仅在 SciDB train 上训练冻结的目标-环境相对差分头 R_psi(z_core, z_std, z_core - z_ctx, z_std - z_ctx, g(b))；
2. 三信号按需升级门控 (Level-1 -> Level-2)：仅当 (1) 整图与 1.25x 预测类别不一致，或 (2) 1.25x 归一化香农熵超源 val 分位阈值，
   或 (3) 检测器颈部特征马氏偏移超源 val 分位阈值时，才追加提取 Core(0.82x) + Ctx(2.50x) 升级至四路差分池化（源域省去 ~70% 扩框开销）；
3. 流形衰减香农确信度对数池化 (Manifold-Attenuated Certainty-Weighted Log-Opinion Pool)：
   u_det(x) = w_0 * (1 - H_norm(p_det)) * exp(-gamma * max(0, d_Maha_z(x)))
   u_k(x)   = w_k * (1 - H_norm(p_k)),  k in {std, core, rel}
   全部超参数 (tau_H, tau_M, gamma, beta) 100% 仅在源域 SciDB val 上标定并冻结！
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'

import sys
import json
import math
import time
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(r'D:\新版柑橘')
EXT = ROOT / '方法路线扩展_20261031'
sys.path.append(str(EXT))
from 共同评测后端 import configure_strict_fp32

WORK_DIR = ROOT / 'Antigravity_CGGAR_真创新攻关_20261125'
sys.path.append(str(WORK_DIR))
from step2_train_and_eval_cggar import (
    coco_eval_full, apply_classwise_nms, compute_maha_z, entropy_norm, TargetContextRelationalHead
)

CACHE_DIR = WORK_DIR / 'cache'
CKPT_DIR = WORK_DIR / 'models'
RES_DIR = WORK_DIR / 'results'
GT_DIR = EXT / '检测数据'
STRONG_DIR = EXT / 'YOLOv8强基线重复_20261109'
E2_OLD_JSON = ROOT / '多路线创新筛选_20261124' / '小论文完整实验_20261124' / 'results' / 'e2_end2end_mgs_symtex_summary.json'


def compute_analytic_opinion_pool(lp_det: np.ndarray, lp_std: np.ndarray, lp_core: np.ndarray, lp_rel: np.ndarray,
                                  h_det: np.ndarray, h_std: np.ndarray, h_core: np.ndarray, h_rel: np.ndarray,
                                  d_maha_z: np.ndarray, esc_mask: np.ndarray, gamma: float) -> np.ndarray:
    """
    流形衰减香农确信度加权对数意见池 (Logarithmic Opinion Pool)：
    零待训过拟合参数，实例级根据各视角香农确信度 (1 - H_norm) 与整图流形偏移 exp(-gamma * d_Maha^+) 自适应融合！
    """
    c_det = np.clip(1.0 - h_det, 0.05, 1.0) * np.exp(-gamma * np.maximum(0.0, d_maha_z))
    c_std = np.clip(1.0 - h_std, 0.05, 1.0)
    c_core = np.clip(1.0 - h_core, 0.05, 1.0)
    c_rel = np.clip(1.0 - h_rel, 0.05, 1.0)

    # Level-2 (escalated 4-view): prior [0.24, 0.24, 0.20, 0.32] * certainty
    u0_4 = 0.24 * c_det
    u1_4 = 0.24 * c_std
    u2_4 = 0.20 * c_core
    u3_4 = 0.32 * c_rel
    s4 = u0_4 + u1_4 + u2_4 + u3_4 + 1e-8
    lp_4v = (u0_4[:, None] * lp_det + u1_4[:, None] * lp_std + u2_4[:, None] * lp_core + u3_4[:, None] * lp_rel) / s4[:, None]

    # Level-1 (non-escalated fast 2-view): prior [0.35, 0.65] * certainty
    u0_2 = 0.35 * c_det
    u1_2 = 0.65 * c_std
    s2 = u0_2 + u1_2 + 1e-8
    lp_2v = (u0_2[:, None] * lp_det + u1_2[:, None] * lp_std) / s2[:, None]

    lp_out = np.where(esc_mask[:, None], lp_4v, lp_2v)
    p_out = np.exp(lp_out - lp_out.max(axis=1, keepdims=True))
    p_out /= p_out.sum(axis=1, keepdims=True)
    return p_out


@torch.no_grad()
def evaluate_seed_tcd_mge(seed: int, old_e2_seed: dict, device='cuda:0'):
    ckpt_path = CKPT_DIR / f'cggar_v2_seed{seed}.pt'
    blob = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    rel_head = TargetContextRelationalHead().to(device).eval()
    rel_head.load_state_dict(blob['rel_head'])
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

        h_det = entropy_norm(p_det).squeeze(-1).cpu().numpy()
        h_std = entropy_norm(p_std).squeeze(-1).cpu().numpy()
        h_core = entropy_norm(p_core).squeeze(-1).cpu().numpy()
        h_rel = entropy_norm(p_rel).squeeze(-1).cpu().numpy()

        c_hat = p_det.argmax(dim=-1)
        d_maha_z = maha_14.gather(1, c_hat.unsqueeze(1)).squeeze(-1).cpu().numpy()
        dis_det_std = (p_det.argmax(-1) != p_std.argmax(-1)).cpu().numpy()

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
            'h_det': h_det, 'h_std': h_std, 'h_core': h_core, 'h_rel': h_rel,
            'd_maha_z': d_maha_z,
            'dis_det_std': dis_det_std,
        }

    # ---------------- STRICT SOURCE-VAL ONLY CALIBRATION ----------------
    v_sc = sp_data['val']
    val_hi = v_sc['yolo_sc'] >= 0.25
    # Calibrate Level-2 Tri-Signal Escalation thresholds ONLY on SciDB val (q=0.82 on entropy & q=0.82 on maha -> ~28% total escalation on val!)
    tau_h_std = float(np.quantile(v_sc['h_std'][val_hi], 0.82))
    tau_maha_z = float(np.quantile(v_sc['d_maha_z'][val_hi], 0.82))
    FG_THR = 0.08  # prune noisy background tail < 0.08

    # Grid search (gamma, beta) STRICTLY on SciDB val AP50-95!
    best_gamma, best_beta, best_val_ap = 0.15, 0.45, -1.0
    v_fg = v_sc['yolo_sc'] >= FG_THR
    v_esc = v_fg & (v_sc['dis_det_std'] | (v_sc['h_std'] > tau_h_std) | (v_sc['d_maha_z'] > tau_maha_z))

    for gm_cand in [0.05, 0.12, 0.20, 0.28]:
        p_val_cand = compute_analytic_opinion_pool(
            v_sc['lp_det'], v_sc['lp_std'], v_sc['lp_core'], v_sc['lp_rel'],
            v_sc['h_det'], v_sc['h_std'], v_sc['h_core'], v_sc['h_rel'],
            v_sc['d_maha_z'], v_esc, gamma=gm_cand
        )
        for bt_cand in [0.35, 0.45, 0.55]:
            rows_v = [dict(r) for r in v_sc['orig_preds']]
            for li, gi in enumerate(v_sc['cand_idxs']):
                if v_fg[li]:
                    pr = p_val_cand[li]
                    nc = int(np.argmax(pr))
                    rows_v[gi]['category_id'] = nc
                    rows_v[gi]['score'] = float((v_sc['yolo_sc'][li] ** bt_cand) * (max(float(pr[nc]), 1e-6) ** (1.0 - bt_cand)))
            rows_v = apply_classwise_nms(rows_v, 0.70)
            st_v, _ = coco_eval_full(v_sc['gt_path'], rows_v)
            if st_v[0] > best_val_ap:
                best_val_ap = st_v[0]
                best_gamma = gm_cand
                best_beta = bt_cand

    # ---------------- FROZEN EVALUATION ACROSS VAL / TESTB / TESTC ----------------
    out_splits = {}
    for sp in ['val', 'testB', 'testC']:
        sc = sp_data[sp]
        n = len(sc['cand_idxs'])
        hi = sc['yolo_sc'] >= 0.25
        fg = sc['yolo_sc'] >= FG_THR

        # Escalation masks for ablation:
        esc_dis_only = fg & sc['dis_det_std']  # Only disagreement
        esc_tri_signal = fg & (sc['dis_det_std'] | (sc['h_std'] > tau_h_std) | (sc['d_maha_z'] > tau_maha_z))  # Full Tri-Signal
        esc_all = fg.copy()

        def eval_cfg(active_m, prob_mat, calib: bool, use_nms: bool, esc_m_opt=None):
            rows = [dict(r) for r in sc['orig_preds']]
            for li, gi in enumerate(sc['cand_idxs']):
                if active_m[li]:
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
            if esc_m_opt is not None:
                crop_cost = float((active_m.sum() + 2 * esc_m_opt.sum()) / max(1, n))
                esc_hi_pct = float((esc_m_opt & hi).sum() / max(1, hi.sum())) * 100.0
            else:
                crop_cost = float(active_m.sum() / max(1, n))
                esc_hi_pct = 0.0
            return {
                'AP50_95': round(stats[0] * 100.0, 4),
                'AP50': round(stats[1] * 100.0, 4),
                'esc_hi_pct': round(esc_hi_pct, 2),
                'crop_cost_vs_fulldino': round(crop_cost, 3),
                'per_cls_AP': per_cls,
            }

        m_all = np.ones(n, dtype=bool)
        p_b3 = 0.65 * sc['p_std_raw'].copy()
        p_b3[np.arange(n), sc['yolo_cls']] += 0.35 * sc['yolo_sc']

        # Abl1: Single 1.25x crop + 14-cls calibrated log-prob + decoupled score calib
        lp_abl1 = 0.35 * sc['lp_det'] + 0.65 * sc['lp_std']
        p_abl1 = np.exp(lp_abl1 - lp_abl1.max(axis=1, keepdims=True)); p_abl1 /= p_abl1.sum(axis=1, keepdims=True)

        # Abl2: Multi-granular + Relational Head (Static Uniform Weights, All 3 crops)
        lp_abl2 = 0.24 * sc['lp_det'] + 0.24 * sc['lp_std'] + 0.20 * sc['lp_core'] + 0.32 * sc['lp_rel']
        p_abl2 = np.exp(lp_abl2 - lp_abl2.max(axis=1, keepdims=True)); p_abl2 /= p_abl2.sum(axis=1, keepdims=True)

        # Abl3: Disagreement-Only Escalation + Opinion Pool (without Manifold/Entropy triggers)
        p_dis_only = compute_analytic_opinion_pool(
            sc['lp_det'], sc['lp_std'], sc['lp_core'], sc['lp_rel'],
            sc['h_det'], sc['h_std'], sc['h_core'], sc['h_rel'],
            sc['d_maha_z'], esc_dis_only, gamma=best_gamma
        )

        # Ours (Selective Tri-Signal Escalation + Manifold-Attenuated Certainty Log-Opinion Pool)
        p_ours_sel = compute_analytic_opinion_pool(
            sc['lp_det'], sc['lp_std'], sc['lp_core'], sc['lp_rel'],
            sc['h_det'], sc['h_std'], sc['h_core'], sc['h_rel'],
            sc['d_maha_z'], esc_tri_signal, gamma=best_gamma
        )

        # Ours-Full (100% Escalated + Manifold-Attenuated Certainty Log-Opinion Pool)
        p_ours_full = compute_analytic_opinion_pool(
            sc['lp_det'], sc['lp_std'], sc['lp_core'], sc['lp_rel'],
            sc['h_det'], sc['h_std'], sc['h_core'], sc['h_rel'],
            sc['d_maha_z'], esc_all, gamma=best_gamma
        )

        st_nat, pc_nat = coco_eval_full(sc['gt_path'], sc['orig_preds'])
        old_full_dino = old_e2_seed['splits'][sp]['S4_Full_DINOv2']
        old_mgs_dino = old_e2_seed['splits'][sp]['Ours_Pro_MGS_DINOv2']

        res_sp = {
            'B0_YOLOv8n_Native': {'AP50_95': round(st_nat[0] * 100.0, 4), 'AP50': round(st_nat[1] * 100.0, 4), 'esc_hi_pct': 0.0, 'crop_cost_vs_fulldino': 0.0, 'per_cls_AP': pc_nat},
            'B1a_Full_DINOv2_GrayPad_Orig': {'AP50_95': old_full_dino['AP50_95'], 'AP50': old_full_dino['AP50'], 'esc_hi_pct': 0.0, 'crop_cost_vs_fulldino': 1.0, 'per_cls_AP': old_full_dino['per_cls_AP']},
            'B1b_Full_DINOv2_SquareCrop': eval_cfg(m_all, sc['p_std_raw'], False, False),
            'B2_MGS_DINOv2_v1': {'AP50_95': old_mgs_dino['AP50_95'], 'AP50': old_mgs_dino['AP50'], 'esc_hi_pct': 0.0, 'crop_cost_vs_fulldino': round(old_mgs_dino['trig_conf25_pct'] / 100.0, 3), 'per_cls_AP': old_mgs_dino['per_cls_AP']},
            'B3_Static_Posterior_Fuse': eval_cfg(m_all, p_b3, False, False),
            'Abl1_SingleCrop_14Cls_Calib': eval_cfg(fg, p_abl1, True, True),
            'Abl2_Disagreement_Esc_Only': eval_cfg(fg, p_dis_only, True, True, esc_m_opt=esc_dis_only),
            'Abl3_TriGranular_StaticWeight': eval_cfg(fg, p_abl2, True, True, esc_m_opt=esc_all),
            'Ours_TCDMGE_Selective': eval_cfg(fg, p_ours_sel, True, True, esc_m_opt=esc_tri_signal),
            'Ours_TCDMGE_Full': eval_cfg(fg, p_ours_full, True, True, esc_m_opt=esc_all),
        }
        out_splits[sp] = res_sp
        print(f"  [Seed {seed} | {sp:5s}] 原生={res_sp['B0_YOLOv8n_Native']['AP50_95']:.2f}% | "
              f"原版Full-DINO={res_sp['B1a_Full_DINOv2_GrayPad_Orig']['AP50_95']:.2f}% | "
              f"静态融合={res_sp['B3_Static_Posterior_Fuse']['AP50_95']:.2f}% | "
              f"Abl1(单框校准)={res_sp['Abl1_SingleCrop_14Cls_Calib']['AP50_95']:.2f}% | "
              f"★Ours(按需TCD-MGE)={res_sp['Ours_TCDMGE_Selective']['AP50_95']:.2f}% (升级率={res_sp['Ours_TCDMGE_Selective']['esc_hi_pct']:.1f}%) | "
              f"★Ours(全升级TCD-MGE)={res_sp['Ours_TCDMGE_Full']['AP50_95']:.2f}%",
              flush=True)

    return {
        'seed': seed,
        'val_calibrated_params': {
            'tau_h_std': round(tau_h_std, 5),
            'tau_maha_z': round(tau_maha_z, 5),
            'gamma': best_gamma,
            'beta': best_beta,
            'fg_thr': FG_THR,
        },
        'rel_head_ckpt_sha256': hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
        'splits': out_splits,
    }


def main():
    t0 = time.monotonic()
    backend = configure_strict_fp32()
    old_e2 = json.loads(E2_OLD_JSON.read_text(encoding='utf-8'))

    per_seed = []
    for seed in [0, 1, 2]:
        print(f"\n================ 评估 TCD-MGE Seed {seed} ================", flush=True)
        old_s = next(r for r in old_e2['逐种子明细'] if r['seed'] == seed)
        per_seed.append(evaluate_seed_tcd_mge(seed, old_s, 'cuda:0'))

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

    print("\n================ 严苛 3/3 随机种子零退步门禁核验 (TCD-MGE) ================", flush=True)
    gate_audit = {}
    baselines = [
        'B0_YOLOv8n_Native', 'B1a_Full_DINOv2_GrayPad_Orig', 'B1b_Full_DINOv2_SquareCrop',
        'B2_MGS_DINOv2_v1', 'B3_Static_Posterior_Fuse', 'Abl1_SingleCrop_14Cls_Calib'
    ]
    for cand in ['Ours_TCDMGE_Selective', 'Ours_TCDMGE_Full']:
        gate_audit[cand] = {}
        for base in baselines:
            wv = [per_seed[s]['splits']['val'][cand]['AP50_95'] - per_seed[s]['splits']['val'][base]['AP50_95'] for s in range(3)]
            wb = [per_seed[s]['splits']['testB'][cand]['AP50_95'] - per_seed[s]['splits']['testB'][base]['AP50_95'] for s in range(3)]
            wc = [per_seed[s]['splits']['testC'][cand]['AP50_95'] - per_seed[s]['splits']['testC'][base]['AP50_95'] for s in range(3)]
            pv, pb, pc = all(d > 0 for d in wv), all(d > 0 for d in wb), all(d > 0 for d in wc)
            gate_audit[cand][base] = {
                'val_seed_deltas': [round(d, 2) for d in wv], 'val_3of3_win': pv,
                'testB_seed_deltas': [round(d, 2) for d in wb], 'testB_3of3_win': pb,
                'testC_seed_deltas': [round(d, 2) for d in wc], 'testC_3of3_win': pc,
            }
            print(f"  [{cand:22s} vs {base:28s}] val={[round(d,2) for d in wv]}({pv}) | "
                  f"testB={[round(d,2) for d in wb]}({pb}) | testC={[round(d,2) for d in wc]}({pc})", flush=True)

    out_p = RES_DIR / 'tcd_mge_scidb_3seed_final.json'
    out_p.write_text(json.dumps({
        '实验名称': 'TCD-MGE_目标环境相对差分与流形衰减确信度池化按需升级_3种子最终核验',
        '耗时秒': round(time.monotonic() - t0, 2),
        '严格后端': backend,
        '三种子汇总': agg,
        '严格3比3门禁核验': gate_audit,
        '逐种子明细': per_seed,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n[完成] 已保存至: {out_p}", flush=True)


if __name__ == '__main__':
    main()
