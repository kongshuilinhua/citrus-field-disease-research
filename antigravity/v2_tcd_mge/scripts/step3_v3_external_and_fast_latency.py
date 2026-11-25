# -*- coding: utf-8 -*-
"""
Step 3 (Final - TCD-MGE): 完全冻结 TCD-MGE 权重与源 val 标定参数下的独立外部真实果园零样本封口盲测
(Zenodo 11148332, 519 图 / 1,765 框) + 单趟零冗余 RTX 4060 Ti 端到端单图延迟实测 + 300 DPI 出版级配图生成。
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
import contextlib
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align
from ultralytics import YOLO
from faster_coco_eval import COCO, COCOeval_faster
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(r'D:\新版柑橘')
EXT = ROOT / '方法路线扩展_20261031'
sys.path.append(str(EXT))
from 共同评测后端 import configure_strict_fp32

WORK_DIR = ROOT / 'Antigravity_CGGAR_真创新攻关_20261125'
sys.path.append(str(WORK_DIR))
from step1_extract_multigranular_cache import YoloFullStateExtractor, DinoExtractor, crop_square_scale, compute_box_geom, letterbox_640
from step2_train_and_eval_cggar import TargetContextRelationalHead, compute_maha_z, entropy_norm, apply_classwise_nms
from step2_v3_analytic_opinion_pool import compute_analytic_opinion_pool
from step3_external_blind_and_latency import build_external_zenodo_gt

CKPT_DIR = WORK_DIR / 'models'
RES_DIR = WORK_DIR / 'results'
FIG_DIR = WORK_DIR / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
IMG_ROOT = Path(r'D:\病虫害论文\复现实验记录\yolo11n_public_compare_20261004\数据副本\images')


@torch.no_grad()
def evaluate_external_tcdmge_3seeds(coco_images, img_paths, gt2_p, gt3_p, s2_final):
    device = 'cuda:0'
    scidb_to_2cls = {0: 0, 1: 0, **{c: 1 for c in range(2, 14)}}
    scidb_to_3cls = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2}

    per_seed_ext = []
    for seed in [0, 1, 2]:
        s_info = next(r for r in s2_final['逐种子明细'] if r['seed'] == seed)
        vp = s_info['val_calibrated_params']
        tau_h, tau_m, gamma, beta, fg_thr = vp['tau_h_std'], vp['tau_maha_z'], vp['gamma'], vp['beta'], vp['fg_thr']

        blob = torch.load(CKPT_DIR / f'cggar_v2_seed{seed}.pt', map_location='cpu', weights_only=False)
        rel_head = TargetContextRelationalHead().to(device).eval(); rel_head.load_state_dict(blob['rel_head'])
        m_cfg = blob['mahalanobis']
        T_DET, T_DINO, T_REL = blob['temps']['T_DET'], blob['temps']['T_DINO'], blob['temps']['T_REL']

        w_yolo = Path(r'D:\citrus_runs') / f'yolov8n_unified_seed{seed}' / 'weights' / 'best.pt'
        yolo_ext = YoloFullStateExtractor(w_yolo, device)
        dino_ext = DinoExtractor(seed, device)

        raw_preds = []
        for start in range(0, len(coco_images), 16):
            grp = coco_images[start:start + 16]
            bgrs = [cv2.imdecode(np.fromfile(str(img_paths[it['id']]), dtype=np.uint8), 1) for it in grp]
            res_b = yolo_ext.yolo.predict(bgrs, imgsz=640, conf=0.05, iou=0.7, max_det=300, device=0, half=False, rect=False, verbose=False)
            for it, r in zip(grp, res_b):
                for bb, sc, cl in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.conf.cpu().tolist(), r.boxes.cls.cpu().tolist()):
                    x1, y1, x2, y2 = bb
                    raw_preds.append({'image_id': it['id'], 'category_id': int(cl), 'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': sc})

        g_pos = defaultdict(list)
        for pos, r in enumerate(raw_preds):
            g_pos[r['image_id']].append(pos)

        n_c = len(raw_preds)
        p14_arr = np.zeros((n_c, 14), dtype=np.float32)
        zneck_arr = np.zeros((n_c, 192), dtype=np.float32)
        geom_arr = np.zeros((n_c, 9), dtype=np.float32)
        c_core_arr = np.zeros((n_c, 3, 224, 224), dtype=np.uint8)
        c_std_arr = np.zeros((n_c, 3, 224, 224), dtype=np.uint8)
        c_ctx_arr = np.zeros((n_c, 3, 224, 224), dtype=np.uint8)

        ordered_iids = list(g_pos.keys())
        for start in range(0, len(ordered_iids), 16):
            sub_iids = ordered_iids[start:start + 16]
            bgrs, bxs_b, tcls_b, pos_b = [], [], [], []
            for iid in sub_iids:
                bgr = cv2.imdecode(np.fromfile(str(img_paths[iid]), dtype=np.uint8), 1)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                plist = g_pos[iid]
                bl = [raw_preds[p]['bbox'] for p in plist]
                cl = [raw_preds[p]['category_id'] for p in plist]
                for p, bb in zip(plist, bl):
                    geom_arr[p] = compute_box_geom(bb, w, h, bl)
                    c_core_arr[p] = crop_square_scale(rgb, bb, 0.82)
                    c_std_arr[p] = crop_square_scale(rgb, bb, 1.25)
                    c_ctx_arr[p] = crop_square_scale(rgb, bb, 2.50)
                bgrs.append(bgr); bxs_b.append(bl); tcls_b.append(cl); pos_b.extend(plist)
            ps, zs = yolo_ext.extract_batch(bgrs, bxs_b, tcls_b)
            for k, p_idx in enumerate(pos_b):
                p14_arr[p_idx] = ps[k]; zneck_arr[p_idx] = zs[k]

        zc, lgc = dino_ext.encode_crops(c_core_arr)
        zs, lgs = dino_ext.encode_crops(c_std_arr)
        zx, _ = dino_ext.encode_crops(c_ctx_arr)
        yolo_ext.close(); del yolo_ext, dino_ext; torch.cuda.empty_cache()

        p14_gpu = torch.from_numpy(p14_arr).to(device)
        geom_gpu = torch.from_numpy(geom_arr).to(device)
        zc_gpu = torch.from_numpy(zc.astype(np.float32)).to(device)
        zs_gpu = torch.from_numpy(zs.astype(np.float32)).to(device)
        zx_gpu = torch.from_numpy(zx.astype(np.float32)).to(device)
        lgc_gpu = torch.from_numpy(lgc).to(device)
        lgs_gpu = torch.from_numpy(lgs).to(device)
        lgr_gpu = rel_head(zc_gpu, zs_gpu, zx_gpu, geom_gpu)

        maha_14 = compute_maha_z(zneck_arr, m_cfg['means'], m_cfg['prec'], m_cfg['m_mu'], m_cfg['m_sd'])
        lp_det = F.log_softmax(torch.logit(p14_gpu.clamp(1e-5, 1.0 - 1e-5)) / T_DET, dim=-1)
        lp_std = F.log_softmax(lgs_gpu / T_DINO, dim=-1)
        lp_core = F.log_softmax(lgc_gpu / T_DINO, dim=-1)
        lp_rel = F.log_softmax(lgr_gpu / T_REL, dim=-1)
        p_det, p_std, p_core, p_rel = lp_det.exp(), lp_std.exp(), lp_core.exp(), lp_rel.exp()

        h_det = entropy_norm(p_det).squeeze(-1).cpu().numpy()
        h_std = entropy_norm(p_std).squeeze(-1).cpu().numpy()
        h_core = entropy_norm(p_core).squeeze(-1).cpu().numpy()
        h_rel = entropy_norm(p_rel).squeeze(-1).cpu().numpy()
        c_hat = p_det.argmax(dim=-1).cpu().numpy()
        d_maha_z = maha_14[np.arange(n_c), c_hat]
        dis_det_std = (c_hat != p_std.argmax(dim=-1).cpu().numpy())

        yolo_sc = np.array([r['score'] for r in raw_preds], dtype=np.float64)
        yolo_cls = np.array([r['category_id'] for r in raw_preds], dtype=np.int64)
        fg = yolo_sc >= fg_thr
        esc_m = fg & (dis_det_std | (h_std > tau_h) | (d_maha_z > tau_m))

        p_ours = compute_analytic_opinion_pool(
            lp_det.cpu().numpy(), lp_std.cpu().numpy(), lp_core.cpu().numpy(), lp_rel.cpu().numpy(),
            h_det, h_std, h_core, h_rel, d_maha_z, esc_m, gamma=gamma
        )
        p_std_raw = F.softmax(lgs_gpu, dim=-1).cpu().numpy()
        p_b3 = 0.65 * p_std_raw.copy()
        p_b3[np.arange(n_c), yolo_cls] += 0.35 * yolo_sc

        def eval_ext_cfg(active_m, prob_mat, calib: bool, use_nms: bool):
            rows_14 = [dict(r) for r in raw_preds]
            for i in range(n_c):
                if active_m[i]:
                    pr = prob_mat[i]
                    nc = int(np.argmax(pr))
                    oc = rows_14[i]['category_id']
                    rows_14[i]['category_id'] = nc
                    if calib:
                        rows_14[i]['score'] = float((yolo_sc[i] ** beta) * (max(float(pr[nc]), 1e-6) ** (1.0 - beta)))
                    elif nc != oc:
                        rows_14[i]['score'] = float(math.sqrt(yolo_sc[i] * max(float(pr[nc]), 1e-6)))
            if use_nms:
                rows_14 = apply_classwise_nms(rows_14, 0.70)

            out_m = {}
            for tag_g, gp, cmap in [('det_2cls', gt2_p, scidb_to_2cls), ('det_3cls', gt3_p, scidb_to_3cls)]:
                mapped = [{'image_id': r['image_id'], 'category_id': cmap[r['category_id']], 'bbox': r['bbox'], 'score': r['score']}
                          for r in rows_14 if r['category_id'] in cmap]
                with contextlib.redirect_stdout(io.StringIO()):
                    gt = COCO(str(gp))
                    ev = COCOeval_faster(gt, gt.loadRes(mapped), 'bbox')
                    ev.evaluate(); ev.accumulate(); ev.summarize()
                out_m[tag_g] = {'AP50': round(ev.stats[1] * 100.0, 2), 'AP50_95': round(ev.stats[0] * 100.0, 2)}
            hi = yolo_sc >= 0.25
            out_m['esc_hi_pct'] = round(float((esc_m & hi).sum() / max(1, hi.sum())) * 100.0, 2)
            return out_m

        m_none = np.zeros(n_c, dtype=bool)
        m_all = np.ones(n_c, dtype=bool)
        sr = {
            'seed': seed,
            'B0_YOLOv8n_Native': eval_ext_cfg(m_none, p_std_raw, False, False),
            'B1_Full_DINOv2_Hard': eval_ext_cfg(m_all, p_std_raw, False, False),
            'B3_Static_Posterior_Fuse': eval_ext_cfg(m_all, p_b3, False, False),
            'Ours_TCDMGE_Selective': eval_ext_cfg(fg, p_ours, True, True),
        }
        print(f"  [外部果园盲测 Seed {seed}] 2cls AP50: 原生={sr['B0_YOLOv8n_Native']['det_2cls']['AP50']:.2f}% | "
              f"Full-DINO={sr['B1_Full_DINOv2_Hard']['det_2cls']['AP50']:.2f}% | "
              f"静态融合={sr['B3_Static_Posterior_Fuse']['det_2cls']['AP50']:.2f}% | "
              f"★Ours(TCD-MGE)={sr['Ours_TCDMGE_Selective']['det_2cls']['AP50']:.2f}% (升级率{sr['Ours_TCDMGE_Selective']['esc_hi_pct']:.1f}%) || "
              f"3cls AP50: Full-DINO={sr['B1_Full_DINOv2_Hard']['det_3cls']['AP50']:.2f}% -> ★Ours={sr['Ours_TCDMGE_Selective']['det_3cls']['AP50']:.2f}%",
              flush=True)
        per_seed_ext.append(sr)
    return per_seed_ext


@torch.no_grad()
def benchmark_single_pass_tcdmge_latency(s2_final):
    device = 'cuda:0'
    s0 = next(r for r in s2_final['逐种子明细'] if r['seed'] == 0)
    vp = s0['val_calibrated_params']
    tau_h, tau_m, fg_thr = vp['tau_h_std'], vp['tau_maha_z'], vp['fg_thr']

    blob = torch.load(CKPT_DIR / 'cggar_v2_seed0.pt', map_location='cpu', weights_only=False)
    rel_head = TargetContextRelationalHead().to(device).eval(); rel_head.load_state_dict(blob['rel_head'])
    m_cfg = blob['mahalanobis']
    m_means_g = torch.from_numpy(m_cfg['means'].astype(np.float32)).to(device)
    m_prec_g = torch.from_numpy(m_cfg['prec'].astype(np.float32)).to(device)
    m_mu, m_sd = float(m_cfg['m_mu']), float(m_cfg['m_sd'])
    T_DET, T_DINO = blob['temps']['T_DET'], blob['temps']['T_DINO']

    yolo = YOLO(r'D:\citrus_runs\yolov8n_unified_seed0\weights\best.pt')
    net = yolo.model.to(device).eval()
    detect = net.model[-1]
    maps = [None, None, None]
    raw_y = [None]
    hooks = [detect.cv3[s][1].register_forward_hook(lambda _m, _i, o, idx=s: maps.__setitem__(idx, o)) for s in range(3)]
    hooks.append(detect.register_forward_hook(lambda _m, _i, o: raw_y.__setitem__(0, o[0])))
    dino_ext = DinoExtractor(0, device)
    strides = [8.0, 16.0, 32.0]

    val_imgs = sorted((IMG_ROOT / 'val').glob('*.jpg'))[:35]
    tc_imgs = sorted((IMG_ROOT / 'testC').glob('*.*'))[:35]

    def run_mode(paths: list, mode: str):
        times = []
        for idx, p in enumerate(paths):
            torch.cuda.synchronize()
            ts = time.perf_counter()
            bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), 1)
            res = yolo.predict([bgr], imgsz=640, conf=0.05, iou=0.7, max_det=300, device=0, half=False, rect=False, verbose=False)[0]
            xyxy = res.boxes.xyxy.cpu().numpy()
            sc_np = res.boxes.conf.cpu().numpy()

            if mode == 'full_dino' and len(xyxy) > 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                bxs = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in xyxy]
                cs = np.stack([crop_square_scale(rgb, bb, 1.25) for bb in bxs], 0)
                x = (torch.from_numpy(cs).to(device).float() / 255.0 - dino_ext.mean) / dino_ext.std
                z = F.normalize(dino_ext.enc.forward_features(x)[:, 0], dim=1)
                _ = dino_ext.head(z)
            elif mode == 'tcdmge' and len(xyxy) > 0:
                fg_ids = np.where(sc_np >= fg_thr)[0]
                if len(fg_ids) > 0:
                    h, w = bgr.shape[:2]
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    _, r_lb, px_lb, py_lb = letterbox_640(bgr)
                    yb = raw_y[0][0]
                    acx, acy, abw, abh = yb[0], yb[1], yb[2], yb[3]
                    ax1 = (acx - abw * 0.5 - px_lb) / r_lb
                    ay1 = (acy - abh * 0.5 - py_lb) / r_lb
                    aw, ah = abw / r_lb, abh / r_lb
                    sc14 = yb[4:]

                    bxs_all = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in xyxy]
                    bxs_fg = [bxs_all[i] for i in fg_ids]
                    a_idxs, rois = [], []
                    for bb in bxs_fg:
                        d_anc = (ax1 - bb[0]).abs() + (ay1 - bb[1]).abs() + (aw - bb[2]).abs() + (ah - bb[3]).abs()
                        a_idxs.append(torch.argmin(d_anc))
                        lx1 = float(np.clip(bb[0] * r_lb + px_lb, 0.0, 639.0))
                        ly1 = float(np.clip(bb[1] * r_lb + py_lb, 0.0, 639.0))
                        lx2 = float(np.clip((bb[0] + bb[2]) * r_lb + px_lb, lx1 + 1.0, 640.0))
                        ly2 = float(np.clip((bb[1] + bb[3]) * r_lb + py_lb, ly1 + 1.0, 640.0))
                        rois.append([0.0, lx1, ly1, lx2, ly2])

                    p14_g = sc14[:, torch.stack(a_idxs)].T
                    rt = torch.tensor(rois, dtype=torch.float32, device=device)
                    sp_f = [F.normalize(roi_align(maps[s].float(), rt, (1, 1), 1.0 / strides[s], aligned=True).flatten(1), dim=1) for s in range(3)]
                    zn_g = F.normalize(torch.cat(sp_f, dim=1), dim=1)

                    cs = np.stack([crop_square_scale(rgb, bb, 1.25) for bb in bxs_fg], 0)
                    xs = (torch.from_numpy(cs).to(device).float() / 255.0 - dino_ext.mean) / dino_ext.std
                    zs_g = F.normalize(dino_ext.enc.forward_features(xs)[:, 0], dim=1)
                    lgs_g = dino_ext.head(zs_g)

                    lp_d = F.log_softmax(torch.logit(p14_g.clamp(1e-5, 1.0 - 1e-5)) / T_DET, dim=-1)
                    lp_s = F.log_softmax(lgs_g / T_DINO, dim=-1)
                    pd, ps = lp_d.exp(), lp_s.exp()
                    hs = entropy_norm(ps).squeeze(-1)
                    ch = pd.argmax(dim=-1)
                    diff_m = zn_g - m_means_g[ch]
                    dm_z = (torch.sqrt(torch.clamp((diff_m @ m_prec_g * diff_m).sum(dim=-1), min=0.0)) - m_mu) / m_sd
                    esc = (ch != ps.argmax(dim=-1)) | (hs > tau_h) | (dm_z > tau_m)
                    e_ids = torch.where(esc)[0].cpu().numpy()

                    if len(e_ids) > 0:
                        geoms = torch.from_numpy(np.stack([compute_box_geom(bxs_fg[j], w, h, bxs_all) for j in e_ids], 0)).to(device)
                        c_cr = [crop_square_scale(rgb, bxs_fg[j], 0.82) for j in e_ids]
                        c_cx = [crop_square_scale(rgb, bxs_fg[j], 2.50) for j in e_ids]
                        x_pair = (torch.from_numpy(np.stack(c_cr + c_cx, 0)).to(device).float() / 255.0 - dino_ext.mean) / dino_ext.std
                        z_pair = F.normalize(dino_ext.enc.forward_features(x_pair)[:, 0], dim=1)
                        k = len(e_ids)
                        zc_sub, zx_sub = z_pair[:k], z_pair[k:]
                        _ = dino_ext.head(zc_sub)
                        _ = rel_head(zc_sub, zs_g[e_ids], zx_sub, geoms)

            torch.cuda.synchronize()
            ms = (time.perf_counter() - ts) * 1000.0
            if idx >= 5:
                times.append(ms)
        return {'mean_ms': round(float(np.mean(times)), 2), 'p95_ms': round(float(np.percentile(times, 95)), 2)}

    _ = run_mode(val_imgs[:8], 'tcdmge')
    lat = {
        'native_yolov8n': {'val': run_mode(val_imgs, 'native'), 'testC': run_mode(tc_imgs, 'native')},
        'full_dinov2': {'val': run_mode(val_imgs, 'full_dino'), 'testC': run_mode(tc_imgs, 'full_dino')},
        'ours_tcdmge': {'val': run_mode(val_imgs, 'tcdmge'), 'testC': run_mode(tc_imgs, 'tcdmge')},
    }
    for h in hooks:
        h.remove()
    return lat


def generate_final_figures(s2_final, ext_agg, lat):
    plt.rcParams['font.sans-serif'] = ['Arial', 'Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    agg = s2_final['三种子汇总']

    # Figure 1: Seed-by-Seed Strict 3/3 Dominance (Individual Seed Markers + Mean Bars) on TestB & TestC
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.4), dpi=300)
    keys = [
        'B0_YOLOv8n_Native', 'B1a_Full_DINOv2_GrayPad_Orig', 'B2_MGS_DINOv2_v1',
        'B3_Static_Posterior_Fuse', 'Abl1_SingleCrop_14Cls_Calib', 'Abl2_Disagreement_Esc_Only', 'Ours_TCDMGE_Selective'
    ]
    lbls = [
        'YOLOv8n\nNative', 'Full DINOv2\n(Hard Overwrite)', 'MGS+DINOv2\n(v1)',
        'Static\nPosterior Fuse', 'Abl-1: 14Cls\n+ScoreCalib', 'Abl-2: Disagree\nEsc Only', 'Ours: TCD-MGE\n(Tri-Signal Esc)'
    ]
    cols = ['#95a5a6', '#e74c3c', '#f39c12', '#5dade2', '#85c1e9', '#76d7c4', '#2ecc71']
    x = np.arange(len(keys))

    for ax, sp, title_sp, ymin, ymax in [
        (ax1, 'testB', '(a) Complex Orchard Shift (TestB) — 3-Seed Individual & Mean', 65, 84.5),
        (ax2, 'testC', '(b) Illumination/Board Shift (TestC) — 3-Seed Individual & Mean', 70, 89.8),
    ]:
        means = [agg[k][sp]['AP50_95_mean'] for k in keys]
        bars = ax.bar(x, means, width=0.56, color=cols, edgecolor='black', linewidth=0.7, alpha=0.88)
        seed_markers = ['o', 's', '^']
        seed_cols = ['#1b2631', '#283747', '#6c3483']
        for s_idx in range(3):
            ys = [agg[k][sp]['seeds'][s_idx] for k in keys]
            ax.plot(x + (s_idx - 1) * 0.13, ys, linestyle='none', marker=seed_markers[s_idx],
                    color=seed_cols[s_idx], markersize=5.8, label=f'Seed {s_idx}' if ax is ax1 else None, zorder=5)
        for idx_k, (b, m) in enumerate(zip(bars, means)):
            top_y = max(m, max(agg[keys[idx_k]][sp]['seeds']))
            ax.text(b.get_x() + b.get_width() / 2, top_y + 0.42, f"{m:.2f}%", ha='center', va='bottom', fontsize=8.4, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(lbls, fontsize=8.0)
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel('Strict FP32 COCO AP50-95 (%)', fontsize=10.5)
        ax.set_title(title_sp, fontsize=11.0, fontweight='bold')
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        if ax is ax1:
            ax.legend(loc='upper left', fontsize=8.5, title='Individual Seeds')

    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig1_tcdmge_3seed_strict_dominance.png', dpi=300)
    plt.close(fig)

    # Figure 2: Unseen External Orchard Blind Test (3/3 Seed Win) + Self-Adaptive Multi-Granularity Escalation & Real Latency
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.2), dpi=300)
    ekeys = ['B0_YOLOv8n_Native', 'B1_Full_DINOv2_Hard', 'B3_Static_Posterior_Fuse', 'Ours_TCDMGE_Selective']
    elbls = ['YOLOv8n\nNative', 'Full DINOv2\n(Strong Baseline)', 'Static\nPosterior Fuse', 'Ours: TCD-MGE\n(Selective Esc)']
    xe = np.arange(len(ekeys))
    ap2_m = [float(np.mean(ext_agg[k]['det_2cls_AP50_seeds'])) for k in ekeys]
    ap3_m = [float(np.mean(ext_agg[k]['det_3cls_AP50_seeds'])) for k in ekeys]

    ax1.bar(xe - 0.18, ap2_m, width=0.34, label='2-Class Orchard Symptom AP50', color='#3498db', edgecolor='black', linewidth=0.7)
    ax1.bar(xe + 0.18, ap3_m, width=0.34, label='3-Class Fine Topology AP50', color='#2ecc71', edgecolor='black', linewidth=0.7)
    for i, (v2, v3) in enumerate(zip(ap2_m, ap3_m)):
        ax1.text(i - 0.18, v2 + 0.35, f"{v2:.2f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
        ax1.text(i + 0.18, v3 + 0.35, f"{v3:.2f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax1.set_xticks(xe)
    ax1.set_xticklabels(elbls, fontsize=8.8)
    ax1.set_ylim(20, 48.5)
    ax1.set_ylabel('Zero-Shot COCO AP50 (%)', fontsize=10.5)
    ax1.set_title('(a) Unseen Orchard Blind Test (Zenodo, 519 Imgs / 1765 Boxes)', fontsize=10.8, fontweight='bold')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax1.legend(loc='upper left', fontsize=8.5)

    doms = ['SciDB Val\n(In-Domain)', 'SciDB TestB\n(Orchard Shift)', 'SciDB TestC\n(Board Shift)', 'External Zenodo\n(Unseen Orchard)']
    esc_rates = [
        agg['Ours_TCDMGE_Selective']['val']['esc_hi_pct_mean'],
        agg['Ours_TCDMGE_Selective']['testB']['esc_hi_pct_mean'],
        agg['Ours_TCDMGE_Selective']['testC']['esc_hi_pct_mean'],
        ext_agg['Ours_TCDMGE_Selective']['esc_hi_pct_mean'],
    ]
    xd = np.arange(len(doms))
    bars2 = ax2.bar(xd, esc_rates, width=0.48, color=['#85c1e9', '#f5b041', '#eb984e', '#e74c3c'], edgecolor='black', linewidth=0.7)
    for b, vr in zip(bars2, esc_rates):
        ax2.text(b.get_x() + b.get_width() / 2, vr + 1.2, f"{vr:.1f}%", ha='center', va='bottom', fontsize=9.0, fontweight='bold')
    ax2.set_xticks(xd)
    ax2.set_xticklabels(doms, fontsize=8.8)
    ax2.set_ylim(0, max(esc_rates) + 16)
    ax2.set_ylabel('Level-2 Multi-Granularity Escalation Rate (%)', fontsize=10.2)
    ax2.set_title(f"(b) Auto-Escalation & Real Latency (Val: {lat['ours_tcdmge']['val']['mean_ms']}ms | Full-DINO: {lat['full_dinov2']['val']['mean_ms']}ms)", fontsize=9.8, fontweight='bold')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig2_tcdmge_external_blind_and_escalation.png', dpi=300)
    plt.close(fig)


def main():
    t0 = time.monotonic()
    configure_strict_fp32()
    s2_final = json.loads((RES_DIR / 'tcd_mge_scidb_3seed_final.json').read_text(encoding='utf-8'))

    print("[1/3] 执行外部真实果园 519 图 / 1,765 框一次性零样本封口盲测 (TCD-MGE)...", flush=True)
    coco_imgs, img_paths, gt2_p, gt3_p = build_external_zenodo_gt()
    ext_seed_res = evaluate_external_tcdmge_3seeds(coco_imgs, img_paths, gt2_p, gt3_p, s2_final)

    ext_agg = {}
    for m in ['B0_YOLOv8n_Native', 'B1_Full_DINOv2_Hard', 'B3_Static_Posterior_Fuse', 'Ours_TCDMGE_Selective']:
        ap2 = [r[m]['det_2cls']['AP50'] for r in ext_seed_res]
        ap2_95 = [r[m]['det_2cls']['AP50_95'] for r in ext_seed_res]
        ap3 = [r[m]['det_3cls']['AP50'] for r in ext_seed_res]
        ap3_95 = [r[m]['det_3cls']['AP50_95'] for r in ext_seed_res]
        escs = [r[m]['esc_hi_pct'] for r in ext_seed_res]
        ext_agg[m] = {
            'det_2cls_AP50': f"{np.mean(ap2):.2f}±{np.std(ap2, ddof=1):.2f}",
            'det_2cls_AP50_seeds': ap2,
            'det_2cls_AP50_95': f"{np.mean(ap2_95):.2f}±{np.std(ap2_95, ddof=1):.2f}",
            'det_3cls_AP50': f"{np.mean(ap3):.2f}±{np.std(ap3, ddof=1):.2f}",
            'det_3cls_AP50_seeds': ap3,
            'det_3cls_AP50_95': f"{np.mean(ap3_95):.2f}±{np.std(ap3_95, ddof=1):.2f}",
            'esc_hi_pct_mean': round(float(np.mean(escs)), 2),
        }

    # Check strict 3/3 seed wins on external blind orchard dataset!
    print("\n================ 外部真实果园盲测 3/3 种子全胜核验 ================", flush=True)
    for base in ['B0_YOLOv8n_Native', 'B1_Full_DINOv2_Hard', 'B3_Static_Posterior_Fuse']:
        d2 = [round(ext_agg['Ours_TCDMGE_Selective']['det_2cls_AP50_seeds'][s] - ext_agg[base]['det_2cls_AP50_seeds'][s], 2) for s in range(3)]
        d3 = [round(ext_agg['Ours_TCDMGE_Selective']['det_3cls_AP50_seeds'][s] - ext_agg[base]['det_3cls_AP50_seeds'][s], 2) for s in range(3)]
        print(f"  [Ours vs {base:25s}] 2cls AP50 逐种子差={d2} (3/3胜={all(x>0 for x in d2)}) | "
              f"3cls AP50 逐种子差={d3} (3/3胜={all(x>0 for x in d3)})", flush=True)

    print("\n[2/3] 测量单趟零冗余 RTX 4060 Ti 端到端单图延迟 (ms)...", flush=True)
    lat = benchmark_single_pass_tcdmge_latency(s2_final)
    print("  单图实测延迟:", json.dumps(lat, ensure_ascii=False), flush=True)

    print("\n[3/3] 生成 300 DPI 论文配图...", flush=True)
    generate_final_figures(s2_final, ext_agg, lat)

    out_p = RES_DIR / 'tcd_mge_external_and_latency_final.json'
    out_p.write_text(json.dumps({
        '实验名称': 'TCD-MGE_外部真实果园封口盲测与单趟实时延迟最终核验',
        '耗时秒': round(time.monotonic() - t0, 2),
        '实测单图延迟_ms': lat,
        '外部果园盲测汇总_3种子': ext_agg,
        '外部果园逐种子明细': ext_seed_res,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n[完成] 已保存至: {out_p}", flush=True)


if __name__ == '__main__':
    main()
