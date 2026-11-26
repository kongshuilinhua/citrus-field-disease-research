# -*- coding: utf-8 -*-
"""
Phase 2B (完全冻结历史权重补充对账与零冗余公平测速):
在不修改任何已锁定权重 (cggar_v2_seed{0,1,2}.pt 与 bgopta_frozen_seed{0,1,2}.pt) 与任何阈值的前提下：
1. 消除测速脚本中 `_, r_lb, px_lb, py_lb = letterbox_640(bgr)` 丢弃返回值却白做一次 4K CPU 缩放的测量瑕疵，
   在同一进程严格 FP32 下公平实测全部 7 档推理路径的真实单图延迟、P95 延迟与每图 DINOv2 前向次数：
   - Tier 1: Native YOLOv8n
   - Tier 2: Full DINOv2 (1-Crop Hard)
   - Tier 3a: Single-Pass BG-OPTA (1.50x Full)
   - Tier 3b: Single-Pass BG-OPTA + Safe-Bypass Gate
   - Tier 4a: Disagreement-Only Escalation (Abl2: 零阈值参数跨视角分歧按需升级)
   - Tier 4b: Tri-Signal TCD-MGE (v2)
   - Tier 5: Full 3-Crop Static Differential (Abl3: 100% 三视角重裁上限参考)
2. 在完全相同的：
   - 修复 9-13 类漏映射后的 Zenodo 11148332 (519图/1,765框，100% 预测框零丢弃)
   - CitrusUAT 受控灰底跨域 5 类诊断 (443图)
   - 真正零接触最终封口盲测集 Sweet Orange V1 (1,389图)
   上补齐已冻结多粒度系列 (Abl2 分歧按需升级 / Abl3 静态三视角 / TCD-MGE v2) 的 3 随机种子结果，
   实现“单趟空间 Token 正交剥离 (BG-OPTA)”与“输入级多粒度隔离 + 流形衰减意见池 (TCD-MGE / Abl2)”的 100% 同口径完整科学对照！
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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torchvision.ops import roi_align
from ultralytics import YOLO

ROOT = Path(r'D:\新版柑橘')
EXT = ROOT / '方法路线扩展_20261031'
WORK_DIR = ROOT / 'Antigravity_BGOPTA_单趟正交Token攻关_20261126'
PREV_DIR = ROOT / 'Antigravity_CGGAR_真创新攻关_20261125'

sys.path.append(str(EXT))
sys.path.append(str(WORK_DIR))
sys.path.append(str(PREV_DIR))

from 共同评测后端 import configure_strict_fp32
from phase1_source_only_train_and_select import BGOPTAHead, crop_single_context_and_masks, apply_classwise_nms, compute_maha_z, CROP_SCALE
from phase2_frozen_oneshot_eval import coco_eval_metrics
from step1_extract_multigranular_cache import YoloFullStateExtractor, DinoExtractor, crop_square_scale, compute_box_geom
from step2_train_and_eval_cggar import TargetContextRelationalHead, entropy_norm
from step2_v3_analytic_opinion_pool import compute_analytic_opinion_pool
from step3_external_blind_and_latency import build_external_zenodo_gt

CKPT_BG = WORK_DIR / 'frozen_models'
CKPT_TCD = PREV_DIR / 'models'
RES_DIR = WORK_DIR / 'results'
IMG_ROOT = Path(r'D:\病虫害论文\复现实验记录\yolo11n_public_compare_20261004\数据副本\images')
AUDIT_INDEX_JSON = Path(r'D:\病虫害论文\新增公开数据_20260916\中文分类与标签使用索引.json')
CITRUSUAT_DIR = ROOT / '复现_baseline' / 'field_test_CitrusUAT'


def fast_letterbox_params(h: int, w: int, new_shape: int = 640):
    """纯算术计算 Ultralytics 640 letterbox 缩放比与边距 (0.0001ms，消除重复 4K 图像缩放冗余)。"""
    r = min(new_shape / float(h), new_shape / float(w))
    nw, nh = int(round(w * r)), int(round(h * r))
    px = int(round((new_shape - nw) * 0.5 - 0.1))
    py = int(round((new_shape - nh) * 0.5 - 0.1))
    return r, float(px), float(py)


@torch.no_grad()
def benchmark_7tier_clean_latency():
    print("\n================ [1/4] RTX 4060 Ti 严格 FP32 七档零冗余公平单图延迟实测 ================", flush=True)
    device = 'cuda:0'
    ckpt0 = torch.load(CKPT_BG / 'bgopta_frozen_seed0.pt', map_location='cpu', weights_only=False)
    hp = ckpt0['val_frozen_hyperparams']
    fg_thr, sb_conf, sb_mar, sb_maha = hp['fg_thr'], hp['safe_bypass_conf'], hp['safe_bypass_margin'], hp['safe_bypass_maha_z']
    m_cfg = ckpt0['mahalanobis']
    m_means_g = torch.from_numpy(m_cfg['means'].astype(np.float32)).to(device)
    m_prec_g = torch.from_numpy(m_cfg['prec'].astype(np.float32)).to(device)
    m_mu, m_sd = float(m_cfg['m_mu']), float(m_cfg['m_sd'])

    blob_tcd = torch.load(CKPT_TCD / 'cggar_v2_seed0.pt', map_location='cpu', weights_only=False)
    rel_head = TargetContextRelationalHead().to(device).eval()
    rel_head.load_state_dict(blob_tcd['rel_head'])
    s2_tcd = json.loads((PREV_DIR / 'results' / 'tcd_mge_scidb_3seed_final.json').read_text(encoding='utf-8'))
    vp_tcd = next(r for r in s2_tcd['逐种子明细'] if r['seed'] == 0)['val_calibrated_params']

    dino_ext = DinoExtractor(0, device)
    bgopta_head = BGOPTAHead(dino_ext.head.weight.data.cpu(), dino_ext.head.bias.data.cpu(), mode='S3_bgopta_full').to(device).eval()
    bgopta_head.load_state_dict(ckpt0['S3_bgopta_full'])

    yolo = YOLO(r'D:\citrus_runs\yolov8n_unified_seed0\weights\best.pt')
    net = yolo.model.to(device).eval()
    detect = net.model[-1]
    maps = [None, None, None]
    raw_y = [None]
    hooks = [detect.cv3[s][1].register_forward_hook(lambda _m, _i, o, idx=s: maps.__setitem__(idx, o)) for s in range(3)]
    hooks.append(detect.register_forward_hook(lambda _m, _i, o: raw_y.__setitem__(0, o[0])))
    strides = [8.0, 16.0, 32.0]

    val_imgs = sorted((IMG_ROOT / 'val').glob('*.jpg'))[:55]
    tc_imgs = sorted((IMG_ROOT / 'testC').glob('*.*'))[:55]

    def extract_anchor_and_neck_fast(h: int, w: int, xyxy, fg_ids):
        r_lb, px_lb, py_lb = fast_letterbox_params(h, w, 640)
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
        return bxs_all, bxs_fg, p14_g, zn_g

    def run_tier(paths: list, tier: str):
        times, dino_calls = [], []
        for idx, p in enumerate(paths):
            torch.cuda.synchronize()
            ts = time.perf_counter()
            n_dino = 0
            bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), 1)
            h, w = bgr.shape[:2]
            res = yolo.predict([bgr], imgsz=640, conf=0.05, iou=0.7, max_det=300, device=0, half=False, rect=False, verbose=False)[0]
            xyxy = res.boxes.xyxy.cpu().numpy()
            sc_np = res.boxes.conf.cpu().numpy()

            if tier == 'Tier2_Full_DINOv2_1Crop' and len(xyxy) > 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                bxs = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in xyxy]
                cs = np.stack([crop_square_scale(rgb, bb, 1.25) for bb in bxs], 0)
                x = (torch.from_numpy(cs).to(device).float() / 255.0 - dino_ext.mean) / dino_ext.std
                z = F.normalize(dino_ext.enc.forward_features(x)[:, 0], dim=1)
                _ = dino_ext.head(z)
                n_dino = len(bxs)

            elif tier in ('Tier3a_BGOPTA_1Pass', 'Tier3b_BGOPTA_SafeBypass') and len(xyxy) > 0:
                fg_ids = np.where(sc_np >= fg_thr)[0]
                if len(fg_ids) > 0:
                    bxs_all, bxs_fg, p14_g, zn_g = extract_anchor_and_neck_fast(h, w, xyxy, fg_ids)
                    pd = F.softmax(torch.logit(p14_g.clamp(1e-5, 1.0 - 1e-5)) / 1.35, dim=-1)
                    if tier == 'Tier3b_BGOPTA_SafeBypass':
                        sc_fg_t = torch.from_numpy(sc_np[fg_ids]).to(device)
                        t2 = torch.topk(pd, 2, dim=-1).values
                        mar = t2[:, 0] - t2[:, 1]
                        ch = pd.argmax(dim=-1)
                        diff_m = zn_g - m_means_g[ch]
                        dm_z = (torch.sqrt(torch.clamp((diff_m @ m_prec_g * diff_m).sum(dim=-1), min=0.0)) - m_mu) / m_sd
                        bypass = (sc_fg_t >= sb_conf) & (mar >= sb_mar) & (dm_z <= sb_maha)
                        act_ids = torch.where(~bypass)[0].cpu().numpy()
                    else:
                        act_ids = np.arange(len(bxs_fg))
                    if len(act_ids) > 0:
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        cs, ms, gs = [], [], []
                        for j in act_ids:
                            c, m = crop_single_context_and_masks(rgb, bxs_fg[j], CROP_SCALE)
                            cs.append(c); ms.append(m)
                            gs.append(compute_box_geom(bxs_fg[j], w, h, bxs_all))
                        xt = (torch.from_numpy(np.stack(cs, 0)).to(device).float() / 255.0 - dino_ext.mean) / dino_ext.std
                        tok = dino_ext.enc.forward_features(xt)
                        _ = bgopta_head(tok, torch.from_numpy(np.stack(ms, 0)).to(device), torch.from_numpy(np.stack(gs, 0)).to(device))
                        n_dino = len(act_ids)

            elif tier in ('Tier4a_Disagreement_Esc_Abl2', 'Tier4b_TriSignal_TCDMGE_v2', 'Tier5_Full_3Crop_Static_Abl3') and len(xyxy) > 0:
                fg_ids = np.where(sc_np >= fg_thr)[0]
                if len(fg_ids) > 0:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    bxs_all, bxs_fg, p14_g, zn_g = extract_anchor_and_neck_fast(h, w, xyxy, fg_ids)
                    if tier == 'Tier5_Full_3Crop_Static_Abl3':
                        geoms = torch.from_numpy(np.stack([compute_box_geom(bb, w, h, bxs_all) for bb in bxs_fg], 0)).to(device)
                        c_all = ([crop_square_scale(rgb, bb, 0.82) for bb in bxs_fg] +
                                 [crop_square_scale(rgb, bb, 1.25) for bb in bxs_fg] +
                                 [crop_square_scale(rgb, bb, 2.50) for bb in bxs_fg])
                        x3 = (torch.from_numpy(np.stack(c_all, 0)).to(device).float() / 255.0 - dino_ext.mean) / dino_ext.std
                        z3 = F.normalize(dino_ext.enc.forward_features(x3)[:, 0], dim=1)
                        k = len(bxs_fg)
                        _ = rel_head(z3[:k], z3[k:2 * k], z3[2 * k:], geoms)
                        n_dino = 3 * k
                    else:
                        cs = np.stack([crop_square_scale(rgb, bb, 1.25) for bb in bxs_fg], 0)
                        xs = (torch.from_numpy(cs).to(device).float() / 255.0 - dino_ext.mean) / dino_ext.std
                        zs_g = F.normalize(dino_ext.enc.forward_features(xs)[:, 0], dim=1)
                        lgs_g = dino_ext.head(zs_g)
                        n_dino = len(bxs_fg)
                        pd = F.softmax(torch.logit(p14_g.clamp(1e-5, 1.0 - 1e-5)) / 1.35, dim=-1)
                        ps = F.softmax(lgs_g / 1.15, dim=-1)
                        ch = pd.argmax(dim=-1)
                        if tier == 'Tier4a_Disagreement_Esc_Abl2':
                            esc = (ch != ps.argmax(dim=-1))
                        else:
                            hs = entropy_norm(ps).squeeze(-1)
                            diff_m = zn_g - m_means_g[ch]
                            dm_z = (torch.sqrt(torch.clamp((diff_m @ m_prec_g * diff_m).sum(dim=-1), min=0.0)) - m_mu) / m_sd
                            esc = (ch != ps.argmax(dim=-1)) | (hs > vp_tcd['tau_h_std']) | (dm_z > vp_tcd['tau_maha_z'])
                        e_ids = torch.where(esc)[0].cpu().numpy()
                        if len(e_ids) > 0:
                            geoms = torch.from_numpy(np.stack([compute_box_geom(bxs_fg[j], w, h, bxs_all) for j in e_ids], 0)).to(device)
                            c_pair = [crop_square_scale(rgb, bxs_fg[j], 0.82) for j in e_ids] + [crop_square_scale(rgb, bxs_fg[j], 2.50) for j in e_ids]
                            xp = (torch.from_numpy(np.stack(c_pair, 0)).to(device).float() / 255.0 - dino_ext.mean) / dino_ext.std
                            zp = F.normalize(dino_ext.enc.forward_features(xp)[:, 0], dim=1)
                            ke = len(e_ids)
                            _ = rel_head(zp[:ke], zs_g[e_ids], zp[ke:], geoms)
                            n_dino += 2 * ke

            torch.cuda.synchronize()
            ms_elapsed = (time.perf_counter() - ts) * 1000.0
            if idx >= 5:
                times.append(ms_elapsed)
                dino_calls.append(n_dino)
        return {
            'mean_ms': round(float(np.mean(times)), 2),
            'p95_ms': round(float(np.percentile(times, 95)), 2),
            'fps': round(1000.0 / float(np.mean(times)), 1),
            'avg_dino_forwards_per_img': round(float(np.mean(dino_calls)), 3),
        }

    _ = run_tier(val_imgs[:8], 'Tier5_Full_3Crop_Static_Abl3')
    tiers = [
        'Tier1_Native_YOLOv8n',
        'Tier2_Full_DINOv2_1Crop',
        'Tier3a_BGOPTA_1Pass',
        'Tier3b_BGOPTA_SafeBypass',
        'Tier4a_Disagreement_Esc_Abl2',
        'Tier4b_TriSignal_TCDMGE_v2',
        'Tier5_Full_3Crop_Static_Abl3',
    ]
    out = {}
    for tr in tiers:
        rv = run_tier(val_imgs, tr)
        rc = run_tier(tc_imgs, tr)
        out[tr] = {'val': rv, 'testC': rc}
        print(f"  [{tr:30s}] val: {rv['mean_ms']:5.2f}ms (P95={rv['p95_ms']:5.2f}ms, DINO={rv['avg_dino_forwards_per_img']:.2f}/图) | "
              f"testC: {rc['mean_ms']:5.2f}ms (P95={rc['p95_ms']:5.2f}ms, DINO={rc['avg_dino_forwards_per_img']:.2f}/图)", flush=True)

    for hk in hooks:
        hk.remove()
    del yolo, dino_ext, bgopta_head, rel_head
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def eval_multigranular_frozen_on_all_externals():
    device = 'cuda:0'
    s2_final = json.loads((PREV_DIR / 'results' / 'tcd_mge_scidb_3seed_final.json').read_text(encoding='utf-8'))

    # Verify frozen cggar_v2_seed{0,1,2}.pt SHA-256 matches GitHub manifest!
    expected_shas = {
        0: 'b277920fab8911f2cbee33bbd2ec4044d3f2fcf60d68584db9b751c5b264e49a',
        1: '47842e4831017911daa4197b1b4442a16bb2f8d28478e747bd1e00d84c2e962b',
        2: '67557b6768837380474b93249e81fc0e7e8870fb8b0530d9225c0edde770c7ae',
    }
    for s, sh in expected_shas.items():
        actual = hashlib.sha256((CKPT_TCD / f'cggar_v2_seed{s}.pt').read_bytes()).hexdigest()
        assert actual == sh, f"cggar_v2_seed{s}.pt SHA mismatch!"
    print("[门禁核验通过] 历史冻结权重 cggar_v2_seed{0,1,2}.pt SHA-256 与 GitHub 清单 100% 一致！", flush=True)

    # 1. Prepare Zenodo 11148332 (Repaired Zero-Drop)
    coco_images, img_paths, gt2_p, gt3_p = build_external_zenodo_gt()
    scidb_to_2cls_full = {0: 0, 1: 0, **{c: 1 for c in range(2, 14)}}
    scidb_to_3cls_repaired = {0: 0, 1: 0, 2: 1, 3: 1, **{c: 2 for c in range(4, 14)}}

    # 2. Prepare CitrusUAT (443 images)
    uat_classes = ['Greasy_spot', 'Healthy', 'HLB', 'Mg', 'Zn']
    uat_groups = [[10], [0, 1], [2, 3], [7], [4]]
    uat_samples = []
    for ci, cname in enumerate(uat_classes):
        for p in sorted((CITRUSUAT_DIR / cname).glob('*.*')):
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                uat_samples.append((ci, p))

    def dec_img(p):
        bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), 1)
        if max(bgr.shape[:2]) > 1280:
            f = 1280.0 / max(bgr.shape[:2])
            bgr = cv2.resize(bgr, (max(1, int(round(bgr.shape[1] * f))), max(1, int(round(bgr.shape[0] * f)))), interpolation=cv2.INTER_AREA)
        return bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    uat_dec = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for (ci, _), (b, r) in zip(uat_samples, ex.map(lambda x: dec_img(x[1]), uat_samples)):
            uat_dec.append((ci, b, r))
    y_uat = np.array([x[0] for x in uat_dec], dtype=np.int64)

    # 3. Prepare Sweet Orange V1 Vault (1,389 images)
    idx_all = json.loads(AUDIT_INDEX_JSON.read_text(encoding='utf-8'))
    lbl_map = {'Healthy leaf': 0, 'Citrus canker': 1, 'Citrus greening': 2}
    sci_groups_v = [[0, 1], [11], [2, 3]]
    vault_recs = [
        r for r in idx_all
        if r.get('数据集') == 'f7cr74mwpj' and r.get('索引去重保留') and not r.get('同库跨标签冲突')
        and r.get('解码状态') == '通过' and r.get('原始标签') in lbl_map
    ]
    v_dec = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for r, (bgr, rgb) in zip(vault_recs, ex.map(lambda r: dec_img(Path(r['路径'])), vault_recs)):
            v_dec.append((lbl_map[r['原始标签']], bgr, rgb))
    y_vault = np.array([x[0] for x in v_dec], dtype=np.int64)

    zen_seeds = {k: {'det_2cls_AP50': [], 'det_2cls_AP50_95': [], 'det_3cls_repaired_AP50': [], 'det_3cls_repaired_AP50_95': [], 'esc_hi_pct': []}
                 for k in ['Abl2_Disagreement_Esc', 'TCDMGE_v2_TriSignal', 'Abl3_Full_3Crop_Static']}
    uat_seeds = {k: {'aligned_5cls_acc': [], 'aligned_5cls_f1': [], 'strict_14cls_acc': []}
                 for k in ['Abl2_Disagreement_Esc', 'TCDMGE_v2_TriSignal', 'Abl3_Full_3Crop_Static']}
    vault_seeds = {k: {'strict_14cls_hit': [], 'aligned_3cls_acc': [], 'aligned_3cls_f1': [], 'esc_pct': []}
                   for k in ['Abl2_Disagreement_Esc', 'TCDMGE_v2_TriSignal', 'Abl3_Full_3Crop_Static']}

    for seed in [0, 1, 2]:
        s_info = next(r for r in s2_final['逐种子明细'] if r['seed'] == seed)
        vp = s_info['val_calibrated_params']
        tau_h, tau_m, gamma, beta, fg_thr = vp['tau_h_std'], vp['tau_maha_z'], vp['gamma'], vp['beta'], vp['fg_thr']

        blob = torch.load(CKPT_TCD / f'cggar_v2_seed{seed}.pt', map_location='cpu', weights_only=False)
        rel_head = TargetContextRelationalHead().to(device).eval()
        rel_head.load_state_dict(blob['rel_head'])
        m_cfg = blob['mahalanobis']
        T_DET, T_DINO, T_REL = blob['temps']['T_DET'], blob['temps']['T_DINO'], blob['temps']['T_REL']

        yolo_ext = YoloFullStateExtractor(Path(r'D:\citrus_runs') / f'yolov8n_unified_seed{seed}' / 'weights' / 'best.pt', device)
        dino_ext = DinoExtractor(seed, device)

        def compute_3variants_probs(p14_arr, zneck_arr, geom_arr, c_core, c_std, c_ctx, fg_mask):
            zc, lgc = dino_ext.encode_crops(c_core)
            zs, lgs = dino_ext.encode_crops(c_std)
            zx, _ = dino_ext.encode_crops(c_ctx)
            lgr = rel_head(torch.from_numpy(zc.astype(np.float32)).to(device),
                           torch.from_numpy(zs.astype(np.float32)).to(device),
                           torch.from_numpy(zx.astype(np.float32)).to(device),
                           torch.from_numpy(geom_arr).to(device)).cpu().numpy()

            maha_14 = compute_maha_z(zneck_arr, m_cfg['means'], m_cfg['prec'], m_cfg['m_mu'], m_cfg['m_sd'])
            lp_det = F.log_softmax(torch.logit(torch.from_numpy(p14_arr).clamp(1e-5, 1.0 - 1e-5)) / T_DET, dim=-1).numpy()
            lp_std = F.log_softmax(torch.from_numpy(lgs) / T_DINO, dim=-1).numpy()
            lp_core = F.log_softmax(torch.from_numpy(lgc) / T_DINO, dim=-1).numpy()
            lp_rel = F.log_softmax(torch.from_numpy(lgr) / T_REL, dim=-1).numpy()

            p_det, p_std, p_core, p_rel = np.exp(lp_det), np.exp(lp_std), np.exp(lp_core), np.exp(lp_rel)
            h_det = -(np.clip(p_det, 1e-7, 1.0) * np.log(np.clip(p_det, 1e-7, 1.0))).sum(axis=1) / math.log(14.0)
            h_std = -(np.clip(p_std, 1e-7, 1.0) * np.log(np.clip(p_std, 1e-7, 1.0))).sum(axis=1) / math.log(14.0)
            h_core = -(np.clip(p_core, 1e-7, 1.0) * np.log(np.clip(p_core, 1e-7, 1.0))).sum(axis=1) / math.log(14.0)
            h_rel = -(np.clip(p_rel, 1e-7, 1.0) * np.log(np.clip(p_rel, 1e-7, 1.0))).sum(axis=1) / math.log(14.0)

            c_hat = p_det.argmax(axis=1)
            d_maha_z = maha_14[np.arange(len(c_hat)), c_hat]
            esc_dis = fg_mask & (c_hat != p_std.argmax(axis=1))
            esc_tri = fg_mask & ((c_hat != p_std.argmax(axis=1)) | (h_std > tau_h) | (d_maha_z > tau_m))

            p_abl2 = compute_analytic_opinion_pool(lp_det, lp_std, lp_core, lp_rel, h_det, h_std, h_core, h_rel, d_maha_z, esc_dis, gamma=gamma)
            p_tcd = compute_analytic_opinion_pool(lp_det, lp_std, lp_core, lp_rel, h_det, h_std, h_core, h_rel, d_maha_z, esc_tri, gamma=gamma)
            lp_abl3 = 0.25 * lp_det + 0.25 * lp_std + 0.20 * lp_core + 0.30 * lp_rel
            p_abl3 = np.exp(lp_abl3 - lp_abl3.max(axis=1, keepdims=True)); p_abl3 /= p_abl3.sum(axis=1, keepdims=True)
            return p_abl2, p_tcd, p_abl3, esc_dis, esc_tri

        # --- (A) Zenodo Repaired Zero-Drop ---
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
        nc = len(raw_preds)
        p14_z, zn_z, gm_z = np.zeros((nc, 14), np.float32), np.zeros((nc, 192), np.float32), np.zeros((nc, 9), np.float32)
        cc_z, cs_z, cx_z = np.zeros((nc, 3, 224, 224), np.uint8), np.zeros((nc, 3, 224, 224), np.uint8), np.zeros((nc, 3, 224, 224), np.uint8)
        oiids = list(g_pos.keys())
        for start in range(0, len(oiids), 16):
            sub_i = oiids[start:start + 16]
            bgrs, bxs_b, tcls_b, pos_b = [], [], [], []
            for iid in sub_i:
                bgr = cv2.imdecode(np.fromfile(str(img_paths[iid]), dtype=np.uint8), 1)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                plist = g_pos[iid]
                bl = [raw_preds[p]['bbox'] for p in plist]
                cl = [raw_preds[p]['category_id'] for p in plist]
                for p, bb in zip(plist, bl):
                    gm_z[p] = compute_box_geom(bb, w, h, bl)
                    cc_z[p] = crop_square_scale(rgb, bb, 0.82)
                    cs_z[p] = crop_square_scale(rgb, bb, 1.25)
                    cx_z[p] = crop_square_scale(rgb, bb, 2.50)
                bgrs.append(bgr); bxs_b.append(bl); tcls_b.append(cl); pos_b.extend(plist)
            ps, zs = yolo_ext.extract_batch(bgrs, bxs_b, tcls_b)
            for k, p_idx in enumerate(pos_b):
                p14_z[p_idx] = ps[k]; zn_z[p_idx] = zs[k]

        ysc_z = np.array([r['score'] for r in raw_preds], dtype=np.float64)
        fg_z = ysc_z >= fg_thr
        hi_z = ysc_z >= 0.25
        pa2, ptc, pa3, ed_z, et_z = compute_3variants_probs(p14_z, zn_z, gm_z, cc_z, cs_z, cx_z, fg_z)

        for mk, pmat, esc_arr in [
            ('Abl2_Disagreement_Esc', pa2, ed_z),
            ('TCDMGE_v2_TriSignal', ptc, et_z),
            ('Abl3_Full_3Crop_Static', pa3, fg_z),
        ]:
            rows_14 = [dict(r) for r in raw_preds]
            for i in range(nc):
                if fg_z[i]:
                    nc_i = int(np.argmax(pmat[i]))
                    rows_14[i]['category_id'] = nc_i
                    rows_14[i]['score'] = float((ysc_z[i] ** beta) * (max(float(pmat[i, nc_i]), 1e-6) ** (1.0 - beta)))
            rows_14 = apply_classwise_nms(rows_14, 0.70)
            m2 = [{'image_id': r['image_id'], 'category_id': scidb_to_2cls_full[r['category_id']], 'bbox': r['bbox'], 'score': r['score']} for r in rows_14]
            m3 = [{'image_id': r['image_id'], 'category_id': scidb_to_3cls_repaired[r['category_id']], 'bbox': r['bbox'], 'score': r['score']} for r in rows_14]
            r2 = coco_eval_metrics(gt2_p, m2)
            r3 = coco_eval_metrics(gt3_p, m3)
            zen_seeds[mk]['det_2cls_AP50'].append(round(r2['AP50'], 2))
            zen_seeds[mk]['det_2cls_AP50_95'].append(round(r2['AP50_95'], 2))
            zen_seeds[mk]['det_3cls_repaired_AP50'].append(round(r3['AP50'], 2))
            zen_seeds[mk]['det_3cls_repaired_AP50_95'].append(round(r3['AP50_95'], 2))
            zen_seeds[mk]['esc_hi_pct'].append(round(float((esc_arr & hi_z).sum() / max(1, hi_z.sum())) * 100.0, 2))

        # --- (B) Helper for Single-Organ Image Benchmarks (CitrusUAT & Sweet Orange Vault) ---
        def run_single_organ_bench(dec_list):
            n_items = len(dec_list)
            p14_a = np.zeros((n_items, 14), np.float32)
            zn_a = np.zeros((n_items, 192), np.float32)
            gm_a = np.zeros((n_items, 9), np.float32)
            cc_a = np.zeros((n_items, 3, 224, 224), np.uint8)
            cs_a = np.zeros((n_items, 3, 224, 224), np.uint8)
            cx_a = np.zeros((n_items, 3, 224, 224), np.uint8)
            for start in range(0, n_items, 24):
                sub = dec_list[start:start + 24]
                bgrs = [x[1] for x in sub]
                res_b = yolo_ext.yolo.predict(bgrs, imgsz=640, conf=0.05, iou=0.7, max_det=20, device=0, half=False, rect=False, verbose=False)
                bxs_b, tcls_b = [], []
                for k, ((_, _, rgb), r) in enumerate(zip(sub, res_b)):
                    idx = start + k
                    h, w = rgb.shape[:2]
                    if len(r.boxes) > 0:
                        x1, y1, x2, y2 = r.boxes.xyxy[0].cpu().tolist()
                        bb = [x1, y1, x2 - x1, y2 - y1]
                        tc = int(r.boxes.cls[0].item())
                    else:
                        bb = [w * 0.06, h * 0.06, w * 0.88, h * 0.88]
                        tc = None
                    bxs_b.append([bb]); tcls_b.append([tc])
                    gm_a[idx] = compute_box_geom(bb, w, h, [bb])
                    cc_a[idx] = crop_square_scale(rgb, bb, 0.82)
                    cs_a[idx] = crop_square_scale(rgb, bb, 1.25)
                    cx_a[idx] = crop_square_scale(rgb, bb, 2.50)
                ps, zs = yolo_ext.extract_batch(bgrs, bxs_b, tcls_b)
                for k in range(len(sub)):
                    p14_a[start + k] = ps[k]; zn_a[start + k] = zs[k]
            return compute_3variants_probs(p14_a, zn_a, gm_a, cc_a, cs_a, cx_a, np.ones(n_items, dtype=bool))

        pa2_u, ptc_u, pa3_u, _, _ = run_single_organ_bench(uat_dec)
        for mk, pmat in [('Abl2_Disagreement_Esc', pa2_u), ('TCDMGE_v2_TriSignal', ptc_u), ('Abl3_Full_3Crop_Static', pa3_u)]:
            c14 = pmat.argmax(axis=1)
            st_ok = [int(c14[i]) in uat_groups[y_uat[i]] for i in range(len(y_uat))]
            p5 = np.stack([pmat[:, g].sum(axis=1) for g in uat_groups], axis=1).argmax(axis=1)
            uat_seeds[mk]['strict_14cls_acc'].append(round(float(np.mean(st_ok) * 100.0), 2))
            uat_seeds[mk]['aligned_5cls_acc'].append(round(float(accuracy_score(y_uat, p5) * 100.0), 2))
            uat_seeds[mk]['aligned_5cls_f1'].append(round(float(f1_score(y_uat, p5, average='macro') * 100.0), 2))

        pa2_v, ptc_v, pa3_v, ed_v, et_v = run_single_organ_bench(v_dec)
        for mk, pmat, earr in [
            ('Abl2_Disagreement_Esc', pa2_v, ed_v),
            ('TCDMGE_v2_TriSignal', ptc_v, et_v),
            ('Abl3_Full_3Crop_Static', pa3_v, np.ones(len(y_vault), dtype=bool)),
        ]:
            c14 = pmat.argmax(axis=1)
            st_ok = [int(c14[i]) in sci_groups_v[y_vault[i]] for i in range(len(y_vault))]
            p3 = np.stack([pmat[:, g].sum(axis=1) for g in sci_groups_v], axis=1).argmax(axis=1)
            vault_seeds[mk]['strict_14cls_hit'].append(round(float(np.mean(st_ok) * 100.0), 2))
            vault_seeds[mk]['aligned_3cls_acc'].append(round(float(accuracy_score(y_vault, p3) * 100.0), 2))
            vault_seeds[mk]['aligned_3cls_f1'].append(round(float(f1_score(y_vault, p3, average='macro') * 100.0), 2))
            vault_seeds[mk]['esc_pct'].append(round(float(earr.mean() * 100.0), 2))

        print(f"  [Seed {seed}] Zenodo(零丢框) Abl2: 2cls={zen_seeds['Abl2_Disagreement_Esc']['det_2cls_AP50'][-1]:.2f} / 3cls={zen_seeds['Abl2_Disagreement_Esc']['det_3cls_repaired_AP50'][-1]:.2f} (升级率={zen_seeds['Abl2_Disagreement_Esc']['esc_hi_pct'][-1]:.1f}%) | "
              f"CitrusUAT 5类F1: Abl2={uat_seeds['Abl2_Disagreement_Esc']['aligned_5cls_f1'][-1]:.2f}% (静态Abl3={uat_seeds['Abl3_Full_3Crop_Static']['aligned_5cls_f1'][-1]:.2f}%) | "
              f"Vault 14类命中: Abl2={vault_seeds['Abl2_Disagreement_Esc']['strict_14cls_hit'][-1]:.2f}% / 3类F1={vault_seeds['Abl2_Disagreement_Esc']['aligned_3cls_f1'][-1]:.2f}%",
              flush=True)

        yolo_ext.close()
        del yolo_ext, dino_ext, rel_head
        torch.cuda.empty_cache()

    def fmt_dict(d):
        return {
            mk: {m: {'mean': round(float(np.mean(v)), 2), 'std': round(float(np.std(v, ddof=1)), 2), 'mean_std': f"{np.mean(v):.2f}±{np.std(v, ddof=1):.2f}", 'seeds': v}
                 for m, v in md.items()}
            for mk, md in d.items()
        }

    return {
        'zenodo_repaired_zerodrop': fmt_dict(zen_seeds),
        'citrusuat_5cls': fmt_dict(uat_seeds),
        'vault_sweet_orange_1389': fmt_dict(vault_seeds),
    }


def main():
    t0 = time.monotonic()
    backend = configure_strict_fp32()
    lat7 = benchmark_7tier_clean_latency()
    ext_mg = eval_multigranular_frozen_on_all_externals()

    out_file = RES_DIR / 'phase2b_frozen_multigranular_and_clean_latency.json'
    out_file.write_text(json.dumps({
        '阶段': 'Phase2B_历史冻结多粒度模型同口径补齐与7档零冗余测速',
        '耗时秒': round(time.monotonic() - t0, 2),
        '严格评测后端': backend,
        'RTX4060Ti_七档零冗余单图延迟': lat7,
        '冻结多粒度模型外部与最终盲测汇总': ext_mg,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n[Phase 2B 完成] 已保存至: {out_file} (耗时 {time.monotonic() - t0:.1f}s)", flush=True)


if __name__ == '__main__':
    main()
