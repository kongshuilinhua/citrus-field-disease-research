# -*- coding: utf-8 -*-
"""
Phase 2 (完全冻结权重与参数后的单次评测、修复版外部评测、CitrusUAT掩膜定量验证、未触碰独立集最终盲测与5档公平测速):
铁律声明：
1. 严格校验并只加载 Phase 1 锁定的 frozen_models/bgopta_frozen_seed{0,1,2}.pt (核验 SHA-256 一致性) 与 val_frozen_hyperparams；
2. 零参数修改、零阈值重调，一次性完成五大评测模块并保存全部逐种子明细 JSON：
   - 模块 A: SciDB val / testB / testC (明确标定为同源跨条件诊断基准) 全消融矩阵 (B0, B1, B3, 1.25x校准, S0, S1普通拼接, S2仅正交拓扑, S3完整BG-OPTA, S3+安全免裁) + 逐框救回/改错四象限统计；
   - 模块 B: RTX 4060 Ti 严格 FP32 五档公平单图推理延迟与 DINOv2 前向调用次数实测 (Native YOLOv8n / Full DINOv2 1-Crop / Full 3-Crop Abl3 / TCD-MGE v2 / Ours BG-OPTA 单趟 & 安全免裁)；
   - 模块 C: 修复 9-13 类漏映射缺陷后的 Zenodo 11148332 (519图/1765框) 外部果园探索评测 (100% 预测框零丢弃 2类 & 3类 AP + 严格错判惩罚匹配 F1)；
   - 模块 D: CitrusUAT (500张受控灰底叶图) 真实物理叶片掩膜定量可解释性 (无需输入掩膜，测量软前景权重落在真实叶片内的能量纯度与四角背景抑制率 + 5类跨域诊断)；
   - 模块 E: 真正零接触最终封口盲测 (Vault Holdout: Sweet Orange V1 去重保留原图 1,389 张端到端整图检测→器官诊断，只跑唯一一次)。
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
from sklearn.metrics import accuracy_score, f1_score
from torchvision.ops import roi_align
from ultralytics import YOLO
from faster_coco_eval import COCO, COCOeval_faster

ROOT = Path(r'D:\新版柑橘')
EXT = ROOT / '方法路线扩展_20261031'
WORK_DIR = ROOT / 'Antigravity_BGOPTA_单趟正交Token攻关_20261126'
PREV_DIR = ROOT / 'Antigravity_CGGAR_真创新攻关_20261125'

sys.path.append(str(EXT))
sys.path.append(str(WORK_DIR))
sys.path.append(str(PREV_DIR))

from 共同评测后端 import configure_strict_fp32
from phase1_source_only_train_and_select import (
    BGOPTAHead, crop_single_context_and_masks, apply_classwise_nms, compute_maha_z, CROP_SCALE
)
from step1_extract_multigranular_cache import (
    YoloFullStateExtractor, DinoExtractor, crop_square_scale, compute_box_geom, letterbox_640
)
from step2_train_and_eval_cggar import TargetContextRelationalHead, entropy_norm
from step3_external_blind_and_latency import build_external_zenodo_gt

CKPT_DIR = WORK_DIR / 'frozen_models'
RES_DIR = WORK_DIR / 'results'
PREV_CACHE = PREV_DIR / 'cache'
GT_DIR = EXT / '检测数据'
STRONG_DIR = EXT / 'YOLOv8强基线重复_20261109'
IMG_ROOT = Path(r'D:\病虫害论文\复现实验记录\yolo11n_public_compare_20261004\数据副本\images')
AUDIT_INDEX_JSON = Path(r'D:\病虫害论文\新增公开数据_20260916\中文分类与标签使用索引.json')
CITRUSUAT_DIR = ROOT / '复现_baseline' / 'field_test_CitrusUAT'


def coco_eval_metrics(gt_path: Path, pred_rows: list):
    if not pred_rows:
        return {'AP50_95': 0.0, 'AP50': 0.0, 'AP75': 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_path))
        ev = COCOeval_faster(gt, gt.loadRes(pred_rows), 'bbox')
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return {
        'AP50_95': round(float(ev.stats[0] * 100.0), 4),
        'AP50': round(float(ev.stats[1] * 100.0), 4),
        'AP75': round(float(ev.stats[2] * 100.0), 4),
    }


def verify_phase1_lock() -> dict:
    p1_path = RES_DIR / 'phase1_source_only_ablation_and_lock.json'
    assert p1_path.exists(), f"缺少 Phase 1 锁定文件: {p1_path}"
    p1 = json.loads(p1_path.read_text(encoding='utf-8'))
    for item in p1['冻结权重与参数清单']:
        cp = Path(item['ckpt'])
        assert cp.exists(), f"缺少冻结权重: {cp}"
        actual_sha = hashlib.sha256(cp.read_bytes()).hexdigest()
        assert actual_sha == item['sha256'], f"SHA-256 不匹配! {cp}: {actual_sha} != {item['sha256']}"
    print("[门禁核验通过] Phase 1 全部 3 个种子冻结权重 SHA-256 100% 吻合，零改动进入单次评测！", flush=True)
    return p1


@torch.no_grad()
def encode_tokens_fp16(enc, crops_u8: np.ndarray, device='cuda:0', batch_size: int = 192) -> torch.Tensor:
    if len(crops_u8) == 0:
        return torch.zeros((0, 257, 384), dtype=torch.float16, device=device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    outs = []
    for i in range(0, len(crops_u8), batch_size):
        x = (torch.from_numpy(crops_u8[i:i + batch_size]).to(device).float() / 255.0 - mean) / std
        outs.append(enc.forward_features(x).half())
    return torch.cat(outs, dim=0)


def load_scidb_scaled_images():
    """一次性解码并预缩放 val / testB / testC 原图 (max_dim<=1280)，与 crop_single_context_and_masks 数学等价且极省内存。"""
    split_maps = {}
    for sp in ['val', 'testB', 'testC']:
        gt = json.loads((GT_DIR / f'{sp}.json').read_text(encoding='utf-8'))
        items = gt['images']

        def dec_one(im):
            p = IMG_ROOT / sp / im['file_name']
            bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), 1)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            md = max(h, w)
            if md > 1280:
                f = 1280.0 / md
                rgb = cv2.resize(rgb, (max(1, int(round(w * f))), max(1, int(round(h * f)))), interpolation=cv2.INTER_AREA)
            else:
                f = 1.0
            return im['id'], (rgb, f)

        mp = {}
        with ThreadPoolExecutor(max_workers=6) as ex:
            for iid, val in ex.map(dec_one, items):
                mp[iid] = val
        split_maps[sp] = (gt, mp)
        print(f"  [SciDB 预解码] {sp:5s}: {len(mp)} 张图像就绪", flush=True)
    return split_maps


def match_gt_boxes_for_quadrant(gt_dict: dict, cand_rows: list, iou_thr: float = 0.50):
    """为高置信匹配候选框关联真实类别，用于精确统计 B类(救回 YOLO 错框)与 C类(改错 YOLO 对框)个数。"""
    anns_by_img = defaultdict(list)
    for a in gt_dict['annotations']:
        anns_by_img[a['image_id']].append(a)
    gt_cls = np.full(len(cand_rows), -1, dtype=np.int64)
    by_img = defaultdict(list)
    for idx, r in enumerate(cand_rows):
        if r['score'] >= 0.25:
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
    return gt_cls


# ==============================================================================
# 模块 A: SciDB val / testB / testC 冻结单次评测与全消融矩阵
# ==============================================================================
@torch.no_grad()
def eval_scidb_frozen_3seeds(p1_json: dict):
    print("\n================ [模块 A] SciDB (val / testB / testC) 冻结单次全消融评测 ================", flush=True)
    device = 'cuda:0'
    meta = json.loads((ROOT / '方法推进_20261027' / '纯CLS对照' / '微调汇总.json').read_text(encoding='utf-8'))
    prev_tcd = json.loads((PREV_DIR / 'results' / 'tcd_mge_scidb_3seed_final.json').read_text(encoding='utf-8'))
    split_maps = load_scidb_scaled_images()

    method_keys = [
        'B0_YOLOv8n_Native',
        'B1_Full_DINOv2_1Crop_Hard',
        'B3_Static_Posterior_Fuse',
        'Ref_SingleCrop_125_Calib',
        'Abl_S0_ClsOnly_150',
        'Abl_S1_PlainPatchConcat_NoProj',
        'Abl_S2_BG_Orthogonal_TopoOnly',
        'Ours_S3_BGOPTA_1Pass',
        'Ours_S3_BGOPTA_SafeBypass',
        'Ref_3Crop_Diff_Static_Abl3',
        'Ref_TCDMGE_v2_3CropEsc',
    ]
    records = {k: {sp: [] for sp in ['val', 'testB', 'testC']} for k in method_keys}
    bypass_stats = {sp: {'dino_call_fg_pct': [], 'dino_call_hi_pct': [], 'avg_dino_crops_per_img': []} for sp in ['val', 'testB', 'testC']}
    quadrant_stats = {sp: {'full_dino_rescued': [], 'full_dino_corrupted': [], 's1_plain_rescued': [], 's1_plain_corrupted': [], 'bgopta_rescued': [], 'bgopta_corrupted': []} for sp in ['testB', 'testC']}

    # Pre-fill exact historical reference seeds from tcd_mge_scidb_3seed_final.json for B1_Hard, Abl3, TCD-MGE
    for sp in ['val', 'testB', 'testC']:
        records['B1_Full_DINOv2_1Crop_Hard'][sp] = prev_tcd['三种子汇总']['B1a_Full_DINOv2_GrayPad_Orig'][sp]['seeds']
        records['Ref_3Crop_Diff_Static_Abl3'][sp] = prev_tcd['三种子汇总']['Abl3_TriGranular_StaticWeight'][sp]['seeds']
        records['Ref_TCDMGE_v2_3CropEsc'][sp] = prev_tcd['三种子汇总']['Ours_TCDMGE_Selective'][sp]['seeds']

    for seed in [0, 1, 2]:
        ckpt = torch.load(CKPT_DIR / f'bgopta_frozen_seed{seed}.pt', map_location='cpu', weights_only=False)
        hp = ckpt['val_frozen_hyperparams']
        al_lock, bt_lock, fg_thr = hp['alpha_det'], hp['beta_score'], hp['fg_thr']
        sb_conf, sb_mar, sb_maha = hp['safe_bypass_conf'], hp['safe_bypass_margin'], hp['safe_bypass_maha_z']
        m_cfg = ckpt['mahalanobis']

        t_info = next(r for r in meta['重复结果'] if r['方法'] == 'LP-FT' and r['种子'] == seed)
        st = torch.load(t_info['权重文件'], map_location='cpu', weights_only=True)
        enc = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False, img_size=224, num_classes=0).to(device).eval()
        enc.load_state_dict(st['骨干'])
        w_h, b_h = st['分类头']['weight'].clone(), st['分类头']['bias'].clone()

        models = {}
        for m_name in ['S0_cls_only', 'S1_plain_concat', 'S2_orth_topo_only', 'S3_bgopta_full']:
            md = BGOPTAHead(w_h, b_h, mode=m_name).to(device).eval()
            md.load_state_dict(ckpt[m_name])
            models[m_name] = md

        for sp in ['val', 'testB', 'testC']:
            gt_dict, rgb_mp = split_maps[sp]
            gt_path = GT_DIR / f'{sp}.json'
            orig_preds = json.loads((STRONG_DIR / f'种子{seed}' / f'{sp}预测.json').read_text(encoding='utf-8'))
            npz = np.load(PREV_CACHE / f'{sp}_cggar_seed{seed}.npz')
            cand_idxs = npz['cand_idxs']
            cand_rows = [orig_preds[i] for i in cand_idxs]
            n_c = len(cand_rows)

            crops_150 = np.empty((n_c, 3, 224, 224), dtype=np.uint8)
            masks_150 = np.empty((n_c, 2, 256), dtype=np.bool_)
            for li, r in enumerate(cand_rows):
                rgb_s, f_s = rgb_mp[r['image_id']]
                bx, by, bw, bh = r['bbox']
                c, m = crop_single_context_and_masks(rgb_s, [bx * f_s, by * f_s, bw * f_s, bh * f_s], CROP_SCALE)
                crops_150[li] = c
                masks_150[li] = m

            tok_gpu = encode_tokens_fp16(enc, crops_150, device)
            masks_gpu = torch.from_numpy(masks_150).to(device)
            geom_gpu = torch.from_numpy(npz['geom']).to(device)
            p14_gpu = torch.from_numpy(npz['p14']).to(device)
            lgs_125_gpu = torch.from_numpy(npz['lg_std']).to(device)

            yolo_sc = np.array([r['score'] for r in cand_rows], dtype=np.float64)
            yolo_cls = np.array([r['category_id'] for r in cand_rows], dtype=np.int64)
            fg_m = yolo_sc >= fg_thr

            lp_det = F.log_softmax(torch.logit(p14_gpu.clamp(1e-5, 1.0 - 1e-5)) / 1.35, dim=-1).cpu().numpy()
            p_det = np.exp(lp_det)
            top2_d = np.sort(p_det, axis=1)[:, -2:]
            mar_det = top2_d[:, 1] - top2_d[:, 0]
            maha_14 = compute_maha_z(npz['zneck'], m_cfg['means'], m_cfg['prec'], m_cfg['m_mu'], m_cfg['m_sd'])
            d_maha_z = maha_14[np.arange(n_c), p_det.argmax(axis=1)]

            # Safe-Bypass Gate (strictly calibrated on val in Phase 1!)
            safe_bypass = fg_m & (yolo_sc >= sb_conf) & (mar_det >= sb_mar) & (d_maha_z <= sb_maha)
            need_dino = fg_m & (~safe_bypass)
            hi_m = yolo_sc >= 0.25
            bypass_stats[sp]['dino_call_fg_pct'].append(round(float(need_dino.sum() / max(1, fg_m.sum())) * 100.0, 2))
            bypass_stats[sp]['dino_call_hi_pct'].append(round(float((need_dino & hi_m).sum() / max(1, hi_m.sum())) * 100.0, 2))
            bypass_stats[sp]['avg_dino_crops_per_img'].append(round(float(need_dino.sum() / len(gt_dict['images'])), 3))

            def eval_post_prob(prob_mat: np.ndarray, calib: bool, use_nms: bool, bt_val: float = bt_lock):
                rows = [dict(r) for r in orig_preds]
                for li, gi in enumerate(cand_idxs):
                    if fg_m[li]:
                        pr = prob_mat[li]
                        nc = int(np.argmax(pr))
                        oc = rows[gi]['category_id']
                        rows[gi]['category_id'] = nc
                        if calib:
                            rows[gi]['score'] = float((yolo_sc[li] ** bt_val) * (max(float(pr[nc]), 1e-6) ** (1.0 - bt_val)))
                        elif nc != oc:
                            rows[gi]['score'] = float(math.sqrt(yolo_sc[li] * max(float(pr[nc]), 1e-6)))
                if use_nms:
                    rows = apply_classwise_nms(rows, 0.70)
                return coco_eval_metrics(gt_path, rows)['AP50_95']

            # 1. Native YOLOv8n
            ap_b0 = coco_eval_metrics(gt_path, orig_preds)['AP50_95']
            records['B0_YOLOv8n_Native'][sp].append(round(ap_b0, 2))

            # 2. B3 Static Posterior Fuse
            p_std_raw = F.softmax(lgs_125_gpu, dim=-1).cpu().numpy()
            p_b3 = 0.65 * p_std_raw.copy()
            p_b3[np.arange(n_c), yolo_cls] += 0.35 * yolo_sc
            # Note: B3 applies to all cand_idxs (score>=0.05) without decoupled calib/nms
            rows_b3 = [dict(r) for r in orig_preds]
            for li, gi in enumerate(cand_idxs):
                nc = int(np.argmax(p_b3[li]))
                if nc != rows_b3[gi]['category_id']:
                    rows_b3[gi]['category_id'] = nc
                    rows_b3[gi]['score'] = float(math.sqrt(yolo_sc[li] * max(float(p_b3[li, nc]), 1e-6)))
            records['B3_Static_Posterior_Fuse'][sp].append(round(coco_eval_metrics(gt_path, rows_b3)['AP50_95'], 2))

            # 3. Ref SingleCrop 1.25x [CLS] + Anchor 14-Cls + Decoupled Calib
            lp_125 = F.log_softmax(lgs_125_gpu / 1.15, dim=-1).cpu().numpy()
            lp_f125 = al_lock * lp_det + (1.0 - al_lock) * lp_125
            p_f125 = np.exp(lp_f125 - lp_f125.max(axis=1, keepdims=True)); p_f125 /= p_f125.sum(axis=1, keepdims=True)
            records['Ref_SingleCrop_125_Calib'][sp].append(round(eval_post_prob(p_f125, True, True, bt_lock), 2))

            # 4. Same-Compute Single-Pass 1.50x Ablations: S0, S1, S2, S3 (identical alpha_lock & beta_lock!)
            p_modes = {}
            for m_key, rec_key in [
                ('S0_cls_only', 'Abl_S0_ClsOnly_150'),
                ('S1_plain_concat', 'Abl_S1_PlainPatchConcat_NoProj'),
                ('S2_orth_topo_only', 'Abl_S2_BG_Orthogonal_TopoOnly'),
                ('S3_bgopta_full', 'Ours_S3_BGOPTA_1Pass'),
            ]:
                lg_m = models[m_key](tok_gpu, masks_gpu, geom_gpu)
                lp_m = F.log_softmax(lg_m / 1.15, dim=-1).cpu().numpy()
                lp_fm = al_lock * lp_det + (1.0 - al_lock) * lp_m
                p_fm = np.exp(lp_fm - lp_fm.max(axis=1, keepdims=True)); p_fm /= p_fm.sum(axis=1, keepdims=True)
                p_modes[m_key] = p_fm
                records[rec_key][sp].append(round(eval_post_prob(p_fm, True, True, bt_lock), 2))

            # 5. Ours S3 BG-OPTA + Safe-Bypass Gate (skips DINOv2 completely on safe_bypass boxes!)
            p_sb = p_modes['S3_bgopta_full'].copy()
            p_sb[safe_bypass] = p_det[safe_bypass]
            records['Ours_S3_BGOPTA_SafeBypass'][sp].append(round(eval_post_prob(p_sb, True, True, bt_lock), 2))

            # 6. Quadrant rescue/corrupt audit on testB / testC
            if sp in ['testB', 'testC']:
                gt_c = match_gt_boxes_for_quadrant(gt_dict, cand_rows, 0.50)
                v_m = gt_c >= 0
                y_ok = (yolo_cls == gt_c) & v_m
                y_err = (yolo_cls != gt_c) & v_m
                d_pred = p_std_raw.argmax(axis=1)
                s1_pred = p_modes['S1_plain_concat'].argmax(axis=1)
                s3_pred = p_sb.argmax(axis=1)
                quadrant_stats[sp]['full_dino_rescued'].append(int((y_err & (d_pred == gt_c)).sum()))
                quadrant_stats[sp]['full_dino_corrupted'].append(int((y_ok & (d_pred != gt_c)).sum()))
                quadrant_stats[sp]['s1_plain_rescued'].append(int((y_err & (s1_pred == gt_c)).sum()))
                quadrant_stats[sp]['s1_plain_corrupted'].append(int((y_ok & (s1_pred != gt_c)).sum()))
                quadrant_stats[sp]['bgopta_rescued'].append(int((y_err & (s3_pred == gt_c)).sum()))
                quadrant_stats[sp]['bgopta_corrupted'].append(int((y_ok & (s3_pred != gt_c)).sum()))

            del tok_gpu
            torch.cuda.empty_cache()

        print(f"  [Seed {seed}] testB: Full-DINO={records['B1_Full_DINOv2_1Crop_Hard']['testB'][seed]:.2f} | S0={records['Abl_S0_ClsOnly_150']['testB'][seed]:.2f} | "
              f"S1(普通拼接)={records['Abl_S1_PlainPatchConcat_NoProj']['testB'][seed]:.2f} | S2(仅正交)={records['Abl_S2_BG_Orthogonal_TopoOnly']['testB'][seed]:.2f} | "
              f"★S3(BG-OPTA)={records['Ours_S3_BGOPTA_1Pass']['testB'][seed]:.2f} (免裁版={records['Ours_S3_BGOPTA_SafeBypass']['testB'][seed]:.2f}) || "
              f"testC: Full-DINO={records['B1_Full_DINOv2_1Crop_Hard']['testC'][seed]:.2f} | S1={records['Abl_S1_PlainPatchConcat_NoProj']['testC'][seed]:.2f} -> ★S3={records['Ours_S3_BGOPTA_1Pass']['testC'][seed]:.2f} (免裁版={records['Ours_S3_BGOPTA_SafeBypass']['testC'][seed]:.2f})",
              flush=True)
        del enc, models
        torch.cuda.empty_cache()

    summary = {}
    b1_ref = records['B1_Full_DINOv2_1Crop_Hard']
    s1_ref = records['Abl_S1_PlainPatchConcat_NoProj']
    for k in method_keys:
        summary[k] = {}
        for sp in ['val', 'testB', 'testC']:
            arr = records[k][sp]
            wins_b1 = sum(1 for a, b in zip(arr, b1_ref[sp]) if a > b)
            wins_s1 = sum(1 for a, b in zip(arr, s1_ref[sp]) if a > b)
            summary[k][sp] = {
                'mean': round(float(np.mean(arr)), 2),
                'std': round(float(np.std(arr, ddof=1)), 2),
                'mean_std': f"{np.mean(arr):.2f}±{np.std(arr, ddof=1):.2f}",
                'seeds': [round(float(x), 2) for x in arr],
                'wins_vs_FullDINO': f"{wins_b1}/3",
                'wins_vs_S1PlainConcat': f"{wins_s1}/3",
            }
    print("\n---------------- [模块 A 汇总: SciDB 3 种子均值±标准差] ----------------", flush=True)
    for k in method_keys:
        print(f"  {k:32s} | val={summary[k]['val']['mean_std']} {summary[k]['val']['seeds']} | "
              f"testB={summary[k]['testB']['mean_std']} {summary[k]['testB']['seeds']} | "
              f"testC={summary[k]['testC']['mean_std']} {summary[k]['testC']['seeds']}", flush=True)

    return {'summary': summary, 'bypass_stats': bypass_stats, 'quadrant_stats': quadrant_stats}


# ==============================================================================
# 模块 B: RTX 4060 Ti 严格 FP32 五档公平单图推理延迟与调用次数实测
# ==============================================================================
@torch.no_grad()
def benchmark_5tier_latency():
    print("\n================ [模块 B] RTX 4060 Ti 严格 FP32 五档公平单图延迟实测 ================", flush=True)
    device = 'cuda:0'
    ckpt0 = torch.load(CKPT_DIR / 'bgopta_frozen_seed0.pt', map_location='cpu', weights_only=False)
    hp = ckpt0['val_frozen_hyperparams']
    fg_thr, sb_conf, sb_mar, sb_maha = hp['fg_thr'], hp['safe_bypass_conf'], hp['safe_bypass_margin'], hp['safe_bypass_maha_z']
    m_cfg = ckpt0['mahalanobis']
    m_means_g = torch.from_numpy(m_cfg['means'].astype(np.float32)).to(device)
    m_prec_g = torch.from_numpy(m_cfg['prec'].astype(np.float32)).to(device)
    m_mu, m_sd = float(m_cfg['m_mu']), float(m_cfg['m_sd'])

    blob_tcd = torch.load(PREV_DIR / 'models' / 'cggar_v2_seed0.pt', map_location='cpu', weights_only=False)
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

    val_imgs = sorted((IMG_ROOT / 'val').glob('*.jpg'))[:45]
    tc_imgs = sorted((IMG_ROOT / 'testC').glob('*.*'))[:45]

    def extract_anchor_and_neck(bgr, xyxy, fg_ids):
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
        return bxs_all, bxs_fg, p14_g, zn_g

    def run_tier(paths: list, tier: str):
        times, dino_calls = [], []
        for idx, p in enumerate(paths):
            torch.cuda.synchronize()
            ts = time.perf_counter()
            n_dino = 0
            bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), 1)
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

            elif tier in ('Tier3_Full_3Crop_Static_Abl3', 'Tier4_TCDMGE_v2_Escalated') and len(xyxy) > 0:
                fg_ids = np.where(sc_np >= fg_thr)[0]
                if len(fg_ids) > 0:
                    h, w = bgr.shape[:2]
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    bxs_all, bxs_fg, p14_g, zn_g = extract_anchor_and_neck(bgr, xyxy, fg_ids)
                    if tier == 'Tier3_Full_3Crop_Static_Abl3':
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
                        hs = entropy_norm(ps).squeeze(-1)
                        ch = pd.argmax(dim=-1)
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

            elif tier in ('Tier5a_Ours_BGOPTA_1Pass', 'Tier5b_Ours_BGOPTA_SafeBypass') and len(xyxy) > 0:
                fg_ids = np.where(sc_np >= fg_thr)[0]
                if len(fg_ids) > 0:
                    h, w = bgr.shape[:2]
                    bxs_all, bxs_fg, p14_g, zn_g = extract_anchor_and_neck(bgr, xyxy, fg_ids)
                    pd = F.softmax(torch.logit(p14_g.clamp(1e-5, 1.0 - 1e-5)) / 1.35, dim=-1)
                    if tier == 'Tier5b_Ours_BGOPTA_SafeBypass':
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
                        md = max(h, w)
                        if md > 1280:
                            f_s = 1280.0 / md
                            rgb_s = cv2.resize(rgb, (max(1, int(round(w * f_s))), max(1, int(round(h * f_s)))), interpolation=cv2.INTER_AREA)
                        else:
                            f_s, rgb_s = 1.0, rgb
                        cs, ms, gs = [], [], []
                        for j in act_ids:
                            bx, by, bw, bh = bxs_fg[j]
                            c, m = crop_single_context_and_masks(rgb_s, [bx * f_s, by * f_s, bw * f_s, bh * f_s], CROP_SCALE)
                            cs.append(c); ms.append(m)
                            gs.append(compute_box_geom(bxs_fg[j], w, h, bxs_all))
                        xt = (torch.from_numpy(np.stack(cs, 0)).to(device).float() / 255.0 - dino_ext.mean) / dino_ext.std
                        tok = dino_ext.enc.forward_features(xt)
                        mt = torch.from_numpy(np.stack(ms, 0)).to(device)
                        gt_b = torch.from_numpy(np.stack(gs, 0)).to(device)
                        _ = bgopta_head(tok, mt, gt_b)
                        n_dino = len(act_ids)

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

    # Global GPU warmup
    _ = run_tier(val_imgs[:8], 'Tier5a_Ours_BGOPTA_1Pass')
    tiers = [
        'Tier1_Native_YOLOv8n',
        'Tier2_Full_DINOv2_1Crop',
        'Tier3_Full_3Crop_Static_Abl3',
        'Tier4_TCDMGE_v2_Escalated',
        'Tier5a_Ours_BGOPTA_1Pass',
        'Tier5b_Ours_BGOPTA_SafeBypass',
    ]
    lat_res = {}
    for tr in tiers:
        rv = run_tier(val_imgs, tr)
        rc = run_tier(tc_imgs, tr)
        lat_res[tr] = {'val': rv, 'testC': rc}
        print(f"  [{tr:30s}] val: {rv['mean_ms']:5.2f}ms (P95={rv['p95_ms']:5.2f}ms, DINO调用={rv['avg_dino_forwards_per_img']:.2f}/图) | "
              f"testC: {rc['mean_ms']:5.2f}ms (P95={rc['p95_ms']:5.2f}ms, DINO调用={rc['avg_dino_forwards_per_img']:.2f}/图)", flush=True)

    for h in hooks:
        h.remove()
    del yolo, dino_ext, bgopta_head, rel_head
    torch.cuda.empty_cache()
    return lat_res


# ==============================================================================
# 模块 C: 修复 9-13 类映射缺陷后的 Zenodo 11148332 (519图/1765框) 外部探索评测
# ==============================================================================
@torch.no_grad()
def eval_zenodo_repaired_3seeds():
    print("\n================ [模块 C] 修复版外部果园探索评测 (Zenodo 11148332, 519图/1765框, 零丢框) ================", flush=True)
    device = 'cuda:0'
    coco_images, img_paths, gt2_p, gt3_p = build_external_zenodo_gt()
    gt3_dict = json.loads(gt3_p.read_text(encoding='utf-8'))

    # 完备无遗漏映射表 (覆盖全部 0..13 共 14 个类别，100% 保留全部预测框，零静默丢弃！)
    # raw Zenodo 3-class taxonomy is: 0=Healthy('health'), 1=Mottling/HLB-suspect('HLB'), 2=Other illness('ill')
    scidb_to_2cls_full = {0: 0, 1: 0, **{c: 1 for c in range(2, 14)}}
    scidb_to_3cls_repaired = {0: 0, 1: 0, 2: 1, 3: 1, **{c: 2 for c in range(4, 14)}}
    # Strict 4..8 narrow mapping for GT-matched classification audit where 9..13 is strictly penalized as -1 (wrong!)
    scidb_to_3cls_strict_penalize = {0: 0, 1: 0, 2: 1, 3: 1, **{c: 2 for c in range(4, 9)}, **{c: -1 for c in range(9, 14)}}

    per_seed = []
    for seed in [0, 1, 2]:
        ckpt = torch.load(CKPT_DIR / f'bgopta_frozen_seed{seed}.pt', map_location='cpu', weights_only=False)
        hp = ckpt['val_frozen_hyperparams']
        al_lock, bt_lock, fg_thr = hp['alpha_det'], hp['beta_score'], hp['fg_thr']
        sb_conf, sb_mar, sb_maha = hp['safe_bypass_conf'], hp['safe_bypass_margin'], hp['safe_bypass_maha_z']
        m_cfg = ckpt['mahalanobis']

        w_yolo = Path(r'D:\citrus_runs') / f'yolov8n_unified_seed{seed}' / 'weights' / 'best.pt'
        yolo_ext = YoloFullStateExtractor(w_yolo, device)
        dino_ext = DinoExtractor(seed, device)
        m_s1 = BGOPTAHead(dino_ext.head.weight.data.cpu(), dino_ext.head.bias.data.cpu(), mode='S1_plain_concat').to(device).eval()
        m_s1.load_state_dict(ckpt['S1_plain_concat'])
        m_s3 = BGOPTAHead(dino_ext.head.weight.data.cpu(), dino_ext.head.bias.data.cpu(), mode='S3_bgopta_full').to(device).eval()
        m_s3.load_state_dict(ckpt['S3_bgopta_full'])

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
        c125_arr = np.zeros((n_c, 3, 224, 224), dtype=np.uint8)
        c150_arr = np.zeros((n_c, 3, 224, 224), dtype=np.uint8)
        m150_arr = np.zeros((n_c, 2, 256), dtype=np.bool_)

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
                    c125_arr[p] = crop_square_scale(rgb, bb, 1.25)
                    c150, m150 = crop_single_context_and_masks(rgb, bb, CROP_SCALE)
                    c150_arr[p] = c150
                    m150_arr[p] = m150
                bgrs.append(bgr); bxs_b.append(bl); tcls_b.append(cl); pos_b.extend(plist)
            ps, zs = yolo_ext.extract_batch(bgrs, bxs_b, tcls_b)
            for k, p_idx in enumerate(pos_b):
                p14_arr[p_idx] = ps[k]; zneck_arr[p_idx] = zs[k]

        _, lgs_125 = dino_ext.encode_crops(c125_arr)
        tok_150 = encode_tokens_fp16(dino_ext.enc, c150_arr, device)
        masks_g = torch.from_numpy(m150_arr).to(device)
        geom_g = torch.from_numpy(geom_arr).to(device)
        lg_s1 = m_s1(tok_150, masks_g, geom_g)
        lg_s3 = m_s3(tok_150, masks_g, geom_g)
        yolo_ext.close()
        del yolo_ext, dino_ext, m_s1, m_s3, tok_150
        torch.cuda.empty_cache()

        yolo_sc = np.array([r['score'] for r in raw_preds], dtype=np.float64)
        yolo_cls = np.array([r['category_id'] for r in raw_preds], dtype=np.int64)
        fg_m = yolo_sc >= fg_thr

        p14_g = torch.from_numpy(p14_arr).to(device)
        lp_det = F.log_softmax(torch.logit(p14_g.clamp(1e-5, 1.0 - 1e-5)) / 1.35, dim=-1).cpu().numpy()
        p_det = np.exp(lp_det)
        t2 = np.sort(p_det, axis=1)[:, -2:]
        mar = t2[:, 1] - t2[:, 0]
        maha_14 = compute_maha_z(zneck_arr, m_cfg['means'], m_cfg['prec'], m_cfg['m_mu'], m_cfg['m_sd'])
        dm_z = maha_14[np.arange(n_c), p_det.argmax(axis=1)]
        safe_bypass = fg_m & (yolo_sc >= sb_conf) & (mar >= sb_mar) & (dm_z <= sb_maha)

        p_125_raw = F.softmax(torch.from_numpy(lgs_125), dim=-1).numpy()
        p_b3 = 0.65 * p_125_raw.copy()
        p_b3[np.arange(n_c), yolo_cls] += 0.35 * yolo_sc

        lp_s1 = F.log_softmax(lg_s1 / 1.15, dim=-1).cpu().numpy()
        p_s1 = np.exp(al_lock * lp_det + (1.0 - al_lock) * lp_s1); p_s1 /= p_s1.sum(axis=1, keepdims=True)

        lp_s3 = F.log_softmax(lg_s3 / 1.15, dim=-1).cpu().numpy()
        p_s3 = np.exp(al_lock * lp_det + (1.0 - al_lock) * lp_s3); p_s3 /= p_s3.sum(axis=1, keepdims=True)

        p_s3_sb = p_s3.copy()
        p_s3_sb[safe_bypass] = p_det[safe_bypass]

        def eval_zen_method(act_m, prob_mat, calib: bool, use_nms: bool):
            rows_14 = [dict(r) for r in raw_preds]
            for i in range(n_c):
                if act_m[i]:
                    pr = prob_mat[i]
                    nc = int(np.argmax(pr))
                    oc = rows_14[i]['category_id']
                    rows_14[i]['category_id'] = nc
                    if calib:
                        rows_14[i]['score'] = float((yolo_sc[i] ** bt_lock) * (max(float(pr[nc]), 1e-6) ** (1.0 - bt_lock)))
                    elif nc != oc:
                        rows_14[i]['score'] = float(math.sqrt(yolo_sc[i] * max(float(pr[nc]), 1e-6)))
            if use_nms:
                rows_14 = apply_classwise_nms(rows_14, 0.70)

            # 100% Zero-Drop 2-class & 3-class COCO AP (assert len(mapped) == len(rows_14)!)
            m2 = [{'image_id': r['image_id'], 'category_id': scidb_to_2cls_full[r['category_id']], 'bbox': r['bbox'], 'score': r['score']} for r in rows_14]
            m3_rep = [{'image_id': r['image_id'], 'category_id': scidb_to_3cls_repaired[r['category_id']], 'bbox': r['bbox'], 'score': r['score']} for r in rows_14]
            assert len(m2) == len(rows_14) and len(m3_rep) == len(rows_14)
            r2 = coco_eval_metrics(gt2_p, m2)
            r3 = coco_eval_metrics(gt3_p, m3_rep)

            # Also compute IoU>=0.5 GT-matched accuracy where 9..13 is strictly penalized as wrong (-1)!
            by_img = defaultdict(list)
            for r in rows_14:
                by_img[r['image_id']].append(r)
            anns_by_img = defaultdict(list)
            for a in gt3_dict['annotations']:
                anns_by_img[a['image_id']].append(a)
            y_true, y_pred_strict, y_pred_rep = [], [], []
            for iid, glist in anns_by_img.items():
                plist = sorted(by_img.get(iid, []), key=lambda x: -x['score'])
                for a in glist:
                    gx, gy, gw, gh = a['bbox']
                    best_iou, best_c14 = 0.0, None
                    for pr_r in plist:
                        px, py, pw, ph = pr_r['bbox']
                        ix1, iy1 = max(gx, px), max(gy, py)
                        ix2, iy2 = min(gx + gw, px + pw), min(gy + gh, py + ph)
                        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                        iou = inter / (gw * gh + pw * ph - inter + 1e-9)
                        if iou > best_iou:
                            best_iou, best_c14 = iou, pr_r['category_id']
                    y_true.append(int(a['category_id']))
                    if best_iou >= 0.50 and best_c14 is not None:
                        y_pred_strict.append(scidb_to_3cls_strict_penalize[best_c14])
                        y_pred_rep.append(scidb_to_3cls_repaired[best_c14])
                    else:
                        y_pred_strict.append(-1)
                        y_pred_rep.append(-1)
            yt = np.array(y_true)
            yp_st = np.array(y_pred_strict)
            yp_rp = np.array(y_pred_rep)
            return {
                'det_2cls_AP50': round(r2['AP50'], 2),
                'det_2cls_AP50_95': round(r2['AP50_95'], 2),
                'det_3cls_repaired_AP50': round(r3['AP50'], 2),
                'det_3cls_repaired_AP50_95': round(r3['AP50_95'], 2),
                'gt_matched_3cls_repaired_acc': round(float((yt == yp_rp).mean() * 100.0), 2),
                'gt_matched_strict_9to13_penalized_acc': round(float((yt == yp_st).mean() * 100.0), 2),
            }

        m_none = np.zeros(n_c, dtype=bool)
        m_all = np.ones(n_c, dtype=bool)
        sr = {
            'seed': seed,
            'dino_call_fg_pct': round(float((fg_m & (~safe_bypass)).sum() / max(1, fg_m.sum())) * 100.0, 2),
            'B0_YOLOv8n_Native': eval_zen_method(m_none, p_125_raw, False, False),
            'B1_Full_DINOv2_1Crop': eval_zen_method(m_all, p_125_raw, False, False),
            'B3_Static_Posterior_Fuse': eval_zen_method(m_all, p_b3, False, False),
            'Abl_S1_PlainPatchConcat': eval_zen_method(fg_m, p_s1, True, True),
            'Ours_S3_BGOPTA_1Pass': eval_zen_method(fg_m, p_s3, True, True),
            'Ours_S3_BGOPTA_SafeBypass': eval_zen_method(fg_m, p_s3_sb, True, True),
        }
        print(f"  [Zenodo 修复版 Seed {seed}] 2cls AP50: B0={sr['B0_YOLOv8n_Native']['det_2cls_AP50']:.2f} | "
              f"Full-DINO={sr['B1_Full_DINOv2_1Crop']['det_2cls_AP50']:.2f} | S1={sr['Abl_S1_PlainPatchConcat']['det_2cls_AP50']:.2f} -> "
              f"★BG-OPTA={sr['Ours_S3_BGOPTA_1Pass']['det_2cls_AP50']:.2f} || 零丢框3cls AP50: Full-DINO={sr['B1_Full_DINOv2_1Crop']['det_3cls_repaired_AP50']:.2f} | "
              f"S1={sr['Abl_S1_PlainPatchConcat']['det_3cls_repaired_AP50']:.2f} -> ★BG-OPTA={sr['Ours_S3_BGOPTA_1Pass']['det_3cls_repaired_AP50']:.2f}",
              flush=True)
        per_seed.append(sr)

    zen_agg = {}
    for mk in ['B0_YOLOv8n_Native', 'B1_Full_DINOv2_1Crop', 'B3_Static_Posterior_Fuse', 'Abl_S1_PlainPatchConcat', 'Ours_S3_BGOPTA_1Pass', 'Ours_S3_BGOPTA_SafeBypass']:
        zen_agg[mk] = {}
        for metric in ['det_2cls_AP50', 'det_2cls_AP50_95', 'det_3cls_repaired_AP50', 'det_3cls_repaired_AP50_95', 'gt_matched_3cls_repaired_acc', 'gt_matched_strict_9to13_penalized_acc']:
            vals = [s[mk][metric] for s in per_seed]
            zen_agg[mk][metric] = {
                'mean': round(float(np.mean(vals)), 2),
                'std': round(float(np.std(vals, ddof=1)), 2),
                'mean_std': f"{np.mean(vals):.2f}±{np.std(vals, ddof=1):.2f}",
                'seeds': vals,
            }
    return {'per_seed': per_seed, 'summary': zen_agg}


# ==============================================================================
# 模块 D: CitrusUAT (500图) 真实物理叶片掩膜能量聚集可解释性与跨域器官诊断
# ==============================================================================
def extract_citrusuat_physical_leaf_patch_mask(rgb_s: np.ndarray, bbox_s: list, scale: float = CROP_SCALE) -> np.ndarray:
    """
    从 CitrusUAT 受控中性灰底板恢复真实物理叶片掩膜并投影至 16x16 补丁网格（全程严禁输入模型，仅作可解释性评价真值！）。
    """
    h, w = rgb_s.shape[:2]
    bdr = np.concatenate([rgb_s[0, :], rgb_s[-1, :], rgb_s[:, 0], rgb_s[:, -1]], axis=0)
    med = np.median(bdr, axis=0).astype(np.float32)
    diff = np.max(np.abs(rgb_s.astype(np.float32) - med), axis=-1)
    sat = np.max(rgb_s, axis=-1).astype(np.float32) - np.min(rgb_s, axis=-1).astype(np.float32)
    raw_fg = ((diff > 14.0) | (sat > 14.0)).astype(np.uint8) * 255

    # Close & fill internal holes of the largest connected leaf component
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(raw_fg, cv2.MORPH_CLOSE, kern)
    n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if n_lbl > 1:
        best_cc = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        leaf_u8 = (lbls == best_cc).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(leaf_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(leaf_u8, cnts, -1, 255, thickness=cv2.FILLED)
    else:
        leaf_u8 = closed

    x, y, bw, bh = bbox_s
    cx, cy = x + bw * 0.5, y + bh * 0.5
    side = max(8, int(math.ceil(max(bw, bh) * scale)))
    x1, y1 = int(math.floor(cx - side * 0.5)), int(math.floor(cy - side * 0.5))
    x2, y2 = x1 + side, y1 + side
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(w, x2), min(h, y2)
    m_canvas = np.zeros((side, side), dtype=np.float32)
    if sx2 > sx1 and sy2 > sy1:
        m_canvas[(sy1 - y1):(sy2 - y1), (sx1 - x1):(sx2 - x1)] = (leaf_u8[sy1:sy2, sx1:sx2] > 127).astype(np.float32)
    patch_gt_fg = cv2.resize(m_canvas, (16, 16), interpolation=cv2.INTER_AREA).reshape(-1)
    return patch_gt_fg  # (256,) in [0, 1]


@torch.no_grad()
def eval_citrusuat_mask_focus_and_transfer():
    print("\n================ [模块 D] CitrusUAT (500图) 叶片掩膜前景能量聚焦验证与跨域诊断 ================", flush=True)
    device = 'cuda:0'
    # 5 aligned CitrusUAT folders -> SciDB exact IDs:
    # Greasy_spot -> {10}, Healthy -> {0, 1}, HLB -> {2, 3}, Mg -> {7}, Zn -> {4}
    uat_classes = ['Greasy_spot', 'Healthy', 'HLB', 'Mg', 'Zn']
    uat_groups = [[10], [0, 1], [2, 3], [7], [4]]

    samples = []
    for ci, cname in enumerate(uat_classes):
        for p in sorted((CITRUSUAT_DIR / cname).glob('*.*')):
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                samples.append((ci, p))

    def dec_uat(item):
        ci, p = item
        bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), 1)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if max(h, w) > 1280:
            f = 1280.0 / max(h, w)
            bgr_s = cv2.resize(bgr, (max(1, int(round(w * f))), max(1, int(round(h * f)))), interpolation=cv2.INTER_AREA)
            rgb_s = cv2.cvtColor(bgr_s, cv2.COLOR_BGR2RGB)
        else:
            bgr_s, rgb_s = bgr, rgb
        return ci, bgr_s, rgb_s

    decoded = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for d in ex.map(dec_uat, samples):
            decoded.append(d)
    y_true = np.array([d[0] for d in decoded], dtype=np.int64)

    mask_focus_seeds = []
    cls_seeds = {k: {'aligned_5cls_acc': [], 'aligned_5cls_f1': [], 'strict_14cls_acc': []}
                 for k in ['B0_YOLOv8n', 'B1_Full_DINOv2', 'Abl_S1_PlainConcat', 'Ours_S3_BGOPTA']}

    for seed in [0, 1, 2]:
        ckpt = torch.load(CKPT_DIR / f'bgopta_frozen_seed{seed}.pt', map_location='cpu', weights_only=False)
        al_lock = ckpt['val_frozen_hyperparams']['alpha_det']
        yolo_ext = YoloFullStateExtractor(Path(r'D:\citrus_runs') / f'yolov8n_unified_seed{seed}' / 'weights' / 'best.pt', device)
        dino_ext = DinoExtractor(seed, device)
        m_s1 = BGOPTAHead(dino_ext.head.weight.data.cpu(), dino_ext.head.bias.data.cpu(), mode='S1_plain_concat').to(device).eval()
        m_s1.load_state_dict(ckpt['S1_plain_concat'])
        m_s3 = BGOPTAHead(dino_ext.head.weight.data.cpu(), dino_ext.head.bias.data.cpu(), mode='S3_bgopta_full').to(device).eval()
        m_s3.load_state_dict(ckpt['S3_bgopta_full'])

        n_s = len(decoded)
        p14_arr = np.zeros((n_s, 14), dtype=np.float32)
        geom_arr = np.zeros((n_s, 9), dtype=np.float32)
        c125_arr = np.zeros((n_s, 3, 224, 224), dtype=np.uint8)
        c150_arr = np.zeros((n_s, 3, 224, 224), dtype=np.uint8)
        m150_arr = np.zeros((n_s, 2, 256), dtype=np.bool_)
        gt_patch_arr = np.zeros((n_s, 256), dtype=np.float32)
        b0_cls_arr = np.zeros(n_s, dtype=np.int64)

        for start in range(0, n_s, 16):
            sub = decoded[start:start + 16]
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
                    tc = -1
                bxs_b.append([bb])
                tcls_b.append([tc if tc >= 0 else None])
                geom_arr[idx] = compute_box_geom(bb, w, h, [bb])
                c125_arr[idx] = crop_square_scale(rgb_s, bb, 1.25)
                c150, m150 = crop_single_context_and_masks(rgb_s, bb, CROP_SCALE)
                c150_arr[idx] = c150
                m150_arr[idx] = m150
                gt_patch_arr[idx] = extract_citrusuat_physical_leaf_patch_mask(rgb_s, bb, CROP_SCALE)

            ps, _ = yolo_ext.extract_batch(bgrs, bxs_b, tcls_b)
            for k in range(len(sub)):
                p14_arr[start + k] = ps[k]
                b0_cls_arr[start + k] = int(np.argmax(ps[k]))

        _, lgs_125 = dino_ext.encode_crops(c125_arr)
        tok_150 = encode_tokens_fp16(dino_ext.enc, c150_arr, device)
        masks_g = torch.from_numpy(m150_arr).to(device)
        geom_g = torch.from_numpy(geom_arr).to(device)
        lg_s1 = m_s1(tok_150, masks_g, geom_g)
        lg_s3, w_fg_s3 = m_s3.forward_with_diag(tok_150, masks_g, geom_g)
        w_fg_np = w_fg_s3.cpu().numpy()
        m_box_np = m150_arr[:, 0].astype(np.float32)

        # Compute Foreground Leaf Energy Purity & Corner Suppression inside M_box
        unif_purity = float(((m_box_np * gt_patch_arr).sum(axis=1) / (m_box_np.sum(axis=1) + 1e-6)).mean() * 100.0)
        bgopta_purity = float(((w_fg_np * gt_patch_arr).sum(axis=1) / (w_fg_np.sum(axis=1) + 1e-6)).mean() * 100.0)
        in_lamina = (m_box_np > 0.5) & (gt_patch_arr >= 0.75)
        in_corner_bg = (m_box_np > 0.5) & (gt_patch_arr <= 0.20)
        mean_w_lam = float(w_fg_np[in_lamina].mean())
        mean_w_cor = float(w_fg_np[in_corner_bg].mean())
        corner_supp_pct = (1.0 - mean_w_cor / max(mean_w_lam, 1e-6)) * 100.0
        mask_focus_seeds.append({
            'seed': seed,
            'uniform_box_leaf_purity_pct': round(unif_purity, 2),
            'bgopta_soft_fg_leaf_purity_pct': round(bgopta_purity, 2),
            'purity_gain_pt': round(bgopta_purity - unif_purity, 2),
            'mean_w_true_lamina': round(mean_w_lam, 4),
            'mean_w_bg_corners_inside_box': round(mean_w_cor, 4),
            'corner_bg_suppression_pct': round(corner_supp_pct, 2),
        })

        lp_det = F.log_softmax(torch.logit(torch.from_numpy(p14_arr).clamp(1e-5, 1.0 - 1e-5)) / 1.35, dim=-1).numpy()
        p_det = np.exp(lp_det)
        p_b1 = F.softmax(torch.from_numpy(lgs_125), dim=-1).numpy()
        p_s1 = np.exp(al_lock * lp_det + (1.0 - al_lock) * F.log_softmax(lg_s1 / 1.15, dim=-1).cpu().numpy())
        p_s3 = np.exp(al_lock * lp_det + (1.0 - al_lock) * F.log_softmax(lg_s3 / 1.15, dim=-1).cpu().numpy())

        for mk, pmat in [('B0_YOLOv8n', p_det), ('B1_Full_DINOv2', p_b1), ('Abl_S1_PlainConcat', p_s1), ('Ours_S3_BGOPTA', p_s3)]:
            raw_c14 = pmat.argmax(axis=1)
            strict_ok = [int(raw_c14[i]) in uat_groups[y_true[i]] for i in range(n_s)]
            grp_scores = np.stack([pmat[:, grp].sum(axis=1) for grp in uat_groups], axis=1)
            pred_5 = grp_scores.argmax(axis=1)
            cls_seeds[mk]['strict_14cls_acc'].append(round(float(np.mean(strict_ok) * 100.0), 2))
            cls_seeds[mk]['aligned_5cls_acc'].append(round(float(accuracy_score(y_true, pred_5) * 100.0), 2))
            cls_seeds[mk]['aligned_5cls_f1'].append(round(float(f1_score(y_true, pred_5, average='macro') * 100.0), 2))

        print(f"  [CitrusUAT Seed {seed}] 框内真实叶片能量纯度: 均匀框池化={unif_purity:.2f}% -> ★BG-OPTA={bgopta_purity:.2f}% "
              f"(四角灰底抑制率={corner_supp_pct:.1f}%) | 5类F1: Full-DINO={cls_seeds['B1_Full_DINOv2']['aligned_5cls_f1'][-1]:.2f}% -> "
              f"★BG-OPTA={cls_seeds['Ours_S3_BGOPTA']['aligned_5cls_f1'][-1]:.2f}%", flush=True)

        yolo_ext.close()
        del yolo_ext, dino_ext, m_s1, m_s3, tok_150
        torch.cuda.empty_cache()

    cls_summary = {}
    for mk, d in cls_seeds.items():
        cls_summary[mk] = {
            m: {'mean_std': f"{np.mean(v):.2f}±{np.std(v, ddof=1):.2f}", 'seeds': v}
            for m, v in d.items()
        }
    return {
        'n_images': len(decoded),
        'mask_focus_per_seed': mask_focus_seeds,
        'mask_focus_mean': {
            'uniform_box_leaf_purity_pct': round(float(np.mean([x['uniform_box_leaf_purity_pct'] for x in mask_focus_seeds])), 2),
            'bgopta_soft_fg_leaf_purity_pct': round(float(np.mean([x['bgopta_soft_fg_leaf_purity_pct'] for x in mask_focus_seeds])), 2),
            'corner_bg_suppression_pct': round(float(np.mean([x['corner_bg_suppression_pct'] for x in mask_focus_seeds])), 2),
        },
        'cross_domain_5cls_summary': cls_summary,
    }


# ==============================================================================
# 模块 E: 真正零接触最终封口盲测 (Vault Holdout: Sweet Orange V1 去重原图 1,389 张)
# ==============================================================================
@torch.no_grad()
def eval_vault_sweet_orange_oneshot():
    print("\n================ [模块 E] 真正零接触最终封口盲测 (Sweet Orange V1 去重原图 1,389 张) ================", flush=True)
    device = 'cuda:0'
    idx_all = json.loads(AUDIT_INDEX_JSON.read_text(encoding='utf-8'))
    # Strictly aligned 3 symptom groups never touched in detection iteration:
    # Healthy leaf -> {0, 1}, Citrus canker -> {11} (S14 Canker), Citrus greening -> {2, 3} (S3/S4 symptom suspect)
    lbl_map = {'Healthy leaf': 0, 'Citrus canker': 1, 'Citrus greening': 2}
    sci_groups = [[0, 1], [11], [2, 3]]

    vault_recs = [
        r for r in idx_all
        if r.get('数据集') == 'f7cr74mwpj' and r.get('索引去重保留') and not r.get('同库跨标签冲突')
        and r.get('解码状态') == '通过' and r.get('原始标签') in lbl_map
    ]

    def dec_v(r):
        p = Path(r['路径'])
        bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), 1)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if max(h, w) > 1280:
            f = 1280.0 / max(h, w)
            bgr = cv2.resize(bgr, (max(1, int(round(w * f))), max(1, int(round(h * f)))), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return lbl_map[r['原始标签']], bgr, rgb

    v_data = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for item in ex.map(dec_v, vault_recs):
            v_data.append(item)
    y_v = np.array([x[0] for x in v_data], dtype=np.int64)
    n_v = len(v_data)
    print(f"  [Vault 盲测集] 共加载 {n_v} 张去重原图 (Healthy={(y_v==0).sum()}, Canker={(y_v==1).sum()}, Greening-suspect={(y_v==2).sum()})", flush=True)

    vault_res = {k: {'strict_14cls_hit': [], 'aligned_3cls_acc': [], 'aligned_3cls_f1': []}
                 for k in ['B0_YOLOv8n_Native', 'B1_Full_DINOv2_1Crop', 'Abl_S1_PlainPatchConcat', 'Ours_S3_BGOPTA_1Pass', 'Ours_S3_BGOPTA_SafeBypass']}
    det_recall_seeds = []

    for seed in [0, 1, 2]:
        ckpt = torch.load(CKPT_DIR / f'bgopta_frozen_seed{seed}.pt', map_location='cpu', weights_only=False)
        hp = ckpt['val_frozen_hyperparams']
        al_lock, sb_conf, sb_mar, sb_maha = hp['alpha_det'], hp['safe_bypass_conf'], hp['safe_bypass_margin'], hp['safe_bypass_maha_z']
        m_cfg = ckpt['mahalanobis']

        yolo_ext = YoloFullStateExtractor(Path(r'D:\citrus_runs') / f'yolov8n_unified_seed{seed}' / 'weights' / 'best.pt', device)
        dino_ext = DinoExtractor(seed, device)
        m_s1 = BGOPTAHead(dino_ext.head.weight.data.cpu(), dino_ext.head.bias.data.cpu(), mode='S1_plain_concat').to(device).eval()
        m_s1.load_state_dict(ckpt['S1_plain_concat'])
        m_s3 = BGOPTAHead(dino_ext.head.weight.data.cpu(), dino_ext.head.bias.data.cpu(), mode='S3_bgopta_full').to(device).eval()
        m_s3.load_state_dict(ckpt['S3_bgopta_full'])

        p14_arr = np.zeros((n_v, 14), dtype=np.float32)
        zneck_arr = np.zeros((n_v, 192), dtype=np.float32)
        geom_arr = np.zeros((n_v, 9), dtype=np.float32)
        c125_arr = np.zeros((n_v, 3, 224, 224), dtype=np.uint8)
        c150_arr = np.zeros((n_v, 3, 224, 224), dtype=np.uint8)
        m150_arr = np.zeros((n_v, 2, 256), dtype=np.bool_)
        yolo_sc_arr = np.zeros(n_v, dtype=np.float64)
        det_hits = 0

        for start in range(0, n_v, 24):
            sub = v_data[start:start + 24]
            bgrs = [x[1] for x in sub]
            res_b = yolo_ext.yolo.predict(bgrs, imgsz=640, conf=0.05, iou=0.7, max_det=20, device=0, half=False, rect=False, verbose=False)
            bxs_b, tcls_b = [], []
            for k, ((_, _, rgb), r) in enumerate(zip(sub, res_b)):
                idx = start + k
                h, w = rgb.shape[:2]
                if len(r.boxes) > 0:
                    det_hits += 1
                    x1, y1, x2, y2 = r.boxes.xyxy[0].cpu().tolist()
                    bb = [x1, y1, x2 - x1, y2 - y1]
                    tc = int(r.boxes.cls[0].item())
                    yolo_sc_arr[idx] = float(r.boxes.conf[0].item())
                else:
                    bb = [w * 0.06, h * 0.06, w * 0.88, h * 0.88]
                    tc = -1
                    yolo_sc_arr[idx] = 0.10
                bxs_b.append([bb])
                tcls_b.append([tc if tc >= 0 else None])
                geom_arr[idx] = compute_box_geom(bb, w, h, [bb])
                c125_arr[idx] = crop_square_scale(rgb, bb, 1.25)
                c150, m150 = crop_single_context_and_masks(rgb, bb, CROP_SCALE)
                c150_arr[idx] = c150
                m150_arr[idx] = m150
            ps, zs = yolo_ext.extract_batch(bgrs, bxs_b, tcls_b)
            for k in range(len(sub)):
                p14_arr[start + k] = ps[k]
                zneck_arr[start + k] = zs[k]

        det_recall_seeds.append(round(det_hits / n_v * 100.0, 2))
        _, lgs_125 = dino_ext.encode_crops(c125_arr)
        tok_150 = encode_tokens_fp16(dino_ext.enc, c150_arr, device)
        masks_g = torch.from_numpy(m150_arr).to(device)
        geom_g = torch.from_numpy(geom_arr).to(device)
        lg_s1 = m_s1(tok_150, masks_g, geom_g)
        lg_s3 = m_s3(tok_150, masks_g, geom_g)

        lp_det = F.log_softmax(torch.logit(torch.from_numpy(p14_arr).clamp(1e-5, 1.0 - 1e-5)) / 1.35, dim=-1).numpy()
        p_det = np.exp(lp_det)
        t2 = np.sort(p_det, axis=1)[:, -2:]
        mar = t2[:, 1] - t2[:, 0]
        maha_14 = compute_maha_z(zneck_arr, m_cfg['means'], m_cfg['prec'], m_cfg['m_mu'], m_cfg['m_sd'])
        dm_z = maha_14[np.arange(n_v), p_det.argmax(axis=1)]
        safe_bypass = (yolo_sc_arr >= sb_conf) & (mar >= sb_mar) & (dm_z <= sb_maha)

        p_b1 = F.softmax(torch.from_numpy(lgs_125), dim=-1).numpy()
        p_s1 = np.exp(al_lock * lp_det + (1.0 - al_lock) * F.log_softmax(lg_s1 / 1.15, dim=-1).cpu().numpy())
        p_s3 = np.exp(al_lock * lp_det + (1.0 - al_lock) * F.log_softmax(lg_s3 / 1.15, dim=-1).cpu().numpy())
        p_s3_sb = p_s3.copy()
        p_s3_sb[safe_bypass] = p_det[safe_bypass]

        for mk, pmat in [
            ('B0_YOLOv8n_Native', p_det),
            ('B1_Full_DINOv2_1Crop', p_b1),
            ('Abl_S1_PlainPatchConcat', p_s1),
            ('Ours_S3_BGOPTA_1Pass', p_s3),
            ('Ours_S3_BGOPTA_SafeBypass', p_s3_sb),
        ]:
            c14 = pmat.argmax(axis=1)
            st_hit = [int(c14[i]) in sci_groups[y_v[i]] for i in range(n_v)]
            g_sc = np.stack([pmat[:, grp].sum(axis=1) for grp in sci_groups], axis=1)
            p3 = g_sc.argmax(axis=1)
            vault_res[mk]['strict_14cls_hit'].append(round(float(np.mean(st_hit) * 100.0), 2))
            vault_res[mk]['aligned_3cls_acc'].append(round(float(accuracy_score(y_v, p3) * 100.0), 2))
            vault_res[mk]['aligned_3cls_f1'].append(round(float(f1_score(y_v, p3, average='macro') * 100.0), 2))

        print(f"  [Vault SweetOrange Seed {seed}] 检出率={det_recall_seeds[-1]:.1f}% | 严格14类零遮蔽命中率: B0={vault_res['B0_YOLOv8n_Native']['strict_14cls_hit'][-1]:.2f}% | "
              f"Full-DINO={vault_res['B1_Full_DINOv2_1Crop']['strict_14cls_hit'][-1]:.2f}% | S1={vault_res['Abl_S1_PlainPatchConcat']['strict_14cls_hit'][-1]:.2f}% -> "
              f"★BG-OPTA={vault_res['Ours_S3_BGOPTA_1Pass']['strict_14cls_hit'][-1]:.2f}% || 3症候群 Macro-F1: "
              f"Full-DINO={vault_res['B1_Full_DINOv2_1Crop']['aligned_3cls_f1'][-1]:.2f}% -> ★BG-OPTA={vault_res['Ours_S3_BGOPTA_1Pass']['aligned_3cls_f1'][-1]:.2f}%",
              flush=True)

        yolo_ext.close()
        del yolo_ext, dino_ext, m_s1, m_s3, tok_150
        torch.cuda.empty_cache()

    vault_summary = {}
    for mk, d in vault_res.items():
        vault_summary[mk] = {
            m: {'mean_std': f"{np.mean(v):.2f}±{np.std(v, ddof=1):.2f}", 'seeds': v}
            for m, v in d.items()
        }
    return {
        'dataset': 'Sweet_Orange_V1_f7cr74mwpj_dedup_holdout',
        'n_images': n_v,
        'class_counts': {'Healthy_leaf': int((y_v == 0).sum()), 'Citrus_canker': int((y_v == 1).sum()), 'Citrus_greening_symptom': int((y_v == 2).sum())},
        'detector_organ_hit_rate_pct': det_recall_seeds,
        'summary': vault_summary,
    }


def main():
    t0 = time.monotonic()
    backend = configure_strict_fp32()
    p1 = verify_phase1_lock()

    mod_a = eval_scidb_frozen_3seeds(p1)
    mod_b = benchmark_5tier_latency()
    mod_c = eval_zenodo_repaired_3seeds()
    mod_d = eval_citrusuat_mask_focus_and_transfer()
    mod_e = eval_vault_sweet_orange_oneshot()

    out_path = RES_DIR / 'phase2_frozen_oneshot_complete_eval.json'
    out_path.write_text(json.dumps({
        '阶段': 'Phase2_完全冻结权重与参数单次全量评测',
        '总耗时秒': round(time.monotonic() - t0, 2),
        '严格评测后端': backend,
        'Phase1锁定校验': p1['冻结权重与参数清单'],
        '模块A_SciDB跨条件诊断与全消融': mod_a,
        '模块B_RTX4060Ti五档公平单图延迟与调用次数': mod_b,
        '模块C_修复版Zenodo外部果园零丢框探索评测': mod_c,
        '模块D_CitrusUAT真实叶片掩膜前景聚焦与跨域诊断': mod_d,
        '模块E_未触碰SweetOrange最终封口盲测': mod_e,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n[Phase 2 全部评测一次性完成] 完整可审计记录已保存至: {out_path} (总耗时 {time.monotonic() - t0:.1f}s)", flush=True)


if __name__ == '__main__':
    main()
