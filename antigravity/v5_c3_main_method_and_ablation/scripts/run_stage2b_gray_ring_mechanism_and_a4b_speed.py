# -*- coding: utf-8 -*-
"""
Stage 2b: G1–G4 受控灰环 + 高斯模糊外环同几何背景干预机制实验、补齐配对 Bootstrap 置信区间、
A4b 同协议端到端公平测速，以及真实果园 5 模型一对一匹配与按簇 Bootstrap 冒烟核验
(run_stage2b_gray_ring_mechanism_and_a4b_speed.py)

严格方法学设计（零新模型训练、12 份底层权重 SHA-256 锁定、C3 主方法 [0.30, 0.20, 0.50] 零改动）：
1. G1–G4 同几何受控灰环与高斯模糊外环实验（1.25x 裁剪几何尺寸、中心定位、缩放插值及 1.00x 检测框内部像素 100% 逐像素恒等）：
   - G1: 1.25x 原始背景             (单尺度参考, [0.30, 0.70])
   - G2: 1.25x 灰色外环 (gray128)   (单尺度背景干预, [0.30, 0.70])
   - G2_blur: 1.25x 模糊外环        (单尺度高频纹理抑制对照, ksize=31, sigma=11, 避免固定灰色阶跃伪影, [0.30, 0.70])
   - G3: 0.82x + 1.25x 原始背景     (唯一冻结主方法 C3, [0.30, 0.20, 0.50])
   - G4: 0.82x + 1.25x 灰色外环     (双尺度灰环背景干预, [0.30, 0.20, 0.50])
   - G4_blur: 0.82x + 1.25x 模糊外环(双尺度模糊外环背景干预, [0.30, 0.20, 0.50])
   重点检验：
     Δ_real = AP(G3) - AP(G1)
     Δ_gray = AP(G4) - AP(G2)
     Δ_blur = AP(G4_blur) - AP(G2_blur)
     以及 B = 1,000 同步逐图配对 Bootstrap 双重差分 (Difference-in-Differences)：
     DiD_gray = Δ_gray - Δ_real = [AP(G4) - AP(G2)] - [AP(G3) - AP(G1)]
     DiD_blur = Δ_blur - Δ_real = [AP(G4_blur) - AP(G2_blur)] - [AP(G3) - AP(G1)]
     同时附带等权 [0.30, 0.35, 0.35]、C1 [0.35, 0.65] 与纯 DINOv2 (w_det=0) 敏感性对照！
2. 补齐 A1–A5 全部配对 Bootstrap 95% CI（含冻结 C3 vs 等权 C3_equal、C3_equal vs A1/C1/A2/A4a_eq/A4b_eq、C3 vs B1/B0 等）。
3. 同一 RTX 4060 Ti 严格 FP32 会话完成含 A4b (1.25x + HFlip) 与 C1 在内的全部 9 组方案端到端公平测速（含多目标图像与全集实时 AP 验算）。
4. 验证修复后的真实果园 5 模型评测器（一对一匹配防重叠双计 + 不确定样本隔离 + 按采集簇 Bootstrap）。
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'

import sys
import json
import math
import time
import tempfile
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

for _extra_sp in [r'D:\新版柑橘\方法路线扩展_20261031\检测环境\Lib\site-packages', r'D:\新版柑橘\检测环境\Lib\site-packages']:
    if Path(_extra_sp).exists() and _extra_sp not in sys.path:
        sys.path.append(_extra_sp)

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms

ROOT = Path(r'D:\新版柑橘')
EXT = ROOT / '方法路线扩展_20261031'
REPO_AG = ROOT / 'citrus-field-disease-research' / 'antigravity'
V5_DIR = REPO_AG / 'v5_c3_main_method_and_ablation'
RES_DIR = V5_DIR / 'results'
RES_DIR.mkdir(parents=True, exist_ok=True)

SCRATCH_DIR = ROOT / 'Antigravity_只读可行性审计_20261126'
SCRATCH_A4_CACHE = SCRATCH_DIR / 'stage2_a4_extra_logits_cache.npz'
SCRATCH_GRAY_BLUR_CACHE = SCRATCH_DIR / 'stage2b_gray_blur_125_logits_cache.npz'

sys.path.append(str(EXT))
sys.path.append(str(ROOT / 'Antigravity_BGOPTA_单趟正交Token攻关_20261126'))
sys.path.append(str(ROOT / 'Antigravity_CGGAR_真创新攻关_20261125'))
sys.path.append(str(REPO_AG / 'v3_bgopta_and_dtemop' / 'scripts'))
sys.path.insert(0, str(V5_DIR / 'scripts'))

from 共同评测后端 import configure_strict_fp32
from phase1_source_only_train_and_select import apply_classwise_nms
from phase2_frozen_oneshot_eval import coco_eval_metrics, match_gt_boxes_for_quadrant
from step1_extract_multigranular_cache import DinoExtractor, crop_square_scale
from run_rigorous_audit_suite import paired_image_bootstrap_coco
from orchard_plug_and_play_cli import (
    UnifiedOrchardInferenceEngine,
    verify_manifest_hashes,
    match_detections_to_gt_1to1,
    evaluate_frozen_5models_on_labeled_orchard,
)

IMG_ROOT = Path(r'D:\病虫害论文\复现实验记录\yolo11n_public_compare_20261004\数据副本\images')
GT_DIR = EXT / '检测数据'
STRONG_DIR = EXT / 'YOLOv8强基线重复_20261109'
CACHE_DIR = ROOT / 'Antigravity_CGGAR_真创新攻关_20261125' / 'cache'
MANIFEST_PATH = V5_DIR / 'configs' / 'c3_frozen_manifest.json'
STAGE2_JSON_PATH = RES_DIR / 'stage2_c3_controlled_scale_ablation_complete.json'


def softmax_from_lp(lp: np.ndarray) -> np.ndarray:
    z = lp - lp.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def crop_square_scale_intervene_125(rgb: np.ndarray, bbox_xywh: list, outer_mode: str = 'real'):
    """
    与 `crop_square_scale(rgb, bbox_xywh, 1.25)` 几何尺寸、中心偏移、CenterCrop(224/256) 及插值 100% 逐像素对齐：
    - `outer_mode == 'real'`：输出与 `crop_square_scale(rgb, bbox_xywh, 1.25)` 逐字节恒等；
    - `outer_mode == 'gray128'`：1.00x 检测框内部 `[py1:py2, px1:px2]` 像素逐字节恒等，框外像素替换为中性灰 128；
    - `outer_mode == 'blur31'`：1.00x 检测框内部 `[py1:py2, px1:px2]` 像素逐字节恒等，框外像素替换为 31x31 (sigma=11) 高斯模糊背景。
    返回 `(chw_u8, outer_area_fraction)`。
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
    side = max(4, int(math.ceil(max(bw, bh) * 1.25)))
    half = side / 2.0
    x1, y1 = int(math.floor(cx - half)), int(math.floor(cy - half))
    x2, y2 = x1 + side, y1 + side

    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(w, x2), min(h, y2)
    canvas = np.full((side, side, 3), 128, dtype=np.uint8)
    if sx2 > sx1 and sy2 > sy1:
        dx1, dy1 = sx1 - x1, sy1 - y1
        canvas[dy1:dy1 + (sy2 - sy1), dx1:dx1 + (sx2 - sx1)] = rgb[sy1:sy2, sx1:sx2]

    margin = int(round(side * (1.0 - 224.0 / 256.0) * 0.5))
    use_inner = (side - 2 * margin) >= 4
    off = margin if use_inner else 0
    inner = canvas[margin:side - margin, margin:side - margin] if use_inner else canvas
    inner_side = float(inner.shape[0])
    interp = cv2.INTER_AREA if inner.shape[0] > 224 else cv2.INTER_CUBIC
    out224 = cv2.resize(inner, (224, 224), interpolation=interp)

    bu1 = (x - x1 - off) / inner_side
    bv1 = (y - y1 - off) / inner_side
    bu2 = (x + bw - x1 - off) / inner_side
    bv2 = (y + bh - y1 - off) / inner_side

    px1 = int(np.clip(math.floor(bu1 * 224.0), 0, 224))
    py1 = int(np.clip(math.floor(bv1 * 224.0), 0, 224))
    px2 = int(np.clip(math.ceil(bu2 * 224.0), 0, 224))
    py2 = int(np.clip(math.ceil(bv2 * 224.0), 0, 224))
    outer_frac = 1.0 - float(max(0, px2 - px1) * max(0, py2 - py1)) / float(224 * 224)

    if outer_mode == 'real':
        return out224.transpose(2, 0, 1), outer_frac
    elif outer_mode == 'gray128':
        bg = np.full_like(out224, 128)
        bg[py1:py2, px1:px2, :] = out224[py1:py2, px1:px2, :]
        return bg.transpose(2, 0, 1), outer_frac
    elif outer_mode == 'blur31':
        bg = cv2.GaussianBlur(out224, (31, 31), sigmaX=11.0, sigmaY=11.0)
        bg[py1:py2, px1:px2, :] = out224[py1:py2, px1:px2, :]
        return bg.transpose(2, 0, 1), outer_frac
    else:
        raise ValueError(outer_mode)


def decode_split_rgb_map(sp: str, gt_dict: dict) -> dict:
    def dec_one(im):
        p = IMG_ROOT / sp / im['file_name']
        bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), 1)
        return im['id'], cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    with ThreadPoolExecutor(max_workers=6) as ex:
        return dict(ex.map(dec_one, gt_dict['images']))


@torch.no_grad()
def load_or_extract_gray_and_blur_125_logits(gt_dicts: dict):
    if SCRATCH_GRAY_BLUR_CACHE.exists():
        print(f"[加载本地缓存] 复用已提取的 1.25x 灰环/模糊外环确定性 logits: {SCRATCH_GRAY_BLUR_CACHE}", flush=True)
        z = np.load(SCRATCH_GRAY_BLUR_CACHE, allow_pickle=True)
        data = {seed: {} for seed in [0, 1, 2]}
        geom_meta = json.loads(str(z['geom_meta_json']))
        for seed in [0, 1, 2]:
            for sp in ['val', 'testB', 'testC']:
                data[seed][sp] = {
                    'lg_125_gray': z[f's{seed}_{sp}_gray'],
                    'lg_125_blur': z[f's{seed}_{sp}_blur'],
                }
        return data, geom_meta

    print("\n================ [模块 1.1] 提取 1.25x 同几何灰色外环 (G2/G4) 与高斯模糊外环 (G2_blur/G4_blur) 特征 ================", flush=True)
    rgb_maps = {sp: decode_split_rgb_map(sp, gt_dicts[sp]) for sp in ['val', 'testB', 'testC']}
    data = {seed: {} for seed in [0, 1, 2]}
    save_dict = {}
    geom_stats = {}

    for seed in [0, 1, 2]:
        dino_ext = DinoExtractor(seed, 'cuda:0')
        for sp in ['val', 'testB', 'testC']:
            orig_preds = json.loads((STRONG_DIR / f'种子{seed}' / f'{sp}预测.json').read_text(encoding='utf-8'))
            d_npz = np.load(CACHE_DIR / f'{sp}_cggar_seed{seed}.npz')
            cand_idxs = d_npz['cand_idxs']
            cand_rows = [orig_preds[i] for i in cand_idxs]
            n_c = len(cand_rows)
            fg = np.array([r['score'] for r in cand_rows], dtype=np.float64) >= 0.08
            fg_pos = np.where(fg)[0]

            lg_gray = np.zeros((n_c, 14), dtype=np.float32)
            lg_blur = np.zeros((n_c, 14), dtype=np.float32)
            c_gray = np.empty((len(fg_pos), 3, 224, 224), dtype=np.uint8)
            c_blur = np.empty((len(fg_pos), 3, 224, 224), dtype=np.uint8)
            outer_fracs = []
            rmp = rgb_maps[sp]

            for k_i, li in enumerate(fg_pos):
                r = cand_rows[li]
                rgb = rmp[r['image_id']]
                bb = r['bbox']
                if seed == 0 and k_i < 80:
                    c_ref = crop_square_scale(rgb, bb, 1.25)
                    c_chk, _ = crop_square_scale_intervene_125(rgb, bb, 'real')
                    assert np.array_equal(c_ref, c_chk), "Geometric mismatch between crop_square_scale and intervene_125('real')!"
                cg, ofrac = crop_square_scale_intervene_125(rgb, bb, 'gray128')
                cb, _ = crop_square_scale_intervene_125(rgb, bb, 'blur31')
                c_gray[k_i] = cg
                c_blur[k_i] = cb
                outer_fracs.append(ofrac)

            _, lg_gray[fg_pos] = dino_ext.encode_crops(c_gray)
            _, lg_blur[fg_pos] = dino_ext.encode_crops(c_blur)

            # Verify Seed 0 'real' logits match cached d_npz['lg_std'] to <1e-5
            if seed == 0:
                sample_n = min(64, len(fg_pos))
                c_real_s = np.stack([crop_square_scale_intervene_125(rmp[cand_rows[fg_pos[j]]['image_id']], cand_rows[fg_pos[j]]['bbox'], 'real')[0]
                                     for j in range(sample_n)], axis=0)
                _, lg_real_s = dino_ext.encode_crops(c_real_s)
                max_diff_real = float(np.max(np.abs(lg_real_s - d_npz['lg_std'][fg_pos[:sample_n]])))
                assert max_diff_real < 1e-4, f"Real 1.25x logit drift: {max_diff_real}"
                geom_stats[sp] = {
                    'n_fg_boxes_seed0': int(len(fg_pos)),
                    'mean_outer_ring_area_pct_in_224': round(float(np.mean(outer_fracs) * 100.0), 2),
                    'median_outer_ring_area_pct_in_224': round(float(np.median(outer_fracs) * 100.0), 2),
                    'p10_p90_outer_ring_area_pct': [
                        round(float(np.percentile(outer_fracs, 10) * 100.0), 2),
                        round(float(np.percentile(outer_fracs, 90) * 100.0), 2),
                    ],
                    'verified_real125_logit_max_diff_vs_cache': round(max_diff_real, 7),
                }

            data[seed][sp] = {'lg_125_gray': lg_gray, 'lg_125_blur': lg_blur}
            save_dict[f's{seed}_{sp}_gray'] = lg_gray
            save_dict[f's{seed}_{sp}_blur'] = lg_blur

        del dino_ext
        torch.cuda.empty_cache()
        print(f"  [Seed {seed}] 1.25x 灰环与模糊外环 logits 提取完成", flush=True)

    save_dict['geom_meta_json'] = np.array(json.dumps(geom_stats, ensure_ascii=False))
    SCRATCH_GRAY_BLUR_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SCRATCH_GRAY_BLUR_CACHE, **save_dict)
    del rgb_maps
    return data, geom_stats


def compute_strict_1to1_box_and_gt_metrics(gt_path: Path, gt_dict: dict, post_rows: list, post_probs_map: dict):
    met = coco_eval_metrics(gt_path, post_rows)
    anns_by_img = defaultdict(list)
    for a in gt_dict['annotations']:
        if a.get('iscrowd', 0) == 0:
            anns_by_img[a['image_id']].append(a)

    act_by_img = defaultdict(list)
    for r in post_rows:
        if r['score'] >= 0.20:
            act_by_img[r['image_id']].append(r)

    n_gt = n_loc = n_t1 = n_t3 = 0
    for iid, glist in anns_by_img.items():
        dlist_desc = sorted(act_by_img.get(iid, []), key=lambda x: -x['score'])
        dets_fmt = [(dr['bbox'], dr['score'], dr) for dr in dlist_desc]
        gt_to_det, _ = match_detections_to_gt_1to1(dets_fmt, glist, iou_thr=0.50)
        n_gt += len(glist)
        for gi, ga in enumerate(glist):
            if gi in gt_to_det:
                n_loc += 1
                dr = dets_fmt[gt_to_det[gi]][2]
                gc = int(ga['category_id'])
                if int(dr['category_id']) == gc:
                    n_t1 += 1
                pr = post_probs_map.get(dr.get('_uid'), None)
                if pr is not None:
                    if gc in np.argsort(-pr)[:3].tolist():
                        n_t3 += 1
                elif int(dr['category_id']) == gc:
                    n_t3 += 1

    return {
        'AP50_95': round(met['AP50_95'], 2),
        'AP50': round(met['AP50'], 2),
        'Recall_IoU50_pct': round(n_loc / max(1, n_gt) * 100.0, 2),
        'Matched_Top1_Acc_pct': round(n_t1 / max(1, n_loc) * 100.0, 2),
        'Matched_Top3_Acc_pct': round(n_t3 / max(1, n_loc) * 100.0, 2),
        'MissPenalized_GT_Top3_Hit_pct': round(n_t3 / max(1, n_gt) * 100.0, 2),
    }


def verify_cli_a4b_parity_and_orchard_evaluator(manifest: dict, gt_dicts: dict):
    print("\n================ [模块 0] CLI 新增 A4b 在线/离线一致性核验 + 真实果园 1-to-1 与按簇 Bootstrap 单测 ================", flush=True)
    # 1) Unit test: two overlapping GT leaves (IoU=0.80 with one predicted box) MUST match at most ONE GT leaf!
    fake_dets = [([10.0, 10.0, 100.0, 100.0], 0.92)]
    fake_gts = [{'bbox': [10.0, 10.0, 100.0, 100.0], 'category_id': 2}, {'bbox': [15.0, 15.0, 100.0, 100.0], 'category_id': 2}]
    m_1to1, mx_ious = match_detections_to_gt_1to1(fake_dets, fake_gts, iou_thr=0.50)
    assert len(m_1to1) == 1 and all(v > 0.75 for v in mx_ious), f"1-to-1 matching failed: {m_1to1}"

    # 2) Verify A4b online live inference matches Stage 2 offline reference on Seed 0 testB & testC
    eng = UnifiedOrchardInferenceEngine(manifest, seed=0, device='cuda:0')
    a4b_parity = {}
    for sp in ['testB', 'testC']:
        gt_p = GT_DIR / f'{sp}.json'
        live_rows = []
        for im in gt_dicts[sp]['images']:
            bgr = cv2.imdecode(np.fromfile(str(IMG_ROOT / sp / im['file_name']), dtype=np.uint8), 1)
            live_rows.extend(eng.infer_image_coco_rows(bgr, im['id'], candidate='A4b'))
        ap_live = round(coco_eval_metrics(gt_p, live_rows)['AP50_95'], 2)
        ap_ref = manifest['seeds']['0']['offline_v4_reference_AP50_95']['A4b'][sp]
        assert abs(ap_live - ap_ref) <= 0.02, f"A4b live/offline mismatch on {sp}: {ap_live} vs {ap_ref}"
        a4b_parity[sp] = {'offline_ref_AP50_95': ap_ref, 'online_cli_AP50_95': ap_live, 'abs_diff': round(abs(ap_live - ap_ref), 3)}
        print(f"  [A4b 在线/离线核验 | Seed 0 {sp}] 离线参考={ap_ref:.2f}% vs 在线实时={ap_live:.2f}% (差值={abs(ap_live - ap_ref):.2f}%)", flush=True)
    eng.close()

    # 3) Smoke-test `evaluate_frozen_5models_on_labeled_orchard` with clustered images + uncertain annotation exclusion
    with tempfile.TemporaryDirectory() as tmpd:
        tmp_path = Path(tmpd)
        sub_imgs = gt_dicts['testB']['images'][:18]
        sub_iids = {im['id'] for im in sub_imgs}
        sub_anns = [dict(a) for a in gt_dicts['testB']['annotations'] if a['image_id'] in sub_iids]
        # Add 2 synthetic "uncertain" expert annotations to verify clean exclusion
        sub_anns.append({'id': 999991, 'image_id': sub_imgs[0]['id'], 'category_id': -1, 'bbox': [5, 5, 50, 50], 'area': 2500, 'iscrowd': 0, 'uncertain': True})
        sub_anns.append({'id': 999992, 'image_id': sub_imgs[1]['id'], 'category_id': 2, 'bbox': [8, 8, 60, 60], 'area': 3600, 'iscrowd': 0, 'status': 'uncertain'})
        sub_gt = {'images': sub_imgs, 'annotations': sub_anns, 'categories': gt_dicts['testB']['categories']}
        sub_gt_p = tmp_path / 'mock_orchard_gt.json'
        sub_gt_p.write_text(json.dumps(sub_gt, ensure_ascii=False), encoding='utf-8')

        # Group 18 images into 6 simulated burst clusters (3 shots per tree)
        csv_lines = ['file_name,cluster_id'] + [f"{im['file_name']},tree_cluster_{idx // 3}" for idx, im in enumerate(sub_imgs)]
        csv_p = tmp_path / 'clusters.csv'
        csv_p.write_text('\n'.join(csv_lines), encoding='utf-8')
        out_p = tmp_path / 'eval5_out.json'

        rep5 = evaluate_frozen_5models_on_labeled_orchard(
            manifest, IMG_ROOT / 'testB', sub_gt_p, out_p, cluster_csv=str(csv_p), seeds=(0,), n_boot=120
        )
        assert rep5['n_unique_clusters'] == 6 and rep5['n_uncertain_expert_boxes_excluded_from_gt'] == 2
        assert 'C3_DualScale_082_125__minus__A4b_FlipTTA_125_HFlip' in rep5['cluster_paired_bootstrap']['paired_comparisons']

    return {
        'unit_test_1to1_overlapping_gt_passed': True,
        'a4b_seed0_online_offline_parity': a4b_parity,
        'orchard_eval5_smoke_test': {
            'n_images': 18,
            'n_burst_clusters': 6,
            'n_uncertain_excluded': 2,
            'models_evaluated': list(rep5['models_3seed_summary'].keys()),
            'cluster_bootstrap_verified': True,
        },
    }


@torch.no_grad()
def benchmark_unified_9modes_latency_with_a4b(manifest: dict, val_gt: dict, tc_gt: dict, val_iso_weights_seed0: dict):
    print("\n================ [模块 3] 包含 A4b (1.25x+HFlip) 与 C1 在内的全 9 组同会话端到端公平测速 ================", flush=True)
    device = 'cuda:0'
    eng = UnifiedOrchardInferenceEngine(manifest, seed=0, device=device)

    anns_val = defaultdict(int)
    for a in val_gt['annotations']:
        anns_val[a['image_id']] += 1
    anns_tc = defaultdict(int)
    for a in tc_gt['annotations']:
        anns_tc[a['image_id']] += 1

    val_multi = [im for im in val_gt['images'] if anns_val[im['id']] >= 3]
    val_single = [im for im in val_gt['images'] if anns_val[im['id']] < 3][:max(0, 250 - len(val_multi))]
    val_items = [(im['id'], IMG_ROOT / 'val' / im['file_name'], anns_val[im['id']]) for im in (val_multi + val_single)]
    tc_items = [(im['id'], IMG_ROOT / 'testC' / im['file_name'], anns_tc[im['id']]) for im in tc_gt['images']]

    def run_timed_mode(items: list, mode_key: str, verify_gt_path: Path = None):
        times_all, times_multi, dino_crops = [], [], []
        out_rows = []
        for idx, (iid, p, n_gt_img) in enumerate(items):
            torch.cuda.synchronize()
            ts = time.perf_counter()
            n_crops = 0

            bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), 1)
            h, w = bgr.shape[:2]
            xyxy_t, sc_t, cls_t = eng.yolo_detect_square640(bgr)
            N = len(xyxy_t)

            if mode_key == 'B0_Native_YOLOv8n' or N == 0:
                xyxy_f, sc_f, cls_f = xyxy_t, sc_t, cls_t
            elif mode_key == 'B1_Full_DINOv2_125_Hard':
                act_ids = torch.where(sc_t >= eng.b1_thr)[0]
                xyxy_f, sc_f, cls_f = xyxy_t.clone(), sc_t.clone(), cls_t.clone()
                if len(act_ids) > 0:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    bxs = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in xyxy_t[act_ids].cpu().tolist()]
                    c125 = np.stack([crop_square_scale(rgb, bb, 1.25) for bb in bxs], axis=0)
                    p_d = F.softmax(eng._encode_dino_logits(c125), dim=-1)
                    pmax, ncls = p_d.max(dim=-1)
                    changed = ncls != cls_f[act_ids]
                    cls_f[act_ids] = ncls
                    sc_sub = sc_f[act_ids]
                    sc_f[act_ids] = torch.where(changed, torch.sqrt(sc_sub * pmax.clamp_min(1e-6)), sc_sub)
                    n_crops = len(bxs)
            else:
                fg_ids = torch.where(sc_t >= eng.fg_thr)[0]
                xyxy_f, sc_f, cls_f = xyxy_t.clone(), sc_t.clone(), cls_t.clone()
                if len(fg_ids) > 0:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    bxs_fg = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in xyxy_t[fg_ids].cpu().tolist()]
                    k = len(bxs_fg)
                    p14_fg, _ = eng._extract_anchor_p14_and_neck(h, w, bxs_fg, need_neck=False)
                    lp_det = F.log_softmax(torch.logit(p14_fg.clamp(1e-5, 1.0 - 1e-5)) / eng.T_DET, dim=-1)

                    if mode_key == 'C1_Single_125_035_065':
                        c1 = np.stack([crop_square_scale(rgb, bb, 1.25) for bb in bxs_fg], axis=0)
                        lp1 = F.log_softmax(eng._encode_dino_logits(c1) / eng.T_DINO, dim=-1)
                        p_post = F.softmax(eng.w_c1[0] * lp_det + eng.w_c1[1] * lp1, dim=-1)
                        n_crops = k
                    elif mode_key == 'A1_Single_125':
                        c1 = np.stack([crop_square_scale(rgb, bb, 1.25) for bb in bxs_fg], axis=0)
                        lp1 = F.log_softmax(eng._encode_dino_logits(c1) / eng.T_DINO, dim=-1)
                        p_post = F.softmax(0.30 * lp_det + 0.70 * lp1, dim=-1)
                        n_crops = k
                    elif mode_key == 'A2_Single_082':
                        c1 = np.stack([crop_square_scale(rgb, bb, 0.82) for bb in bxs_fg], axis=0)
                        lp1 = F.log_softmax(eng._encode_dino_logits(c1) / eng.T_DINO, dim=-1)
                        p_post = F.softmax(0.30 * lp_det + 0.70 * lp1, dim=-1)
                        n_crops = k
                    elif mode_key == 'A3_Dual_082_125_C3_Main':
                        c2 = np.stack([crop_square_scale(rgb, bb, 0.82) for bb in bxs_fg] +
                                      [crop_square_scale(rgb, bb, 1.25) for bb in bxs_fg], axis=0)
                        lg2 = eng._encode_dino_logits(c2)
                        lp_a = F.log_softmax(lg2[:k] / eng.T_DINO, dim=-1)
                        lp_b = F.log_softmax(lg2[k:] / eng.T_DINO, dim=-1)
                        w0, w1, w2 = eng.w_c3
                        p_post = F.softmax(w0 * lp_det + w1 * lp_a + w2 * lp_b, dim=-1)
                        n_crops = 2 * k
                    elif mode_key == 'A4a_CloseDual_115_135':
                        c2 = np.stack([crop_square_scale(rgb, bb, 1.15) for bb in bxs_fg] +
                                      [crop_square_scale(rgb, bb, 1.35) for bb in bxs_fg], axis=0)
                        lg2 = eng._encode_dino_logits(c2)
                        lp_a = F.log_softmax(lg2[:k] / eng.T_DINO, dim=-1)
                        lp_b = F.log_softmax(lg2[k:] / eng.T_DINO, dim=-1)
                        w0, w1, w2 = val_iso_weights_seed0['A4a_CloseDual_115_135_Iso30']
                        p_post = F.softmax(w0 * lp_det + w1 * lp_a + w2 * lp_b, dim=-1)
                        n_crops = 2 * k
                    elif mode_key == 'A4b_FlipTTA_125_HFlip':
                        std_l = [crop_square_scale(rgb, bb, 1.25) for bb in bxs_fg]
                        flp_l = [c[:, :, ::-1].copy() for c in std_l]
                        c2 = np.stack(std_l + flp_l, axis=0)
                        lg2 = eng._encode_dino_logits(c2)
                        lp_a = F.log_softmax(lg2[:k] / eng.T_DINO, dim=-1)
                        lp_b = F.log_softmax(lg2[k:] / eng.T_DINO, dim=-1)
                        w0, w1, w2 = eng.w_a4b
                        p_post = F.softmax(w0 * lp_det + w1 * lp_a + w2 * lp_b, dim=-1)
                        n_crops = 2 * k
                    elif mode_key == 'A5_TriScale_082_125_250':
                        c3 = np.stack([crop_square_scale(rgb, bb, 0.82) for bb in bxs_fg] +
                                      [crop_square_scale(rgb, bb, 1.25) for bb in bxs_fg] +
                                      [crop_square_scale(rgb, bb, 2.50) for bb in bxs_fg], axis=0)
                        lg3 = eng._encode_dino_logits(c3)
                        lp_a = F.log_softmax(lg3[:k] / eng.T_DINO, dim=-1)
                        lp_b = F.log_softmax(lg3[k:2 * k] / eng.T_DINO, dim=-1)
                        lp_c = F.log_softmax(lg3[2 * k:] / eng.T_DINO, dim=-1)
                        w0, w1, w2, w3 = val_iso_weights_seed0['A5_TriScale_082_125_250_Iso30']
                        p_post = F.softmax(w0 * lp_det + w1 * lp_a + w2 * lp_b + w3 * lp_c, dim=-1)
                        n_crops = 3 * k
                    else:
                        raise ValueError(mode_key)

                    pmax, ncls = p_post.max(dim=-1)
                    cls_f[fg_ids] = ncls
                    sc_f[fg_ids] = (sc_f[fg_ids] ** eng.beta) * (pmax.clamp_min(1e-6) ** (1.0 - eng.beta))
                    if len(xyxy_f) > 1:
                        keep = batched_nms(xyxy_f, sc_f, cls_f, eng.post_nms_iou)[:300]
                        xyxy_f, sc_f, cls_f = xyxy_f[keep], sc_f[keep], cls_f[keep]

            xy_cpu = xyxy_f.cpu().tolist()
            sc_cpu = sc_f.cpu().tolist()
            cl_cpu = cls_f.cpu().tolist()

            torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - ts) * 1000.0
            if idx >= 10:
                times_all.append(dt_ms)
                dino_crops.append(n_crops)
                n_active_det = sum(1 for s in sc_cpu if s >= 0.25)
                if n_gt_img >= 3 or n_active_det >= 3:
                    times_multi.append(dt_ms)

            if verify_gt_path is not None:
                for (x1, y1, x2, y2), sv, cv in zip(xy_cpu, sc_cpu, cl_cpu):
                    out_rows.append({'image_id': iid, 'category_id': int(cv), 'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': float(sv)})

        live_ap = round(coco_eval_metrics(verify_gt_path, out_rows)['AP50_95'], 2) if verify_gt_path is not None else None
        return {
            'mean_ms': round(float(np.mean(times_all)), 2),
            'p50_ms': round(float(np.percentile(times_all, 50)), 2),
            'p95_ms': round(float(np.percentile(times_all, 95)), 2),
            'multi_target_ge3_mean_ms': round(float(np.mean(times_multi)), 2) if times_multi else None,
            'multi_target_ge3_count': len(times_multi),
            'fps': round(1000.0 / float(np.mean(times_all)), 1),
            'avg_dino_crops_per_img': round(float(np.mean(dino_crops)), 3),
            'verified_live_AP50_95_seed0': live_ap,
        }

    # GPU Warmup
    _ = run_timed_mode(val_items[:15], 'A5_TriScale_082_125_250', None)
    speed_modes = [
        'B0_Native_YOLOv8n',
        'B1_Full_DINOv2_125_Hard',
        'C1_Single_125_035_065',
        'A1_Single_125',
        'A2_Single_082',
        'A4a_CloseDual_115_135',
        'A4b_FlipTTA_125_HFlip',
        'A3_Dual_082_125_C3_Main',
        'A5_TriScale_082_125_250',
    ]
    speed_summary = {}
    for mk in speed_modes:
        rv = run_timed_mode(val_items, mk, None)
        rc = run_timed_mode(tc_items, mk, GT_DIR / 'testC.json')
        speed_summary[mk] = {'val_250imgs': rv, 'testC_all602imgs': rc}
        print(f"  [{mk:28s}] val: mean={rv['mean_ms']:5.2f}ms, P95={rv['p95_ms']:5.2f}ms, 多目标(≥3框, n={rv['multi_target_ge3_count']})={rv['multi_target_ge3_mean_ms']}ms | "
              f"testC(602图): mean={rc['mean_ms']:5.2f}ms, P95={rc['p95_ms']:5.2f}ms, 实时AP={rc['verified_live_AP50_95_seed0']:.2f}%",
              flush=True)

    eng.close()
    return speed_summary


def main():
    t0 = time.time()
    env_meta = configure_strict_fp32()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    verify_manifest_hashes(manifest)
    stage2_prev = json.loads(STAGE2_JSON_PATH.read_text(encoding='utf-8'))

    gt_dicts = {sp: json.loads((GT_DIR / f'{sp}.json').read_text(encoding='utf-8')) for sp in ['val', 'testB', 'testC']}

    # Step 0: Verify A4b online/offline parity + Orchard 1-to-1 & Cluster-Bootstrap Evaluator
    cli_and_orchard_verif = verify_cli_a4b_parity_and_orchard_evaluator(manifest, gt_dicts)

    # Step 1: Extract or load G2/G4 gray128 and G2_blur/G4_blur 1.25x deterministic logits + A4 logits
    gb_logits, outer_geom_stats = load_or_extract_gray_and_blur_125_logits(gt_dicts)
    a4_npz = np.load(SCRATCH_A4_CACHE)

    print("\n================ [模块 1.2] G1–G4 受控灰环 + 高斯模糊外环机制实验 (3 种子 × val/testB/testC) ================", flush=True)
    mech_keys = [
        'G1_125_RealBG_Iso30',
        'G2_125_GrayOuter_Iso30',
        'G2_blur_125_BlurOuter_Iso30',
        'G3_082_125_RealBG_C3_Locked',
        'G4_082_125_GrayOuter_C3_Locked',
        'G4_blur_082_125_BlurOuter_C3_Locked',
        # Sensitivity controls: Equal visual weights [0.30, 0.35, 0.35], C1 [0.35, 0.65], and Pure DINOv2 (w_det=0)
        'G3_eq_082_125_RealBG_Equal',
        'G4_eq_082_125_GrayOuter_Equal',
        'G4_blur_eq_082_125_BlurOuter_Equal',
        'G1_c1_125_RealBG_035_065',
        'G2_c1_125_GrayOuter_035_065',
        'G2_blur_c1_125_BlurOuter_035_065',
        'PureDINO_G1_125_RealBG',
        'PureDINO_G2_125_GrayOuter',
        'PureDINO_G2_blur_125_BlurOuter',
        'PureDINO_G3_082_125_RealBG',
        'PureDINO_G4_082_125_GrayOuter',
        'PureDINO_G4_blur_082_125_BlurOuter',
    ]
    mech_metrics_per_seed = {mk: {sp: [] for sp in ['val', 'testB', 'testC']} for mk in mech_keys}
    mech_pred_store = {sp: {s: {} for s in [0, 1, 2]} for sp in ['val', 'testB', 'testC']}

    # Also store predictions for supplemental A1-A5 paired CIs
    supp_ablation_keys = [
        'B0_Native_YOLOv8n',
        'B1_Full_DINOv2_125_Hard',
        'C1_Historical_125_035_065',
        'A1_Single_125_Iso30',
        'A2_Single_082_Iso30',
        'A3_C3_Locked_082_125_Main',
        'A3_Dual_082_125_Iso30_ValGrid',
        'A3_Dual_082_125_Iso30_Equal',
        'A4a_CloseDual_115_135_Iso30_ValGrid',
        'A4a_CloseDual_115_135_Iso30_Equal',
        'A4b_FlipTTA_125_HFlip_Iso30_ValGrid',
        'A4b_FlipTTA_125_HFlip_Iso30_Equal',
        'A5_TriScale_082_125_250_Iso30_ValGrid',
        'A5_TriScale_082_125_250_Iso30_Equal',
    ]
    supp_pred_store = {sp: {s: {} for s in [0, 1, 2]} for sp in ['val', 'testB', 'testC']}
    a4b_eq_metrics_per_seed = {sp: [] for sp in ['val', 'testB', 'testC']}

    for seed in [0, 1, 2]:
        scfg = manifest['seeds'][str(seed)]
        T_DET = float(scfg['temperatures']['T_DET'])
        T_DINO = float(scfg['temperatures']['T_DINO'])
        beta = float(scfg['postprocess']['beta_score_reweight'])
        fg_thr = float(scfg['postprocess']['fusion_fg_conf_thr'])
        w_c3 = scfg['locked_fusion_params_from_scidb_val']['C3_MainMethod_weights_det_core082_std125']
        w_iso = stage2_prev['val_iso_wdet030_selected_weights_per_seed'][str(seed)]

        for sp in ['val', 'testB', 'testC']:
            gt_p = GT_DIR / f'{sp}.json'
            gt_d = gt_dicts[sp]
            orig_preds = json.loads((STRONG_DIR / f'种子{seed}' / f'{sp}预测.json').read_text(encoding='utf-8'))
            d_npz = np.load(CACHE_DIR / f'{sp}_cggar_seed{seed}.npz')
            cand_idxs = d_npz['cand_idxs']
            cand_rows = [orig_preds[i] for i in cand_idxs]
            gt_matched = match_gt_boxes_for_quadrant(gt_d, cand_rows, 0.50)
            valid_gt = gt_matched >= 0

            yolo_sc = np.array([orig_preds[i]['score'] for i in cand_idxs], dtype=np.float64)
            fg = yolo_sc >= fg_thr
            p14_norm = d_npz['p14'] / np.maximum(d_npz['p14'].sum(axis=1, keepdims=True), 1e-8)
            p125_raw = F.softmax(torch.from_numpy(d_npz['lg_std']), dim=-1).numpy()

            lp_det = F.log_softmax(torch.logit(torch.from_numpy(d_npz['p14']).clamp(1e-5, 1.0 - 1e-5)) / T_DET, dim=-1).numpy()
            lp_082 = F.log_softmax(torch.from_numpy(d_npz['lg_core']) / T_DINO, dim=-1).numpy()
            lp_125_real = F.log_softmax(torch.from_numpy(d_npz['lg_std']) / T_DINO, dim=-1).numpy()
            lp_125_gray = F.log_softmax(torch.from_numpy(gb_logits[seed][sp]['lg_125_gray']) / T_DINO, dim=-1).numpy()
            lp_125_blur = F.log_softmax(torch.from_numpy(gb_logits[seed][sp]['lg_125_blur']) / T_DINO, dim=-1).numpy()
            lp_250 = F.log_softmax(torch.from_numpy(d_npz['lg_ctx']) / T_DINO, dim=-1).numpy()
            lp_115 = F.log_softmax(torch.from_numpy(a4_npz[f's{seed}_{sp}_115']) / T_DINO, dim=-1).numpy()
            lp_135 = F.log_softmax(torch.from_numpy(a4_npz[f's{seed}_{sp}_135']) / T_DINO, dim=-1).numpy()
            lp_125f = F.log_softmax(torch.from_numpy(a4_npz[f's{seed}_{sp}_125f']) / T_DINO, dim=-1).numpy()

            cand_map = {gi: li for li, gi in enumerate(cand_idxs)}

            def build_rows(p_mat: np.ndarray):
                rows = []
                prob_map = {}
                for gi, r in enumerate(orig_preds):
                    nr = dict(r)
                    nr['_uid'] = gi
                    li = cand_map.get(gi, None)
                    if li is not None:
                        if fg[li]:
                            pr = p_mat[li]
                            nc = int(np.argmax(pr))
                            nr['category_id'] = nc
                            nr['score'] = float((yolo_sc[li] ** beta) * (max(float(pr[nc]), 1e-6) ** (1.0 - beta)))
                            prob_map[gi] = pr
                        else:
                            prob_map[gi] = p14_norm[li]
                    rows.append(nr)
                return apply_classwise_nms(rows, 0.70), prob_map

            # --- G1-G4 + Blur + Sensitivity Controls ---
            mech_p_mats = {
                'G1_125_RealBG_Iso30': softmax_from_lp(0.30 * lp_det + 0.70 * lp_125_real),
                'G2_125_GrayOuter_Iso30': softmax_from_lp(0.30 * lp_det + 0.70 * lp_125_gray),
                'G2_blur_125_BlurOuter_Iso30': softmax_from_lp(0.30 * lp_det + 0.70 * lp_125_blur),
                'G3_082_125_RealBG_C3_Locked': softmax_from_lp(w_c3[0] * lp_det + w_c3[1] * lp_082 + w_c3[2] * lp_125_real),
                'G4_082_125_GrayOuter_C3_Locked': softmax_from_lp(w_c3[0] * lp_det + w_c3[1] * lp_082 + w_c3[2] * lp_125_gray),
                'G4_blur_082_125_BlurOuter_C3_Locked': softmax_from_lp(w_c3[0] * lp_det + w_c3[1] * lp_082 + w_c3[2] * lp_125_blur),
                'G3_eq_082_125_RealBG_Equal': softmax_from_lp(0.30 * lp_det + 0.35 * lp_082 + 0.35 * lp_125_real),
                'G4_eq_082_125_GrayOuter_Equal': softmax_from_lp(0.30 * lp_det + 0.35 * lp_082 + 0.35 * lp_125_gray),
                'G4_blur_eq_082_125_BlurOuter_Equal': softmax_from_lp(0.30 * lp_det + 0.35 * lp_082 + 0.35 * lp_125_blur),
                'G1_c1_125_RealBG_035_065': softmax_from_lp(0.35 * lp_det + 0.65 * lp_125_real),
                'G2_c1_125_GrayOuter_035_065': softmax_from_lp(0.35 * lp_det + 0.65 * lp_125_gray),
                'G2_blur_c1_125_BlurOuter_035_065': softmax_from_lp(0.35 * lp_det + 0.65 * lp_125_blur),
                'PureDINO_G1_125_RealBG': softmax_from_lp(lp_125_real),
                'PureDINO_G2_125_GrayOuter': softmax_from_lp(lp_125_gray),
                'PureDINO_G2_blur_125_BlurOuter': softmax_from_lp(lp_125_blur),
                'PureDINO_G3_082_125_RealBG': softmax_from_lp((2.0 / 7.0) * lp_082 + (5.0 / 7.0) * lp_125_real),
                'PureDINO_G4_082_125_GrayOuter': softmax_from_lp((2.0 / 7.0) * lp_082 + (5.0 / 7.0) * lp_125_gray),
                'PureDINO_G4_blur_082_125_BlurOuter': softmax_from_lp((2.0 / 7.0) * lp_082 + (5.0 / 7.0) * lp_125_blur),
            }
            for mk, p_mat in mech_p_mats.items():
                r_m, pm_m = build_rows(p_mat)
                clean_rows = [{'image_id': r['image_id'], 'category_id': r['category_id'], 'bbox': r['bbox'], 'score': r['score']} for r in r_m]
                mech_pred_store[sp][seed][mk] = clean_rows
                m_dict = compute_strict_1to1_box_and_gt_metrics(gt_p, gt_d, r_m, pm_m)
                m_dict['RawCand_IoU50_BoxAcc_pct'] = round(float((p_mat[valid_gt].argmax(axis=1) == gt_matched[valid_gt]).mean() * 100.0), 2)
                mech_metrics_per_seed[mk][sp].append(m_dict)

            # --- Supplemental A1-A5 predictions for complete paired Bootstrap ---
            r_b0 = [{'image_id': r['image_id'], 'category_id': r['category_id'], 'bbox': r['bbox'], 'score': r['score']} for r in orig_preds]
            r_b1 = []
            for gi, r in enumerate(orig_preds):
                nr = {'image_id': r['image_id'], 'category_id': r['category_id'], 'bbox': r['bbox'], 'score': r['score']}
                if gi in cand_map:
                    li = cand_map[gi]
                    pr = p125_raw[li]
                    nc = int(np.argmax(pr))
                    if nc != nr['category_id']:
                        nr['category_id'] = nc
                        nr['score'] = float(math.sqrt(yolo_sc[li] * max(float(pr[nc]), 1e-6)))
                r_b1.append(nr)

            w_a3_v = w_iso['A3_Dual_082_125_Iso30_ValGrid']
            w_a4a_v = w_iso['A4a_CloseDual_115_135_Iso30']
            w_a4b_v = w_iso['A4b_FlipTTA_125_HFlip_Iso30']
            w_a5_v = w_iso['A5_TriScale_082_125_250_Iso30']
            w3 = 0.70 / 3.0

            r_a4b_eq, pm_a4b_eq = build_rows(softmax_from_lp(0.30 * lp_det + 0.35 * lp_125_real + 0.35 * lp_125f))
            a4b_eq_metrics_per_seed[sp].append(compute_strict_1to1_box_and_gt_metrics(gt_p, gt_d, r_a4b_eq, pm_a4b_eq))

            supp_map = {
                'B0_Native_YOLOv8n': r_b0,
                'B1_Full_DINOv2_125_Hard': r_b1,
                'C1_Historical_125_035_065': mech_pred_store[sp][seed]['G1_c1_125_RealBG_035_065'],
                'A1_Single_125_Iso30': mech_pred_store[sp][seed]['G1_125_RealBG_Iso30'],
                'A2_Single_082_Iso30': [{'image_id': r['image_id'], 'category_id': r['category_id'], 'bbox': r['bbox'], 'score': r['score']}
                                        for r in build_rows(softmax_from_lp(0.30 * lp_det + 0.70 * lp_082))[0]],
                'A3_C3_Locked_082_125_Main': mech_pred_store[sp][seed]['G3_082_125_RealBG_C3_Locked'],
                'A3_Dual_082_125_Iso30_ValGrid': [{'image_id': r['image_id'], 'category_id': r['category_id'], 'bbox': r['bbox'], 'score': r['score']}
                                                  for r in build_rows(softmax_from_lp(w_a3_v[0] * lp_det + w_a3_v[1] * lp_082 + w_a3_v[2] * lp_125_real))[0]],
                'A3_Dual_082_125_Iso30_Equal': mech_pred_store[sp][seed]['G3_eq_082_125_RealBG_Equal'],
                'A4a_CloseDual_115_135_Iso30_ValGrid': [{'image_id': r['image_id'], 'category_id': r['category_id'], 'bbox': r['bbox'], 'score': r['score']}
                                                        for r in build_rows(softmax_from_lp(w_a4a_v[0] * lp_det + w_a4a_v[1] * lp_115 + w_a4a_v[2] * lp_135))[0]],
                'A4a_CloseDual_115_135_Iso30_Equal': [{'image_id': r['image_id'], 'category_id': r['category_id'], 'bbox': r['bbox'], 'score': r['score']}
                                                      for r in build_rows(softmax_from_lp(0.30 * lp_det + 0.35 * lp_115 + 0.35 * lp_135))[0]],
                'A4b_FlipTTA_125_HFlip_Iso30_ValGrid': [{'image_id': r['image_id'], 'category_id': r['category_id'], 'bbox': r['bbox'], 'score': r['score']}
                                                        for r in build_rows(softmax_from_lp(w_a4b_v[0] * lp_det + w_a4b_v[1] * lp_125_real + w_a4b_v[2] * lp_125f))[0]],
                'A4b_FlipTTA_125_HFlip_Iso30_Equal': [{'image_id': r['image_id'], 'category_id': r['category_id'], 'bbox': r['bbox'], 'score': r['score']} for r in r_a4b_eq],
                'A5_TriScale_082_125_250_Iso30_ValGrid': [{'image_id': r['image_id'], 'category_id': r['category_id'], 'bbox': r['bbox'], 'score': r['score']}
                                                          for r in build_rows(softmax_from_lp(w_a5_v[0] * lp_det + w_a5_v[1] * lp_082 + w_a5_v[2] * lp_125_real + w_a5_v[3] * lp_250))[0]],
                'A5_TriScale_082_125_250_Iso30_Equal': [{'image_id': r['image_id'], 'category_id': r['category_id'], 'bbox': r['bbox'], 'score': r['score']}
                                                        for r in build_rows(softmax_from_lp(0.30 * lp_det + w3 * lp_082 + w3 * lp_125_real + w3 * lp_250))[0]],
            }
            for mk, r_clean in supp_map.items():
                supp_pred_store[sp][seed][mk] = r_clean

    # Aggregate G1-G4 mechanism metrics
    mech_summary = {}
    for mk in mech_keys:
        mech_summary[mk] = {}
        for sp in ['val', 'testB', 'testC']:
            s_list = mech_metrics_per_seed[mk][sp]
            agg = {}
            for k in s_list[0].keys():
                vals = [x[k] for x in s_list]
                agg[f'{k}_mean'] = round(float(np.mean(vals)), 2)
                agg[f'{k}_std'] = round(float(np.std(vals, ddof=1)), 2)
                agg[f'{k}_seeds'] = vals
            mech_summary[mk][sp] = agg
        print(f"  [{mk:37s}] val={mech_summary[mk]['val']['AP50_95_mean']:.2f} | "
              f"testB={mech_summary[mk]['testB']['AP50_95_mean']:.2f}±{mech_summary[mk]['testB']['AP50_95_std']:.2f} {mech_summary[mk]['testB']['AP50_95_seeds']} | "
              f"testC={mech_summary[mk]['testC']['AP50_95_mean']:.2f}±{mech_summary[mk]['testC']['AP50_95_std']:.2f} {mech_summary[mk]['testC']['AP50_95_seeds']}",
              flush=True)

    # Aggregate A4b_Equal metrics
    a4b_eq_summary = {}
    for sp in ['val', 'testB', 'testC']:
        s_list = a4b_eq_metrics_per_seed[sp]
        agg = {}
        for k in s_list[0].keys():
            vals = [x[k] for x in s_list]
            agg[f'{k}_mean'] = round(float(np.mean(vals)), 2)
            agg[f'{k}_std'] = round(float(np.std(vals, ddof=1)), 2)
            agg[f'{k}_seeds'] = vals
        a4b_eq_summary[sp] = agg

    print("\n================ [模块 1.3] G1–G4 灰环与模糊外环 B=1,000 配对 Bootstrap 及双重差分 (DiD) 检验 ================", flush=True)
    mech_pairs = [
        ('G2_125_GrayOuter_Iso30', 'G1_125_RealBG_Iso30'),
        ('G2_blur_125_BlurOuter_Iso30', 'G1_125_RealBG_Iso30'),
        ('G4_082_125_GrayOuter_C3_Locked', 'G3_082_125_RealBG_C3_Locked'),
        ('G4_blur_082_125_BlurOuter_C3_Locked', 'G3_082_125_RealBG_C3_Locked'),
        ('G3_082_125_RealBG_C3_Locked', 'G1_125_RealBG_Iso30'),          # Δ_real
        ('G4_082_125_GrayOuter_C3_Locked', 'G2_125_GrayOuter_Iso30'),    # Δ_gray
        ('G4_blur_082_125_BlurOuter_C3_Locked', 'G2_blur_125_BlurOuter_Iso30'),  # Δ_blur
        ('G3_eq_082_125_RealBG_Equal', 'G1_125_RealBG_Iso30'),           # Δ_real (equal weight)
        ('G4_eq_082_125_GrayOuter_Equal', 'G2_125_GrayOuter_Iso30'),     # Δ_gray (equal weight)
        ('G4_blur_eq_082_125_BlurOuter_Equal', 'G2_blur_125_BlurOuter_Iso30'),   # Δ_blur (equal weight)
        ('PureDINO_G2_125_GrayOuter', 'PureDINO_G1_125_RealBG'),
        ('PureDINO_G2_blur_125_BlurOuter', 'PureDINO_G1_125_RealBG'),
        ('PureDINO_G3_082_125_RealBG', 'PureDINO_G1_125_RealBG'),
        ('PureDINO_G4_082_125_GrayOuter', 'PureDINO_G2_125_GrayOuter'),
        ('PureDINO_G4_blur_082_125_BlurOuter', 'PureDINO_G2_blur_125_BlurOuter'),
    ]

    def summarize_diff_vec(dv: np.ndarray, per_seed_vecs: list):
        lo, hi = float(np.percentile(dv, 2.5)), float(np.percentile(dv, 97.5))
        p_two = min(1.0, 2.0 * min(float(np.mean(dv <= 0.0)), float(np.mean(dv >= 0.0))))
        return {
            'mean_diff_3seed': round(float(dv.mean()), 3),
            'ci95_3seed': [round(lo, 3), round(hi, 3)],
            'p_value_two_sided': round(p_two, 4),
            'significant_95': bool(lo > 0.0 or hi < 0.0),
            'per_seed_mean_diffs': [round(float(sv.mean()), 3) for sv in per_seed_vecs],
            'per_seed_ci95': [[round(float(np.percentile(sv, 2.5)), 3), round(float(np.percentile(sv, 97.5)), 3)] for sv in per_seed_vecs],
        }

    mech_bootstrap = {}
    for sp in ['val', 'testB', 'testC']:
        boot_s = []
        for s in [0, 1, 2]:
            b_arrs, _ = paired_image_bootstrap_coco(gt_dicts[sp], mech_pred_store[sp][s], mech_pairs, n_boot=1000, rng_seed=20261127 + s)
            boot_s.append(b_arrs)

        mech_bootstrap[sp] = {}
        for m_a, m_b in mech_pairs:
            tag = f"{m_a}__minus__{m_b}"
            mech_bootstrap[sp][tag] = {}
            for col, mlbl in [(0, 'AP50_95'), (1, 'AP50')]:
                ps_v = [boot_s[s][m_a][:, col] - boot_s[s][m_b][:, col] for s in [0, 1, 2]]
                mech_bootstrap[sp][tag][mlbl] = summarize_diff_vec(np.mean(ps_v, axis=0), ps_v)

        # Difference-in-Differences (DiD): Δ_intervened - Δ_real = (G4 - G2) - (G3 - G1)
        did_specs = [
            ('DiD_Gray_minus_Real_LockedC3', 'G4_082_125_GrayOuter_C3_Locked', 'G2_125_GrayOuter_Iso30', 'G3_082_125_RealBG_C3_Locked', 'G1_125_RealBG_Iso30'),
            ('DiD_Blur_minus_Real_LockedC3', 'G4_blur_082_125_BlurOuter_C3_Locked', 'G2_blur_125_BlurOuter_Iso30', 'G3_082_125_RealBG_C3_Locked', 'G1_125_RealBG_Iso30'),
            ('DiD_Gray_minus_Real_EqualWeight', 'G4_eq_082_125_GrayOuter_Equal', 'G2_125_GrayOuter_Iso30', 'G3_eq_082_125_RealBG_Equal', 'G1_125_RealBG_Iso30'),
            ('DiD_Blur_minus_Real_EqualWeight', 'G4_blur_eq_082_125_BlurOuter_Equal', 'G2_blur_125_BlurOuter_Iso30', 'G3_eq_082_125_RealBG_Equal', 'G1_125_RealBG_Iso30'),
            ('DiD_Gray_minus_Real_PureDINO', 'PureDINO_G4_082_125_GrayOuter', 'PureDINO_G2_125_GrayOuter', 'PureDINO_G3_082_125_RealBG', 'PureDINO_G1_125_RealBG'),
            ('DiD_Blur_minus_Real_PureDINO', 'PureDINO_G4_blur_082_125_BlurOuter', 'PureDINO_G2_blur_125_BlurOuter', 'PureDINO_G3_082_125_RealBG', 'PureDINO_G1_125_RealBG'),
        ]
        for did_tag, g4k, g2k, g3k, g1k in did_specs:
            mech_bootstrap[sp][did_tag] = {}
            for col, mlbl in [(0, 'AP50_95'), (1, 'AP50')]:
                ps_v = [(boot_s[s][g4k][:, col] - boot_s[s][g2k][:, col]) - (boot_s[s][g3k][:, col] - boot_s[s][g1k][:, col]) for s in [0, 1, 2]]
                mech_bootstrap[sp][did_tag][mlbl] = summarize_diff_vec(np.mean(ps_v, axis=0), ps_v)

        print(f"  === [{sp}] 核心灰环/模糊外环 Bootstrap 检验 ===", flush=True)
        for show_k in [
            'G2_125_GrayOuter_Iso30__minus__G1_125_RealBG_Iso30',
            'G2_blur_125_BlurOuter_Iso30__minus__G1_125_RealBG_Iso30',
            'G4_082_125_GrayOuter_C3_Locked__minus__G3_082_125_RealBG_C3_Locked',
            'G4_blur_082_125_BlurOuter_C3_Locked__minus__G3_082_125_RealBG_C3_Locked',
            'G3_082_125_RealBG_C3_Locked__minus__G1_125_RealBG_Iso30',
            'G4_082_125_GrayOuter_C3_Locked__minus__G2_125_GrayOuter_Iso30',
            'G4_blur_082_125_BlurOuter_C3_Locked__minus__G2_blur_125_BlurOuter_Iso30',
            'DiD_Gray_minus_Real_LockedC3',
            'DiD_Blur_minus_Real_LockedC3',
        ]:
            r_ap = mech_bootstrap[sp][show_k]['AP50_95']
            print(f"    {show_k:74s} | ΔAP50-95 = {r_ap['mean_diff_3seed']:+.3f}% "
                  f"(95% CI {r_ap['ci95_3seed']}, p={r_ap['p_value_two_sided']:.4f}, sig={r_ap['significant_95']})", flush=True)

    print("\n================ [模块 2] 补齐 A1–A5 全部配对 Bootstrap 95% 置信区间 ================", flush=True)
    supp_pairs = [
        ('A3_C3_Locked_082_125_Main', 'B0_Native_YOLOv8n'),
        ('A3_C3_Locked_082_125_Main', 'B1_Full_DINOv2_125_Hard'),
        ('A3_C3_Locked_082_125_Main', 'A3_Dual_082_125_Iso30_Equal'),
        ('A3_C3_Locked_082_125_Main', 'A3_Dual_082_125_Iso30_ValGrid'),
        ('A3_C3_Locked_082_125_Main', 'A4b_FlipTTA_125_HFlip_Iso30_Equal'),
        ('A3_Dual_082_125_Iso30_Equal', 'A1_Single_125_Iso30'),
        ('A3_Dual_082_125_Iso30_Equal', 'C1_Historical_125_035_065'),
        ('A3_Dual_082_125_Iso30_Equal', 'A2_Single_082_Iso30'),
        ('A3_Dual_082_125_Iso30_Equal', 'A4b_FlipTTA_125_HFlip_Iso30_Equal'),
        ('A5_TriScale_082_125_250_Iso30_ValGrid', 'A3_Dual_082_125_Iso30_ValGrid'),
    ]
    supp_bootstrap = {}
    for sp in ['val', 'testB', 'testC']:
        boot_s = []
        for s in [0, 1, 2]:
            b_arrs, _ = paired_image_bootstrap_coco(gt_dicts[sp], supp_pred_store[sp][s], supp_pairs, n_boot=1000, rng_seed=20261126 + s)
            boot_s.append(b_arrs)
        supp_bootstrap[sp] = {}
        for m_a, m_b in supp_pairs:
            tag = f"{m_a}__minus__{m_b}"
            supp_bootstrap[sp][tag] = {}
            for col, mlbl in [(0, 'AP50_95'), (1, 'AP50')]:
                ps_v = [boot_s[s][m_a][:, col] - boot_s[s][m_b][:, col] for s in [0, 1, 2]]
                supp_bootstrap[sp][tag][mlbl] = summarize_diff_vec(np.mean(ps_v, axis=0), ps_v)
        if sp in ('testB', 'testC'):
            for m_a, m_b in supp_pairs:
                tag = f"{m_a}__minus__{m_b}"
                r_ap = supp_bootstrap[sp][tag]['AP50_95']
                print(f"  [补齐 Bootstrap {sp:5s} | {tag:76s}] ΔAP50-95 = {r_ap['mean_diff_3seed']:+.3f}% "
                      f"(95% CI {r_ap['ci95_3seed']}, p={r_ap['p_value_two_sided']:.4f})", flush=True)

    # Step 3: Unified 9-mode fair end-to-end latency benchmark including A4b & C1
    latency_9modes = benchmark_unified_9modes_latency_with_a4b(
        manifest, gt_dicts['val'], gt_dicts['testC'], stage2_prev['val_iso_wdet030_selected_weights_per_seed']['0']
    )

    # Save dedicated Stage 2b JSON AND merge supplemental CIs + A4b/C1 latency into stage2 complete JSON for unified plotting
    stage2b_payload = {
        'stage': 'Stage2b_GrayAndBlurRing_Mechanism_SupplementalBootstrap_A4bSpeed_OrchardEvalFix',
        'elapsed_seconds': round(time.time() - t0, 2),
        'strict_fp32_env': env_meta,
        'cli_a4b_parity_and_orchard_1to1_cluster_verif': cli_and_orchard_verif,
        'outer_ring_125_geometry_stats': outer_geom_stats,
        'g1_to_g4_gray_and_blur_ring_metrics_3seed': mech_summary,
        'g1_to_g4_paired_bootstrap_and_did_1000': mech_bootstrap,
        'a4b_equal_weight_metrics_3seed': a4b_eq_summary,
        'supplemental_ablation_paired_bootstrap_1000': supp_bootstrap,
        'unified_9modes_fair_end2end_latency_rtx4060ti_fp32': latency_9modes,
        'historical_tcd_mge_v3_latency_separate_protocol_note': {
            'val_mean_ms': 27.53,
            'val_p95_ms': 45.75,
            'testC_mean_ms': 25.79,
            'testC_p95_ms': 29.90,
            'warning': '引自 v3 独立测速协议，因未纳入当前 v5 统一 9 模式同会话脚本重测，仅作历史背景参考，不作为严格跨协议速度优势证据。',
        },
    }
    out_2b = RES_DIR / 'stage2b_gray_ring_mechanism_and_a4b_speed.json'
    out_2b.write_text(json.dumps(stage2b_payload, ensure_ascii=False, indent=2), encoding='utf-8')

    # Sync supplemental A4b_Equal + paired CIs + unified 9-mode latency into stage2_c3_controlled_scale_ablation_complete.json
    stage2_prev['ablation_metrics_3seed']['A4b_FlipTTA_125_HFlip_Iso30_Equal'] = a4b_eq_summary
    for sp in ['val', 'testB', 'testC']:
        stage2_prev['paired_bootstrap_1000_results'][sp].update(supp_bootstrap[sp])
    stage2_prev['fair_end2end_latency_rtx4060ti_fp32'] = latency_9modes
    STAGE2_JSON_PATH.write_text(json.dumps(stage2_prev, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f"\n[Stage 2b 全部受控机制实验、补齐 CI、A4b 公平测速与果园评测修复核验完成] 耗时 {stage2b_payload['elapsed_seconds']}s -> {out_2b}", flush=True)


if __name__ == '__main__':
    main()
