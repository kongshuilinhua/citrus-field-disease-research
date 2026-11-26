# -*- coding: utf-8 -*-
"""
第二阶段 (Stage 2): 以 C3 (0.82x + 1.25x) 为核心的严格受控尺度消融 (A1–A5)、逐图配对 Bootstrap 与端到端公平测速
(run_stage2_controlled_scale_ablation_and_speed.py)

关键方法学严密控制（消除检测器权重 w_det 混杂变量！）：
1. 为什么必须固定总检测器权重 w_det = 0.30 (即 30% YOLOv8n + 70% DINOv2 总权重) 进行主受控对比？
   - 在干净源域 SciDB val 上，YOLOv8n 与 DINOv2 准确率均超 98%，若不固定 w_det 而自由搜索，不同视角组合会偶然跳到
     w_det=0.45 或 w_det=0.30，从而把“检测器权重变化”混入“裁剪尺度变化”！
   - 因此，主受控消融严格固定 w_det = 0.30（与冻结 C3 [0.30, 0.20, 0.50] 总视觉权重 0.70 完全一致！），在视觉内部通过源域
     SciDB val 锁定（或等权对照）分配 0.70 的视觉权重：
       * A1 (1.25x 标准单尺度):        0.30 * lp_det + 0.70 * lp_125  (及 C1 历史基准 [0.35, 0.65])
       * A2 (0.82x 紧致单尺度):        0.30 * lp_det + 0.70 * lp_082
       * A3 / C3 (0.82x + 1.25x 主方法): 冻结 [0.30, 0.20, 0.50] (同时汇报等权 [0.30, 0.35, 0.35] 与 val 子网格选权)
       * A4a (1.15x + 1.35x 相近双裁剪 TTA): 固定 w_det=0.30 下 val 选权 & [0.30, 0.20, 0.50] / [0.30, 0.35, 0.35] 对照
       * A4b (1.25x + 水平翻转 常规双视角 TTA): 固定 w_det=0.30 下 val 选权 & [0.30, 0.35, 0.35] 对照
       * A5 (0.82x + 1.25x + 2.50x 三尺度): 固定 w_det=0.30 下 val 选权 & [0.30, 0.20, 0.35, 0.15] 对照
   - 同时完整保留“不固定 w_det 的自由 val 网格”结果备查，做到 100% 透明零隐瞒！
2. 零新模型训练（全部复用 12 份 SHA-256 锁定权重，额外 MLP 参数恒为 0），三随机种子 + B=1,000 逐图配对 Bootstrap + 含多目标公平测速。
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
V4_DIR = REPO_AG / 'v4_orchard_candidate_arsenal'
V5_DIR = REPO_AG / 'v5_c3_main_method_and_ablation'
RES_DIR = V5_DIR / 'results'
RES_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_SCRATCH_CACHE = ROOT / 'Antigravity_只读可行性审计_20261126' / 'stage2_a4_extra_logits_cache.npz'

sys.path.append(str(EXT))
sys.path.append(str(ROOT / 'Antigravity_BGOPTA_单趟正交Token攻关_20261126'))
sys.path.append(str(ROOT / 'Antigravity_CGGAR_真创新攻关_20261125'))
sys.path.append(str(REPO_AG / 'v3_bgopta_and_dtemop' / 'scripts'))
sys.path.append(str(V5_DIR / 'scripts'))

from 共同评测后端 import configure_strict_fp32
from phase1_source_only_train_and_select import apply_classwise_nms
from phase2_frozen_oneshot_eval import coco_eval_metrics
from step1_extract_multigranular_cache import DinoExtractor, crop_square_scale
from run_rigorous_audit_suite import paired_image_bootstrap_coco
from orchard_plug_and_play_cli import UnifiedOrchardInferenceEngine, verify_manifest_hashes

IMG_ROOT = Path(r'D:\病虫害论文\复现实验记录\yolo11n_public_compare_20261004\数据副本\images')
GT_DIR = EXT / '检测数据'
STRONG_DIR = EXT / 'YOLOv8强基线重复_20261109'
CACHE_DIR = ROOT / 'Antigravity_CGGAR_真创新攻关_20261125' / 'cache'
MANIFEST_PATH = V5_DIR / 'configs' / 'c3_frozen_manifest.json'


def softmax_from_lp(lp: np.ndarray) -> np.ndarray:
    z = lp - lp.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def generate_iso_wdet_dino_grid(n_dino_views: int, w_det: float = 0.30, step: float = 0.05):
    """固定总检测器权重 w_det = 0.30，在剩余 0.70 视觉总权重上按 0.05 步长分配各 DINOv2 视角权重 (每个纳入视角至少 0.10 防止退化为单视角)。"""
    rem = round(1.0 - w_det, 4)
    n_rem = int(round(rem / step))
    out = []
    if n_dino_views == 1:
        out.append((w_det, rem))
    elif n_dino_views == 2:
        for i in range(2, n_rem - 1):
            w1 = round(i * step, 4)
            w2 = round(rem - w1, 4)
            out.append((w_det, w1, w2))
    elif n_dino_views == 3:
        for i in range(2, n_rem - 3):
            for j in range(2, n_rem - i - 1):
                w1 = round(i * step, 4)
                w2 = round(j * step, 4)
                w3 = round(rem - w1 - w2, 4)
                if w3 >= 0.10 - 1e-6:
                    out.append((w_det, w1, w2, w3))
    return out


def decode_split_rgb_map(sp: str, gt_dict: dict) -> dict:
    def dec_one(im):
        p = IMG_ROOT / sp / im['file_name']
        bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), 1)
        return im['id'], cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    with ThreadPoolExecutor(max_workers=6) as ex:
        return dict(ex.map(dec_one, gt_dict['images']))


@torch.no_grad()
def load_or_extract_extra_a4_logits(gt_dicts: dict) -> dict:
    if LOCAL_SCRATCH_CACHE.exists():
        print(f"[加载本地缓存] 复用已提取的 A4 确定性特征缓存: {LOCAL_SCRATCH_CACHE}", flush=True)
        z = np.load(LOCAL_SCRATCH_CACHE)
        extra = {seed: {} for seed in [0, 1, 2]}
        for seed in [0, 1, 2]:
            for sp in ['val', 'testB', 'testC']:
                extra[seed][sp] = {
                    'lg_115': z[f's{seed}_{sp}_115'],
                    'lg_135': z[f's{seed}_{sp}_135'],
                    'lg_125_flip': z[f's{seed}_{sp}_125f'],
                }
        return extra

    print("\n================ [预备步骤] 提取 A4 相近双尺度 (1.15x, 1.35x) 与水平翻转 (1.25x-HFlip) 确定性特征 ================", flush=True)
    rgb_maps = {sp: decode_split_rgb_map(sp, gt_dicts[sp]) for sp in ['val', 'testB', 'testC']}
    extra = {seed: {} for seed in [0, 1, 2]}
    save_dict = {}

    for seed in [0, 1, 2]:
        dino_ext = DinoExtractor(seed, 'cuda:0')
        for sp in ['val', 'testB', 'testC']:
            orig_preds = json.loads((STRONG_DIR / f'种子{seed}' / f'{sp}预测.json').read_text(encoding='utf-8'))
            d_npz = np.load(CACHE_DIR / f'{sp}_cggar_seed{seed}.npz')
            cand_idxs = d_npz['cand_idxs']
            cand_rows = [orig_preds[i] for i in cand_idxs]
            n_c = len(cand_rows)
            fg = np.array([r['score'] for r in cand_rows], dtype=np.float64) >= 0.08

            lg_115 = np.zeros((n_c, 14), dtype=np.float32)
            lg_135 = np.zeros((n_c, 14), dtype=np.float32)
            lg_125_flip = np.zeros((n_c, 14), dtype=np.float32)

            fg_pos = np.where(fg)[0]
            c115 = np.empty((len(fg_pos), 3, 224, 224), dtype=np.uint8)
            c135 = np.empty((len(fg_pos), 3, 224, 224), dtype=np.uint8)
            c125f = np.empty((len(fg_pos), 3, 224, 224), dtype=np.uint8)
            rmp = rgb_maps[sp]

            for k_i, li in enumerate(fg_pos):
                r = cand_rows[li]
                rgb = rmp[r['image_id']]
                bb = r['bbox']
                c115[k_i] = crop_square_scale(rgb, bb, 1.15)
                c135[k_i] = crop_square_scale(rgb, bb, 1.35)
                c_std = crop_square_scale(rgb, bb, 1.25)
                c125f[k_i] = c_std[:, :, ::-1].copy()

            _, lg_115[fg_pos] = dino_ext.encode_crops(c115)
            _, lg_135[fg_pos] = dino_ext.encode_crops(c135)
            _, lg_125_flip[fg_pos] = dino_ext.encode_crops(c125f)
            extra[seed][sp] = {'lg_115': lg_115, 'lg_135': lg_135, 'lg_125_flip': lg_125_flip}
            save_dict[f's{seed}_{sp}_115'] = lg_115
            save_dict[f's{seed}_{sp}_135'] = lg_135
            save_dict[f's{seed}_{sp}_125f'] = lg_125_flip

        del dino_ext
        torch.cuda.empty_cache()
        print(f"  [Seed {seed}] 1.15x / 1.35x / 1.25x-HFlip 确定性 logits 提取完成", flush=True)

    LOCAL_SCRATCH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(LOCAL_SCRATCH_CACHE, **save_dict)
    del rgb_maps
    return extra


def compute_unified_box_and_gt_metrics(gt_path: Path, gt_dict: dict, post_rows: list, post_probs_map: dict):
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
        dlist = sorted(act_by_img.get(iid, []), key=lambda x: -x['score'])
        used_d = set()
        for ga in glist:
            n_gt += 1
            gx, gy, gw, gh = ga['bbox']
            gc = int(ga['category_id'])
            best_iou, best_j = 0.0, -1
            for j, dr in enumerate(dlist):
                if j in used_d:
                    continue
                dx, dy, dw, dh = dr['bbox']
                ix1, iy1 = max(gx, dx), max(gy, dy)
                ix2, iy2 = min(gx + gw, dx + dw), min(gy + gh, dy + dh)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                iou = inter / (gw * gh + dw * dh - inter + 1e-9)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= 0.50 and best_j >= 0:
                used_d.add(best_j)
                n_loc += 1
                dr = dlist[best_j]
                if int(dr['category_id']) == gc:
                    n_t1 += 1
                pr = post_probs_map.get(dr.get('_uid'), None)
                if pr is not None:
                    top3 = np.argsort(-pr)[:3].tolist()
                    if gc in top3:
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


@torch.no_grad()
def benchmark_fair_end2end_ablation_latency(manifest: dict, val_gt: dict, tc_gt: dict, val_iso_weights_seed0: dict):
    print("\n================ [模块 B] B0/B1 与 A1–A5 完整端到端公平测速 (含多目标图像延迟 & 全集实时 AP 核验) ================", flush=True)
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

    def run_timed_ablation(items: list, mode_key: str, verify_gt_path: Path = None):
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

                    if mode_key == 'A1_Single_125':
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
                        w0, w1, w2 = eng.w_c3  # [0.30, 0.20, 0.50]
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

    _ = run_timed_ablation(val_items[:12], 'A5_TriScale_082_125_250', None)
    speed_modes = [
        'B0_Native_YOLOv8n',
        'B1_Full_DINOv2_125_Hard',
        'A1_Single_125',
        'A2_Single_082',
        'A3_Dual_082_125_C3_Main',
        'A4a_CloseDual_115_135',
        'A5_TriScale_082_125_250',
    ]
    speed_summary = {}
    for mk in speed_modes:
        rv = run_timed_ablation(val_items, mk, None)
        rc = run_timed_ablation(tc_items, mk, GT_DIR / 'testC.json')
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

    gt_dicts = {sp: json.loads((GT_DIR / f'{sp}.json').read_text(encoding='utf-8')) for sp in ['val', 'testB', 'testC']}
    extra_logits = load_or_extract_extra_a4_logits(gt_dicts)

    iso_grid_2v = generate_iso_wdet_dino_grid(2, w_det=0.30, step=0.05)
    iso_grid_3v = generate_iso_wdet_dino_grid(3, w_det=0.30, step=0.05)

    print("\n================ [模块 A] 严格等检测器权重 (w_det=0.30) 受控尺度消融 A1–A5 (零混杂变量，仅源域 val 选权) ================", flush=True)
    ablation_keys = [
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
        'A5_TriScale_082_125_250_Iso30_ValGrid',
        'A5_TriScale_082_125_250_Iso30_Equal',
    ]
    pred_rows_store = {sp: {s: {} for s in [0, 1, 2]} for sp in ['val', 'testB', 'testC']}
    metrics_per_seed = {mk: {sp: [] for sp in ['val', 'testB', 'testC']} for mk in ablation_keys}
    val_iso_weights = {}

    for seed in [0, 1, 2]:
        scfg = manifest['seeds'][str(seed)]
        T_DET = float(scfg['temperatures']['T_DET'])
        T_DINO = float(scfg['temperatures']['T_DINO'])
        beta = float(scfg['postprocess']['beta_score_reweight'])
        fg_thr = float(scfg['postprocess']['fusion_fg_conf_thr'])
        w_c3_lock = scfg['locked_fusion_params_from_scidb_val']['C3_MainMethod_weights_det_core082_std125']

        sp_packs = {}
        for sp in ['val', 'testB', 'testC']:
            orig_preds = json.loads((STRONG_DIR / f'种子{seed}' / f'{sp}预测.json').read_text(encoding='utf-8'))
            d_npz = np.load(CACHE_DIR / f'{sp}_cggar_seed{seed}.npz')
            cand_idxs = d_npz['cand_idxs']
            yolo_sc = np.array([orig_preds[i]['score'] for i in cand_idxs], dtype=np.float64)
            fg = yolo_sc >= fg_thr

            lp_det = F.log_softmax(torch.logit(torch.from_numpy(d_npz['p14']).clamp(1e-5, 1.0 - 1e-5)) / T_DET, dim=-1).numpy()
            lp_082 = F.log_softmax(torch.from_numpy(d_npz['lg_core']) / T_DINO, dim=-1).numpy()
            lp_125 = F.log_softmax(torch.from_numpy(d_npz['lg_std']) / T_DINO, dim=-1).numpy()
            lp_250 = F.log_softmax(torch.from_numpy(d_npz['lg_ctx']) / T_DINO, dim=-1).numpy()
            lp_115 = F.log_softmax(torch.from_numpy(extra_logits[seed][sp]['lg_115']) / T_DINO, dim=-1).numpy()
            lp_135 = F.log_softmax(torch.from_numpy(extra_logits[seed][sp]['lg_135']) / T_DINO, dim=-1).numpy()
            lp_125f = F.log_softmax(torch.from_numpy(extra_logits[seed][sp]['lg_125_flip']) / T_DINO, dim=-1).numpy()

            sp_packs[sp] = {
                'orig_preds': orig_preds,
                'cand_idxs': cand_idxs,
                'yolo_sc': yolo_sc,
                'fg': fg,
                'p14_norm': d_npz['p14'] / np.maximum(d_npz['p14'].sum(axis=1, keepdims=True), 1e-8),
                'p125_raw': F.softmax(torch.from_numpy(d_npz['lg_std']), dim=-1).numpy(),
                'lp_det': lp_det,
                'lp_082': lp_082,
                'lp_125': lp_125,
                'lp_250': lp_250,
                'lp_115': lp_115,
                'lp_135': lp_135,
                'lp_125f': lp_125f,
            }

        def build_calibrated_rows(pk: dict, p_mat: np.ndarray):
            rows = []
            prob_map = {}
            cand_map = {gi: li for li, gi in enumerate(pk['cand_idxs'])}
            for gi, r in enumerate(pk['orig_preds']):
                nr = dict(r)
                nr['_uid'] = gi
                li = cand_map.get(gi, None)
                if li is not None:
                    if pk['fg'][li]:
                        pr = p_mat[li]
                        nc = int(np.argmax(pr))
                        nr['category_id'] = nc
                        nr['score'] = float((pk['yolo_sc'][li] ** beta) * (max(float(pr[nc]), 1e-6) ** (1.0 - beta)))
                        prob_map[gi] = pr
                    else:
                        prob_map[gi] = pk['p14_norm'][li]
                rows.append(nr)
            return apply_classwise_nms(rows, 0.70), prob_map

        vpk = sp_packs['val']
        val_gt_p = GT_DIR / 'val.json'

        def select_best_iso_on_val(grid_w, lp_tuple):
            best_w, best_ap = grid_w[0], -1.0
            for w in grid_w:
                lp_mix = sum(wk * lpk for wk, lpk in zip(w, lp_tuple))
                r_nms, _ = build_calibrated_rows(vpk, softmax_from_lp(lp_mix))
                ap = coco_eval_metrics(val_gt_p, r_nms)['AP50_95']
                if ap > best_ap + 1e-5:
                    best_ap = ap
                    best_w = w
            return list(best_w), round(best_ap, 3)

        w_a3_iso, _ = select_best_iso_on_val(iso_grid_2v, (vpk['lp_det'], vpk['lp_082'], vpk['lp_125']))
        w_a4a_iso, _ = select_best_iso_on_val(iso_grid_2v, (vpk['lp_det'], vpk['lp_115'], vpk['lp_135']))
        w_a4b_iso, _ = select_best_iso_on_val(iso_grid_2v, (vpk['lp_det'], vpk['lp_125'], vpk['lp_125f']))
        w_a5_iso, _ = select_best_iso_on_val(iso_grid_3v, (vpk['lp_det'], vpk['lp_082'], vpk['lp_125'], vpk['lp_250']))

        val_iso_weights[str(seed)] = {
            'A1_Single_125_Iso30': [0.30, 0.70],
            'A2_Single_082_Iso30': [0.30, 0.70],
            'A3_C3_Locked_082_125_Main': w_c3_lock,
            'A3_Dual_082_125_Iso30_ValGrid': w_a3_iso,
            'A4a_CloseDual_115_135_Iso30': w_a4a_iso,
            'A4b_FlipTTA_125_HFlip_Iso30': w_a4b_iso,
            'A5_TriScale_082_125_250_Iso30': w_a5_iso,
        }
        print(f"  [Seed {seed} | 固定w_det=0.30 源val选权] A3={w_a3_iso} (冻结C3={w_c3_lock}) | "
              f"A4a(1.15+1.35)={w_a4a_iso} | A4b(HFlip)={w_a4b_iso} | A5(三尺度)={w_a5_iso}", flush=True)

        for sp in ['val', 'testB', 'testC']:
            pk = sp_packs[sp]
            gt_p = GT_DIR / f'{sp}.json'
            gt_d = gt_dicts[sp]
            cand_map = {gi: li for li, gi in enumerate(pk['cand_idxs'])}

            r_b0, pm_b0 = [], {}
            for gi, r in enumerate(pk['orig_preds']):
                nr = dict(r); nr['_uid'] = gi
                if gi in cand_map:
                    pm_b0[gi] = pk['p14_norm'][cand_map[gi]]
                r_b0.append(nr)

            r_b1, pm_b1 = [], {}
            for gi, r in enumerate(pk['orig_preds']):
                nr = dict(r); nr['_uid'] = gi
                if gi in cand_map:
                    li = cand_map[gi]
                    pr = pk['p125_raw'][li]
                    pm_b1[gi] = pr
                    nc = int(np.argmax(pr))
                    if nc != nr['category_id']:
                        nr['category_id'] = nc
                        nr['score'] = float(math.sqrt(pk['yolo_sc'][li] * max(float(pr[nc]), 1e-6)))
                r_b1.append(nr)

            w3 = 0.70 / 3.0
            mode_defs = {
                'B0_Native_YOLOv8n': (r_b0, pm_b0),
                'B1_Full_DINOv2_125_Hard': (r_b1, pm_b1),
                'C1_Historical_125_035_065': build_calibrated_rows(pk, softmax_from_lp(0.35 * pk['lp_det'] + 0.65 * pk['lp_125'])),
                'A1_Single_125_Iso30': build_calibrated_rows(pk, softmax_from_lp(0.30 * pk['lp_det'] + 0.70 * pk['lp_125'])),
                'A2_Single_082_Iso30': build_calibrated_rows(pk, softmax_from_lp(0.30 * pk['lp_det'] + 0.70 * pk['lp_082'])),
                'A3_C3_Locked_082_125_Main': build_calibrated_rows(pk, softmax_from_lp(w_c3_lock[0] * pk['lp_det'] + w_c3_lock[1] * pk['lp_082'] + w_c3_lock[2] * pk['lp_125'])),
                'A3_Dual_082_125_Iso30_ValGrid': build_calibrated_rows(pk, softmax_from_lp(w_a3_iso[0] * pk['lp_det'] + w_a3_iso[1] * pk['lp_082'] + w_a3_iso[2] * pk['lp_125'])),
                'A3_Dual_082_125_Iso30_Equal': build_calibrated_rows(pk, softmax_from_lp(0.30 * pk['lp_det'] + 0.35 * pk['lp_082'] + 0.35 * pk['lp_125'])),
                'A4a_CloseDual_115_135_Iso30_ValGrid': build_calibrated_rows(pk, softmax_from_lp(w_a4a_iso[0] * pk['lp_det'] + w_a4a_iso[1] * pk['lp_115'] + w_a4a_iso[2] * pk['lp_135'])),
                'A4a_CloseDual_115_135_Iso30_Equal': build_calibrated_rows(pk, softmax_from_lp(0.30 * pk['lp_det'] + 0.35 * pk['lp_115'] + 0.35 * pk['lp_135'])),
                'A4b_FlipTTA_125_HFlip_Iso30_ValGrid': build_calibrated_rows(pk, softmax_from_lp(w_a4b_iso[0] * pk['lp_det'] + w_a4b_iso[1] * pk['lp_125'] + w_a4b_iso[2] * pk['lp_125f'])),
                'A5_TriScale_082_125_250_Iso30_ValGrid': build_calibrated_rows(pk, softmax_from_lp(w_a5_iso[0] * pk['lp_det'] + w_a5_iso[1] * pk['lp_082'] + w_a5_iso[2] * pk['lp_125'] + w_a5_iso[3] * pk['lp_250'])),
                'A5_TriScale_082_125_250_Iso30_Equal': build_calibrated_rows(pk, softmax_from_lp(0.30 * pk['lp_det'] + w3 * pk['lp_082'] + w3 * pk['lp_125'] + w3 * pk['lp_250'])),
            }

            for mk, (r_m, pm_m) in mode_defs.items():
                clean_rows = [{'image_id': r['image_id'], 'category_id': r['category_id'], 'bbox': r['bbox'], 'score': r['score']} for r in r_m]
                pred_rows_store[sp][seed][mk] = clean_rows
                metrics_per_seed[mk][sp].append(compute_unified_box_and_gt_metrics(gt_p, gt_d, r_m, pm_m))

    ablation_summary = {}
    for mk in ablation_keys:
        ablation_summary[mk] = {}
        for sp in ['val', 'testB', 'testC']:
            s_list = metrics_per_seed[mk][sp]
            agg = {}
            for k in s_list[0].keys():
                vals = [x[k] for x in s_list]
                agg[f'{k}_mean'] = round(float(np.mean(vals)), 2)
                agg[f'{k}_std'] = round(float(np.std(vals, ddof=1)), 2)
                agg[f'{k}_seeds'] = vals
            ablation_summary[mk][sp] = agg
        print(f"  [{mk:37s}] val={ablation_summary[mk]['val']['AP50_95_mean']:.2f} | "
              f"testB={ablation_summary[mk]['testB']['AP50_95_mean']:.2f}±{ablation_summary[mk]['testB']['AP50_95_std']:.2f} {ablation_summary[mk]['testB']['AP50_95_seeds']} | "
              f"testC={ablation_summary[mk]['testC']['AP50_95_mean']:.2f}±{ablation_summary[mk]['testC']['AP50_95_std']:.2f} {ablation_summary[mk]['testC']['AP50_95_seeds']}",
              flush=True)

    print("\n================ [模块 A.2] 严格等 w_det=0.30 下 B=1,000 逐图配对 Bootstrap 假设检验 ================", flush=True)
    pairs_to_test = [
        ('A3_C3_Locked_082_125_Main', 'A1_Single_125_Iso30'),
        ('A3_C3_Locked_082_125_Main', 'C1_Historical_125_035_065'),
        ('A3_C3_Locked_082_125_Main', 'A2_Single_082_Iso30'),
        ('A3_C3_Locked_082_125_Main', 'A4a_CloseDual_115_135_Iso30_ValGrid'),
        ('A3_C3_Locked_082_125_Main', 'A4a_CloseDual_115_135_Iso30_Equal'),
        ('A3_C3_Locked_082_125_Main', 'A4b_FlipTTA_125_HFlip_Iso30_ValGrid'),
        ('A3_Dual_082_125_Iso30_ValGrid', 'A4a_CloseDual_115_135_Iso30_ValGrid'),
        ('A3_Dual_082_125_Iso30_Equal', 'A4a_CloseDual_115_135_Iso30_Equal'),
        ('A5_TriScale_082_125_250_Iso30_ValGrid', 'A3_C3_Locked_082_125_Main'),
        ('A5_TriScale_082_125_250_Iso30_Equal', 'A3_Dual_082_125_Iso30_Equal'),
        ('A2_Single_082_Iso30', 'A1_Single_125_Iso30'),
    ]

    bootstrap_summary = {}
    for sp in ['val', 'testB', 'testC']:
        boot_seeds = []
        per_seed_pairs = []
        for s in [0, 1, 2]:
            b_arrs, p_out = paired_image_bootstrap_coco(gt_dicts[sp], pred_rows_store[sp][s], pairs_to_test, n_boot=1000, rng_seed=20261126 + s)
            boot_seeds.append(b_arrs)
            per_seed_pairs.append(p_out)

        bootstrap_summary[sp] = {}
        for m_a, m_b in pairs_to_test:
            tag = f"{m_a}__minus__{m_b}"
            bootstrap_summary[sp][tag] = {}
            for col, mlbl in [(0, 'AP50_95'), (1, 'AP50')]:
                diffs_3s = np.mean([boot_seeds[s][m_a][:, col] - boot_seeds[s][m_b][:, col] for s in [0, 1, 2]], axis=0)
                ci_lo, ci_hi = float(np.percentile(diffs_3s, 2.5)), float(np.percentile(diffs_3s, 97.5))
                p_two = min(1.0, 2.0 * min(float(np.mean(diffs_3s <= 0.0)), float(np.mean(diffs_3s >= 0.0))))
                bootstrap_summary[sp][tag][mlbl] = {
                    'mean_diff_3seed': round(float(diffs_3s.mean()), 3),
                    'ci95_3seed': [round(ci_lo, 3), round(ci_hi, 3)],
                    'p_value_two_sided': round(p_two, 4),
                    'significant_95': bool(ci_lo > 0.0 or ci_hi < 0.0),
                    'per_seed_mean_diffs': [per_seed_pairs[s][tag][mlbl]['boot_mean_diff'] for s in [0, 1, 2]],
                    'per_seed_ci95': [per_seed_pairs[s][tag][mlbl]['ci95'] for s in [0, 1, 2]],
                }
        if sp in ('testB', 'testC'):
            for m_a, m_b in pairs_to_test:
                tag = f"{m_a}__minus__{m_b}"
                r_ap = bootstrap_summary[sp][tag]['AP50_95']
                print(f"  [Bootstrap {sp:5s} | {tag:76s}] ΔAP50-95 = {r_ap['mean_diff_3seed']:+.3f}% "
                      f"(95% CI {r_ap['ci95_3seed']}, p={r_ap['p_value_two_sided']:.4f}, sig={r_ap['significant_95']})", flush=True)

    latency_summary = benchmark_fair_end2end_ablation_latency(manifest, gt_dicts['val'], gt_dicts['testC'], val_iso_weights['0'])

    out_payload = {
        'stage': 'Stage2_IsoWdet030_Controlled_Scale_Ablation_A1_to_A5_and_Fair_Latency',
        'elapsed_seconds': round(time.time() - t0, 2),
        'strict_fp32_env': env_meta,
        'val_iso_wdet030_selected_weights_per_seed': val_iso_weights,
        'ablation_metrics_3seed': ablation_summary,
        'paired_bootstrap_1000_results': bootstrap_summary,
        'fair_end2end_latency_rtx4060ti_fp32': latency_summary,
    }
    out_file = RES_DIR / 'stage2_c3_controlled_scale_ablation_complete.json'
    out_file.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n[Stage 2 全部严格受控实验与测速完成] 耗时 {out_payload['elapsed_seconds']}s，保存至: {out_file}", flush=True)


if __name__ == '__main__':
    main()
