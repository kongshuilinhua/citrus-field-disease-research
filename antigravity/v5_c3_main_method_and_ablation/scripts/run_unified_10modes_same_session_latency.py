# -*- coding: utf-8 -*-
"""
统一同会话 10 模式端到端公平测速脚本 (run_unified_10modes_same_session_latency.py)
严格杜绝跨会话 GPU 功耗/频率波动导致的延迟混比：在同一进程、同一预热状态下顺序评测
B0, B1, C1, A1, G2_blur, A2, A4a, A4b, C3, A5 共 10 组方案，并回写至 stage2 与 stage2b JSON。
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'

import sys
import json
import time
from pathlib import Path
from collections import defaultdict

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
V5_DIR = ROOT / 'citrus-field-disease-research' / 'antigravity' / 'v5_c3_main_method_and_ablation'
RES_DIR = V5_DIR / 'results'

sys.path.append(str(EXT))
sys.path.append(str(ROOT / 'Antigravity_BGOPTA_单趟正交Token攻关_20261126'))
sys.path.append(str(ROOT / 'Antigravity_CGGAR_真创新攻关_20261125'))
sys.path.insert(0, str(V5_DIR / 'scripts'))

from 共同评测后端 import configure_strict_fp32
from phase2_frozen_oneshot_eval import coco_eval_metrics
from step1_extract_multigranular_cache import crop_square_scale
from orchard_plug_and_play_cli import UnifiedOrchardInferenceEngine, crop_125_with_outer_blur

IMG_ROOT = Path(r'D:\病虫害论文\复现实验记录\yolo11n_public_compare_20261004\数据副本\images')
GT_DIR = EXT / '检测数据'
MANIFEST_PATH = V5_DIR / 'configs' / 'c3_frozen_manifest.json'
STAGE2_JSON_PATH = RES_DIR / 'stage2_c3_controlled_scale_ablation_complete.json'
STAGE2B_JSON_PATH = RES_DIR / 'stage2b_gray_ring_mechanism_and_a4b_speed.json'


@torch.no_grad()
def main():
    configure_strict_fp32()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    s2 = json.loads(STAGE2_JSON_PATH.read_text(encoding='utf-8'))
    s2b = json.loads(STAGE2B_JSON_PATH.read_text(encoding='utf-8'))
    val_iso_s0 = s2['val_iso_wdet030_selected_weights_per_seed']['0']

    val_gt = json.loads((GT_DIR / 'val.json').read_text(encoding='utf-8'))
    tc_gt = json.loads((GT_DIR / 'testC.json').read_text(encoding='utf-8'))

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

    eng = UnifiedOrchardInferenceEngine(manifest, seed=0, device='cuda:0')

    def run_timed_mode(items: list, mode_key: str, verify_gt_path: Path = None):
        times_all, times_multi, dino_crops, out_rows = [], [], [], []
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
                    elif mode_key == 'G2_blur_Single_125_BlurRing':
                        c1 = np.stack([crop_125_with_outer_blur(rgb, bb, eng.blur_ksize, eng.blur_sigma) for bb in bxs_fg], axis=0)
                        lp1 = F.log_softmax(eng._encode_dino_logits(c1) / eng.T_DINO, dim=-1)
                        p_post = F.softmax(eng.w_g2_blur[0] * lp_det + eng.w_g2_blur[1] * lp1, dim=-1)
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
                        w0, w1, w2 = eng.w_c3_eq  # same compute cost for [0.30,0.35,0.35] and [0.30,0.20,0.50]
                        p_post = F.softmax(w0 * lp_det + w1 * lp_a + w2 * lp_b, dim=-1)
                        n_crops = 2 * k
                    elif mode_key == 'A4a_CloseDual_115_135':
                        c2 = np.stack([crop_square_scale(rgb, bb, 1.15) for bb in bxs_fg] +
                                      [crop_square_scale(rgb, bb, 1.35) for bb in bxs_fg], axis=0)
                        lg2 = eng._encode_dino_logits(c2)
                        lp_a = F.log_softmax(lg2[:k] / eng.T_DINO, dim=-1)
                        lp_b = F.log_softmax(lg2[k:] / eng.T_DINO, dim=-1)
                        w0, w1, w2 = val_iso_s0['A4a_CloseDual_115_135_Iso30']
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
                        w0, w1, w2, w3 = val_iso_s0['A5_TriScale_082_125_250_Iso30']
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

            xy_cpu, sc_cpu, cl_cpu = xyxy_f.cpu().tolist(), sc_f.cpu().tolist(), cls_f.cpu().tolist()
            torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - ts) * 1000.0
            if idx >= 10:
                times_all.append(dt_ms)
                dino_crops.append(n_crops)
                if n_gt_img >= 3 or sum(1 for s in sc_cpu if s >= 0.25) >= 3:
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

    # Heavy GPU warmup so GPU boosts to full clock before timing mode 1
    _ = run_timed_mode(val_items[:35], 'A5_TriScale_082_125_250', None)
    modes_10 = [
        'B0_Native_YOLOv8n',
        'B1_Full_DINOv2_125_Hard',
        'C1_Single_125_035_065',
        'A1_Single_125',
        'G2_blur_Single_125_BlurRing',
        'A2_Single_082',
        'A4a_CloseDual_115_135',
        'A4b_FlipTTA_125_HFlip',
        'A3_Dual_082_125_C3_Main',
        'A5_TriScale_082_125_250',
    ]
    lat_10 = {}
    print("================ [全 10 模式同会话公平端到端测速] ================", flush=True)
    for mk in modes_10:
        rv = run_timed_mode(val_items, mk, None)
        rc = run_timed_mode(tc_items, mk, GT_DIR / 'testC.json')
        lat_10[mk] = {'val_250imgs': rv, 'testC_all602imgs': rc}
        print(f"  [{mk:28s}] val: mean={rv['mean_ms']:5.2f}ms, P95={rv['p95_ms']:5.2f}ms, 多目标={rv['multi_target_ge3_mean_ms']}ms | "
              f"testC: mean={rc['mean_ms']:5.2f}ms, P95={rc['p95_ms']:5.2f}ms, 实时AP={rc['verified_live_AP50_95_seed0']:.2f}%", flush=True)

    eng.close()
    s2['fair_end2end_latency_rtx4060ti_fp32'] = lat_10
    s2b['unified_9modes_fair_end2end_latency_rtx4060ti_fp32'] = lat_10
    s2b['v1_1_statistical_wrapup']['g2_blur_end2end_latency_rtx4060ti_fp32'] = lat_10['G2_blur_Single_125_BlurRing']
    STAGE2_JSON_PATH.write_text(json.dumps(s2, ensure_ascii=False, indent=2), encoding='utf-8')
    STAGE2B_JSON_PATH.write_text(json.dumps(s2b, ensure_ascii=False, indent=2), encoding='utf-8')
    print("[10 模式同会话测速更新完毕]", flush=True)


if __name__ == '__main__':
    main()
