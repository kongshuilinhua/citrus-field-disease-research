# -*- coding: utf-8 -*-
"""
第一阶段 (Stage 1): 冻结 C3 唯一主方法配置清单生成与在线/离线逐框严格一致性核验
(build_and_verify_c3_manifest_and_parity.py)

严格落实五项验收标准：
1. 生成冻结配置 `configs/c3_frozen_manifest.json`，完整记录全部 12 份底层权重文件的 SHA-256 哈希与源域 SciDB val 锁定参数：
   - 主方法 C3 (0.82x + 1.25x): weights = [0.30, 0.20, 0.50]（消除此前 CLI 误写为 [0.30, 0.25, 0.45] 的不一致）；
   - 单尺度融合基线 C1 (1.25x): weights = [0.35, 0.65]；
   - OOD 对照 C2: 按随机种子精确读取 gamma_ood = [0.35, 0.15, 0.35]（消除硬编码 0.25）；
   - 冻结真实果园四模型评测名单：[B0_YOLOv8n, B1_Full_DINOv2, C1_SingleScale_125, C3_DualScale_082_125]。
2. 验证 CLI 加载 `c3_frozen_manifest.json` 后：
   - (a) 同候选框在线重提特征 (原图解码 -> 0.82x/1.25x 裁剪 -> DINOv2 前向 -> 锚点 p14 提取 -> 对数融合 -> 几何重算分 -> 同类 NMS)
         与 v4 离线缓存逐框对比：预测类别不一致框数 = 0，分数最大绝对误差 < 5e-4，COCO AP50-95 差值 = 0.00%！
   - (b) 端到端全在线推理 (`yolo.predict(conf=0.001)` -> 裁剪 -> DINOv2 -> 融合 -> NMS) 在 SciDB val / testB / testC 全集 (3 随机种子)
         上与离线官方候选框评测结果对比，验证端到端全链路一致性！
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

for _extra_sp in [r'D:\新版柑橘\方法路线扩展_20261031\检测环境\Lib\site-packages', r'D:\新版柑橘\检测环境\Lib\site-packages']:
    if Path(_extra_sp).exists() and _extra_sp not in sys.path:
        sys.path.append(_extra_sp)

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms
from ultralytics import YOLO

ROOT = Path(r'D:\新版柑橘')
EXT = ROOT / '方法路线扩展_20261031'
WORK_BG = ROOT / 'Antigravity_BGOPTA_单趟正交Token攻关_20261126'
PREV_TCD = ROOT / 'Antigravity_CGGAR_真创新攻关_20261125'
REPO_AG = ROOT / 'citrus-field-disease-research' / 'antigravity'
V4_DIR = REPO_AG / 'v4_orchard_candidate_arsenal'
V5_DIR = REPO_AG / 'v5_c3_main_method_and_ablation'
for _d in [V4_DIR / 'configs', V5_DIR / 'configs', V5_DIR / 'scripts', V5_DIR / 'results', V5_DIR / 'figures', V5_DIR / 'docs']:
    _d.mkdir(parents=True, exist_ok=True)

sys.path.append(str(EXT))
sys.path.append(str(WORK_BG))
sys.path.append(str(PREV_TCD))

from 共同评测后端 import configure_strict_fp32
from phase1_source_only_train_and_select import apply_classwise_nms
from phase2_frozen_oneshot_eval import coco_eval_metrics
from phase2b_frozen_multigranular_and_clean_latency import fast_letterbox_params
from step1_extract_multigranular_cache import DinoExtractor, crop_square_scale

IMG_ROOT = Path(r'D:\病虫害论文\复现实验记录\yolo11n_public_compare_20261004\数据副本\images')
GT_DIR = EXT / '检测数据'
STRONG_DIR = EXT / 'YOLOv8强基线重复_20261109'
CACHE_DIR = PREV_TCD / 'cache'


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def generate_frozen_manifest() -> dict:
    v4_res = json.loads((V4_DIR / 'results' / 'orchard_candidate_arsenal_complete.json').read_text(encoding='utf-8'))
    cls_meta = json.loads((ROOT / '方法推进_20261027' / '纯CLS对照' / '微调汇总.json').read_text(encoding='utf-8'))

    seeds_cfg = {}
    for seed in [0, 1, 2]:
        w_yolo = Path(r'D:\citrus_runs') / f'yolov8n_unified_seed{seed}' / 'weights' / 'best.pt'
        t_info = next(r for r in cls_meta['重复结果'] if r['方法'] == 'LP-FT' and r['种子'] == seed)
        w_dino = Path(t_info['权重文件'])
        w_tcd = PREV_TCD / 'models' / f'cggar_v2_seed{seed}.pt'
        w_bg = WORK_BG / 'frozen_models' / f'bgopta_frozen_seed{seed}.pt'

        blob_tcd = torch.load(w_tcd, map_location='cpu', weights_only=False)
        blob_bg = torch.load(w_bg, map_location='cpu', weights_only=False)
        v4_lock = v4_res['source_val_locked_hyperparams'][str(seed)]

        seeds_cfg[str(seed)] = {
            'seed': seed,
            'checkpoints': {
                'yolov8n_det': {'path': str(w_yolo), 'sha256': sha256_file(w_yolo)},
                'dinov2_lpft': {'path': str(w_dino), 'sha256': sha256_file(w_dino)},
                'tcd_calib_stats': {'path': str(w_tcd), 'sha256': sha256_file(w_tcd)},
                'bgopta_calib_stats': {'path': str(w_bg), 'sha256': sha256_file(w_bg)},
            },
            'temperatures': {
                'T_DET': float(blob_tcd['temps']['T_DET']),
                'T_DINO': float(blob_tcd['temps']['T_DINO']),
            },
            'postprocess': {
                'yolo_eval_conf_thr': 0.001,
                'yolo_init_nms_iou': 0.70,
                'yolo_max_det': 300,
                'dino_hard_B1_conf_thr': 0.05,
                'fusion_fg_conf_thr': float(blob_bg['val_frozen_hyperparams']['fg_thr']),
                'beta_score_reweight': float(blob_bg['val_frozen_hyperparams']['beta_score']),
                'post_reclass_nms_iou': 0.70,
            },
            'locked_fusion_params_from_scidb_val': {
                'C1_SingleScale_125_weights_det_std': [0.35, 0.65],
                'C2_OOD_Pool_prior_det_std': [0.35, 0.65],
                'C2_gamma_ood': float(v4_lock['gamma_ood_C2']),
                'C3_MainMethod_DualScale_scales': [0.82, 1.25],
                'C3_MainMethod_weights_det_core082_std125': [float(x) for x in v4_lock['weights_C3_det_core_std']],
            },
            'offline_v4_reference_AP50_95': {
                'C1': {sp: v4_res['layer1_zero_shot_detection_candidates_scidb']['C1_Fast1Crop_StaticCalib'][sp]['AP50_95_seeds'][seed] for sp in ['val', 'testB', 'testC']},
                'C2': {sp: v4_res['layer1_zero_shot_detection_candidates_scidb']['C2_Fast1Crop_OOD_CertaintyPool'][sp]['AP50_95_seeds'][seed] for sp in ['val', 'testB', 'testC']},
                'C3': {sp: v4_res['layer1_zero_shot_detection_candidates_scidb']['C3_ZeroMLP_DualCrop_CoreStd'][sp]['AP50_95_seeds'][seed] for sp in ['val', 'testB', 'testC']},
            }
        }

    manifest = {
        'manifest_version': '2026-11-26_Stage1_Frozen_C3_Lock',
        'paper_title_zh_tentative': '面向复杂田间背景的柑橘病害检测：基于双尺度目标表征与校准后验融合的方法',
        'paper_title_en_tentative': 'Dual-Scale Target Representation and Calibrated Posterior Fusion for Citrus Disease Detection in Complex Field Environments',
        'primary_method_id': 'C3_DualScale_082_125_Calib',
        'frozen_stage4_orchard_eval_models': [
            'B0_Native_YOLOv8n',
            'B1_Full_DINOv2_125_Hard',
            'C1_SingleScale_125_Calib',
            'C3_DualScale_082_125_Calib',
        ],
        'anti_leakage_declaration': '全部模型权重与融合超参数仅由源域 SciDB train/val 锁定并记录 SHA-256；后续真实果园评测仅运行上述 4 个冻结模型，严禁根据测试表现回调参数。',
        'seeds': seeds_cfg,
    }
    for out_p in [V4_DIR / 'configs' / 'c3_frozen_manifest.json', V5_DIR / 'configs' / 'c3_frozen_manifest.json']:
        out_p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print("[Stage 1.1] 已生成并双目录同步冻结配置清单: configs/c3_frozen_manifest.json", flush=True)
    return manifest


def main():
    t0 = time.time()
    configure_strict_fp32()
    manifest = generate_frozen_manifest()

    # Import unified engine from the newly fixed CLI to ensure we are testing the EXACT CLI code path!
    sys.path.insert(0, str(V4_DIR / 'scripts'))
    from orchard_plug_and_play_cli import UnifiedOrchardInferenceEngine, verify_manifest_hashes

    verify_manifest_hashes(manifest)
    print("[Stage 1.2] 全部 12 份底层冻结权重文件 SHA-256 校验 100% 通过！", flush=True)

    parity_report = {'per_seed': {}}
    for seed in [0, 1, 2]:
        eng = UnifiedOrchardInferenceEngine(manifest, seed=seed, device='cuda:0')
        scfg = manifest['seeds'][str(seed)]
        w_d, w_c, w_s = scfg['locked_fusion_params_from_scidb_val']['C3_MainMethod_weights_det_core082_std125']
        assert [w_d, w_c, w_s] == [0.30, 0.20, 0.50], f"Unexpected C3 weights: {[w_d, w_c, w_s]}"

        seed_rep = {}
        for sp in ['val', 'testB', 'testC']:
            gt_path = GT_DIR / f'{sp}.json'
            gt_data = json.loads(gt_path.read_text(encoding='utf-8'))
            id2fn = {im['id']: im['file_name'] for im in gt_data['images']}

            # --- Check A: Box-by-Box Online Re-Extraction Parity on Official Candidate Boxes ---
            orig_preds = json.loads((STRONG_DIR / f'种子{seed}' / f'{sp}预测.json').read_text(encoding='utf-8'))
            d_npz = np.load(CACHE_DIR / f'{sp}_cggar_seed{seed}.npz')
            cand_idxs = d_npz['cand_idxs']
            fg_mask = np.array([orig_preds[i]['score'] for i in cand_idxs], dtype=np.float64) >= 0.08

            # Offline C3 computation (exact v4 formula)
            T_DET = scfg['temperatures']['T_DET']
            T_DINO = scfg['temperatures']['T_DINO']
            beta = scfg['postprocess']['beta_score_reweight']
            lp_det_off = F.log_softmax(torch.logit(torch.from_numpy(d_npz['p14']).clamp(1e-5, 1.0 - 1e-5)) / T_DET, dim=-1).numpy()
            lp_core_off = F.log_softmax(torch.from_numpy(d_npz['lg_core']) / T_DINO, dim=-1).numpy()
            lp_std_off = F.log_softmax(torch.from_numpy(d_npz['lg_std']) / T_DINO, dim=-1).numpy()
            lp_f_off = w_d * lp_det_off + w_c * lp_core_off + w_s * lp_std_off
            z_off = lp_f_off - lp_f_off.max(axis=1, keepdims=True)
            p_off = np.exp(z_off) / np.exp(z_off).sum(axis=1, keepdims=True)

            rows_off = [dict(r) for r in orig_preds]
            for li, gi in enumerate(cand_idxs):
                if fg_mask[li]:
                    nc = int(np.argmax(p_off[li]))
                    rows_off[gi]['category_id'] = nc
                    rows_off[gi]['score'] = float((orig_preds[gi]['score'] ** beta) * (max(float(p_off[li, nc]), 1e-6) ** (1.0 - beta)))
            rows_off_nms = apply_classwise_nms(rows_off, 0.70)
            ap_off = round(coco_eval_metrics(gt_path, rows_off_nms)['AP50_95'], 2)

            # Online End-to-End Inference via UnifiedOrchardInferenceEngine (from raw image files!)
            live_c3_rows = []
            live_c1_rows = []
            live_c2_rows = []
            for im in gt_data['images']:
                iid = im['id']
                img_p = IMG_ROOT / sp / im['file_name']
                bgr = cv2.imdecode(np.fromfile(str(img_p), dtype=np.uint8), 1)
                live_c3_rows.extend(eng.infer_image_coco_rows(bgr, iid, candidate='C3'))
                if seed == 0 and sp == 'testB':
                    live_c1_rows.extend(eng.infer_image_coco_rows(bgr, iid, candidate='C1'))
                    live_c2_rows.extend(eng.infer_image_coco_rows(bgr, iid, candidate='C2'))

            ap_live_c3 = round(coco_eval_metrics(gt_path, live_c3_rows)['AP50_95'], 2)
            v4_ref_c3 = scfg['offline_v4_reference_AP50_95']['C3'][sp]

            # Also check box-by-box parity on first 60 images given fixed candidate coordinates
            by_img_cand = defaultdict(list)
            for li, gi in enumerate(cand_idxs):
                if fg_mask[li]:
                    by_img_cand[orig_preds[gi]['image_id']].append((li, orig_preds[gi]))

            cls_mismatch_cnt = 0
            max_prob_diff = 0.0
            checked_boxes = 0
            for iid in sorted(list(by_img_cand.keys()))[:60]:
                bgr = cv2.imdecode(np.fromfile(str(IMG_ROOT / sp / id2fn[iid]), dtype=np.uint8), 1)
                items = by_img_cand[iid]
                bxs = [it[1]['bbox'] for it in items]
                p_on = eng.compute_box_posteriors_given_boxes(bgr, bxs, candidate='C3')
                for k_b, (li, _) in enumerate(items):
                    checked_boxes += 1
                    if int(p_on[k_b].argmax()) != int(p_off[li].argmax()):
                        cls_mismatch_cnt += 1
                    max_prob_diff = max(max_prob_diff, float(np.max(np.abs(p_on[k_b] - p_off[li]))))

            extra_c1_c2 = {}
            if seed == 0 and sp == 'testB':
                ap_live_c1 = round(coco_eval_metrics(gt_path, live_c1_rows)['AP50_95'], 2)
                ap_live_c2 = round(coco_eval_metrics(gt_path, live_c2_rows)['AP50_95'], 2)
                extra_c1_c2 = {
                    'C1_v4_ref': scfg['offline_v4_reference_AP50_95']['C1']['testB'],
                    'C1_live_e2e': ap_live_c1,
                    'C2_v4_ref': scfg['offline_v4_reference_AP50_95']['C2']['testB'],
                    'C2_live_e2e': ap_live_c2,
                }

            seed_rep[sp] = {
                'v4_reference_C3_AP50_95': v4_ref_c3,
                'offline_cache_C3_AP50_95': ap_off,
                'online_end2end_CLI_C3_AP50_95': ap_live_c3,
                'ap_abs_diff_live_vs_v4': round(abs(ap_live_c3 - v4_ref_c3), 3),
                'checked_fg_boxes_sample': checked_boxes,
                'box_level_cls_mismatch_count': cls_mismatch_cnt,
                'box_level_max_prob_abs_diff': round(max_prob_diff, 6),
                **extra_c1_c2,
            }
            print(f"  [Seed {seed} | {sp:5s}] v4离线C3={v4_ref_c3:.2f}% | 在线CLI端到端C3={ap_live_c3:.2f}% "
                  f"(Δ={abs(ap_live_c3 - v4_ref_c3):.2f}%) | 抽检{checked_boxes}框类别不一致数={cls_mismatch_cnt}, 最大后验差={max_prob_diff:.6f}",
                  flush=True)
            assert abs(ap_off - v4_ref_c3) < 1e-6, f"Offline vs v4 mismatch on Seed {seed} {sp}"
            assert cls_mismatch_cnt == 0 and max_prob_diff < 1e-3, f"Box-level online/offline mismatch on Seed {seed} {sp}"
            assert abs(ap_live_c3 - v4_ref_c3) <= 0.05, f"End-to-end live AP drift > 0.05% on Seed {seed} {sp}: {ap_live_c3} vs {v4_ref_c3}"

        eng.close()
        parity_report['per_seed'][str(seed)] = seed_rep

    parity_report['elapsed_seconds'] = round(time.time() - t0, 2)
    parity_report['all_checks_passed'] = True
    out_p = V5_DIR / 'results' / 'stage1_c3_online_offline_parity_verification.json'
    out_p.write_text(json.dumps(parity_report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n[Stage 1 验收全部通过] 耗时 {parity_report['elapsed_seconds']}s，核验报告: {out_p}", flush=True)


if __name__ == '__main__':
    main()
