# -*- coding: utf-8 -*-
"""
第一阶段统计收尾脚本 (run_stage2b_v11_statistical_wrapup.py)：
不重跑特征提取（直接复用本地确定性 logits 缓存），快速完成 5 项统计收尾与核验：
1. 补齐决定“两刀是否必要”的关键 B = 1,000 配对 Bootstrap 检验（输出逐种子差值、逐种子 CI、逐种子双侧 p 值 + 3 种子均值）：
   - G3 − G2            (冻结 C3 [0.3,0.2,0.5] vs 单刀灰环 G2 [0.3,0.7])
   - G3 − G2_blur       (冻结 C3 [0.3,0.2,0.5] vs 单刀模糊遮罩 G2_blur [0.3,0.7] —— 核心判决项！)
   - G3_eq − G2         (零调参等权 C3_equal [0.3,0.35,0.35] vs 单刀灰环 G2 [0.3,0.7])
   - G3_eq − G2_blur    (零调参等权 C3_equal [0.3,0.35,0.35] vs 单刀模糊遮罩 G2_blur [0.3,0.7] —— 零自由度核心判决项！)
   - G3 − G2_c1 / G3 − G2_blur_c1 / PureDINO_G3 − PureDINO_G2 / PureDINO_G3 − PureDINO_G2_blur
   - C3 − A4b           (冻结 C3 vs A4b_ValGrid / A4b_Equal) 及 C3_equal − A4b_ValGrid / A4b_Equal
   - G2_blur − A4b      (单刀模糊遮罩 vs 双刀翻转 TTA)
   并为既有 DiD 等全部检验补充 `per_seed_p_values`！
2. 核验新增第 6 预注册盲测模型 `G2_blur` 与 `C3_equal` 在 CLI 中的在线/离线一致性（Seed 0 testB/testC 零误差）及 6 模型按簇评测器冒烟测试。
3. 补测 `G2_blur` 在 RTX 4060 Ti FP32 上的端到端单图延迟（连同 C3 / A4b / A1 同协议对照）。
4. 将全部结果追加回写至 `results/stage2b_gray_ring_mechanism_and_a4b_speed.json` 与 `stage2_c3_controlled_scale_ablation_complete.json`。
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'

import sys
import json
import time
import tempfile
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
REPO_AG = ROOT / 'citrus-field-disease-research' / 'antigravity'
V5_DIR = REPO_AG / 'v5_c3_main_method_and_ablation'
RES_DIR = V5_DIR / 'results'

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
from phase2_frozen_oneshot_eval import coco_eval_metrics
from run_rigorous_audit_suite import paired_image_bootstrap_coco
from orchard_plug_and_play_cli import (
    UnifiedOrchardInferenceEngine,
    verify_manifest_hashes,
    crop_125_with_outer_blur,
    evaluate_frozen_6models_on_labeled_orchard,
)

IMG_ROOT = Path(r'D:\病虫害论文\复现实验记录\yolo11n_public_compare_20261004\数据副本\images')
GT_DIR = EXT / '检测数据'
STRONG_DIR = EXT / 'YOLOv8强基线重复_20261109'
CACHE_DIR = ROOT / 'Antigravity_CGGAR_真创新攻关_20261125' / 'cache'
MANIFEST_PATH = V5_DIR / 'configs' / 'c3_frozen_manifest.json'
STAGE2_JSON_PATH = RES_DIR / 'stage2_c3_controlled_scale_ablation_complete.json'
STAGE2B_JSON_PATH = RES_DIR / 'stage2b_gray_ring_mechanism_and_a4b_speed.json'


def softmax_from_lp(lp: np.ndarray) -> np.ndarray:
    z = lp - lp.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def summarize_diff_with_per_seed_p(dv_3s: np.ndarray, per_seed_vecs: list):
    lo, hi = float(np.percentile(dv_3s, 2.5)), float(np.percentile(dv_3s, 97.5))
    p_two = min(1.0, 2.0 * min(float(np.mean(dv_3s <= 0.0)), float(np.mean(dv_3s >= 0.0))))
    ps_means, ps_cis, ps_ps = [], [], []
    for sv in per_seed_vecs:
        ps_means.append(round(float(sv.mean()), 3))
        ps_cis.append([round(float(np.percentile(sv, 2.5)), 3), round(float(np.percentile(sv, 97.5)), 3)])
        ps_ps.append(round(min(1.0, 2.0 * min(float(np.mean(sv <= 0.0)), float(np.mean(sv >= 0.0)))), 4))
    return {
        'mean_diff_3seed': round(float(dv_3s.mean()), 3),
        'ci95_3seed': [round(lo, 3), round(hi, 3)],
        'p_value_two_sided': round(p_two, 4),
        'significant_95': bool(lo > 0.0 or hi < 0.0),
        'per_seed_mean_diffs': ps_means,
        'per_seed_ci95': ps_cis,
        'per_seed_p_values': ps_ps,
    }


@torch.no_grad()
def verify_g2_blur_parity_latency_and_eval6(manifest: dict, gt_dicts: dict):
    print("\n================ [模块 1] 核验 G2_blur / C3_equal 在线一致性、G2_blur 端到端测速与 6 模型评测器 ================", flush=True)
    eng = UnifiedOrchardInferenceEngine(manifest, seed=0, device='cuda:0')

    # 1) Verify G2_blur & C3_equal online vs offline parity on Seed 0 testB & testC
    parity_out = {}
    for cand in ['G2_blur', 'C3_equal']:
        parity_out[cand] = {}
        for sp in ['testB', 'testC']:
            gt_p = GT_DIR / f'{sp}.json'
            rows_live = []
            for im in gt_dicts[sp]['images']:
                bgr = cv2.imdecode(np.fromfile(str(IMG_ROOT / sp / im['file_name']), dtype=np.uint8), 1)
                rows_live.extend(eng.infer_image_coco_rows(bgr, im['id'], candidate=cand))
            ap_live = round(coco_eval_metrics(gt_p, rows_live)['AP50_95'], 2)
            ap_ref = manifest['seeds']['0']['offline_v4_reference_AP50_95'][cand][sp]
            assert abs(ap_live - ap_ref) <= 0.02, f"{cand} parity mismatch on {sp}: {ap_live} vs {ap_ref}"
            parity_out[cand][sp] = {'offline_ref_AP50_95': ap_ref, 'online_cli_AP50_95': ap_live, 'abs_diff': round(abs(ap_live - ap_ref), 3)}
            print(f"  [{cand:10s} 在线/离线核验 | Seed 0 {sp}] 离线={ap_ref:.2f}% vs 在线={ap_live:.2f}% (差值={abs(ap_live - ap_ref):.2f}%)", flush=True)

    # 2) Measure G2_blur end-to-end latency on identical val_250imgs and testC_all602imgs
    anns_val = defaultdict(int)
    for a in gt_dicts['val']['annotations']:
        anns_val[a['image_id']] += 1
    anns_tc = defaultdict(int)
    for a in gt_dicts['testC']['annotations']:
        anns_tc[a['image_id']] += 1

    val_multi = [im for im in gt_dicts['val']['images'] if anns_val[im['id']] >= 3]
    val_single = [im for im in gt_dicts['val']['images'] if anns_val[im['id']] < 3][:max(0, 250 - len(val_multi))]
    val_items = [(im['id'], IMG_ROOT / 'val' / im['file_name'], anns_val[im['id']]) for im in (val_multi + val_single)]
    tc_items = [(im['id'], IMG_ROOT / 'testC' / im['file_name'], anns_tc[im['id']]) for im in gt_dicts['testC']['images']]

    def time_g2_blur(items: list, verify_gt_path: Path = None):
        times_all, times_multi, dino_crops, out_rows = [], [], [], []
        for idx, (iid, p, n_gt_img) in enumerate(items):
            torch.cuda.synchronize()
            ts = time.perf_counter()
            n_crops = 0
            bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), 1)
            h, w = bgr.shape[:2]
            xyxy_t, sc_t, cls_t = eng.yolo_detect_square640(bgr)
            fg_ids = torch.where(sc_t >= eng.fg_thr)[0]
            xyxy_f, sc_f, cls_f = xyxy_t.clone(), sc_t.clone(), cls_t.clone()
            if len(fg_ids) > 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                bxs_fg = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in xyxy_t[fg_ids].cpu().tolist()]
                p14_fg, _ = eng._extract_anchor_p14_and_neck(h, w, bxs_fg, need_neck=False)
                lp_det = F.log_softmax(torch.logit(p14_fg.clamp(1e-5, 1.0 - 1e-5)) / eng.T_DET, dim=-1)
                cb = np.stack([crop_125_with_outer_blur(rgb, bb, eng.blur_ksize, eng.blur_sigma) for bb in bxs_fg], axis=0)
                lp1 = F.log_softmax(eng._encode_dino_logits(cb) / eng.T_DINO, dim=-1)
                p_post = F.softmax(eng.w_g2_blur[0] * lp_det + eng.w_g2_blur[1] * lp1, dim=-1)
                n_crops = len(bxs_fg)
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

    _ = time_g2_blur(val_items[:15], None)
    rv_g2b = time_g2_blur(val_items, None)
    rc_g2b = time_g2_blur(tc_items, GT_DIR / 'testC.json')
    print(f"  [G2_blur_Single_125_BlurRing ] val: mean={rv_g2b['mean_ms']:5.2f}ms, P95={rv_g2b['p95_ms']:5.2f}ms, "
          f"多目标={rv_g2b['multi_target_ge3_mean_ms']}ms | testC: mean={rc_g2b['mean_ms']:5.2f}ms, P95={rc_g2b['p95_ms']:5.2f}ms, "
          f"实时AP={rc_g2b['verified_live_AP50_95_seed0']:.2f}%", flush=True)
    eng.close()

    # 3) Smoke-test 6-model (+ C3_equal) cluster-bootstrap orchard evaluator
    with tempfile.TemporaryDirectory() as tmpd:
        tmp_path = Path(tmpd)
        sub_imgs = gt_dicts['testB']['images'][:12]
        sub_iids = {im['id'] for im in sub_imgs}
        sub_anns = [dict(a) for a in gt_dicts['testB']['annotations'] if a['image_id'] in sub_iids]
        sub_gt = {'images': sub_imgs, 'annotations': sub_anns, 'categories': gt_dicts['testB']['categories']}
        sub_gt_p = tmp_path / 'mock_gt.json'
        sub_gt_p.write_text(json.dumps(sub_gt, ensure_ascii=False), encoding='utf-8')
        csv_p = tmp_path / 'clusters.csv'
        csv_p.write_text('\n'.join(['file_name,cluster_id'] + [f"{im['file_name']},c_{i // 3}" for i, im in enumerate(sub_imgs)]), encoding='utf-8')
        rep6 = evaluate_frozen_6models_on_labeled_orchard(
            manifest, IMG_ROOT / 'testB', sub_gt_p, tmp_path / 'out6.json', cluster_csv=str(csv_p), seeds=(0,), n_boot=100
        )
        assert 'G2_blur_Single125_BlurRing' in rep6['models_per_seed_and_mean']
        assert 'C3_DualScale_082_125__minus__G2_blur_Single125_BlurRing' in rep6['cluster_paired_bootstrap']['paired_comparisons']

    return parity_out, {'val_250imgs': rv_g2b, 'testC_all602imgs': rc_g2b}


def main():
    t0 = time.time()
    configure_strict_fp32()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    verify_manifest_hashes(manifest)
    s2 = json.loads(STAGE2_JSON_PATH.read_text(encoding='utf-8'))
    s2b = json.loads(STAGE2B_JSON_PATH.read_text(encoding='utf-8'))

    gt_dicts = {sp: json.loads((GT_DIR / f'{sp}.json').read_text(encoding='utf-8')) for sp in ['val', 'testB', 'testC']}
    parity_g2b, lat_g2b = verify_g2_blur_parity_latency_and_eval6(manifest, gt_dicts)

    print("\n================ [模块 2] 复用确定性特征缓存补齐 G3−G2 / G3−G2_blur / C3−A4b 全部 B=1,000 配对检验 ================", flush=True)
    gb_z = np.load(SCRATCH_GRAY_BLUR_CACHE, allow_pickle=True)
    a4_z = np.load(SCRATCH_A4_CACHE)

    pred_store = {sp: {s: {} for s in [0, 1, 2]} for sp in ['val', 'testB', 'testC']}
    for seed in [0, 1, 2]:
        scfg = manifest['seeds'][str(seed)]
        T_DET = float(scfg['temperatures']['T_DET'])
        T_DINO = float(scfg['temperatures']['T_DINO'])
        beta = float(scfg['postprocess']['beta_score_reweight'])
        fg_thr = float(scfg['postprocess']['fusion_fg_conf_thr'])
        w_c3 = scfg['locked_fusion_params_from_scidb_val']['C3_MainMethod_weights_det_core082_std125']
        w_a4b_v = s2['val_iso_wdet030_selected_weights_per_seed'][str(seed)]['A4b_FlipTTA_125_HFlip_Iso30']

        for sp in ['val', 'testB', 'testC']:
            orig_preds = json.loads((STRONG_DIR / f'种子{seed}' / f'{sp}预测.json').read_text(encoding='utf-8'))
            d_npz = np.load(CACHE_DIR / f'{sp}_cggar_seed{seed}.npz')
            cand_idxs = d_npz['cand_idxs']
            yolo_sc = np.array([orig_preds[i]['score'] for i in cand_idxs], dtype=np.float64)
            fg = yolo_sc >= fg_thr
            cand_map = {gi: li for li, gi in enumerate(cand_idxs)}

            lp_det = F.log_softmax(torch.logit(torch.from_numpy(d_npz['p14']).clamp(1e-5, 1.0 - 1e-5)) / T_DET, dim=-1).numpy()
            lp_082 = F.log_softmax(torch.from_numpy(d_npz['lg_core']) / T_DINO, dim=-1).numpy()
            lp_125 = F.log_softmax(torch.from_numpy(d_npz['lg_std']) / T_DINO, dim=-1).numpy()
            lp_gray = F.log_softmax(torch.from_numpy(gb_z[f's{seed}_{sp}_gray']) / T_DINO, dim=-1).numpy()
            lp_blur = F.log_softmax(torch.from_numpy(gb_z[f's{seed}_{sp}_blur']) / T_DINO, dim=-1).numpy()
            lp_125f = F.log_softmax(torch.from_numpy(a4_z[f's{seed}_{sp}_125f']) / T_DINO, dim=-1).numpy()

            def build_clean(p_mat: np.ndarray):
                rows = []
                for gi, r in enumerate(orig_preds):
                    nr = {'image_id': r['image_id'], 'category_id': r['category_id'], 'bbox': r['bbox'], 'score': r['score']}
                    li = cand_map.get(gi, None)
                    if li is not None and fg[li]:
                        pr = p_mat[li]
                        nc = int(np.argmax(pr))
                        nr['category_id'] = nc
                        nr['score'] = float((yolo_sc[li] ** beta) * (max(float(pr[nc]), 1e-6) ** (1.0 - beta)))
                    rows.append(nr)
                return [{'image_id': x['image_id'], 'category_id': x['category_id'], 'bbox': x['bbox'], 'score': x['score']}
                        for x in apply_classwise_nms(rows, 0.70)]

            p_defs = {
                'G1_125_RealBG_Iso30': softmax_from_lp(0.30 * lp_det + 0.70 * lp_125),
                'G2_125_GrayOuter_Iso30': softmax_from_lp(0.30 * lp_det + 0.70 * lp_gray),
                'G2_blur_125_BlurOuter_Iso30': softmax_from_lp(0.30 * lp_det + 0.70 * lp_blur),
                'G3_082_125_RealBG_C3_Locked': softmax_from_lp(w_c3[0] * lp_det + w_c3[1] * lp_082 + w_c3[2] * lp_125),
                'G4_082_125_GrayOuter_C3_Locked': softmax_from_lp(w_c3[0] * lp_det + w_c3[1] * lp_082 + w_c3[2] * lp_gray),
                'G4_blur_082_125_BlurOuter_C3_Locked': softmax_from_lp(w_c3[0] * lp_det + w_c3[1] * lp_082 + w_c3[2] * lp_blur),
                'G3_eq_082_125_RealBG_Equal': softmax_from_lp(0.30 * lp_det + 0.35 * lp_082 + 0.35 * lp_125),
                'G4_eq_082_125_GrayOuter_Equal': softmax_from_lp(0.30 * lp_det + 0.35 * lp_082 + 0.35 * lp_gray),
                'G4_blur_eq_082_125_BlurOuter_Equal': softmax_from_lp(0.30 * lp_det + 0.35 * lp_082 + 0.35 * lp_blur),
                'G2_c1_125_GrayOuter_035_065': softmax_from_lp(0.35 * lp_det + 0.65 * lp_gray),
                'G2_blur_c1_125_BlurOuter_035_065': softmax_from_lp(0.35 * lp_det + 0.65 * lp_blur),
                'A4b_FlipTTA_125_HFlip_Iso30_ValGrid': softmax_from_lp(w_a4b_v[0] * lp_det + w_a4b_v[1] * lp_125 + w_a4b_v[2] * lp_125f),
                'A4b_FlipTTA_125_HFlip_Iso30_Equal': softmax_from_lp(0.30 * lp_det + 0.35 * lp_125 + 0.35 * lp_125f),
                'PureDINO_G2_125_GrayOuter': softmax_from_lp(lp_gray),
                'PureDINO_G2_blur_125_BlurOuter': softmax_from_lp(lp_blur),
                'PureDINO_G3_082_125_RealBG': softmax_from_lp((2.0 / 7.0) * lp_082 + (5.0 / 7.0) * lp_125),
            }
            for mk, pm in p_defs.items():
                pred_store[sp][seed][mk] = build_clean(pm)

    decisive_pairs = [
        # Decisive "Is the 2nd crop necessary over 1-crop + mask?" pairs:
        ('G3_082_125_RealBG_C3_Locked', 'G2_125_GrayOuter_Iso30'),
        ('G3_082_125_RealBG_C3_Locked', 'G2_blur_125_BlurOuter_Iso30'),
        ('G3_eq_082_125_RealBG_Equal', 'G2_125_GrayOuter_Iso30'),
        ('G3_eq_082_125_RealBG_Equal', 'G2_blur_125_BlurOuter_Iso30'),
        ('G3_082_125_RealBG_C3_Locked', 'G2_c1_125_GrayOuter_035_065'),
        ('G3_082_125_RealBG_C3_Locked', 'G2_blur_c1_125_BlurOuter_035_065'),
        ('PureDINO_G3_082_125_RealBG', 'PureDINO_G2_125_GrayOuter'),
        ('PureDINO_G3_082_125_RealBG', 'PureDINO_G2_blur_125_BlurOuter'),
        # Explicit C3 / C3_equal / G2_blur vs A4b pairs in stage2b JSON:
        ('G3_082_125_RealBG_C3_Locked', 'A4b_FlipTTA_125_HFlip_Iso30_ValGrid'),
        ('G3_082_125_RealBG_C3_Locked', 'A4b_FlipTTA_125_HFlip_Iso30_Equal'),
        ('G3_eq_082_125_RealBG_Equal', 'A4b_FlipTTA_125_HFlip_Iso30_ValGrid'),
        ('G3_eq_082_125_RealBG_Equal', 'A4b_FlipTTA_125_HFlip_Iso30_Equal'),
        ('G2_blur_125_BlurOuter_Iso30', 'A4b_FlipTTA_125_HFlip_Iso30_ValGrid'),
        ('G2_blur_125_BlurOuter_Iso30', 'A4b_FlipTTA_125_HFlip_Iso30_Equal'),
        # Also re-compute core DiD inputs with identical RNG seed (20261127+s) so per_seed_p_values are populated!
        ('G2_125_GrayOuter_Iso30', 'G1_125_RealBG_Iso30'),
        ('G2_blur_125_BlurOuter_Iso30', 'G1_125_RealBG_Iso30'),
        ('G3_082_125_RealBG_C3_Locked', 'G1_125_RealBG_Iso30'),
        ('G4_082_125_GrayOuter_C3_Locked', 'G2_125_GrayOuter_Iso30'),
        ('G4_blur_082_125_BlurOuter_C3_Locked', 'G2_blur_125_BlurOuter_Iso30'),
        ('G3_eq_082_125_RealBG_Equal', 'G1_125_RealBG_Iso30'),
        ('G4_eq_082_125_GrayOuter_Equal', 'G2_125_GrayOuter_Iso30'),
        ('G4_blur_eq_082_125_BlurOuter_Equal', 'G2_blur_125_BlurOuter_Iso30'),
    ]

    for sp in ['val', 'testB', 'testC']:
        boot_s = []
        for s in [0, 1, 2]:
            b_arrs, _ = paired_image_bootstrap_coco(gt_dicts[sp], pred_store[sp][s], decisive_pairs, n_boot=1000, rng_seed=20261127 + s)
            boot_s.append(b_arrs)

        for m_a, m_b in decisive_pairs:
            tag = f"{m_a}__minus__{m_b}"
            s2b['g1_to_g4_paired_bootstrap_and_did_1000'][sp].setdefault(tag, {})
            for col, mlbl in [(0, 'AP50_95'), (1, 'AP50')]:
                ps_v = [boot_s[s][m_a][:, col] - boot_s[s][m_b][:, col] for s in [0, 1, 2]]
                s2b['g1_to_g4_paired_bootstrap_and_did_1000'][sp][tag][mlbl] = summarize_diff_with_per_seed_p(np.mean(ps_v, axis=0), ps_v)

        # Update DiDs with per_seed_p_values too
        did_updates = [
            ('DiD_Gray_minus_Real_LockedC3', 'G4_082_125_GrayOuter_C3_Locked', 'G2_125_GrayOuter_Iso30', 'G3_082_125_RealBG_C3_Locked', 'G1_125_RealBG_Iso30'),
            ('DiD_Blur_minus_Real_LockedC3', 'G4_blur_082_125_BlurOuter_C3_Locked', 'G2_blur_125_BlurOuter_Iso30', 'G3_082_125_RealBG_C3_Locked', 'G1_125_RealBG_Iso30'),
            ('DiD_Gray_minus_Real_EqualWeight', 'G4_eq_082_125_GrayOuter_Equal', 'G2_125_GrayOuter_Iso30', 'G3_eq_082_125_RealBG_Equal', 'G1_125_RealBG_Iso30'),
            ('DiD_Blur_minus_Real_EqualWeight', 'G4_blur_eq_082_125_BlurOuter_Equal', 'G2_blur_125_BlurOuter_Iso30', 'G3_eq_082_125_RealBG_Equal', 'G1_125_RealBG_Iso30'),
        ]
        for did_tag, g4k, g2k, g3k, g1k in did_updates:
            for col, mlbl in [(0, 'AP50_95'), (1, 'AP50')]:
                ps_v = [(boot_s[s][g4k][:, col] - boot_s[s][g2k][:, col]) - (boot_s[s][g3k][:, col] - boot_s[s][g1k][:, col]) for s in [0, 1, 2]]
                s2b['g1_to_g4_paired_bootstrap_and_did_1000'][sp][did_tag][mlbl] = summarize_diff_with_per_seed_p(np.mean(ps_v, axis=0), ps_v)

        if sp in ('testB', 'testC'):
            print(f"\n  === [{sp}] 决定“两刀是否必要”及 A4b 逐种子+均值 Bootstrap 核查 ===", flush=True)
            for k_show in [
                'G3_082_125_RealBG_C3_Locked__minus__G2_125_GrayOuter_Iso30',
                'G3_082_125_RealBG_C3_Locked__minus__G2_blur_125_BlurOuter_Iso30',
                'G3_eq_082_125_RealBG_Equal__minus__G2_125_GrayOuter_Iso30',
                'G3_eq_082_125_RealBG_Equal__minus__G2_blur_125_BlurOuter_Iso30',
                'G3_082_125_RealBG_C3_Locked__minus__A4b_FlipTTA_125_HFlip_Iso30_ValGrid',
                'G3_eq_082_125_RealBG_Equal__minus__A4b_FlipTTA_125_HFlip_Iso30_ValGrid',
                'G3_eq_082_125_RealBG_Equal__minus__A4b_FlipTTA_125_HFlip_Iso30_Equal',
                'G2_blur_125_BlurOuter_Iso30__minus__A4b_FlipTTA_125_HFlip_Iso30_ValGrid',
                'DiD_Gray_minus_Real_LockedC3',
                'DiD_Blur_minus_Real_LockedC3',
                'DiD_Gray_minus_Real_EqualWeight',
                'DiD_Blur_minus_Real_EqualWeight',
            ]:
                r = s2b['g1_to_g4_paired_bootstrap_and_did_1000'][sp][k_show]['AP50_95']
                print(f"    {k_show:74s} | 3种子均值={r['mean_diff_3seed']:+.3f}% CI={r['ci95_3seed']} p={r['p_value_two_sided']:.4f} | "
                      f"逐种子={r['per_seed_mean_diffs']} 逐种子p={r['per_seed_p_values']}", flush=True)

    # Record v1.1 wrapup additions into stage2b & stage2 JSON
    s2b['v1_1_statistical_wrapup'] = {
        'completed_at': '2026-11-27',
        'g2_blur_and_c3_equal_cli_parity': parity_g2b,
        'g2_blur_end2end_latency_rtx4060ti_fp32': lat_g2b,
        'decisive_question_verdict': {
            'question': '第二次裁剪 (0.82x) 相对“单刀 1.25x + 框外高斯模糊遮罩 (G2_blur)”到底多带来多少，显不显著？',
            'testB_C3_frozen_minus_G2_blur': s2b['g1_to_g4_paired_bootstrap_and_did_1000']['testB']['G3_082_125_RealBG_C3_Locked__minus__G2_blur_125_BlurOuter_Iso30']['AP50_95'],
            'testB_C3_equal_minus_G2_blur': s2b['g1_to_g4_paired_bootstrap_and_did_1000']['testB']['G3_eq_082_125_RealBG_Equal__minus__G2_blur_125_BlurOuter_Iso30']['AP50_95'],
            'testC_C3_frozen_minus_G2_blur': s2b['g1_to_g4_paired_bootstrap_and_did_1000']['testC']['G3_082_125_RealBG_C3_Locked__minus__G2_blur_125_BlurOuter_Iso30']['AP50_95'],
            'testC_C3_equal_minus_G2_blur': s2b['g1_to_g4_paired_bootstrap_and_did_1000']['testC']['G3_eq_082_125_RealBG_Equal__minus__G2_blur_125_BlurOuter_Iso30']['AP50_95'],
        },
        'errata_logged': [
            '勘误 1：DiD 收缩为 2/3 种子现象——testB 上冻结 C3 的 DiD_gray 逐种子为 [+0.505, -1.989, -1.455] (Seed 0 无收缩)，DiD_blur 逐种子为 [-0.154, -1.011, -1.503]。',
            '勘误 2：等权口径 [0.30, 0.35, 0.35] 下 DiD 未达统计显著——testB 上 DiD_gray_eq = -0.621% (p=0.122)，DiD_blur_eq = -0.688% (p=0.098)，撤回此前“等权口径同样显著收缩”的不准确表述。',
        ],
    }
    s2b['unified_9modes_fair_end2end_latency_rtx4060ti_fp32']['G2_blur_Single_125_BlurRing'] = lat_g2b
    s2['fair_end2end_latency_rtx4060ti_fp32']['G2_blur_Single_125_BlurRing'] = lat_g2b

    STAGE2B_JSON_PATH.write_text(json.dumps(s2b, ensure_ascii=False, indent=2), encoding='utf-8')
    STAGE2_JSON_PATH.write_text(json.dumps(s2, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n[v1.1 统计收尾脚本全部完成] 耗时 {time.time() - t0:.1f}s -> 已更新 {STAGE2B_JSON_PATH}", flush=True)


if __name__ == '__main__':
    main()
