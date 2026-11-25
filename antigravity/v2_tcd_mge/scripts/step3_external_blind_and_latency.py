# -*- coding: utf-8 -*-
"""
Step 3: 完全冻结权重与源域 val 阈值下的独立外部果园零样本盲测 (Zenodo 11148332, 519 图 / 1,765 框)
以及 RTX 4060 Ti 端到端真实单图推理延迟实测。
严格遵守零泄漏原则：直接加载 Step 2 冻结保存的 cggar_seed{0,1,2}.pt 与 cggar_scidb_3seed_ablation_and_gate.json 中的源 val 阈值！
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
from faster_coco_eval import COCO, COCOeval_faster

ROOT = Path(r'D:\新版柑橘')
EXT = ROOT / '方法路线扩展_20261031'
sys.path.append(str(EXT))
from 共同评测后端 import configure_strict_fp32

WORK_DIR = ROOT / 'Antigravity_CGGAR_真创新攻关_20261125'
sys.path.append(str(WORK_DIR))
from step1_extract_multigranular_cache import YoloFullStateExtractor, DinoExtractor, crop_square_scale, compute_box_geom
from step2_train_and_eval_cggar import (
    TargetContextRelationalHead, Stage1GainRiskGate, CounterfactualCrossGranularityArbitrator,
    compute_maha_z, build_u1_features, build_meta_features, entropy_norm, apply_classwise_nms
)

CKPT_DIR = WORK_DIR / 'models'
RES_DIR = WORK_DIR / 'results'
AUDIT_INDEX_JSON = Path(r'D:\病虫害论文\新增公开数据_20260916\中文分类与标签使用索引.json')
ZENODO_COCO_DIR = Path(r'D:\病虫害论文\新增公开数据_20260916\Zenodo_11148332\第二包解压\orange_detection_COCO')
IMG_ROOT = Path(r'D:\病虫害论文\复现实验记录\yolo11n_public_compare_20261004\数据副本\images')


def build_external_zenodo_gt():
    records = json.loads(AUDIT_INDEX_JSON.read_text(encoding='utf-8'))
    coco_recs = [r for r in records if r.get('数据集') == 'zenodo11148332_coco' and r.get('索引去重保留') and not r.get('同库跨标签冲突')]
    raw_to_2cls = {1: 0, 0: 1, 2: 1}
    raw_to_3cls = {1: 0, 0: 1, 2: 2}

    coco_images, anns_2cls, anns_3cls = [], [], []
    img_paths = {}
    ann_id, img_id = 1, 1
    for r in coco_recs:
        ip = Path(r['路径'])
        lp = ZENODO_COCO_DIR / 'labels' / ip.parent.name / (ip.stem + '.txt')
        if not lp.exists():
            continue
        bgr = cv2.imdecode(np.fromfile(str(ip), dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        b_in = []
        for ln in lp.read_text(encoding='utf-8').splitlines():
            t = ln.split()
            if len(t) >= 5:
                rc = int(float(t[0]))
                cx, cy, bw, bh = map(float, t[1:5])
                x1, y1 = max(0.0, (cx - bw * 0.5) * w), max(0.0, (cy - bh * 0.5) * h)
                pw, ph = min(float(w) - x1, bw * w), min(float(h) - y1, bh * h)
                if pw > 2.0 and ph > 2.0 and rc in raw_to_3cls:
                    b_in.append((rc, [x1, y1, pw, ph]))
        if not b_in:
            continue
        coco_images.append({'id': img_id, 'file_name': ip.name, 'width': w, 'height': h})
        img_paths[img_id] = ip
        for rc, bb in b_in:
            anns_2cls.append({'id': ann_id, 'image_id': img_id, 'category_id': raw_to_2cls[rc], 'bbox': bb, 'area': bb[2] * bb[3], 'iscrowd': 0})
            anns_3cls.append({'id': ann_id, 'image_id': img_id, 'category_id': raw_to_3cls[rc], 'bbox': bb, 'area': bb[2] * bb[3], 'iscrowd': 0})
            ann_id += 1
        img_id += 1

    gt2_p = WORK_DIR / 'cache' / 'zenodo_2cls_gt.json'
    gt3_p = WORK_DIR / 'cache' / 'zenodo_3cls_gt.json'
    gt2_p.write_text(json.dumps({'images': coco_images, 'annotations': anns_2cls, 'categories': [{'id': 0, 'name': 'Healthy'}, {'id': 1, 'name': 'Diseased'}]}), encoding='utf-8')
    gt3_p.write_text(json.dumps({'images': coco_images, 'annotations': anns_3cls, 'categories': [{'id': 0, 'name': 'Healthy'}, {'id': 1, 'name': 'Mottling'}, {'id': 2, 'name': 'Deficiency'}]}), encoding='utf-8')
    return coco_images, img_paths, gt2_p, gt3_p


@torch.no_grad()
def evaluate_external_blind_3seeds(coco_images, img_paths, gt2_p, gt3_p, step2_json):
    device = 'cuda:0'
    scidb_to_2cls = {0: 0, 1: 0, **{c: 1 for c in range(2, 14)}}
    scidb_to_3cls = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2}

    per_seed_ext = []
    for seed in [0, 1, 2]:
        s2_info = next(r for r in step2_json['逐种子明细'] if r['seed'] == seed)
        tau_cggar = s2_info['tau_cggar_on_val']
        beta = s2_info['selected_beta_on_val']

        blob = torch.load(CKPT_DIR / f'cggar_seed{seed}.pt', map_location='cpu', weights_only=False)
        rel_head = TargetContextRelationalHead().to(device).eval(); rel_head.load_state_dict(blob['rel_head'])
        gate = Stage1GainRiskGate(29).to(device).eval(); gate.load_state_dict(blob['gate'])
        arb = CounterfactualCrossGranularityArbitrator(24).to(device).eval(); arb.load_state_dict(blob['arb'])
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

        maha_14 = torch.from_numpy(compute_maha_z(zneck_arr, m_cfg['means'], m_cfg['prec'], m_cfg['m_mu'], m_cfg['m_sd'])).to(device)
        lp_det = F.log_softmax(torch.logit(p14_gpu.clamp(1e-5, 1.0 - 1e-5)) / T_DET, dim=-1)
        lp_std = F.log_softmax(lgs_gpu / T_DINO, dim=-1)
        lp_core = F.log_softmax(lgc_gpu / T_DINO, dim=-1)
        lp_rel = F.log_softmax(lgr_gpu / T_REL, dim=-1)
        p_det, p_std, p_core, p_rel = lp_det.exp(), lp_std.exp(), lp_core.exp(), lp_rel.exp()

        u1 = build_u1_features(p14_gpu, p_det, maha_14, geom_gpu)
        r_logit, gain_pred = gate(u1)
        trig_score = (torch.sigmoid(r_logit) + 0.5 * F.softplus(gain_pred)).squeeze(-1).cpu().numpy()

        esc_cond = ((p_det.argmax(-1) != p_std.argmax(-1)) | (entropy_norm(p_std).squeeze(-1) > 0.16)).float().unsqueeze(1)
        meta_esc = build_meta_features(p14_gpu, p_det, p_std, p_core, p_rel, zc_gpu, zs_gpu, zx_gpu, maha_14, geom_gpu, esc_cond)
        lp_arb, _ = arb(meta_esc, esc_cond, lp_det, lp_std, lp_core, lp_rel)
        p_arb = lp_arb.exp().cpu().numpy()
        p_std_raw = F.softmax(lgs_gpu, dim=-1).cpu().numpy()

        yolo_sc = np.array([r['score'] for r in raw_preds], dtype=np.float64)
        yolo_cls = np.array([r['category_id'] for r in raw_preds], dtype=np.int64)
        p_b3 = 0.65 * p_std_raw.copy()
        p_b3[np.arange(n_c), yolo_cls] += 0.35 * yolo_sc

        def eval_ext_cfg(mask, prob_mat, calib: bool, use_nms: bool):
            rows_14 = [dict(r) for r in raw_preds]
            for i in range(n_c):
                if mask[i]:
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
            out_m['trig_pct'] = round(float((mask & hi).sum() / max(1, hi.sum())) * 100.0, 2)
            return out_m

        m_none = np.zeros(n_c, dtype=bool)
        m_all = np.ones(n_c, dtype=bool)
        m_cg = trig_score > tau_cggar

        sr = {
            'seed': seed,
            'B0_YOLOv8n_Native': eval_ext_cfg(m_none, p_std_raw, False, False),
            'B1_Full_DINOv2_Hard': eval_ext_cfg(m_all, p_std_raw, False, False),
            'B3_Static_Posterior_Fuse': eval_ext_cfg(m_all, p_b3, False, False),
            'Ours_CGGAR_Selective': eval_ext_cfg(m_cg, p_arb, True, True),
        }
        print(f"  [外部果园盲测 Seed {seed}] 2cls AP50: 原生={sr['B0_YOLOv8n_Native']['det_2cls']['AP50']:.2f}% | "
              f"Full-DINO={sr['B1_Full_DINOv2_Hard']['det_2cls']['AP50']:.2f}% | "
              f"静态融合={sr['B3_Static_Posterior_Fuse']['det_2cls']['AP50']:.2f}% | "
              f"★Ours(CG-GAR)={sr['Ours_CGGAR_Selective']['det_2cls']['AP50']:.2f}% || "
              f"3cls AP50: Full-DINO={sr['B1_Full_DINOv2_Hard']['det_3cls']['AP50']:.2f}% -> ★Ours={sr['Ours_CGGAR_Selective']['det_3cls']['AP50']:.2f}%",
              flush=True)
        per_seed_ext.append(sr)
    return per_seed_ext


@torch.no_grad()
def benchmark_cggar_latency(step2_json):
    device = 'cuda:0'
    s0 = next(r for r in step2_json['逐种子明细'] if r['seed'] == 0)
    tau_cggar = s0['tau_cggar_on_val']
    blob = torch.load(CKPT_DIR / 'cggar_seed0.pt', map_location='cpu', weights_only=False)
    rel_head = TargetContextRelationalHead().to(device).eval(); rel_head.load_state_dict(blob['rel_head'])
    gate = Stage1GainRiskGate(29).to(device).eval(); gate.load_state_dict(blob['gate'])
    arb = CounterfactualCrossGranularityArbitrator(24).to(device).eval(); arb.load_state_dict(blob['arb'])
    m_cfg = blob['mahalanobis']
    T_DET, T_DINO, T_REL = blob['temps']['T_DET'], blob['temps']['T_DINO'], blob['temps']['T_REL']

    yolo_ext = YoloFullStateExtractor(Path(r'D:\citrus_runs\yolov8n_unified_seed0\weights\best.pt'), device)
    dino_ext = DinoExtractor(0, device)

    val_imgs = sorted((IMG_ROOT / 'val').glob('*.jpg'))[:35]
    tc_imgs = sorted((IMG_ROOT / 'testC').glob('*.*'))[:35]

    def run_lat(paths: list, mode: str):
        times = []
        for idx, p in enumerate(paths):
            torch.cuda.synchronize()
            ts = time.perf_counter()
            bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), 1)
            res = yolo_ext.yolo.predict([bgr], imgsz=640, conf=0.05, iou=0.7, max_det=300, device=0, half=False, rect=False, verbose=False)[0]
            xyxy = res.boxes.xyxy.cpu().numpy()
            cl = res.boxes.cls.cpu().numpy().astype(np.int64)
            bxs = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in xyxy]

            if mode == 'full_dino' and bxs:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                cs = np.stack([crop_square_scale(rgb, bb, 1.25) for bb in bxs], 0)
                _ = dino_ext.encode_crops(cs, 64)
            elif mode == 'cggar' and bxs:
                h, w = bgr.shape[:2]
                p14, zn = yolo_ext.extract_batch([bgr], [bxs], [cl.tolist()])
                geoms = np.stack([compute_box_geom(bb, w, h, bxs) for bb in bxs], 0)
                p14_g = torch.from_numpy(p14).to(device)
                gm_g = torch.from_numpy(geoms).to(device)
                m14_g = torch.from_numpy(compute_maha_z(zn, m_cfg['means'], m_cfg['prec'], m_cfg['m_mu'], m_cfg['m_sd'])).to(device)
                lp_d = F.log_softmax(torch.logit(p14_g.clamp(1e-5, 1.0 - 1e-5)) / T_DET, dim=-1)
                pd = lp_d.exp()
                rl, gp = gate(build_u1_features(p14_g, pd, m14_g, gm_g))
                ts_arr = (torch.sigmoid(rl) + 0.5 * F.softplus(gp)).squeeze(-1).cpu().numpy()
                t_ids = np.where(ts_arr > tau_cggar)[0]
                if len(t_ids) > 0:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    c_std = np.stack([crop_square_scale(rgb, bxs[i], 1.25) for i in t_ids], 0)
                    zs, lgs = dino_ext.encode_crops(c_std, 64)
                    zs_g = torch.from_numpy(zs.astype(np.float32)).to(device)
                    lgs_g = torch.from_numpy(lgs).to(device)
                    lp_s = F.log_softmax(lgs_g / T_DINO, dim=-1)
                    ps = lp_s.exp()
                    esc = ((pd[t_ids].argmax(-1) != ps.argmax(-1)) | (entropy_norm(ps).squeeze(-1) > 0.16))
                    e_sub = np.where(esc.cpu().numpy())[0]
                    if len(e_sub) > 0:
                        c_cr = np.stack([crop_square_scale(rgb, bxs[t_ids[j]], 0.82) for j in e_sub], 0)
                        c_cx = np.stack([crop_square_scale(rgb, bxs[t_ids[j]], 2.50) for j in e_sub], 0)
                        zcr, lgcr = dino_ext.encode_crops(np.concatenate([c_cr, c_cx], 0), 64)
                        k = len(e_sub)
                        zc_g = torch.from_numpy(zcr[:k].astype(np.float32)).to(device)
                        zx_g = torch.from_numpy(zcr[k:].astype(np.float32)).to(device)
                        _ = rel_head(zc_g, zs_g[e_sub], zx_g, gm_g[t_ids[e_sub]])

            torch.cuda.synchronize()
            ms = (time.perf_counter() - ts) * 1000.0
            if idx >= 5:
                times.append(ms)
        return {'mean_ms': round(float(np.mean(times)), 2), 'p95_ms': round(float(np.percentile(times, 95)), 2)}

    _ = run_lat(val_imgs[:8], 'cggar')
    lat = {
        'native_yolov8n': {'val': run_lat(val_imgs, 'native'), 'testC': run_lat(tc_imgs, 'native')},
        'full_dinov2': {'val': run_lat(val_imgs, 'full_dino'), 'testC': run_lat(tc_imgs, 'full_dino')},
        'ours_cggar': {'val': run_lat(val_imgs, 'cggar'), 'testC': run_lat(tc_imgs, 'cggar')},
    }
    yolo_ext.close()
    return lat


def main():
    t0 = time.monotonic()
    configure_strict_fp32()
    s2 = json.loads((RES_DIR / 'cggar_scidb_3seed_ablation_and_gate.json').read_text(encoding='utf-8'))

    print("[1/2] 执行外部真实果园 519 图 / 1,765 框一次性零样本封口盲测...", flush=True)
    coco_imgs, img_paths, gt2_p, gt3_p = build_external_zenodo_gt()
    ext_seed_res = evaluate_external_blind_3seeds(coco_imgs, img_paths, gt2_p, gt3_p, s2)

    ext_agg = {}
    for m in ['B0_YOLOv8n_Native', 'B1_Full_DINOv2_Hard', 'B3_Static_Posterior_Fuse', 'Ours_CGGAR_Selective']:
        ap2 = [r[m]['det_2cls']['AP50'] for r in ext_seed_res]
        ap2_95 = [r[m]['det_2cls']['AP50_95'] for r in ext_seed_res]
        ap3 = [r[m]['det_3cls']['AP50'] for r in ext_seed_res]
        ap3_95 = [r[m]['det_3cls']['AP50_95'] for r in ext_seed_res]
        trs = [r[m]['trig_pct'] for r in ext_seed_res]
        ext_agg[m] = {
            'det_2cls_AP50': f"{np.mean(ap2):.2f}±{np.std(ap2, ddof=1):.2f}",
            'det_2cls_AP50_seeds': ap2,
            'det_2cls_AP50_95': f"{np.mean(ap2_95):.2f}±{np.std(ap2_95, ddof=1):.2f}",
            'det_3cls_AP50': f"{np.mean(ap3):.2f}±{np.std(ap3, ddof=1):.2f}",
            'det_3cls_AP50_seeds': ap3,
            'det_3cls_AP50_95': f"{np.mean(ap3_95):.2f}±{np.std(ap3_95, ddof=1):.2f}",
            'trig_pct_mean': round(float(np.mean(trs)), 2),
        }

    print("\n[2/2] 测量 RTX 4060 Ti 端到端单图真实延迟 (ms)...", flush=True)
    lat = benchmark_cggar_latency(s2)
    print("  实测延迟:", json.dumps(lat, ensure_ascii=False), flush=True)

    out_p = RES_DIR / 'cggar_external_blind_and_latency.json'
    out_p.write_text(json.dumps({
        '实验名称': 'CG-GAR_外部真实果园封口盲测与RTX4060Ti单图延迟',
        '耗时秒': round(time.monotonic() - t0, 2),
        '实测单图延迟_ms': lat,
        '外部果园盲测汇总_3种子': ext_agg,
        '外部果园逐种子明细': ext_seed_res,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n[完成] 已保存至: {out_p}", flush=True)


if __name__ == '__main__':
    main()
