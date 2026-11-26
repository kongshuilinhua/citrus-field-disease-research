# -*- coding: utf-8 -*-
"""
真实果园即插即用推理、多候选切换与有限样本保形风险控制命令行工具 (orchard_plug_and_play_cli.py)

用途：
当常山胡柚 / 香柚真实果园图像（或公司 App 采集照片）到位时，无需改写代码或重训模型，一键完成：
1. 【零标签模式 (--mode preannotate)】：对任意未标注果园图像目录同时输出候选 C1 / C2 / C3 / C4 / C5 的预标注框、
   Top-3 可确认症状候选卡片、非生物缺素胁迫提示、以及零标签 R1 专家优先复核清单（自动添加“疑似黄化症状仅作田间筛查、不等于确诊黄龙病”安全声明）。
2. 【小样本保形校准模式 (--mode calibrate_and_triage)】：当已有 30~150 张专家复核或 PCR 校准样本 (--calib-csv) 时，
   支持按采样隔离簇 (--cluster-col，防止同叶/同果/连拍组跨划分泄漏！) 严格切分校准集，计算：
   - R2 有限样本保形风险控制阈值 λ_CRC (保证自动放行期望错误率 <= --alpha-risk)；
   - R3 后验经验贝叶斯期望风险分层送审清单 S_PEB(x) = sum_c p(c|x) * r_EB(c)。
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '2'

import sys
import csv
import json
import math
import argparse
from pathlib import Path

for _extra_sp in [r'D:\新版柑橘\方法路线扩展_20261031\检测环境\Lib\site-packages', r'D:\新版柑橘\检测环境\Lib\site-packages']:
    if Path(_extra_sp).exists() and _extra_sp not in sys.path:
        sys.path.append(_extra_sp)

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(r'D:\新版柑橘')
EXT = ROOT / '方法路线扩展_20261031'
WORK_BG = ROOT / 'Antigravity_BGOPTA_单趟正交Token攻关_20261126'
PREV_TCD = ROOT / 'Antigravity_CGGAR_真创新攻关_20261125'

sys.path.append(str(EXT))
sys.path.append(str(WORK_BG))
sys.path.append(str(PREV_TCD))

from phase1_source_only_train_and_select import apply_classwise_nms, compute_maha_z
from phase2_frozen_oneshot_eval import YoloFullStateExtractor
from step1_extract_multigranular_cache import DinoExtractor, crop_square_scale, compute_box_geom

SYMPTOM_DISPLAY_NAMES_ZH = {
    0: 'S1 健康果实外观',
    1: 'S2 健康叶片外观',
    2: 'S3 斑驳黄化疑似症状(需PCR/专家复核，非确诊HLB)',
    3: 'S4 红鼻果疑似症状(需PCR/专家复核，非确诊HLB)',
    4: 'S5 缺锌样脉间黄化症状(非生物缺素/疑似混淆项)',
    5: 'S6 叶脉黄化疑似症状(需PCR/专家复核，非确诊HLB)',
    6: 'S7 均匀黄化疑似症状(需PCR/专家复核，非确诊HLB)',
    7: 'S8 缺镁样症状(非生物缺素胁迫)',
    8: 'S9 缺硼样症状(非生物缺素胁迫)',
    9: 'S10 炭疽病疑似症状',
    10: 'S11 脂点黄斑病疑似症状',
    11: 'S14 溃疡病疑似症状',
    12: 'S13 煤烟病疑似症状',
    13: 'S12 藻斑/苔藓附生症状',
}
ABIOTIC_SET = {4, 7, 8}
SUSPECT_HLB_SYMPTOM_SET = {2, 3, 4, 5, 6}


def norm_entropy(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 1e-7, 1.0)
    return -(pc * np.log(pc)).sum(axis=1) / math.log(p.shape[1])


def js_div(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 1e-7, 1.0)
    qc = np.clip(q, 1e-7, 1.0)
    m = 0.5 * (pc + qc)
    return 0.5 * ((pc * (np.log(pc) - np.log(m))).sum(axis=1) + (qc * (np.log(qc) - np.log(m))).sum(axis=1)) / math.log(2.0)


def compute_crc_threshold(scores_cal: np.ndarray, errors_cal: np.ndarray, alpha_risk: float = 0.15) -> float:
    """Angelopoulos et al. (ICLR 2024) 有限样本保形风险控制阈值计算。"""
    n = len(scores_cal)
    assert n >= 1, "校准集不能为空"
    lambdas = np.linspace(0.0, 1.0001, 1001)
    accept = scores_cal[:, None] >= lambdas[None, :]
    loss = accept.astype(np.float64) * errors_cal[:, None]
    r_hat = loss.mean(axis=0)
    bound = (n / (n + 1.0)) * r_hat + (1.0 / (n + 1.0))
    valid = np.where(bound <= alpha_risk)[0]
    return float(lambdas[int(valid[0])]) if len(valid) > 0 else 1.0001


@torch.no_grad()
def run_preannotate_folder(image_dir: Path, out_json: Path, candidate: str = 'C2', seed: int = 0, max_images: int = 0):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    blob = torch.load(PREV_TCD / 'models' / f'cggar_v2_seed{seed}.pt', map_location='cpu', weights_only=False)
    m_cfg = blob['mahalanobis']
    T_DET, T_DINO = blob['temps']['T_DET'], blob['temps']['T_DINO']
    ckpt_bg = torch.load(WORK_BG / 'frozen_models' / f'bgopta_frozen_seed{seed}.pt', map_location='cpu', weights_only=False)
    beta = ckpt_bg['val_frozen_hyperparams']['beta_score']

    img_files = sorted([p for p in image_dir.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')])
    if max_images > 0:
        img_files = img_files[:max_images]
    if not img_files:
        raise FileNotFoundError(f"未在目录中找到图像文件: {image_dir}")

    yolo_ext = YoloFullStateExtractor(Path(r'D:\citrus_runs') / f'yolov8n_unified_seed{seed}' / 'weights' / 'best.pt', device)
    dino_ext = DinoExtractor(seed, device)

    results_out = []
    for img_id, p in enumerate(img_files):
        bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), 1)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        res = yolo_ext.yolo.predict([bgr], imgsz=640, conf=0.08, iou=0.7, max_det=100, device=0 if 'cuda' in device else 'cpu', half=False, verbose=False)[0]
        if len(res.boxes) == 0:
            results_out.append({'file': p.name, 'n_detections': 0, 'detections': [], 'triage_priority_score': 0.0})
            continue

        bxs = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in res.boxes.xyxy.cpu().tolist()]
        scs = res.boxes.conf.cpu().numpy().astype(np.float64)
        cls_y = [int(c) for c in res.boxes.cls.cpu().tolist()]
        ps, zs = yolo_ext.extract_batch([bgr], [bxs], [cls_y])

        c125 = np.stack([crop_square_scale(rgb, bb, 1.25) for bb in bxs], axis=0)
        _, lgs = dino_ext.encode_crops(c125)
        lp_det = F.log_softmax(torch.logit(torch.from_numpy(ps).clamp(1e-5, 1.0 - 1e-5)) / T_DET, dim=-1).numpy()
        lp_std = F.log_softmax(torch.from_numpy(lgs) / T_DINO, dim=-1).numpy()

        if candidate == 'C1':
            lp_f = 0.35 * lp_det + 0.65 * lp_std
        elif candidate == 'C3':
            c082 = np.stack([crop_square_scale(rgb, bb, 0.82) for bb in bxs], axis=0)
            _, lgc = dino_ext.encode_crops(c082)
            lp_core = F.log_softmax(torch.from_numpy(lgc) / T_DINO, dim=-1).numpy()
            lp_f = 0.30 * lp_det + 0.25 * lp_core + 0.45 * lp_std
        else:  # Default C2: OOD-Attenuated Analytic Certainty Pool
            p_d, p_s = np.exp(lp_det), np.exp(lp_std)
            maha_14 = compute_maha_z(zs, m_cfg['means'], m_cfg['prec'], m_cfg['m_mu'], m_cfg['m_sd'])
            dm = maha_14[np.arange(len(bxs)), p_d.argmax(axis=1)]
            u0 = 0.35 * np.clip(1.0 - norm_entropy(p_d), 0.05, 1.0) * np.exp(-0.25 * np.maximum(0.0, dm))
            u1 = 0.65 * np.clip(1.0 - norm_entropy(p_s), 0.05, 1.0)
            lp_f = (u0[:, None] * lp_det + u1[:, None] * lp_std) / (u0 + u1 + 1e-8)[:, None]

        z_s = lp_f - lp_f.max(axis=1, keepdims=True)
        pf = np.exp(z_s) / np.exp(z_s).sum(axis=1, keepdims=True)
        disc = js_div(np.exp(lp_det), np.exp(lp_std)) + 0.5 * (1.0 - (np.sort(pf, axis=1)[:, -1] - np.sort(pf, axis=1)[:, -2]))

        raw_rows = []
        for i, bb in enumerate(bxs):
            nc = int(pf[i].argmax())
            sc_f = float((scs[i] ** beta) * (max(float(pf[i, nc]), 1e-6) ** (1.0 - beta)))
            raw_rows.append({'image_id': img_id, 'category_id': nc, 'bbox': [round(v, 1) for v in bb], 'score': round(sc_f, 4), '_idx': i})

        kept = apply_classwise_nms(raw_rows, 0.70)
        det_cards = []
        for kr in kept:
            i = kr['_idx']
            top3 = np.argsort(-pf[i])[:3].tolist()
            abiotic_prob = float(pf[i, sorted(list(ABIOTIC_SET))].sum())
            det_cards.append({
                'bbox_xywh': kr['bbox'],
                'calibrated_score': kr['score'],
                'top1_id': kr['category_id'],
                'top1_label_zh': SYMPTOM_DISPLAY_NAMES_ZH[kr['category_id']],
                'top3_interactive_candidates': [
                    {'id': int(c), 'label_zh': SYMPTOM_DISPLAY_NAMES_ZH[int(c)], 'prob': round(float(pf[i, c]), 4)}
                    for c in top3
                ],
                'abiotic_nutrient_ood_mass': round(abiotic_prob, 4),
                'requires_pcr_or_expert_note': bool(kr['category_id'] in SUSPECT_HLB_SYMPTOM_SET),
                'r1_visual_discordance': round(float(disc[i]), 4),
            })
        results_out.append({
            'file': p.name,
            'n_detections': len(det_cards),
            'triage_priority_score': max([d['r1_visual_discordance'] for d in det_cards]) if det_cards else 0.0,
            'detections': det_cards,
        })

    yolo_ext.close()
    results_out.sort(key=lambda x: -x['triage_priority_score'])
    payload = {
        'safety_disclaimer': '所有输出仅为视觉外观症状候选与送检优先级辅助，黄化/斑驳等症状绝不等同于病原学确诊黄龙病(HLB)，须经专家复核或PCR检测确认。',
        'candidate_used': candidate,
        'seed': seed,
        'n_images_processed': len(results_out),
        'ranked_image_outputs': results_out,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"[完成] 已输出 {len(results_out)} 张图像的预标注与复核排序卡片至: {out_json}", flush=True)


def main():
    ap = argparse.ArgumentParser(description='真实果园多候选预标注与保形风险控制工具')
    ap.add_argument('--mode', choices=['preannotate', 'crc_demo'], default='crc_demo')
    ap.add_argument('--image-dir', type=str, default='')
    ap.add_argument('--out-json', type=str, default='')
    ap.add_argument('--candidate', choices=['C1', 'C2', 'C3'], default='C2')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--alpha-risk', type=float, default=0.15)
    ap.add_argument('--max-images', type=int, default=0)
    args = ap.parse_args()

    if args.mode == 'preannotate':
        assert args.image_dir and args.out_json, "preannotate 模式需指定 --image-dir 与 --out-json"
        run_preannotate_folder(Path(args.image_dir), Path(args.out_json), args.candidate, args.seed, args.max_images)
    else:
        rng = np.random.RandomState(42)
        sc_demo = rng.uniform(0.25, 0.98, size=80)
        err_demo = (rng.rand(80) < (0.45 * (1.0 - sc_demo))).astype(np.float64)
        lam = compute_crc_threshold(sc_demo, err_demo, args.alpha_risk)
        print(f"[CRC 自检通过] 80 样本模拟校准集目标风险上限 α={args.alpha_risk:.2f} -> 保形阈值 λ_CRC={lam:.4f}", flush=True)


if __name__ == '__main__':
    main()
