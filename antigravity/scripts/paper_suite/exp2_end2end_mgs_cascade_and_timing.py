# -*- coding: utf-8 -*-
"""
实验 E2: 端到端完整检测 3×3 配对随机种子评测 (YOLOv8n seed s × 分类器 seed s) 与真实单图推理延迟测定
严格同进程 FP32 方形 640 COCO AP50-95 / AP50 口径；所有门控阈值仅在源 val 预测分布按 q=0.80 无监督标定。
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
import contextlib
import io
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.ops import roi_align
import timm
from ultralytics import YOLO
from faster_coco_eval import COCO, COCOeval_faster

ROOT = Path(r'D:\新版柑橘')
EXT = ROOT / '方法路线扩展_20261031'
sys.path.append(str(EXT))
from 共同评测后端 import configure_strict_fp32

EXP_DIR = ROOT / '多路线创新筛选_20261124' / '小论文完整实验_20261124'
sys.path.append(str(EXP_DIR))
from exp1_symtex_ablation_and_kd import DeployedPrunedStudent

CKPT_DIR = EXP_DIR / 'checkpoints'
RES_DIR = EXP_DIR / 'results'
IMG_ROOT = Path(r'D:\病虫害论文\复现实验记录\yolo11n_public_compare_20261004\数据副本\images')
GT_DIR = EXT / '检测数据'
STRONG_DIR = EXT / 'YOLOv8强基线重复_20261109'


def coco_eval_stats(gt_path: Path, pred_rows: list):
    if not pred_rows:
        return [0.0] * 12, [-1.0] * 14
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_path))
        dt = gt.loadRes(pred_rows)
        ev = COCOeval_faster(gt, dt, 'bbox')
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        # Extract per-class AP50-95
        prec = ev.eval['precision']  # [T, R, K, A, M]
        per_cls = []
        for k in range(14):
            v = prec[:, :, k, 0, -1]
            v = v[v > -1]
            per_cls.append(round(float(v.mean() * 100.0), 2) if v.size else 0.0)
    return ev.stats.tolist(), per_cls


def letterbox_640(bgr: np.ndarray):
    h0, w0 = bgr.shape[:2]
    r = min(640.0 / h0, 640.0 / w0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    dw, dh = (640 - nw) / 2.0, (640 - nh) / 2.0
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR) if (w0, h0) != (nw, nh) else bgr
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, r, left, top


class YoloNeckExtractor:
    def __init__(self, weight_path: Path, device='cuda:0'):
        self.device = device
        self.yolo = YOLO(str(weight_path))
        self.net = self.yolo.model.to(device).eval()
        self.detect = self.net.model[-1]
        self.maps = [None, None, None]
        self.hooks = [self.detect.cv3[s][1].register_forward_hook(self._hk(s)) for s in range(3)]

    def _hk(self, s):
        def fn(_m, _i, out):
            self.maps[s] = out
        return fn

    def close(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    @torch.no_grad()
    def extract_boxes(self, img_records: list, boxes_per_img: list, batch_size: int = 32):
        out_feats = [None] * len(img_records)
        strides = [8.0, 16.0, 32.0]
        for start in range(0, len(img_records), batch_size):
            sub_imgs = img_records[start:start + batch_size]
            sub_boxes = boxes_per_img[start:start + batch_size]
            tensors, metas = [], []
            for _, p in sub_imgs:
                bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
                padded, r, px, py = letterbox_640(bgr)
                rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
                tensors.append(torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0)
                metas.append((r, px, py))

            bx = torch.stack(tensors, dim=0).to(self.device)
            self.maps = [None, None, None]
            _ = self.net(bx)

            rois, counts = [], []
            for bi, (blist, (r, px, py)) in enumerate(zip(sub_boxes, metas)):
                counts.append(len(blist))
                for (x1, y1, x2, y2) in blist:
                    lx1 = float(np.clip(x1 * r + px, 0.0, 639.0))
                    ly1 = float(np.clip(y1 * r + py, 0.0, 639.0))
                    lx2 = float(np.clip(x2 * r + px, lx1 + 1.0, 640.0))
                    ly2 = float(np.clip(y2 * r + py, ly1 + 1.0, 640.0))
                    rois.append([float(bi), lx1, ly1, lx2, ly2])

            if not rois:
                for bi in range(len(sub_imgs)):
                    out_feats[start + bi] = np.zeros((0, 192), dtype=np.float32)
                continue

            rt = torch.tensor(rois, dtype=torch.float32, device=self.device)
            sp = [F.normalize(roi_align(self.maps[s].float(), rt, (1, 1), 1.0 / strides[s], aligned=True).flatten(1), dim=1) for s in range(3)]
            cat = F.normalize(torch.cat(sp, dim=1), dim=1).cpu().numpy()
            cur = 0
            for bi, c in enumerate(counts):
                out_feats[start + bi] = cat[cur:cur + c]
                cur += c
        return out_feats


def load_train_gt_boxes():
    train_txt = (EXT / '统一检测数据' / 'train.txt').read_text(encoding='utf-8').splitlines()
    img_records, boxes_per_img, labels = [], [], []
    for idx, line in enumerate(train_txt):
        p_str = line.strip()
        if not p_str:
            continue
        img_p = Path(p_str)
        parts = list(img_p.parts)
        parts[parts.index('images')] = 'labels'
        lbl_p = Path(*parts).with_suffix('.txt')
        bgr = cv2.imdecode(np.fromfile(str(img_p), dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = bgr.shape[:2]
        bl = []
        if lbl_p.exists():
            for row in lbl_p.read_text(encoding='utf-8').splitlines():
                t = row.split()
                if len(t) >= 5:
                    cid = int(float(t[0]))
                    cx, cy, bw, bh = map(float, t[1:5])
                    x1, y1, x2, y2 = (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h
                    if x2 > x1 + 1 and y2 > y1 + 1:
                        bl.append([x1, y1, x2, y2])
                        labels.append(cid)
        if bl:
            img_records.append((idx, img_p))
            boxes_per_img.append(bl)
    return img_records, boxes_per_img, np.array(labels, dtype=np.int64)


def fit_mahalanobis(feats: np.ndarray, labels: np.ndarray, reg: float = 1e-2):
    d = feats.shape[1]
    means = np.zeros((14, d), dtype=np.float64)
    centered = []
    for c in range(14):
        Xc = feats[labels == c].astype(np.float64)
        means[c] = Xc.mean(axis=0)
        centered.append(Xc - means[c])
    cov = (np.concatenate(centered, 0).T @ np.concatenate(centered, 0)) / max(1, len(feats) - 1)
    prec = np.linalg.inv(cov + reg * np.eye(d))
    return means, prec


def maha_dist(feats: np.ndarray, pred_cls: np.ndarray, means: np.ndarray, prec: np.ndarray):
    if len(feats) == 0:
        return np.zeros((0,), dtype=np.float64)
    diff = feats.astype(np.float64) - means[pred_cls]
    return np.sqrt(np.maximum(0.0, np.sum((diff @ prec) * diff, axis=1)))


def extract_crop_array(split: str, coco_images: list, cand_rows: list):
    tf = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
    ])
    id2img = {im['id']: im for im in coco_images}
    folder = IMG_ROOT / split
    grouped = defaultdict(list)
    for idx, r in enumerate(cand_rows):
        grouped[r['image_id']].append(idx)

    crops = np.zeros((len(cand_rows), 3, 224, 224), dtype=np.uint8)
    valid = np.zeros(len(cand_rows), dtype=bool)
    for iid, idxs in grouped.items():
        bgr = cv2.imdecode(np.fromfile(str(folder / id2img[iid]['file_name']), dtype=np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        for idx in idxs:
            x, y, bw, bh = cand_rows[idx]['bbox']
            l, t = max(0, math.floor(x)), max(0, math.floor(y))
            r, b = min(w, math.ceil(x + bw)), min(h, math.ceil(y + bh))
            if r > l and b > t:
                c = Image.fromarray(rgb[t:b, l:r])
                side = math.ceil(max(c.size) * 1.25)
                sq = Image.new('RGB', (side, side), (128, 128, 128))
                sq.paste(c, ((side - c.width) // 2, (side - c.height) // 2))
                crops[idx] = np.asarray(tf(sq)).transpose(2, 0, 1)
                valid[idx] = True
    return crops, valid


@torch.no_grad()
def predict_pruned_student(ckpt_path: Path, crops_u8: np.ndarray, valid: np.ndarray, device='cuda:0'):
    st = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    m = DeployedPrunedStudent(14).load_pruned(st['pruned_state']).to(device).eval()
    cls_out = np.zeros(len(crops_u8), dtype=np.int64)
    conf_out = np.zeros(len(crops_u8), dtype=np.float64)
    v_ids = np.where(valid)[0]
    for i in range(0, len(v_ids), 128):
        b = v_ids[i:i + 128]
        x = torch.from_numpy(crops_u8[b]).to(device).float() / 255.0
        probs = m(x)[0].softmax(dim=1)
        cf, cl = probs.max(dim=1)
        cls_out[b] = cl.cpu().numpy()
        conf_out[b] = cf.cpu().numpy()
    del m
    return cls_out, conf_out


@torch.no_grad()
def predict_dinov2_teacher(seed: int, crops_u8: np.ndarray, valid: np.ndarray, device='cuda:0'):
    meta = json.loads((ROOT / '方法推进_20261027' / '纯CLS对照' / '微调汇总.json').read_text(encoding='utf-8'))
    t_info = next(r for r in meta['重复结果'] if r['方法'] == 'LP-FT' and r['种子'] == seed)
    state = torch.load(t_info['权重文件'], map_location='cpu', weights_only=True)
    enc = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False, img_size=224, num_classes=0).to(device).eval()
    enc.load_state_dict(state['骨干'])
    head = nn.Linear(384, 14).to(device).eval()
    head.load_state_dict(state['分类头'])
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)
    cls_out = np.zeros(len(crops_u8), dtype=np.int64)
    conf_out = np.zeros(len(crops_u8), dtype=np.float64)
    v_ids = np.where(valid)[0]
    for i in range(0, len(v_ids), 64):
        b = v_ids[i:i + 64]
        x = (torch.from_numpy(crops_u8[b]).to(device).float() / 255.0 - mean) / std
        z = F.normalize(enc.forward_features(x)[:, 0], dim=1)
        probs = head(z).softmax(dim=1)
        cf, cl = probs.max(dim=1)
        cls_out[b] = cl.cpu().numpy()
        conf_out[b] = cf.cpu().numpy()
    del enc, head
    return cls_out, conf_out


@torch.no_grad()
def benchmark_real_single_image_latency(means: np.ndarray, prec: np.ndarray, tau_maha: float):
    """在 RTX 4060 Ti 上实测单图端到端推理延迟（含磁盘读图解码、YOLOv8n 前向、马氏距离计算、按真实比例触发裁图分类与同步计时）。"""
    device = 'cuda:0'
    extractor = YoloNeckExtractor(Path(r'D:\citrus_runs\yolov8n_unified_seed0\weights\best.pt'), device)
    st = torch.load(CKPT_DIR / 'Ours_SymTex_KD_Pruned_seed0.pt', map_location='cpu', weights_only=True)
    light_cls = DeployedPrunedStudent(14).load_pruned(st['pruned_state']).to(device).eval()

    meta = json.loads((ROOT / '方法推进_20261027' / '纯CLS对照' / '微调汇总.json').read_text(encoding='utf-8'))
    t_info = next(r for r in meta['重复结果'] if r['方法'] == 'LP-FT' and r['种子'] == 0)
    t_state = torch.load(t_info['权重文件'], map_location='cpu', weights_only=True)
    dino_enc = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False, img_size=224, num_classes=0).to(device).eval()
    dino_enc.load_state_dict(t_state['骨干'])
    dino_head = nn.Linear(384, 14).to(device).eval()
    dino_head.load_state_dict(t_state['分类头'])

    tf = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(224)])
    mean_t = torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)
    strides = [8.0, 16.0, 32.0]

    # Sample 30 images from val (in-domain) and 30 images from testC (shifted domain)
    val_imgs = sorted((IMG_ROOT / 'val').glob('*.jpg'))[:30]
    tc_imgs = sorted((IMG_ROOT / 'testC').glob('*.*'))[:30]

    def run_mode(img_paths: list, mode: str):
        times = []
        for idx, p in enumerate(img_paths):
            torch.cuda.synchronize()
            t_s = time.perf_counter()
            bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
            extractor.maps = [None, None, None]
            res = extractor.yolo.predict([bgr], imgsz=640, conf=0.05, iou=0.7, max_det=300, device=0, half=False, rect=False, verbose=False)[0]
            boxes_xyxy = res.boxes.xyxy.cpu().numpy()
            cls_np = res.boxes.cls.cpu().numpy().astype(np.int64)

            if mode != 'native' and len(boxes_xyxy) > 0:
                if mode.startswith('mgs_'):
                    _, r, px, py = letterbox_640(bgr)
                    rois = []
                    for (x1, y1, x2, y2) in boxes_xyxy:
                        lx1, ly1 = float(np.clip(x1 * r + px, 0, 639)), float(np.clip(y1 * r + py, 0, 639))
                        lx2, ly2 = float(np.clip(x2 * r + px, lx1 + 1, 640)), float(np.clip(y2 * r + py, ly1 + 1, 640))
                        rois.append([0.0, lx1, ly1, lx2, ly2])
                    rt = torch.tensor(rois, dtype=torch.float32, device=device)
                    sp = [F.normalize(roi_align(extractor.maps[s].float(), rt, (1, 1), 1.0 / strides[s], aligned=True).flatten(1), dim=1) for s in range(3)]
                    z_neck = F.normalize(torch.cat(sp, dim=1), dim=1).cpu().numpy()
                    dm = maha_dist(z_neck, cls_np, means, prec)
                    trig_ids = np.where(dm > tau_maha)[0]
                else:
                    trig_ids = np.arange(len(boxes_xyxy))

                if len(trig_ids) > 0:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    h, w = rgb.shape[:2]
                    crop_list = []
                    for ti in trig_ids:
                        x1, y1, x2, y2 = boxes_xyxy[ti]
                        l, t, rr, bb = max(0, math.floor(x1)), max(0, math.floor(y1)), min(w, math.ceil(x2)), min(h, math.ceil(y2))
                        if rr > l and bb > t:
                            c = Image.fromarray(rgb[t:bb, l:rr])
                            side = math.ceil(max(c.size) * 1.25)
                            sq = Image.new('RGB', (side, side), (128, 128, 128))
                            sq.paste(c, ((side - c.width) // 2, (side - c.height) // 2))
                            crop_list.append(np.asarray(tf(sq)).transpose(2, 0, 1))
                    if crop_list:
                        cx = torch.from_numpy(np.stack(crop_list, 0)).to(device).float() / 255.0
                        if 'light' in mode:
                            _ = light_cls(cx)
                        else:
                            zn = F.normalize(dino_enc.forward_features((cx - mean_t) / std_t)[:, 0], dim=1)
                            _ = dino_head(zn)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t_s) * 1000.0
            if idx >= 5:  # exclude initial 5 warmup frames
                times.append(elapsed_ms)
        return {'mean_ms': round(float(np.mean(times)), 2), 'p95_ms': round(float(np.percentile(times, 95)), 2)}

    # Warmup GPU
    _ = run_mode(val_imgs[:8], 'mgs_pro')
    timings = {}
    for m_key in ['native', 'full_light', 'mgs_light', 'full_pro', 'mgs_pro']:
        timings[m_key] = {
            'in_domain_val': run_mode(val_imgs, m_key),
            'shifted_testC': run_mode(tc_imgs, m_key),
        }
    extractor.close()
    del extractor, light_cls, dino_enc, dino_head
    torch.cuda.empty_cache()
    return timings


def main():
    t0 = time.monotonic()
    backend = configure_strict_fp32()
    train_imgs, train_boxes, train_labels = load_train_gt_boxes()

    seed_results = []
    seed0_manifold = None

    for seed in [0, 1, 2]:
        print(f"\n================ 端到端配对评测 Seed {seed} ================", flush=True)
        w_yolo = Path(r'D:\citrus_runs') / f'yolov8n_unified_seed{seed}' / 'weights' / 'best.pt'
        ext = YoloNeckExtractor(w_yolo, 'cuda:0')
        tr_feats = np.concatenate(ext.extract_boxes(train_imgs, train_boxes, batch_size=64), axis=0)
        means, prec = fit_mahalanobis(tr_feats, train_labels, reg=1e-2)

        splits_cache = {}
        for sp in ['val', 'testB', 'testC']:
            gt_path = GT_DIR / f'{sp}.json'
            coco_gt = json.loads(gt_path.read_text(encoding='utf-8'))
            orig_preds = json.loads((STRONG_DIR / f'种子{seed}' / f'{sp}预测.json').read_text(encoding='utf-8'))
            cand_idxs = [i for i, r in enumerate(orig_preds) if r['score'] >= 0.05]
            cand_rows = [orig_preds[i] for i in cand_idxs]

            id2img = {im['id']: im for im in coco_gt['images']}
            g_pos = defaultdict(list)
            for pos, r in enumerate(cand_rows):
                g_pos[r['image_id']].append(pos)
            img_recs, b_per_img, pos_order = [], [], []
            for iid, plist in g_pos.items():
                img_recs.append((iid, IMG_ROOT / sp / id2img[iid]['file_name']))
                b_per_img.append([[cand_rows[p]['bbox'][0], cand_rows[p]['bbox'][1],
                                   cand_rows[p]['bbox'][0] + cand_rows[p]['bbox'][2],
                                   cand_rows[p]['bbox'][1] + cand_rows[p]['bbox'][3]] for p in plist])
                pos_order.extend(plist)

            f_list = ext.extract_boxes(img_recs, b_per_img, batch_size=32)
            flat_f = np.concatenate(f_list, 0) if f_list else np.zeros((0, 192), dtype=np.float32)
            cand_f = np.zeros_like(flat_f)
            for k, op in enumerate(pos_order):
                cand_f[op] = flat_f[k]

            yolo_cls = np.array([r['category_id'] for r in cand_rows], dtype=np.int64)
            yolo_sc = np.array([r['score'] for r in cand_rows], dtype=np.float64)
            d_maha = maha_dist(cand_f, yolo_cls, means, prec)
            crops_u8, valid_c = extract_crop_array(sp, coco_gt['images'], cand_rows)

            # Classify crops with paired seed classifiers:
            b1_cls, b1_cf = predict_pruned_student(CKPT_DIR / f'B1_Vanilla_Pruned_seed{seed}.pt', crops_u8, valid_c)
            vkd_cls, vkd_cf = predict_pruned_student(CKPT_DIR / f'Vanilla_KD_seed{seed}.pt', crops_u8, valid_c)
            sym_cls, sym_cf = predict_pruned_student(CKPT_DIR / f'SymTex_Pruned_seed{seed}.pt', crops_u8, valid_c)
            skd_cls, skd_cf = predict_pruned_student(CKPT_DIR / f'Ours_SymTex_KD_Pruned_seed{seed}.pt', crops_u8, valid_c)
            dino_cls, dino_cf = predict_dinov2_teacher(seed, crops_u8, valid_c)

            splits_cache[sp] = {
                'gt_path': gt_path,
                'orig_preds': orig_preds,
                'cand_idxs': cand_idxs,
                'yolo_sc': yolo_sc,
                'd_maha': d_maha,
                'low_conf': 1.0 - yolo_sc,
                'valid_c': valid_c,
                'preds_map': {
                    'b1': (b1_cls, b1_cf),
                    'vkd': (vkd_cls, vkd_cf),
                    'sym': (sym_cls, sym_cf),
                    'skd': (skd_cls, skd_cf),
                    'dino': (dino_cls, dino_cf),
                }
            }
        ext.close()
        del ext
        torch.cuda.empty_cache()

        val_hi = splits_cache['val']['yolo_sc'] >= 0.25
        tau_maha_80 = float(np.quantile(splits_cache['val']['d_maha'][val_hi], 0.80))
        tau_conf_80 = float(np.quantile(splits_cache['val']['low_conf'][val_hi], 0.80))
        if seed == 0:
            seed0_manifold = (means, prec, tau_maha_80)

        s_rep = {'seed': seed, 'tau_maha_80': tau_maha_80, 'splits': {}}
        for sp in ['val', 'testB', 'testC']:
            sc = splits_cache[sp]
            n = len(sc['cand_idxs'])

            def eval_cfg(mask: np.ndarray, cls_key: str, calib: bool = True):
                rows = [dict(r) for r in sc['orig_preds']]
                c_arr, cf_arr = sc['preds_map'][cls_key] if cls_key else (None, None)
                trig = 0
                for li, gi in enumerate(sc['cand_idxs']):
                    if mask[li] and sc['valid_c'][li]:
                        trig += 1
                        nc = int(c_arr[li])
                        oc = rows[gi]['category_id']
                        rows[gi]['category_id'] = nc
                        if calib and nc != oc:
                            rows[gi]['score'] = float(math.sqrt(rows[gi]['score'] * cf_arr[li]))
                stats, per_cls = coco_eval_stats(sc['gt_path'], rows)
                hi_m = sc['yolo_sc'] >= 0.25
                trig_hi = float((mask & hi_m).sum() / max(1, hi_m.sum())) * 100.0
                return {
                    'AP50_95': round(stats[0] * 100.0, 4),
                    'AP50': round(stats[1] * 100.0, 4),
                    'trig_conf25_pct': round(trig_hi, 2),
                    'per_cls_AP': per_cls,
                }

            m_none = np.zeros(n, dtype=bool)
            m_all = np.ones(n, dtype=bool)
            m_conf = sc['low_conf'] > tau_conf_80
            m_maha = sc['d_maha'] > tau_maha_80

            s_rep['splits'][sp] = {
                'S0_YOLOv8n_Native': eval_cfg(m_none, ''),
                'S1_Full_Vanilla_MBV3': eval_cfg(m_all, 'b1', True),
                'S1b_LowConf_Vanilla_MBV3': eval_cfg(m_conf, 'b1', True),
                'S1c_MGS_Vanilla_MBV3': eval_cfg(m_maha, 'b1', True),
                'S2a_Full_SymTex_NoKD': eval_cfg(m_all, 'sym', True),
                'S2b_LowConf_SymTex_NoKD': eval_cfg(m_conf, 'sym', True),
                'Ours_Light_MGS_SymTex_NoKD': eval_cfg(m_maha, 'sym', True),
                'S2c_Full_Vanilla_KD': eval_cfg(m_all, 'vkd', True),
                'S2d_LowConf_Vanilla_KD': eval_cfg(m_conf, 'vkd', True),
                'Ours_Light_MGS_VanillaKD': eval_cfg(m_maha, 'vkd', True),
                'S2_Full_SymTex_KD': eval_cfg(m_all, 'skd', True),
                'S3_LowConf_SymTex_KD': eval_cfg(m_conf, 'skd', True),
                'Ours_Light_MGS_SymTex_KD': eval_cfg(m_maha, 'skd', True),
                'S4_Full_DINOv2': eval_cfg(m_all, 'dino', True),
                'S4_LowConf_DINOv2': eval_cfg(m_conf, 'dino', True),
                'Ours_Pro_MGS_DINOv2': eval_cfg(m_maha, 'dino', True),
            }
            r = s_rep['splits'][sp]
            print(f"  [{sp:5s}] 原生={r['S0_YOLOv8n_Native']['AP50_95']:.2f}% | "
                  f"MGS+B1={r['S1c_MGS_Vanilla_MBV3']['AP50_95']:.2f}% | "
                  f"MGS+SymTex(无教师)={r['Ours_Light_MGS_SymTex_NoKD']['AP50_95']:.2f}% | "
                  f"MGS+VanillaKD={r['Ours_Light_MGS_VanillaKD']['AP50_95']:.2f}% | "
                  f"MGS+SymTexKD={r['Ours_Light_MGS_SymTex_KD']['AP50_95']:.2f}% | "
                  f"★Ours-Pro(DINO)={r['Ours_Pro_MGS_DINOv2']['AP50_95']:.2f}%", flush=True)
        seed_results.append(s_rep)

    print("\n================ 测量 RTX 4060 Ti 真实单图端到端延迟 (ms) ================", flush=True)
    timings = benchmark_real_single_image_latency(*seed0_manifold)
    print("  耗时结果:", json.dumps(timings, ensure_ascii=False), flush=True)

    # Aggregate 3-seed summary
    sys_names = list(seed_results[0]['splits']['val'].keys())
    agg = {}
    for sn in sys_names:
        agg[sn] = {}
        for sp in ['val', 'testB', 'testC']:
            aps = [s['splits'][sp][sn]['AP50_95'] for s in seed_results]
            ap50s = [s['splits'][sp][sn]['AP50'] for s in seed_results]
            trigs = [s['splits'][sp][sn]['trig_conf25_pct'] for s in seed_results]
            agg[sn][sp] = {
                'AP50_95_str': f"{np.mean(aps):.2f}±{np.std(aps, ddof=1):.2f}",
                'AP50_95_mean': round(float(np.mean(aps)), 4),
                'AP50_95_std': round(float(np.std(aps, ddof=1)), 4),
                'AP50_str': f"{np.mean(ap50s):.2f}±{np.std(ap50s, ddof=1):.2f}",
                'trig_pct_mean': round(float(np.mean(trigs)), 2),
                'seeds': aps,
            }

    out_p = RES_DIR / 'e2_end2end_mgs_symtex_summary.json'
    out_p.write_text(json.dumps({
        '实验名称': 'E2_端到端MGS选择性级联与SymTex合体3x3配对种子评测及实测延迟',
        '总耗时秒': round(time.monotonic() - t0, 2),
        '严格后端': backend,
        '实测单图延迟_ms': timings,
        '三种子汇总': agg,
        '逐种子明细': seed_results,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n[完成] 实验 E2 已保存至: {out_p}", flush=True)


if __name__ == '__main__':
    main()
