# -*- coding: utf-8 -*-
"""
Probe-A1: 颈部特征流形马氏距离驱动的风险校准选择性级联检测验证
严格遵守项目评测规范：
- 同进程严格 FP32（allow_tf32=False, cudnn.benchmark=False）
- 马氏距离均值与协方差仅由源 train 拟合，门控阈值仅由源 val 分位点确定（严禁接触 testB/testC 调参）
- 完整覆盖 YOLOv8n 强基线 3 个随机种子 (seed 0, 1, 2)
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
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.ops import roi_align
import timm
from ultralytics import YOLO
from faster_coco_eval import COCO, COCOeval_faster
from sklearn.metrics import roc_auc_score

ROOT = Path(r'D:\新版柑橘')
EXT = ROOT / '方法路线扩展_20261031'
sys.path.append(str(EXT))
from 共同评测后端 import configure_strict_fp32

OUT_DIR = ROOT / '多路线创新筛选_20261124' / '探针A_选择性路由与证据拒识' / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_ROOT = Path(r'D:\病虫害论文\复现实验记录\yolo11n_public_compare_20261004\数据副本\images')
GT_DIR = EXT / '检测数据'
STRONG_DIR = EXT / 'YOLOv8强基线重复_20261109'


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coco_eval_stats(gt_path: Path, pred_rows: list) -> list:
    if not pred_rows:
        return [0.0] * 12
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_path))
        dt = gt.loadRes(pred_rows)
        ev = COCOeval_faster(gt, dt, 'bbox')
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev.stats.tolist()


def letterbox_640(bgr: np.ndarray):
    """与 Ultralytics 默认方形 letterbox(640, auto=False, scaleFill=False, scaleup=True) 一致的几何变换。"""
    h0, w0 = bgr.shape[:2]
    r = min(640.0 / h0, 640.0 / w0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    dw, dh = (640 - new_w) / 2.0, (640 - new_h) / 2.0
    if (w0, h0) != (new_w, new_h):
        resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = bgr
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    assert padded.shape[:2] == (640, 640), f"Unexpected padded shape: {padded.shape}"
    return padded, r, left, top


class YoloNeckFeatureExtractor:
    """提取 YOLOv8 Detect 头 cv3 各尺度末层线性分类器前的特征图，并对给定框提取多尺度对齐特征向量。"""
    def __init__(self, weight_path: Path, device: str = 'cuda:0'):
        self.device = device
        self.yolo = YOLO(str(weight_path))
        self.net = self.yolo.model.to(device).eval()
        self.detect = self.net.model[-1]
        self.penultimate_maps = [None, None, None]
        self.hooks = []
        for s_idx in range(3):
            # cv3[s_idx] = Sequential(Conv, Conv, nn.Conv2d(c3, nc, 1))
            target_module = self.detect.cv3[s_idx][1]
            self.hooks.append(target_module.register_forward_hook(self._make_hook(s_idx)))

    def _make_hook(self, s_idx: int):
        def _hook(_module, _inp, out):
            self.penultimate_maps[s_idx] = out
        return _hook

    def close(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    @torch.no_grad()
    def extract_boxes_features(self, img_records: list, boxes_per_img: list, batch_size: int = 32):
        """
        img_records: list of (img_id, Path)
        boxes_per_img: list of list of [x1, y1, x2, y2] in original image coordinates
        Returns: list of np.ndarray shape (len_boxes, D) aligned with boxes_per_img
        """
        out_feats = [None] * len(img_records)
        for start in range(0, len(img_records), batch_size):
            sub_imgs = img_records[start:start + batch_size]
            sub_boxes = boxes_per_img[start:start + batch_size]
            tensors = []
            transforms_meta = []
            for _, p in sub_imgs:
                bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
                assert bgr is not None, f"Failed to decode {p}"
                padded, r, pad_x, pad_y = letterbox_640(bgr)
                rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
                tensors.append(torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0)
                transforms_meta.append((r, pad_x, pad_y))

            batch_x = torch.stack(tensors, dim=0).to(self.device)
            self.penultimate_maps = [None, None, None]
            _ = self.net(batch_x)
            assert all(m is not None for m in self.penultimate_maps), "Hook failed to capture cv3 features"

            # Build ROI tensor [batch_idx, x1_640, y1_640, x2_640, y2_640]
            rois = []
            counts = []
            for b_i, (b_list, (r, pad_x, pad_y)) in enumerate(zip(sub_boxes, transforms_meta)):
                counts.append(len(b_list))
                for (x1, y1, x2, y2) in b_list:
                    lx1 = float(np.clip(x1 * r + pad_x, 0.0, 639.0))
                    ly1 = float(np.clip(y1 * r + pad_y, 0.0, 639.0))
                    lx2 = float(np.clip(x2 * r + pad_x, lx1 + 1.0, 640.0))
                    ly2 = float(np.clip(y2 * r + pad_y, ly1 + 1.0, 640.0))
                    rois.append([float(b_i), lx1, ly1, lx2, ly2])

            if not rois:
                for b_i in range(len(sub_imgs)):
                    out_feats[start + b_i] = np.zeros((0, 192), dtype=np.float32)
                continue

            roi_tensor = torch.tensor(rois, dtype=torch.float32, device=self.device)
            scale_pooled = []
            strides = [8.0, 16.0, 32.0]
            for s_idx, fmap in enumerate(self.penultimate_maps):
                # 1x1 ROIAlign over box region at each FPN level -> (TotalBoxes, 64)
                pooled = roi_align(fmap.float(), roi_tensor, output_size=(1, 1),
                                   spatial_scale=1.0 / strides[s_idx], aligned=True).squeeze(-1).squeeze(-1)
                scale_pooled.append(F.normalize(pooled, dim=1))
            concat_feat = F.normalize(torch.cat(scale_pooled, dim=1), dim=1).cpu().numpy()

            cursor = 0
            for b_i, cnt in enumerate(counts):
                out_feats[start + b_i] = concat_feat[cursor:cursor + cnt]
                cursor += cnt
        return out_feats


def fit_class_mahalanobis(feats: np.ndarray, labels: np.ndarray, num_classes: int = 14, reg: float = 1e-2):
    """在源训练集框特征上拟合类条件高斯均值与正则化共享精度矩阵（零目标域泄漏）。"""
    d = feats.shape[1]
    means = np.zeros((num_classes, d), dtype=np.float64)
    centered_list = []
    for c in range(num_classes):
        mask = (labels == c)
        assert mask.sum() > 0, f"Class {c} empty in train"
        Xc = feats[mask].astype(np.float64)
        means[c] = Xc.mean(axis=0)
        centered_list.append(Xc - means[c])
    centered = np.concatenate(centered_list, axis=0)
    cov = (centered.T @ centered) / max(1, len(centered) - 1)
    cov_reg = cov + reg * np.eye(d, dtype=np.float64)
    precision = np.linalg.inv(cov_reg)
    return means, precision


def mahalanobis_scores(feats: np.ndarray, pred_cls: np.ndarray, means: np.ndarray, precision: np.ndarray):
    """计算每个候选框相对其预测类别原型的马氏距离，以及相对最近任意类原型的最小马氏距离。"""
    if len(feats) == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    X = feats.astype(np.float64)
    diff_pred = X - means[pred_cls]
    d_pred = np.sqrt(np.maximum(0.0, np.sum((diff_pred @ precision) * diff_pred, axis=1)))

    # Minimum distance across all 14 class prototypes (pure OOD manifold distance)
    all_d = []
    for c in range(means.shape[0]):
        dc = X - means[c]
        all_d.append(np.sqrt(np.maximum(0.0, np.sum((dc @ precision) * dc, axis=1))))
    d_min = np.min(np.stack(all_d, axis=1), axis=1)
    return d_pred, d_min


def load_train_gt_boxes():
    """读取统一检测数据 train.txt 的 7136 张图及其 YOLO GT 框，用于拟合源域颈部特征分布。"""
    train_txt = (EXT / '统一检测数据' / 'train.txt').read_text(encoding='utf-8').splitlines()
    img_records = []
    boxes_per_img = []
    labels_per_img = []
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
        b_list, c_list = [], []
        if lbl_p.exists():
            for row in lbl_p.read_text(encoding='utf-8').splitlines():
                toks = row.split()
                if len(toks) >= 5:
                    cls_id = int(float(toks[0]))
                    cx, cy, bw, bh = map(float, toks[1:5])
                    x1 = (cx - bw / 2.0) * w
                    y1 = (cy - bh / 2.0) * h
                    x2 = (cx + bw / 2.0) * w
                    y2 = (cy + bh / 2.0) * h
                    if x2 > x1 + 1 and y2 > y1 + 1:
                        b_list.append([x1, y1, x2, y2])
                        c_list.append(cls_id)
        if b_list:
            img_records.append((idx, img_p))
            boxes_per_img.append(b_list)
            labels_per_img.append(c_list)
    return img_records, boxes_per_img, labels_per_img


def load_dinov2_teacher(device='cuda:0'):
    meta = json.loads((ROOT / '方法推进_20261027' / '纯CLS对照' / '微调汇总.json').read_text(encoding='utf-8'))
    teacher_info = next(r for r in meta['重复结果'] if r['方法'] == 'LP-FT' and r['种子'] == 0)
    w_path = Path(teacher_info['权重文件'])
    state = torch.load(w_path, map_location='cpu', weights_only=True)
    enc = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False, img_size=224, num_classes=0).to(device).eval()
    enc.load_state_dict(state['骨干'])
    head = torch.nn.Linear(384, 14).to(device).eval()
    head.load_state_dict(state['分类头'])
    return enc, head, sha256_file(w_path)


@torch.no_grad()
def classify_crops_dinov2(enc, head, split: str, coco_images: list, candidate_rows: list, device='cuda:0'):
    """对候选预测框（score >= 0.05）提取与前期基线完全一致的 1.25x 方形灰底裁图并计算 DINOv2 预测类别与概率。"""
    tf = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
    ])
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)

    id2img = {im['id']: im for im in coco_images}
    folder = IMG_ROOT / split

    # Group by image_id to decode each image once
    grouped_indices = defaultdict(list)
    for idx, r in enumerate(candidate_rows):
        grouped_indices[r['image_id']].append(idx)

    crop_tensors = [None] * len(candidate_rows)
    valid_mask = np.zeros(len(candidate_rows), dtype=bool)

    for iid, idxs in grouped_indices.items():
        im_meta = id2img[iid]
        bgr = cv2.imdecode(np.fromfile(str(folder / im_meta['file_name']), dtype=np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        for idx in idxs:
            x, y, bw, bh = candidate_rows[idx]['bbox']
            left, top = max(0, math.floor(x)), max(0, math.floor(y))
            right, bottom = min(w, math.ceil(x + bw)), min(h, math.ceil(y + bh))
            if right > left and bottom > top:
                crop = Image.fromarray(rgb[top:bottom, left:right])
                side = math.ceil(max(crop.size) * 1.25)
                sq = Image.new('RGB', (side, side), (128, 128, 128))
                sq.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
                crop_tensors[idx] = np.asarray(tf(sq)).transpose(2, 0, 1)
                valid_mask[idx] = True

    valid_idxs = np.where(valid_mask)[0]
    pred_cls = np.array([r['category_id'] for r in candidate_rows], dtype=np.int64)
    pred_conf = np.array([r['score'] for r in candidate_rows], dtype=np.float64)

    for start in range(0, len(valid_idxs), 64):
        batch_ids = valid_idxs[start:start + 64]
        arr = np.stack([crop_tensors[i] for i in batch_ids], axis=0)
        x = (torch.from_numpy(arr).to(device).float() / 255.0 - mean) / std
        z = F.normalize(enc.forward_features(x)[:, 0], dim=1)
        probs = head(z).softmax(dim=1)
        conf, cl = probs.max(dim=1)
        pred_cls[batch_ids] = cl.cpu().numpy()
        pred_conf[batch_ids] = conf.cpu().numpy()

    return pred_cls, pred_conf, valid_mask


def match_box_error_labels(coco_gt: dict, candidate_rows: list):
    """在评测诊断中标记候选框是否与某 GT IoU>=0.5 定位命中但原类别判错（仅用于计算门控检出错框 AUROC，绝不参与路由决策）。"""
    gt_by_img = defaultdict(list)
    for ann in coco_gt['annotations']:
        gt_by_img[ann['image_id']].append(ann)

    is_matched = np.zeros(len(candidate_rows), dtype=bool)
    is_cls_wrong = np.zeros(len(candidate_rows), dtype=bool)
    for i, r in enumerate(candidate_rows):
        gts = gt_by_img[r['image_id']]
        if not gts:
            continue
        rx1, ry1, rw, rh = r['bbox']
        rx2, ry2 = rx1 + rw, ry1 + rh
        best_iou, best_cat = 0.0, -1
        for g in gts:
            gx1, gy1, gw, gh = g['bbox']
            gx2, gy2 = gx1 + gw, gy1 + gh
            iw = max(0.0, min(rx2, gx2) - max(rx1, gx1))
            ih = max(0.0, min(ry2, gy2) - max(ry1, gy1))
            inter = iw * ih
            union = rw * rh + gw * gh - inter
            iou_v = inter / union if union > 0 else 0.0
            if iou_v > best_iou:
                best_iou = iou_v
                best_cat = g['category_id']
        if best_iou >= 0.5:
            is_matched[i] = True
            is_cls_wrong[i] = (r['category_id'] != best_cat)
    return is_matched, is_cls_wrong


def main():
    t0 = time.monotonic()
    backend = configure_strict_fp32()
    print("Strict FP32 Backend:", backend, flush=True)

    print("[1/4] 读取源训练集 7136 张图真实框列表...", flush=True)
    train_imgs, train_boxes, train_labels_nested = load_train_gt_boxes()
    train_labels = np.array([c for sub in train_labels_nested for c in sub], dtype=np.int64)
    print(f"  源训练有框图数: {len(train_imgs)}, 总框数: {len(train_labels)}", flush=True)

    print("[2/4] 加载固定 DINOv2 裁图分类教师 (seed0)...", flush=True)
    enc, head, teacher_sha = load_dinov2_teacher('cuda:0')

    # Pre-cache DINOv2 crop classifications per (seed, split) for candidate boxes (score >= 0.05)
    SCORE_CAND_MIN = 0.05
    QUANTILES = [0.85, 0.80]  # 仅在源 val 候选框分布上取分位点阈值（即源 val 仅触发约 15% 或 20%）

    all_seed_results = []

    for seed in [0, 1, 2]:
        print(f"\n================ 处理 YOLOv8n 强基线 Seed {seed} ================", flush=True)
        w_path = Path(r'D:\citrus_runs') / f'yolov8n_unified_seed{seed}' / 'weights' / 'best.pt'
        extractor = YoloNeckFeatureExtractor(w_path, 'cuda:0')

        # 1. Extract train neck features & fit Mahalanobis manifold
        t_feat = time.monotonic()
        train_feat_list = extractor.extract_boxes_features(train_imgs, train_boxes, batch_size=64)
        train_feats = np.concatenate(train_feat_list, axis=0)
        assert len(train_feats) == len(train_labels)
        means, precision = fit_class_mahalanobis(train_feats, train_labels, num_classes=14, reg=1e-2)
        print(f"  Seed {seed} 源 train 颈部特征流形拟合完成 ({time.monotonic() - t_feat:.1f}s), 特征维度={train_feats.shape[1]}", flush=True)

        # 2. Extract neck features & crop predictions for val, testB, testC
        split_data = {}
        for split in ['val', 'testB', 'testC']:
            gt_path = GT_DIR / f'{split}.json'
            coco_gt = json.loads(gt_path.read_text(encoding='utf-8'))
            pred_path = STRONG_DIR / f'种子{seed}' / f'{split}预测.json'
            orig_preds = json.loads(pred_path.read_text(encoding='utf-8'))

            cand_idxs = [i for i, r in enumerate(orig_preds) if r['score'] >= SCORE_CAND_MIN]
            cand_rows = [orig_preds[i] for i in cand_idxs]

            # Group candidate boxes per image for neck feature extraction
            id2img = {im['id']: im for im in coco_gt['images']}
            grouped_pos = defaultdict(list)
            for pos, r in enumerate(cand_rows):
                grouped_pos[r['image_id']].append(pos)

            img_recs, b_per_img, pos_order = [], [], []
            for iid, pos_list in grouped_pos.items():
                img_recs.append((iid, IMG_ROOT / split / id2img[iid]['file_name']))
                b_per_img.append([
                    [cand_rows[p]['bbox'][0], cand_rows[p]['bbox'][1],
                     cand_rows[p]['bbox'][0] + cand_rows[p]['bbox'][2],
                     cand_rows[p]['bbox'][1] + cand_rows[p]['bbox'][3]]
                    for p in pos_list
                ])
                pos_order.extend(pos_list)

            feats_per_img = extractor.extract_boxes_features(img_recs, b_per_img, batch_size=32)
            flat_feats = np.concatenate(feats_per_img, axis=0) if feats_per_img else np.zeros((0, 192), dtype=np.float32)
            cand_feats = np.zeros_like(flat_feats)
            for k, orig_pos in enumerate(pos_order):
                cand_feats[orig_pos] = flat_feats[k]

            yolo_cls = np.array([r['category_id'] for r in cand_rows], dtype=np.int64)
            yolo_sc = np.array([r['score'] for r in cand_rows], dtype=np.float64)
            d_pred, d_min = mahalanobis_scores(cand_feats, yolo_cls, means, precision)

            # Binary uncertainty baseline: 1 - yolo_score
            low_conf_score = 1.0 - yolo_sc

            # Crop teacher classification
            crop_cls, crop_conf, valid_crop = classify_crops_dinov2(enc, head, split, coco_gt['images'], cand_rows, 'cuda:0')
            is_matched, is_cls_wrong = match_box_error_labels(coco_gt, cand_rows)

            split_data[split] = {
                'gt_path': gt_path,
                'orig_preds': orig_preds,
                'cand_idxs': cand_idxs,
                'yolo_cls': yolo_cls,
                'yolo_sc': yolo_sc,
                'd_pred': d_pred,
                'd_min': d_min,
                'low_conf_score': low_conf_score,
                'crop_cls': crop_cls,
                'crop_conf': crop_conf,
                'valid_crop': valid_crop,
                'is_matched': is_matched,
                'is_cls_wrong': is_cls_wrong,
            }

        extractor.close()
        del extractor
        torch.cuda.empty_cache()

        # 3. Calibrate thresholds STRICTLY on source val!
        val_info = split_data['val']
        # Per-class or global quantile on val matched/high-score distribution (strictly unsupervised on val predictions, no val GT used!)
        val_high_mask = val_info['yolo_sc'] >= 0.25
        if val_high_mask.sum() < 10:
            val_high_mask = np.ones(len(val_info['yolo_sc']), dtype=bool)

        seed_summary = {'seed': seed, 'splits': {}, 'gate_diagnostics': {}}

        # Diagnostic AUROC: separating (val matched correct) vs (testB/C matched wrong) using d_pred vs (1 - yolo_sc)
        val_correct_mask = val_info['is_matched'] & (~val_info['is_cls_wrong'])
        for t_split in ['testB', 'testC']:
            ti = split_data[t_split]
            t_wrong_mask = ti['is_matched'] & ti['is_cls_wrong']
            y_diag = np.concatenate([np.zeros(val_correct_mask.sum()), np.ones(t_wrong_mask.sum())])
            auroc_maha = float(roc_auc_score(y_diag, np.concatenate([val_info['d_pred'][val_correct_mask], ti['d_pred'][t_wrong_mask]])))
            auroc_conf = float(roc_auc_score(y_diag, np.concatenate([val_info['low_conf_score'][val_correct_mask], ti['low_conf_score'][t_wrong_mask]])))
            # Within-split error detection AUROC among matched boxes with score >= 0.25
            m_sub = ti['is_matched'] & (ti['yolo_sc'] >= 0.25)
            auroc_within_maha = float(roc_auc_score(ti['is_cls_wrong'][m_sub], ti['d_pred'][m_sub])) if len(np.unique(ti['is_cls_wrong'][m_sub])) > 1 else 0.5
            auroc_within_conf = float(roc_auc_score(ti['is_cls_wrong'][m_sub], ti['low_conf_score'][m_sub])) if len(np.unique(ti['is_cls_wrong'][m_sub])) > 1 else 0.5
            seed_summary['gate_diagnostics'][t_split] = {
                'cross_domain_error_auroc_mahalanobis': auroc_maha,
                'cross_domain_error_auroc_low_conf': auroc_conf,
                'within_split_highconf_error_auroc_mahalanobis': auroc_within_maha,
                'within_split_highconf_error_auroc_low_conf': auroc_within_conf,
            }
            print(f"  [{t_split} 错框识别 AUROC] 流形马氏距离={auroc_maha:.4f} (域内高分错框={auroc_within_maha:.4f}) vs 低置信度={auroc_conf:.4f} (域内={auroc_within_conf:.4f})", flush=True)

        # Evaluate detection AP across schemes
        tau_maha_q85 = float(np.quantile(val_info['d_pred'][val_high_mask], 0.85))
        tau_maha_q80 = float(np.quantile(val_info['d_pred'][val_high_mask], 0.80))
        tau_conf_q85 = float(np.quantile(val_info['low_conf_score'][val_high_mask], 0.85))
        tau_conf_q80 = float(np.quantile(val_info['low_conf_score'][val_high_mask], 0.80))

        for split in ['val', 'testB', 'testC']:
            sd = split_data[split]
            orig = sd['orig_preds']
            c_idxs = sd['cand_idxs']

            def build_routed_preds(trigger_mask: np.ndarray, calib_score: bool = False):
                rows = [dict(r) for r in orig]
                triggered_cnt = 0
                for local_i, global_i in enumerate(c_idxs):
                    if trigger_mask[local_i] and sd['valid_crop'][local_i]:
                        triggered_cnt += 1
                        new_c = int(sd['crop_cls'][local_i])
                        old_c = rows[global_i]['category_id']
                        rows[global_i]['category_id'] = new_c
                        if calib_score and new_c != old_c:
                            # 仅当重分类纠正了被背景抑制的类别时，采用几何均值校准分数
                            rows[global_i]['score'] = float(math.sqrt(rows[global_i]['score'] * sd['crop_conf'][local_i]))
                trig_rate_high = float((trigger_mask & (sd['yolo_sc'] >= 0.25)).sum() / max(1, (sd['yolo_sc'] >= 0.25).sum()))
                trig_rate_all = float(triggered_cnt / max(1, len(c_idxs)))
                stats = coco_eval_stats(sd['gt_path'], rows)
                return {
                    'AP50_95': round(stats[0] * 100.0, 4),
                    'AP50': round(stats[1] * 100.0, 4),
                    'triggered_boxes': triggered_cnt,
                    'total_cand_boxes': len(c_idxs),
                    'trigger_rate_cand_pct': round(trig_rate_all * 100.0, 2),
                    'trigger_rate_conf25_pct': round(trig_rate_high * 100.0, 2),
                }

            n_cand = len(c_idxs)
            res_split = {
                'B0_原生单阶段': build_routed_preds(np.zeros(n_cand, dtype=bool), False),
                'B1_全量重分类_保原分': build_routed_preds(np.ones(n_cand, dtype=bool), False),
                'B2_全量重分类_几何校准分': build_routed_preds(np.ones(n_cand, dtype=bool), True),
                'B3_低置信度门控_q85': build_routed_preds(sd['low_conf_score'] > tau_conf_q85, False),
                'B3_低置信度门控_q80': build_routed_preds(sd['low_conf_score'] > tau_conf_q80, False),
                'Ours_流形马氏门控_q85_保原分': build_routed_preds(sd['d_pred'] > tau_maha_q85, False),
                'Ours_流形马氏门控_q80_保原分': build_routed_preds(sd['d_pred'] > tau_maha_q80, False),
                'Ours_流形马氏门控_q80_几何校准': build_routed_preds(sd['d_pred'] > tau_maha_q80, True),
            }
            seed_summary['splits'][split] = res_split
            print(f"  [{split}] 原生={res_split['B0_原生单阶段']['AP50_95']:.2f}% | "
                  f"全量重分类={res_split['B1_全量重分类_保原分']['AP50_95']:.2f}% | "
                  f"低置信q80={res_split['B3_低置信度门控_q80']['AP50_95']:.2f}% (触发{res_split['B3_低置信度门控_q80']['trigger_rate_conf25_pct']:.1f}%) | "
                  f"马氏q80={res_split['Ours_流形马氏门控_q80_保原分']['AP50_95']:.2f}% / 校准={res_split['Ours_流形马氏门控_q80_几何校准']['AP50_95']:.2f}% "
                  f"(高分框自适应触发={res_split['Ours_流形马氏门控_q80_保原分']['trigger_rate_conf25_pct']:.1f}%, 全候选触发={res_split['Ours_流形马氏门控_q80_保原分']['trigger_rate_cand_pct']:.1f}%)",
                  flush=True)

        all_seed_results.append(seed_summary)

    # Aggregate 3-seed mean ± std
    methods = list(all_seed_results[0]['splits']['val'].keys())
    agg = {}
    for m_name in methods:
        agg[m_name] = {}
        for split in ['val', 'testB', 'testC']:
            aps = [s['splits'][split][m_name]['AP50_95'] for s in all_seed_results]
            ap50s = [s['splits'][split][m_name]['AP50'] for s in all_seed_results]
            trigs = [s['splits'][split][m_name]['trigger_rate_conf25_pct'] for s in all_seed_results]
            trigs_all = [s['splits'][split][m_name]['trigger_rate_cand_pct'] for s in all_seed_results]
            agg[m_name][split] = {
                'AP50_95_mean': round(float(np.mean(aps)), 4),
                'AP50_95_std': round(float(np.std(aps, ddof=1)), 4),
                'AP50_95_seeds': aps,
                'AP50_mean': round(float(np.mean(ap50s)), 4),
                'trigger_conf25_mean_pct': round(float(np.mean(trigs)), 2),
                'trigger_cand_mean_pct': round(float(np.mean(trigs_all)), 2),
            }

    final_out = {
        '实验名称': 'Probe-A1_颈部特征流形马氏距离选择性级联',
        '耗时秒': round(time.monotonic() - t0, 2),
        '严格后端': backend,
        '教师权重SHA256': teacher_sha,
        '候选框最低分门限': SCORE_CAND_MIN,
        '阈值标定说明': '所有门控阈值仅在源val预测框(score>=0.25)分布按0.85/0.80分位数无监督标定，零testB/testC信息泄漏',
        '三种子汇总': agg,
        '逐种子明细': all_seed_results,
    }
    out_json = OUT_DIR / 'probe_a1_selective_routing_summary.json'
    out_json.write_text(json.dumps(final_out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n[完成] Probe-A1 结果已保存至: {out_json}", flush=True)


if __name__ == '__main__':
    main()
