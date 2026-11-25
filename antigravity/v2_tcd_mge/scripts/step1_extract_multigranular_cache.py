# -*- coding: utf-8 -*-
"""
Step 1: 提取 3 随机种子 (0, 1, 2) 锚点级 14 类完整检测概率、颈部特征、几何描述子、
以及三粒度视图 (Core 0.82x / Std 1.25x / Ctx 2.50x) + 源域图像级反事实退化视图特征缓存。
严格红线：
- 反事实退化样本仅在 SciDB train 上合成；
- val / testB / testC 仅提取冻结候选框确定性视图特征，零数据增强、零标签窥探。
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'

import sys
import json
import math
import time
import random
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

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

ROOT = Path(r'D:\新版柑橘')
EXT = ROOT / '方法路线扩展_20261031'
sys.path.append(str(EXT))
from 共同评测后端 import configure_strict_fp32

WORK_DIR = ROOT / 'Antigravity_CGGAR_真创新攻关_20261125'
CACHE_DIR = WORK_DIR / 'cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

IMG_ROOT = Path(r'D:\病虫害论文\复现实验记录\yolo11n_public_compare_20261004\数据副本\images')
GT_DIR = EXT / '检测数据'
STRONG_DIR = EXT / 'YOLOv8强基线重复_20261109'
KD_CKPT_DIR = ROOT / '多路线创新筛选_20261124' / '小论文完整实验_20261124' / 'checkpoints'

TF_224 = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
])


def letterbox_640(bgr: np.ndarray, blur_down: bool = False, gamma: float = 1.0):
    h0, w0 = bgr.shape[:2]
    src = bgr
    if blur_down:
        # 模拟整图小尺度远拍与运动模糊 (诱发 B 类检测器细粒度退化)
        sw, sh = max(32, int(w0 * 0.42)), max(32, int(h0 * 0.42))
        src = cv2.resize(cv2.resize(src, (sw, sh), interpolation=cv2.INTER_AREA), (w0, h0), interpolation=cv2.INTER_LINEAR)
    if abs(gamma - 1.0) > 1e-3:
        lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
        src = cv2.LUT(src, lut)
    r = min(640.0 / h0, 640.0 / w0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    dw, dh = (640 - nw) / 2.0, (640 - nh) / 2.0
    resized = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_LINEAR) if (w0, h0) != (nw, nh) else src
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, r, left, top


def crop_square_scale(rgb: np.ndarray, bbox_xywh: list, scale: float) -> np.ndarray:
    """以 bbox 中心为锚点，按 scale 倍最大边截取方形上下文（大尺寸框先降采样防 10000x10000 巨幅画布），输出 224x224 CHW uint8。"""
    h, w = rgb.shape[:2]
    x, y, bw, bh = bbox_xywh
    # 若原图分辨率极高 (>1280)，先按比例在局部或缩放图上裁取，避免 2.5x 产生近万像素空白画布
    max_dim = max(h, w)
    if max_dim > 1280:
        f = 1280.0 / max_dim
        rgb = cv2.resize(rgb, (max(1, int(round(w * f))), max(1, int(round(h * f)))), interpolation=cv2.INTER_AREA)
        h, w = rgb.shape[:2]
        x, y, bw, bh = x * f, y * f, bw * f, bh * f

    cx, cy = x + bw * 0.5, y + bh * 0.5
    side = max(4, int(math.ceil(max(bw, bh) * scale)))
    half = side / 2.0
    x1, y1 = int(math.floor(cx - half)), int(math.floor(cy - half))
    x2, y2 = x1 + side, y1 + side

    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(w, x2), min(h, y2)
    canvas = np.full((side, side, 3), 128, dtype=np.uint8)
    if sx2 > sx1 and sy2 > sy1:
        dx1, dy1 = sx1 - x1, sy1 - y1
        canvas[dy1:dy1 + (sy2 - sy1), dx1:dx1 + (sx2 - sx1)] = rgb[sy1:sy2, sx1:sx2]
    # 等价于 Resize(256) + CenterCrop(224): 中央 224/256 = 87.5% 视场缩放至 224x224
    margin = int(round(side * (1.0 - 224.0 / 256.0) * 0.5))
    inner = canvas[margin:side - margin, margin:side - margin] if (side - 2 * margin) >= 4 else canvas
    interp = cv2.INTER_AREA if inner.shape[0] > 224 else cv2.INTER_CUBIC
    out224 = cv2.resize(inner, (224, 224), interpolation=interp)
    return out224.transpose(2, 0, 1)


def make_counterfactual_bad_crop(rgb: np.ndarray, bbox_xywh: list, rng: np.random.RandomState) -> np.ndarray:
    """
    仅用于源训练集 SciDB train：合成局部紧裁框的真实物理退化（诱发 C 类：YOLO 整图对，但局部紧框因截断/偏心遮挡/强反光判错）。
    """
    h, w = rgb.shape[:2]
    x, y, bw, bh = bbox_xywh
    mode = rng.randint(0, 3)
    if mode == 0:
        # 边界截断/偏心裁剪：只取叶片一角 (45%~62%)，丢失整叶对称拓扑
        sub_w, sub_h = max(4.0, bw * rng.uniform(0.42, 0.60)), max(4.0, bh * rng.uniform(0.42, 0.60))
        ox = x + (0.0 if rng.rand() < 0.5 else bw - sub_w)
        oy = y + (0.0 if rng.rand() < 0.5 else bh - sub_h)
        arr = crop_square_scale(rgb, [ox, oy, sub_w, sub_h], 1.15)
    elif mode == 1:
        # 强曝光/白板反光与色偏模拟
        arr = crop_square_scale(rgb, bbox_xywh, 1.25).astype(np.float32) / 255.0
        gm = rng.choice([0.38, 0.48, 2.1, 2.5])
        gains = rng.uniform(0.75, 1.30, size=(3, 1, 1)).astype(np.float32)
        arr = np.clip((arr ** gm) * gains, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
    else:
        # 邻叶/手部/阴影大面积遮挡 (遮住紧框 45%~60% 区域)
        arr = crop_square_scale(rgb, bbox_xywh, 1.25).copy()
        occ_h = rng.randint(100, 145)
        occ_w = rng.randint(100, 145)
        y0 = 0 if rng.rand() < 0.5 else 224 - occ_h
        x0 = 0 if rng.rand() < 0.5 else 224 - occ_w
        palette = [[235, 238, 240], [35, 55, 30], [180, 145, 120]]
        fill_col = np.array(palette[rng.randint(0, 3)], dtype=np.uint8)[:, None, None]
        arr[:, y0:y0 + occ_h, x0:x0 + occ_w] = fill_col
    return arr


def compute_box_geom(bbox_xywh: list, img_w: int, img_h: int, other_boxes: list) -> np.ndarray:
    """计算 9 维尺度与边界截断几何描述子 g(b)（零类别泄漏）。"""
    x, y, bw, bh = bbox_xywh
    rw = bw / max(1.0, float(img_w))
    rh = bh / max(1.0, float(img_h))
    r_area = rw * rh
    log_ar = math.log(max(1e-3, bw / max(1.0, bh)))
    dl = max(0.0, x) / max(1.0, float(img_w))
    dt = max(0.0, y) / max(1.0, float(img_h))
    dr = max(0.0, float(img_w) - (x + bw)) / max(1.0, float(img_w))
    db = max(0.0, float(img_h) - (y + bh)) / max(1.0, float(img_h))
    min_border = min(dl, dt, dr, db)
    max_ov = 0.0
    for ob in other_boxes:
        if ob is bbox_xywh:
            continue
        ix1, iy1 = max(x, ob[0]), max(y, ob[1])
        ix2, iy2 = min(x + bw, ob[0] + ob[2]), min(y + bh, ob[1] + ob[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter > 0:
            ov = inter / (bw * bh + ob[2] * ob[3] - inter + 1e-9)
            if ov < 0.999 and ov > max_ov:
                max_ov = ov
    return np.array([rw, rh, r_area, log_ar, dl, dt, min_border, float(min_border < 0.015), max_ov], dtype=np.float32)


class YoloFullStateExtractor:
    """一次前向同时提取 8400 锚点的 14 类原始概率分布 p_det 与 192 维多尺度颈部特征 z_neck。"""
    def __init__(self, weight_path: Path, device='cuda:0'):
        self.device = device
        self.yolo = YOLO(str(weight_path))
        self.net = self.yolo.model.to(device).eval()
        self.detect = self.net.model[-1]
        self.maps = [None, None, None]
        self.raw_out = None
        self.hooks = [self.detect.cv3[s][1].register_forward_hook(self._hk_map(s)) for s in range(3)]
        self.hooks.append(self.detect.register_forward_hook(self._hk_det))

    def _hk_map(self, s):
        def fn(_m, _i, out):
            self.maps[s] = out.detach()
        return fn

    def _hk_det(self, _m, _i, out):
        self.raw_out = out[0].detach()  # (B, 18, 8400)

    def close(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    @torch.no_grad()
    def extract_batch(self, bgr_list: list, boxes_xywh_per_img: list, target_cls_per_img: list = None,
                      blur_down: bool = False, gamma: float = 1.0):
        strides = [8.0, 16.0, 32.0]
        tensors, metas = [], []
        for bgr in bgr_list:
            pad, r, px, py = letterbox_640(bgr, blur_down=blur_down, gamma=gamma)
            rgb = cv2.cvtColor(pad, cv2.COLOR_BGR2RGB)
            tensors.append(torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0)
            metas.append((r, px, py))

        bx = torch.stack(tensors, 0).to(self.device)
        _ = self.net(bx)
        y_all = self.raw_out.cpu().numpy()  # (B, 18, 8400)

        p14_list, rois, counts = [], [], []
        for bi, (boxes, (r, px, py)) in enumerate(zip(boxes_xywh_per_img, metas)):
            counts.append(len(boxes))
            yb = y_all[bi]
            acx, acy, abw, abh = yb[0], yb[1], yb[2], yb[3]
            ax1 = (acx - abw * 0.5 - px) / r
            ay1 = (acy - abh * 0.5 - py) / r
            aw = abw / r
            ah = abh / r
            sc14 = yb[4:]  # (14, 8400)

            t_cls_list = target_cls_per_img[bi] if target_cls_per_img is not None else [None] * len(boxes)
            for (x, y, bw, bh), tc in zip(boxes, t_cls_list):
                if tc is not None and tc >= 0:
                    # Candidate box from official JSON: match exact anchor by coords + predicted class score
                    d = np.abs(ax1 - x) + np.abs(ay1 - y) + np.abs(aw - bw) + np.abs(ah - bh)
                else:
                    # GT box on train: find anchor with highest IoU to GT box!
                    ix1 = np.maximum(ax1, x)
                    iy1 = np.maximum(ay1, y)
                    ix2 = np.minimum(ax1 + aw, x + bw)
                    iy2 = np.minimum(ay1 + ah, y + bh)
                    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
                    iou_v = inter / (aw * ah + bw * bh - inter + 1e-9)
                    # Weight IoU * max_conf slightly so we pick the active positive anchor on that GT
                    d = -(iou_v + 0.05 * sc14.max(axis=0))
                a_idx = int(np.argmin(d))
                p14_list.append(sc14[:, a_idx].astype(np.float32))

                lx1 = float(np.clip(x * r + px, 0.0, 639.0))
                ly1 = float(np.clip(y * r + py, 0.0, 639.0))
                lx2 = float(np.clip((x + bw) * r + px, lx1 + 1.0, 640.0))
                ly2 = float(np.clip((y + bh) * r + py, ly1 + 1.0, 640.0))
                rois.append([float(bi), lx1, ly1, lx2, ly2])

        if not rois:
            return np.zeros((0, 14), dtype=np.float32), np.zeros((0, 192), dtype=np.float32)

        rt = torch.tensor(rois, dtype=torch.float32, device=self.device)
        sp = [F.normalize(roi_align(self.maps[s].float(), rt, (1, 1), 1.0 / strides[s], aligned=True).flatten(1), dim=1) for s in range(3)]
        z_neck = F.normalize(torch.cat(sp, dim=1), dim=1).cpu().numpy().astype(np.float32)
        return np.stack(p14_list, 0), z_neck


class DinoExtractor:
    def __init__(self, seed: int, device='cuda:0'):
        self.device = device
        meta = json.loads((ROOT / '方法推进_20261027' / '纯CLS对照' / '微调汇总.json').read_text(encoding='utf-8'))
        t_info = next(r for r in meta['重复结果'] if r['方法'] == 'LP-FT' and r['种子'] == seed)
        st = torch.load(t_info['权重文件'], map_location='cpu', weights_only=True)
        self.enc = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False, img_size=224, num_classes=0).to(device).eval()
        self.enc.load_state_dict(st['骨干'])
        self.head = nn.Linear(384, 14).to(device).eval()
        self.head.load_state_dict(st['分类头'])
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode_crops(self, crops_u8: np.ndarray, batch_size: int = 256):
        if len(crops_u8) == 0:
            return np.zeros((0, 384), dtype=np.float32), np.zeros((0, 14), dtype=np.float32)
        zs, lgs = [], []
        for i in range(0, len(crops_u8), batch_size):
            x = (torch.from_numpy(crops_u8[i:i + batch_size]).to(self.device).float() / 255.0 - self.mean) / self.std
            z = F.normalize(self.enc.forward_features(x)[:, 0], dim=1)
            lg = self.head(z)
            zs.append(z.cpu().numpy().astype(np.float16))
            lgs.append(lg.cpu().numpy().astype(np.float32))
        return np.concatenate(zs, 0), np.concatenate(lgs, 0)


def load_train_items():
    lines = [l.strip() for l in (EXT / '统一检测数据' / 'train.txt').read_text(encoding='utf-8').splitlines() if l.strip()]
    items = []
    for img_idx, p_str in enumerate(lines):
        ip = Path(p_str)
        parts = list(ip.parts)
        parts[parts.index('images')] = 'labels'
        lp = Path(*parts).with_suffix('.txt')
        if not lp.exists():
            continue
        raw_rows = [r.split() for r in lp.read_text(encoding='utf-8').splitlines() if len(r.split()) >= 5]
        if raw_rows:
            items.append((img_idx, ip, raw_rows))
    return items


def build_train_crops_and_geom(train_items: list):
    """多线程一次性解码 SciDB train 图像，生成 3 粒度标准视图 + 1 个图像级反事实退化视图 + 几何特征。"""
    print(f"[Train 图像解码与多粒度裁图] 共 {len(train_items)} 张源训练图...", flush=True)
    rng_master = np.random.RandomState(20261125)
    seeds_per_img = rng_master.randint(0, 10**9, size=len(train_items))

    def proc_one(args):
        (img_idx, ip, raw_rows), r_seed = args
        r_loc = np.random.RandomState(int(r_seed))
        bgr = cv2.imdecode(np.fromfile(str(ip), dtype=np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        # 提前将超清大图缩至 <=1280 一次供 4 个视图复用，速度提升 30 倍！
        if max(h, w) > 1280:
            f_pre = 1280.0 / max(h, w)
            rgb_fast = cv2.resize(rgb, (max(1, int(round(w * f_pre))), max(1, int(round(h * f_pre)))), interpolation=cv2.INTER_AREA)
        else:
            f_pre = 1.0
            rgb_fast = rgb
        boxes, boxes_fast, labels = [], [], []
        for t in raw_rows:
            cid = int(float(t[0]))
            cx, cy, bw, bh = map(float, t[1:5])
            px, py, pw, ph = (cx - bw * 0.5) * w, (cy - bh * 0.5) * h, bw * w, bh * h
            if pw > 2.0 and ph > 2.0:
                jx = px + r_loc.uniform(-0.025, 0.025) * pw
                jy = py + r_loc.uniform(-0.025, 0.025) * ph
                jw = pw * r_loc.uniform(0.96, 1.04)
                jh = ph * r_loc.uniform(0.96, 1.04)
                boxes.append([jx, jy, jw, jh])
                boxes_fast.append([jx * f_pre, jy * f_pre, jw * f_pre, jh * f_pre])
                labels.append(cid)
        if not boxes:
            return None
        geoms = [compute_box_geom(b, w, h, boxes) for b in boxes]
        c_core = [crop_square_scale(rgb_fast, bf, 0.82) for bf in boxes_fast]
        c_std = [crop_square_scale(rgb_fast, bf, 1.25) for bf in boxes_fast]
        c_ctx = [crop_square_scale(rgb_fast, bf, 2.50) for bf in boxes_fast]
        c_cf = [make_counterfactual_bad_crop(rgb_fast, bf, r_loc) for bf in boxes_fast]
        return img_idx, ip, boxes, labels, geoms, c_core, c_std, c_ctx, c_cf

    img_ids, img_paths, labels_all, geoms_all = [], [], [], []
    core_list, std_list, ctx_list, cf_list = [], [], [], []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for k_done, out in enumerate(ex.map(proc_one, zip(train_items, seeds_per_img)), 1):
            if out is None:
                continue
            img_idx, ip, boxes, labels, geoms, cc, cs, cx, ccf = out
            img_ids.extend([img_idx] * len(boxes))
            img_paths.append((img_idx, ip, boxes))
            labels_all.extend(labels)
            geoms_all.extend(geoms)
            core_list.extend(cc)
            std_list.extend(cs)
            ctx_list.extend(cx)
            cf_list.extend(ccf)
            if k_done % 2000 == 0:
                print(f"  已处理源训练图 {k_done}/{len(train_items)}...", flush=True)

    return {
        'img_paths_boxes': img_paths,
        'box_img_ids': np.array(img_ids, dtype=np.int32),
        'labels': np.array(labels_all, dtype=np.int64),
        'geom': np.stack(geoms_all, 0),
        'crops_core': np.stack(core_list, 0),
        'crops_std': np.stack(std_list, 0),
        'crops_ctx': np.stack(ctx_list, 0),
        'crops_cf_bad': np.stack(cf_list, 0),
    }


def main():
    t0 = time.monotonic()
    configure_strict_fp32()

    train_items = load_train_items()
    tr_pack = build_train_crops_and_geom(train_items)
    print(f"  源训练集有效框总数: {len(tr_pack['labels'])} (耗时 {time.monotonic() - t0:.1f}s)", flush=True)

    for seed in [0, 1, 2]:
        ts = time.monotonic()
        print(f"\n================ 构建 Seed {seed} 多粒度特征缓存 ================", flush=True)
        w_yolo = Path(r'D:\citrus_runs') / f'yolov8n_unified_seed{seed}' / 'weights' / 'best.pt'
        yolo_ext = YoloFullStateExtractor(w_yolo, 'cuda:0')
        dino_ext = DinoExtractor(seed, 'cuda:0')

        # 1. Extract YOLO clean & YOLO degraded (blur+gamma counterfactual Mode B) on SciDB train
        tr_p14_clean, tr_zneck_clean = [], []
        tr_p14_deg, tr_zneck_deg = [], []
        ipb = tr_pack['img_paths_boxes']
        for start in range(0, len(ipb), 48):
            chunk = ipb[start:start + 48]
            bgrs = [cv2.imdecode(np.fromfile(str(it[1]), dtype=np.uint8), 1) for it in chunk]
            bxs = [it[2] for it in chunk]
            p_c, z_c = yolo_ext.extract_batch(bgrs, bxs, None, blur_down=False, gamma=1.0)
            gm = 0.52 if ((start // 48) % 2 == 0) else 1.95
            p_d, z_d = yolo_ext.extract_batch(bgrs, bxs, None, blur_down=True, gamma=gm)
            tr_p14_clean.append(p_c); tr_zneck_clean.append(z_c)
            tr_p14_deg.append(p_d); tr_zneck_deg.append(z_d)

        # 2. Encode SciDB train crops with DINOv2 (seed)
        z_core, lg_core = dino_ext.encode_crops(tr_pack['crops_core'])
        z_std, lg_std = dino_ext.encode_crops(tr_pack['crops_std'])
        z_ctx, lg_ctx = dino_ext.encode_crops(tr_pack['crops_ctx'])
        z_cf, lg_cf = dino_ext.encode_crops(tr_pack['crops_cf_bad'])

        np.savez_compressed(
            CACHE_DIR / f'train_cggar_seed{seed}.npz',
            box_img_ids=tr_pack['box_img_ids'],
            labels=tr_pack['labels'],
            geom=tr_pack['geom'],
            p14_clean=np.concatenate(tr_p14_clean, 0),
            zneck_clean=np.concatenate(tr_zneck_clean, 0),
            p14_deg=np.concatenate(tr_p14_deg, 0),
            zneck_deg=np.concatenate(tr_zneck_deg, 0),
            z_core=z_core, lg_core=lg_core,
            z_std=z_std, lg_std=lg_std,
            z_ctx=z_ctx, lg_ctx=lg_ctx,
            z_cf_bad=z_cf, lg_cf_bad=lg_cf,
        )
        print(f"  [Seed {seed}] train 缓存已保存 ({time.monotonic() - ts:.1f}s)", flush=True)

        # 3. Extract deterministic features for val / testB / testC official candidate boxes (score >= 0.05)
        for sp in ['val', 'testB', 'testC']:
            gt_data = json.loads((GT_DIR / f'{sp}.json').read_text(encoding='utf-8'))
            orig_preds = json.loads((STRONG_DIR / f'种子{seed}' / f'{sp}预测.json').read_text(encoding='utf-8'))
            cand_idxs = np.array([i for i, r in enumerate(orig_preds) if r['score'] >= 0.05], dtype=np.int32)
            cand_rows = [orig_preds[i] for i in cand_idxs]
            id2img = {im['id']: im for im in gt_data['images']}

            g_pos = defaultdict(list)
            for pos, r in enumerate(cand_rows):
                g_pos[r['image_id']].append(pos)

            n_c = len(cand_rows)
            p14_arr = np.zeros((n_c, 14), dtype=np.float32)
            zneck_arr = np.zeros((n_c, 192), dtype=np.float32)
            geom_arr = np.zeros((n_c, 9), dtype=np.float32)
            c_core_arr = np.zeros((n_c, 3, 224, 224), dtype=np.uint8)
            c_std_arr = np.zeros((n_c, 3, 224, 224), dtype=np.uint8)
            c_ctx_arr = np.zeros((n_c, 3, 224, 224), dtype=np.uint8)

            ordered_iids = list(g_pos.keys())
            for start in range(0, len(ordered_iids), 16):
                sub_iids = ordered_iids[start:start + 16]
                bgrs, bxs_batch, tcls_batch, pos_batch = [], [], [], []
                for iid in sub_iids:
                    im_meta = id2img[iid]
                    bgr = cv2.imdecode(np.fromfile(str(IMG_ROOT / sp / im_meta['file_name']), dtype=np.uint8), 1)
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    h, w = rgb.shape[:2]
                    plist = g_pos[iid]
                    b_list = [cand_rows[p]['bbox'] for p in plist]
                    c_list = [cand_rows[p]['category_id'] for p in plist]
                    for p, bb in zip(plist, b_list):
                        geom_arr[p] = compute_box_geom(bb, w, h, b_list)
                        c_core_arr[p] = crop_square_scale(rgb, bb, 0.82)
                        c_std_arr[p] = crop_square_scale(rgb, bb, 1.25)
                        c_ctx_arr[p] = crop_square_scale(rgb, bb, 2.50)
                    bgrs.append(bgr)
                    bxs_batch.append(b_list)
                    tcls_batch.append(c_list)
                    pos_batch.extend(plist)

                p_sub, z_sub = yolo_ext.extract_batch(bgrs, bxs_batch, tcls_batch)
                for k, p_idx in enumerate(pos_batch):
                    p14_arr[p_idx] = p_sub[k]
                    zneck_arr[p_idx] = z_sub[k]

            zc, lgc = dino_ext.encode_crops(c_core_arr)
            zs, lgs = dino_ext.encode_crops(c_std_arr)
            zx, lgx = dino_ext.encode_crops(c_ctx_arr)

            np.savez_compressed(
                CACHE_DIR / f'{sp}_cggar_seed{seed}.npz',
                cand_idxs=cand_idxs,
                geom=geom_arr,
                p14=p14_arr,
                zneck=zneck_arr,
                z_core=zc, lg_core=lgc,
                z_std=zs, lg_std=lgs,
                z_ctx=zx, lg_ctx=lgx,
            )
            print(f"  [Seed {seed} | {sp:5s}] 候选框={n_c} 缓存完成", flush=True)

        yolo_ext.close()
        del yolo_ext, dino_ext
        torch.cuda.empty_cache()

    print(f"\n[全部完成] Step 1 总耗时: {time.monotonic() - t0:.1f}s", flush=True)


if __name__ == '__main__':
    main()
