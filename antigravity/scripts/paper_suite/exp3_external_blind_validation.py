# -*- coding: utf-8 -*-
"""
实验 E3: 完全冻结权重与源 val 阈值下的多源独立公开数据一次性零样本封口盲测 (3 随机种子 0/1/2)
严格遵守学术诚信与数据审计边界（《01_新增数据审计与入口.md》）：
- 零目标域训练、零目标域选参、零阈值重标定（直接继承 E1 冻结权重与 E2 源域 val q=0.80 冻结门控阈值）；
- 全部样本严格采用《中文分类与标签使用索引.json》中 "索引去重保留=True 且 同库跨标签冲突=False" 的像素唯一图像（与 SciDB 零像素重叠）：
  1. 外部分类盲测集 A — Sweet Orange V1 (f7cr74mwpj, N=1,135): Healthy leaf (547) vs Citrus canker (588);
  2. 外部分类盲测集 B — Zenodo 11148332 Citrus (zenodo11148332_citrus, N=560): anthrax 炭疽病候选 (219) vs canker 溃疡病候选 (341);
  3. 外部跨树种分类盲测集 C — Pomelo Leaf V2 (vk7dnf8nrj, Citrus maxima 柚叶原图子集, N=292): Healthy (241) vs Yellow Mosaic 黄化症状 (51);
  4. 外部真实果园带框检测盲测集 D — Zenodo 11148332 Orange Detection (zenodo11148332_coco, N=519 张真实果园图, 1,766 个经归一化坐标核验的 YOLO 标注框):
     - 注：依审计规范，该公开包未附 PCR 病原确诊报告，且原始 COCO JSON 像素尺寸与缩放后图像不一致，故严格采用经核验对齐的 YOLO 归一化标注框结合图像真实宽高恢复坐标，并明确定义为“真实果园症状级外部盲测”：
       (i) 2 类果园症状检测 (Healthy Leaf vs Chlorosis/Yellowing Leaf, N=519 图, 1,766 框);
       (ii) 3 类细粒度病理拓扑框级分类与检测 (Healthy 健康叶 vs Asymmetric Blotchy Mottling 斑驳黄化疑似 vs Nutritional Deficiency 缺素对称黄化疑似)。
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
from sklearn.metrics import f1_score
from faster_coco_eval import COCO, COCOeval_faster

ROOT = Path(r'D:\新版柑橘')
EXT = ROOT / '方法路线扩展_20261031'
sys.path.append(str(EXT))
from 共同评测后端 import configure_strict_fp32

EXP_DIR = ROOT / '多路线创新筛选_20261124' / '小论文完整实验_20261124'
sys.path.append(str(EXP_DIR))
from exp1_symtex_ablation_and_kd import DeployedPrunedStudent
from exp2_end2end_mgs_cascade_and_timing import YoloNeckExtractor, load_train_gt_boxes, fit_mahalanobis, maha_dist

CKPT_DIR = EXP_DIR / 'checkpoints'
RES_DIR = EXP_DIR / 'results'
BLIND_DET_DIR = EXP_DIR / 'external_det_blind'
BLIND_DET_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_INDEX_JSON = Path(r'D:\病虫害论文\新增公开数据_20260916\中文分类与标签使用索引.json')
ZENODO_COCO_DIR = Path(r'D:\病虫害论文\新增公开数据_20260916\Zenodo_11148332\第二包解压\orange_detection_COCO')


def load_deduplicated_external_datasets():
    """直接读取《中文分类与标签使用索引.json》像素去重保留样本，一次加载 3 个独立分类盲测集与 1 个真实果园带框检测盲测集。"""
    records = json.loads(AUDIT_INDEX_JSON.read_text(encoding='utf-8'))
    tf = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
    ])

    sweet_imgs, sweet_y = [], []    # 0: Healthy (SciDB 1), 1: Canker (SciDB 11)
    zenodo_imgs, zenodo_y = [], []  # 0: Anthracnose (SciDB 9), 1: Canker (SciDB 11)
    pomelo_imgs, pomelo_y = [], []  # 0: Healthy, 1: Yellowing symptom cluster
    coco_det_records = []

    for r in records:
        if not r.get('索引去重保留', False) or r.get('同库跨标签冲突', False):
            continue
        ds = r.get('数据集', '')
        lbl = r.get('原始标签', '')
        p = Path(r['路径'])
        if ds == 'f7cr74mwpj' and lbl in ['Healthy leaf', 'Citrus canker']:
            bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                sweet_imgs.append(np.asarray(tf(Image.fromarray(rgb))).transpose(2, 0, 1))
                sweet_y.append(0 if lbl == 'Healthy leaf' else 1)
        elif ds == 'zenodo11148332_citrus' and lbl in ['anthrax', 'canker']:
            bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                zenodo_imgs.append(np.asarray(tf(Image.fromarray(rgb))).transpose(2, 0, 1))
                zenodo_y.append(0 if lbl == 'anthrax' else 1)
        elif ds == 'vk7dnf8nrj' and lbl in ['Pomelo_Leaf_Healthy', 'pomelo_Leaf_Yellow_Mosaic']:
            bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pomelo_imgs.append(np.asarray(tf(Image.fromarray(rgb))).transpose(2, 0, 1))
                pomelo_y.append(0 if lbl == 'Pomelo_Leaf_Healthy' else 1)
        elif ds == 'zenodo11148332_coco':
            coco_det_records.append(r)

    cls_sets = {
        'SweetOrange_V1': (np.stack(sweet_imgs, 0), np.array(sweet_y, dtype=np.int64)),
        'Zenodo11148332_Citrus': (np.stack(zenodo_imgs, 0), np.array(zenodo_y, dtype=np.int64)),
        'Pomelo_V2': (np.stack(pomelo_imgs, 0), np.array(pomelo_y, dtype=np.int64)),
    }
    return cls_sets, coco_det_records


@torch.no_grad()
def eval_external_cls_models(cls_sets: dict, device='cuda:0'):
    sw_x, sw_y = torch.from_numpy(cls_sets['SweetOrange_V1'][0]).to(device), cls_sets['SweetOrange_V1'][1]
    zn_x, zn_y = torch.from_numpy(cls_sets['Zenodo11148332_Citrus'][0]).to(device), cls_sets['Zenodo11148332_Citrus'][1]
    pm_x, pm_y = torch.from_numpy(cls_sets['Pomelo_V2'][0]).to(device), cls_sets['Pomelo_V2'][1]

    tags = ['B1_Vanilla_Pruned', 'Abl_BASD_Only', 'Abl_IIHFR_Only', 'SymTex_Pruned', 'Vanilla_KD', 'Ours_SymTex_KD_Pruned']
    results = {}

    for tag in tags:
        sw_f1s, sw_accs = [], []
        zn_f1s, zn_accs = [], []
        pm_f1s, pm_accs = [], []

        for s in [0, 1, 2]:
            st = torch.load(CKPT_DIR / f'{tag}_seed{s}.pt', map_location='cpu', weights_only=True)
            m = DeployedPrunedStudent(14).load_pruned(st['pruned_state']).to(device).eval()

            def get_probs(x_gpu):
                ps = []
                for i in range(0, len(x_gpu), 128):
                    lg, _ = m(x_gpu[i:i + 128].float() / 255.0)
                    ps.append(lg.softmax(dim=1).cpu().numpy())
                return np.concatenate(ps, 0)

            # 1) Sweet Orange: Healthy leaf (SciDB 1) vs Citrus canker (SciDB 11)
            p_sw = get_probs(sw_x)
            pred_sw = (p_sw[:, 11] > p_sw[:, 1]).astype(np.int64)
            sw_accs.append(float((pred_sw == sw_y).mean()) * 100.0)
            sw_f1s.append(float(f1_score(sw_y, pred_sw, average='macro')) * 100.0)

            # 2) Zenodo Citrus: Anthracnose (SciDB 9) vs Canker (SciDB 11)
            p_zn = get_probs(zn_x)
            pred_zn = (p_zn[:, 11] > p_zn[:, 9]).astype(np.int64)
            zn_accs.append(float((pred_zn == zn_y).mean()) * 100.0)
            zn_f1s.append(float(f1_score(zn_y, pred_zn, average='macro')) * 100.0)

            # 3) Pomelo (Citrus maxima): Healthy leaf (SciDB {0,1}) vs Yellowing symptom cluster (SciDB {2,4,5,6,7,8})
            p_pm = get_probs(pm_x)
            prob_healthy = p_pm[:, [0, 1]].sum(axis=1)
            prob_yellow = p_pm[:, [2, 4, 5, 6, 7, 8]].sum(axis=1)
            pred_pm = (prob_yellow > prob_healthy).astype(np.int64)
            pm_accs.append(float((pred_pm == pm_y).mean()) * 100.0)
            pm_f1s.append(float(f1_score(pm_y, pred_pm, average='macro')) * 100.0)
            del m

        results[tag] = {
            'SweetOrange_N': int(len(sw_y)),
            'SweetOrange_Acc': f"{np.mean(sw_accs):.2f}±{np.std(sw_accs, ddof=1):.2f}",
            'SweetOrange_MacroF1': f"{np.mean(sw_f1s):.2f}±{np.std(sw_f1s, ddof=1):.2f}",
            'SweetOrange_F1_mean': round(float(np.mean(sw_f1s)), 2),
            'ZenodoCitrus_N': int(len(zn_y)),
            'ZenodoCitrus_Acc': f"{np.mean(zn_accs):.2f}±{np.std(zn_accs, ddof=1):.2f}",
            'ZenodoCitrus_MacroF1': f"{np.mean(zn_f1s):.2f}±{np.std(zn_f1s, ddof=1):.2f}",
            'Zenodo_F1_mean': round(float(np.mean(zn_f1s)), 2),
            'Pomelo_N': int(len(pm_y)),
            'Pomelo_Acc': f"{np.mean(pm_accs):.2f}±{np.std(pm_accs, ddof=1):.2f}",
            'Pomelo_MacroF1': f"{np.mean(pm_f1s):.2f}±{np.std(pm_f1s, ddof=1):.2f}",
            'Pomelo_F1_mean': round(float(np.mean(pm_f1s)), 2),
            'Avg_3External_F1': round(float(np.mean([np.mean(sw_f1s), np.mean(zn_f1s), np.mean(pm_f1s)])), 2),
            'seeds_sweet_f1': [round(v, 2) for v in sw_f1s],
            'seeds_zenodo_f1': [round(v, 2) for v in zn_f1s],
            'seeds_pomelo_f1': [round(v, 2) for v in pm_f1s],
        }
        print(f"  [外部盲测分类 | {tag:22s}] SweetOrange(N={len(sw_y)}) F1={results[tag]['SweetOrange_MacroF1']} | "
              f"Zenodo(N={len(zn_y)}) F1={results[tag]['ZenodoCitrus_MacroF1']} | "
              f"Pomelo(N={len(pm_y)}) F1={results[tag]['Pomelo_MacroF1']} | 三库均F1={results[tag]['Avg_3External_F1']:.2f}%", flush=True)
    return results


def evaluate_external_orchard_detection_blind(coco_det_records: list):
    """
    在 Zenodo 11148332 真实果园带框检测集 (N=519 像素去重图, 1,766 个经核验对齐的归一化 YOLO 标注框) 上执行冻结模型零样本盲测：
    1) 2 类真实果园症状检测 (0: Healthy Leaf vs 1: Chlorosis/Diseased Leaf) COCO AP50 & AP50-95;
    2) 3 类细粒度病理拓扑检测 (0: Healthy Leaf, 1: Blotchy Mottling HLB-Suspect [S3], 2: Symmetric Nutritional Deficiency [S5-S9]) COCO AP50 & 真实框级分类 Macro-F1。
    """
    coco_images = []
    anns_2cls = []
    anns_3cls = []
    gt_crop_imgs = []
    gt_crop_y3 = []
    img_paths_by_id = {}

    tf = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(224)])

    # YOLO raw class in classes.txt: 0=HLB (blotchy mottling suspect), 1=healthy_leaf, 2=nutrition deficiency
    raw_to_2cls = {1: 0, 0: 1, 2: 1}
    raw_to_3cls = {1: 0, 0: 1, 2: 2}

    # SciDB 14-class -> external 2-class & 3-class symptom taxonomy
    scidb_to_2cls = {0: 0, 1: 0, **{c: 1 for c in range(2, 14)}}
    # SciDB S3 (id=2) & S4 (id=3) -> 1 (HLB-suspect mottling); S5..S9 (ids 4..8) -> 2 (Nutritional deficiency chlorosis); S1/S2 (0,1) -> 0 (Healthy)
    scidb_to_3cls = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2}

    ann_id = 1
    img_id = 1
    for r in coco_det_records:
        ip = Path(r['路径'])
        lp = ZENODO_COCO_DIR / 'labels' / ip.parent.name / (ip.stem + '.txt')
        if not lp.exists():
            continue
        bgr = cv2.imdecode(np.fromfile(str(ip), dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        boxes_in_img = []
        for ln in lp.read_text(encoding='utf-8').splitlines():
            toks = ln.split()
            if len(toks) >= 5:
                rc = int(float(toks[0]))
                cx, cy, bw, bh = map(float, toks[1:5])
                x1 = max(0.0, (cx - bw / 2.0) * w)
                y1 = max(0.0, (cy - bh / 2.0) * h)
                pw = min(float(w) - x1, bw * w)
                ph = min(float(h) - y1, bh * h)
                if pw > 2.0 and ph > 2.0 and rc in raw_to_3cls:
                    boxes_in_img.append((rc, [x1, y1, pw, ph]))
                    l, t, rr, bb = max(0, math.floor(x1)), max(0, math.floor(y1)), min(w, math.ceil(x1 + pw)), min(h, math.ceil(y1 + ph))
                    if rr > l and bb > t:
                        c = Image.fromarray(rgb[t:bb, l:rr])
                        side = math.ceil(max(c.size) * 1.25)
                        sq = Image.new('RGB', (side, side), (128, 128, 128))
                        sq.paste(c, ((side - c.width) // 2, (side - c.height) // 2))
                        gt_crop_imgs.append(np.asarray(tf(sq)).transpose(2, 0, 1))
                        gt_crop_y3.append(raw_to_3cls[rc])
        if not boxes_in_img:
            continue
        coco_images.append({'id': img_id, 'file_name': ip.name, 'width': w, 'height': h})
        img_paths_by_id[img_id] = ip
        for rc, bbox in boxes_in_img:
            anns_2cls.append({'id': ann_id, 'image_id': img_id, 'category_id': raw_to_2cls[rc], 'bbox': bbox, 'area': bbox[2] * bbox[3], 'iscrowd': 0})
            anns_3cls.append({'id': ann_id, 'image_id': img_id, 'category_id': raw_to_3cls[rc], 'bbox': bbox, 'area': bbox[2] * bbox[3], 'iscrowd': 0})
            ann_id += 1
        img_id += 1

    gt2_path = BLIND_DET_DIR / 'zenodo_orchard_2cls_gt.json'
    gt3_path = BLIND_DET_DIR / 'zenodo_orchard_3cls_gt.json'
    gt2_path.write_text(json.dumps({'images': coco_images, 'annotations': anns_2cls,
                                    'categories': [{'id': 0, 'name': 'Healthy_Leaf'}, {'id': 1, 'name': 'Chlorosis_Diseased_Leaf'}]}), encoding='utf-8')
    gt3_path.write_text(json.dumps({'images': coco_images, 'annotations': anns_3cls,
                                    'categories': [{'id': 0, 'name': 'Healthy_Leaf'}, {'id': 1, 'name': 'Mottling_Suspect'}, {'id': 2, 'name': 'Nutritional_Deficiency'}]}), encoding='utf-8')
    print(f"  真实果园带框盲测集构建完成: 唯一果园图={len(coco_images)}, 真实标注框={len(anns_3cls)}", flush=True)

    # First evaluate GT-box 3-class fine-grained pathology topology classification (Healthy vs Mottling vs Nutritional Deficiency) across E1 variants!
    gt_x_gpu = torch.from_numpy(np.stack(gt_crop_imgs, 0)).to('cuda:0')
    gt_y3_np = np.array(gt_crop_y3, dtype=np.int64)
    gt_box_cls_res = {}
    for tag in ['B1_Vanilla_Pruned', 'Abl_BASD_Only', 'Abl_IIHFR_Only', 'SymTex_Pruned', 'Vanilla_KD', 'Ours_SymTex_KD_Pruned']:
        f1s, accs = [], []
        for s in [0, 1, 2]:
            st = torch.load(CKPT_DIR / f'{tag}_seed{s}.pt', map_location='cpu', weights_only=True)
            m = DeployedPrunedStudent(14).load_pruned(st['pruned_state']).to('cuda:0').eval()
            ps = []
            with torch.no_grad():
                for i in range(0, len(gt_x_gpu), 128):
                    ps.append(m(gt_x_gpu[i:i + 128].float() / 255.0)[0].softmax(dim=1).cpu().numpy())
            p14 = np.concatenate(ps, 0)
            # Group probabilities over compatible SciDB clusters: 0=Healthy{0,1}, 1=Mottling{2,3}, 2=Deficiency{4,5,6,7,8}
            p3 = np.stack([p14[:, [0, 1]].sum(1), p14[:, [2, 3]].sum(1), p14[:, [4, 5, 6, 7, 8]].sum(1)], axis=1)
            pred3 = p3.argmax(axis=1)
            accs.append(float((pred3 == gt_y3_np).mean()) * 100.0)
            f1s.append(float(f1_score(gt_y3_np, pred3, average='macro')) * 100.0)
            del m
        gt_box_cls_res[tag] = {
            'Acc_3cls': f"{np.mean(accs):.2f}±{np.std(accs, ddof=1):.2f}",
            'MacroF1_3cls': f"{np.mean(f1s):.2f}±{np.std(f1s, ddof=1):.2f}",
            'F1_mean': round(float(np.mean(f1s)), 2),
        }
        print(f"  [果园1766框细粒度3类拓扑盲测 | {tag:22s}] Acc={gt_box_cls_res[tag]['Acc_3cls']} | Macro-F1={gt_box_cls_res[tag]['MacroF1_3cls']}", flush=True)
    del gt_x_gpu
    torch.cuda.empty_cache()

    # Second: End-to-End Frozen Detection Blind Evaluation (3 seeds, frozen source-val q=0.80 thresholds)
    e2_data = json.loads((RES_DIR / 'e2_end2end_mgs_symtex_summary.json').read_text(encoding='utf-8'))
    train_imgs, train_boxes, train_labels = load_train_gt_boxes()
    mean_t = torch.tensor([0.485, 0.456, 0.406], device='cuda:0').reshape(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225], device='cuda:0').reshape(1, 3, 1, 1)

    seed_det_res = []
    for seed in [0, 1, 2]:
        tau_maha_80 = next(r['tau_maha_80'] for r in e2_data['逐种子明细'] if r['seed'] == seed)
        val_preds = json.loads((EXT / 'YOLOv8强基线重复_20261109' / f'种子{seed}' / 'val预测.json').read_text(encoding='utf-8'))
        val_sc = np.array([r['score'] for r in val_preds if r['score'] >= 0.25])
        tau_conf_80 = float(np.quantile(1.0 - val_sc, 0.80))

        w_yolo = Path(r'D:\citrus_runs') / f'yolov8n_unified_seed{seed}' / 'weights' / 'best.pt'
        ext = YoloNeckExtractor(w_yolo, 'cuda:0')
        tr_feats = np.concatenate(ext.extract_boxes(train_imgs, train_boxes, batch_size=64), axis=0)
        means, prec = fit_mahalanobis(tr_feats, train_labels, reg=1e-2)

        raw_preds = []
        for start in range(0, len(coco_images), 16):
            grp = coco_images[start:start + 16]
            bgrs = [cv2.imdecode(np.fromfile(str(img_paths_by_id[it['id']]), dtype=np.uint8), 1) for it in grp]
            res_batch = ext.yolo.predict(bgrs, imgsz=640, conf=0.05, iou=0.7, max_det=300, device=0, half=False, rect=False, verbose=False)
            for it, r in zip(grp, res_batch):
                for bb, sc, cl in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.conf.cpu().tolist(), r.boxes.cls.cpu().tolist()):
                    x1, y1, x2, y2 = bb
                    raw_preds.append({'image_id': it['id'], 'category_id': int(cl), 'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': sc})

        g_pos = defaultdict(list)
        for pos, r in enumerate(raw_preds):
            g_pos[r['image_id']].append(pos)
        img_recs, b_per_img, pos_order = [], [], []
        for iid, plist in g_pos.items():
            img_recs.append((iid, img_paths_by_id[iid]))
            b_per_img.append([[raw_preds[p]['bbox'][0], raw_preds[p]['bbox'][1],
                               raw_preds[p]['bbox'][0] + raw_preds[p]['bbox'][2],
                               raw_preds[p]['bbox'][1] + raw_preds[p]['bbox'][3]] for p in plist])
            pos_order.extend(plist)

        f_list = ext.extract_boxes(img_recs, b_per_img, batch_size=32)
        flat_f = np.concatenate(f_list, 0) if f_list else np.zeros((0, 192), dtype=np.float32)
        cand_f = np.zeros_like(flat_f)
        for k, op in enumerate(pos_order):
            cand_f[op] = flat_f[k]

        yolo_cls = np.array([r['category_id'] for r in raw_preds], dtype=np.int64)
        yolo_sc = np.array([r['score'] for r in raw_preds], dtype=np.float64)
        d_maha = maha_dist(cand_f, yolo_cls, means, prec)
        low_conf = 1.0 - yolo_sc

        crops_u8 = np.zeros((len(raw_preds), 3, 224, 224), dtype=np.uint8)
        valid_c = np.zeros(len(raw_preds), dtype=bool)
        for iid, plist in g_pos.items():
            bgr = cv2.imdecode(np.fromfile(str(img_paths_by_id[iid]), dtype=np.uint8), 1)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            for idx in plist:
                x, y, bw, bh = raw_preds[idx]['bbox']
                l, t, rr, bb = max(0, math.floor(x)), max(0, math.floor(y)), min(w, math.ceil(x + bw)), min(h, math.ceil(y + bh))
                if rr > l and bb > t:
                    c = Image.fromarray(rgb[t:bb, l:rr])
                    side = math.ceil(max(c.size) * 1.25)
                    sq = Image.new('RGB', (side, side), (128, 128, 128))
                    sq.paste(c, ((side - c.width) // 2, (side - c.height) // 2))
                    crops_u8[idx] = np.asarray(tf(sq)).transpose(2, 0, 1)
                    valid_c[idx] = True

        def pred_student_tag(tag_name):
            st = torch.load(CKPT_DIR / f'{tag_name}_seed{seed}.pt', map_location='cpu', weights_only=True)
            ms = DeployedPrunedStudent(14).load_pruned(st['pruned_state']).to('cuda:0').eval()
            ca, cfa = np.zeros(len(raw_preds), dtype=np.int64), np.zeros(len(raw_preds), dtype=np.float64)
            v_ids = np.where(valid_c)[0]
            with torch.no_grad():
                for i in range(0, len(v_ids), 128):
                    b = v_ids[i:i + 128]
                    pr = ms(torch.from_numpy(crops_u8[b]).to('cuda:0').float() / 255.0)[0].softmax(dim=1)
                    cf, cl = pr.max(dim=1)
                    ca[b] = cl.cpu().numpy(); cfa[b] = cf.cpu().numpy()
            del ms
            return ca, cfa

        b1_cls, b1_cf = pred_student_tag('B1_Vanilla_Pruned')
        sym_cls, sym_cf = pred_student_tag('SymTex_Pruned')
        vkd_cls, vkd_cf = pred_student_tag('Vanilla_KD')
        skd_cls, skd_cf = pred_student_tag('Ours_SymTex_KD_Pruned')

        meta = json.loads((ROOT / '方法推进_20261027' / '纯CLS对照' / '微调汇总.json').read_text(encoding='utf-8'))
        t_info = next(r for r in meta['重复结果'] if r['方法'] == 'LP-FT' and r['种子'] == seed)
        t_state = torch.load(t_info['权重文件'], map_location='cpu', weights_only=True)
        d_enc = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False, img_size=224, num_classes=0).to('cuda:0').eval()
        d_enc.load_state_dict(t_state['骨干'])
        d_head = nn.Linear(384, 14).to('cuda:0').eval(); d_head.load_state_dict(t_state['分类头'])
        dino_cls, dino_cf = np.zeros(len(raw_preds), dtype=np.int64), np.zeros(len(raw_preds), dtype=np.float64)
        v_ids = np.where(valid_c)[0]
        with torch.no_grad():
            for i in range(0, len(v_ids), 64):
                b = v_ids[i:i + 64]
                x = (torch.from_numpy(crops_u8[b]).to('cuda:0').float() / 255.0 - mean_t) / std_t
                pr = d_head(F.normalize(d_enc.forward_features(x)[:, 0], dim=1)).softmax(dim=1)
                cf, cl = pr.max(dim=1)
                dino_cls[b] = cl.cpu().numpy(); dino_cf[b] = cf.cpu().numpy()
        del d_enc, d_head
        ext.close(); del ext; torch.cuda.empty_cache()

        def eval_coco_map(mask, c_arr, cf_arr, gt_p, cmap):
            mapped = []
            for i, r in enumerate(raw_preds):
                c14 = int(c_arr[i]) if (mask[i] and valid_c[i]) else r['category_id']
                sc_val = float(math.sqrt(r['score'] * cf_arr[i])) if (mask[i] and valid_c[i] and c14 != r['category_id']) else r['score']
                if c14 in cmap:
                    mapped.append({'image_id': r['image_id'], 'category_id': cmap[c14], 'bbox': r['bbox'], 'score': sc_val})
            if not mapped:
                return {'AP50': 0.0, 'AP50_95': 0.0, 'trig_pct': 0.0}
            with contextlib.redirect_stdout(io.StringIO()):
                gt = COCO(str(gt_p))
                ev = COCOeval_faster(gt, gt.loadRes(mapped), 'bbox')
                ev.evaluate(); ev.accumulate(); ev.summarize()
            hi = yolo_sc >= 0.25
            tpct = float((mask & hi).sum() / max(1, hi.sum())) * 100.0
            return {'AP50_95': round(ev.stats[0] * 100.0, 2), 'AP50': round(ev.stats[1] * 100.0, 2), 'trig_pct': round(tpct, 2)}

        n = len(raw_preds)
        cfgs = [
            ('YOLOv8n_Native', np.zeros(n, bool), yolo_cls, yolo_sc),
            ('Full_Vanilla_MBV3', np.ones(n, bool), b1_cls, b1_cf),
            ('LowConf_SymTex_NoKD', low_conf > tau_conf_80, sym_cls, sym_cf),
            ('MGS_SymTex_NoKD', d_maha > tau_maha_80, sym_cls, sym_cf),
            ('LowConf_Vanilla_KD', low_conf > tau_conf_80, vkd_cls, vkd_cf),
            ('MGS_Vanilla_KD', d_maha > tau_maha_80, vkd_cls, vkd_cf),
            ('LowConf_SymTex_KD', low_conf > tau_conf_80, skd_cls, skd_cf),
            ('Ours_Light_MGS_SymTex_KD', d_maha > tau_maha_80, skd_cls, skd_cf),
            ('LowConf_DINOv2', low_conf > tau_conf_80, dino_cls, dino_cf),
            ('Ours_Pro_MGS_DINOv2', d_maha > tau_maha_80, dino_cls, dino_cf),
        ]
        sr = {'seed': seed, 'det_2cls': {}, 'det_3cls': {}}
        for name, msk, ca, cfa in cfgs:
            sr['det_2cls'][name] = eval_coco_map(msk, ca, cfa, gt2_path, scidb_to_2cls)
            sr['det_3cls'][name] = eval_coco_map(msk, ca, cfa, gt3_path, scidb_to_3cls)
        print(f"  [果园带框检测盲测 Seed {seed} | 3类细粒度 AP50] 原生={sr['det_3cls']['YOLOv8n_Native']['AP50']:.2f}% | "
              f"低置信Light={sr['det_3cls']['LowConf_SymTex_KD']['AP50']:.2f}% | "
              f"★Ours-Light={sr['det_3cls']['Ours_Light_MGS_SymTex_KD']['AP50']:.2f}% (触发{sr['det_3cls']['Ours_Light_MGS_SymTex_KD']['trig_pct']:.1f}%) | "
              f"★Ours-Pro={sr['det_3cls']['Ours_Pro_MGS_DINOv2']['AP50']:.2f}%", flush=True)
        seed_det_res.append(sr)

    def agg_mode(mode_key):
        out = {}
        for k in seed_det_res[0][mode_key].keys():
            ap50s = [r[mode_key][k]['AP50'] for r in seed_det_res]
            aps = [r[mode_key][k]['AP50_95'] for r in seed_det_res]
            trs = [r[mode_key][k]['trig_pct'] for r in seed_det_res]
            out[k] = {
                'AP50_str': f"{np.mean(ap50s):.2f}±{np.std(ap50s, ddof=1):.2f}",
                'AP50_95_str': f"{np.mean(aps):.2f}±{np.std(aps, ddof=1):.2f}",
                'AP50_mean': round(float(np.mean(ap50s)), 2),
                'AP50_95_mean': round(float(np.mean(aps)), 2),
                'trig_pct_mean': round(float(np.mean(trs)), 2),
            }
        return out

    return {
        'dataset': 'Zenodo_11148332_orange_detection_COCO (Deduplicated & Normalized-Box Verified)',
        'caveat': '公开果园数据未附PCR确诊报告，严格作为跨来源症状级外部盲测报告（健康叶 vs 斑驳黄化疑似 vs 缺素对称黄化疑似）',
        'image_count': len(coco_images),
        'box_count': len(anns_3cls),
        'gt_box_3cls_topology_classification': gt_box_cls_res,
        'end2end_det_2cls_summary': agg_mode('det_2cls'),
        'end2end_det_3cls_summary': agg_mode('det_3cls'),
        'per_seed': seed_det_res,
    }


def main():
    t0 = time.monotonic()
    configure_strict_fp32()

    print("[1/2] 加载像素去重后的多源外部公开数据并执行零样本分类盲测 (SweetOrange + ZenodoCitrus + Pomelo)...", flush=True)
    cls_sets, coco_det_records = load_deduplicated_external_datasets()
    cls_blind_res = eval_external_cls_models(cls_sets, 'cuda:0')

    print("\n[2/2] 执行 Zenodo 11148332 真实果园带框检测集 (519图/1766框) 一次性零样本封口盲测...", flush=True)
    det_blind_res = evaluate_external_orchard_detection_blind(coco_det_records)

    out_p = RES_DIR / 'e3_external_blind_validation_summary.json'
    out_p.write_text(json.dumps({
        '实验名称': 'E3_冻结模型四源独立公开集一次性零样本封口盲测_3种子',
        '耗时秒': round(time.monotonic() - t0, 2),
        '外部独立分类盲测': cls_blind_res,
        '外部真实果园检测盲测': det_blind_res,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n[完成] 实验 E3 已保存至: {out_p}", flush=True)


if __name__ == '__main__':
    main()
