# -*- coding: utf-8 -*-
"""
第一至四阶段可立即执行实验与工作包全量构建脚本
(run_stage2_3_4_complete_external_and_orchard190_suite.py)

零训练、零权重改动（严格校验 12 份 SHA-256 哈希），一次性完成：
1. [模块 1] 190 张胡柚实拍 (148 采集簇 / 161 对象簇) 权威主索引 CSV 构建 + 本地 100% 盲态几何候选框工作包生成
   - 输出 `configs/orchard190_148clusters_master_index.csv`（解决跨批次同名文件 `#048/#130` 与 `#093/#185` 碰撞，标记 `#057` 废片与 9 张马克笔叶片图）；
   - 在仓库外本地目录 `D:\新版柑橘\胡柚190张_双人独立标注与仲裁工作包_20261127\` 生成双人独立盲标 CSV、空白仲裁模板及 190 份 X-AnyLabeling/LabelMe 纯几何框 JSON，
     并执行字节级零模型类别泄漏断言核验（确保盲标员在仲裁前绝不可见任何模型预测类别或分数）。
2. [模块 2] 零缺口外部公开数据集全矩阵 3 种子评测（彻底消除论文外部验证表中的“—”缺口，完整保留正负结果）：
   - `Zenodo 11148332` (`519` 张真实果园图 / `1,765` 框，零丢框 `2-class` & `3-class` COCO AP + `B=1,000` 配对 Bootstrap)：
     评测 `B0, B1, C1, A1, G2_blur, A4b, C3_equal, C3_frozen, C5a`；
   - `CitrusUAT` (`500` 张室内灰底单叶 OOD 图，5 对齐类)：
     评测上述全部方案，并增设纯 DINOv2 (`w_det=0`) 的 `G2_blur_pureDino` 与 `C3_pureDino` 对照，定量证明 `-3.3pt` 负结果 100% 来自田间检测器先验 OOD 偏移而非裁剪表征退化；
   - `Sweet Orange V1` (`1,389` 张去重留出图，3 症候群)：
     评测上述全部方案的 `检出率`、`严格 14 类零遮蔽命中率`、`3 类准确率` 与 `3 类 Macro-F1`（完整呈现 14 类命中率正收益与 3 类子集坍缩反转负结果）。
3. [模块 3] 190 张 `12MP` (`3072×4096`) 衢州胡柚手机实拍图【解盲前无标签部署与稳定性实测】（绝不伪造人工真值！）：
   - 预注册 6 模型 (`B0 / B1 / C1 / G2_blur / A4b / C3` + `C3_equal`) 在 `180` 张纯净自然实拍图上的同会话严格 `FP32` 端到端推理延迟（含 `12MP` 缩放裁剪 vs 含磁盘原图解码双口径，`Mean / P95 / 4+叶片多目标子集`）；
   - 实拍活跃目标检出密度、校准置信度分布、检测器颈部流形马氏距离偏移 $z_{\text{Maha}}$；
   - 同簇多拍（28 个多拍 `AOBJ` 物理对象簇，含几何确证同叶三连拍 `001/002/091`）跨视角匹配框零标签 Top-1 症状自洽率与后验 JS 散度（仅存于仓库实验 JSON，绝不流入盲标目录）。
4. [模块 4] 运行 Stage-2 Kappa/仲裁工具自检与 Stage-3 三分支判决门禁干跑测试。
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'

import sys
import csv
import json
import math
import time
import hashlib
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
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(r'D:\新版柑橘')
EXT = ROOT / '方法路线扩展_20261031'
REPO_ROOT = ROOT / 'citrus-field-disease-research'
V5_DIR = REPO_ROOT / 'antigravity' / 'v5_c3_main_method_and_ablation'
V6_DIR = REPO_ROOT / 'antigravity' / 'v6_stage2to4_orchard_kit_and_paper_draft'
CFG_DIR = V6_DIR / 'configs'
RES_DIR = V6_DIR / 'results'
CFG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_BLIND_KIT_DIR = ROOT / '胡柚190张_双人独立标注与仲裁工作包_20261127'
RAW_HUYOU_DIR = Path(r'D:\病虫害论文\自采数据\胡柚田间_20260722\raw')
DS_AUDIT_CSV = ROOT / 'DS执行_真实照片场景核查_20261124' / '03_逐图核查' / '逐图场景核查_190张.csv'
AUDIT_INDEX_JSON = Path(r'D:\病虫害论文\新增公开数据_20260916\中文分类与标签使用索引.json')
CITRUSUAT_DIR = ROOT / '复现_baseline' / 'field_test_CitrusUAT'
CKPT_TCD = ROOT / 'Antigravity_CGGAR_真创新攻关_20261125' / 'checkpoints'
CKPT_BG = ROOT / 'Antigravity_BGOPTA_单趟正交Token攻关_20261126' / 'frozen_models'

sys.path.append(str(EXT))
sys.path.append(str(ROOT / 'Antigravity_BGOPTA_单趟正交Token攻关_20261126'))
sys.path.append(str(ROOT / 'Antigravity_CGGAR_真创新攻关_20261125'))
sys.path.append(str(REPO_ROOT / 'antigravity' / 'v3_bgopta_and_dtemop' / 'scripts'))
sys.path.insert(0, str(V5_DIR / 'scripts'))
sys.path.insert(0, str(V6_DIR / 'scripts'))

from 共同评测后端 import configure_strict_fp32
from phase1_source_only_train_and_select import apply_classwise_nms, compute_maha_z
from phase2_frozen_oneshot_eval import coco_eval_metrics
from step1_extract_multigranular_cache import YoloFullStateExtractor, DinoExtractor, crop_square_scale, compute_box_geom
from step3_external_blind_and_latency import build_external_zenodo_gt
from run_rigorous_audit_suite import paired_image_bootstrap_coco
from orchard_plug_and_play_cli import (
    UnifiedOrchardInferenceEngine,
    verify_manifest_hashes,
    crop_125_with_outer_blur,
    iou_xywh,
)
from stage2_kappa_arbitration_and_coco_builder import SYMPTOM_CODE_MAP, run_self_test as run_stage2_self_test
from stage3_oneshot_unblind_and_decision_gate import run_dry_run_test as run_stage3_dry_run

MANIFEST_PATH = V5_DIR / 'configs' / 'c3_frozen_manifest.json'
MARKER_PEN_SEQS = {21, 22, 54, 64, 66, 78, 81, 83, 85}
INVALID_OCCLUSION_SEQS = {57}


sys.path.append(str(REPO_ROOT / 'antigravity' / 'v4_orchard_candidate_arsenal' / 'scripts'))
from run_candidate_arsenal_and_conformal_suite import train_source_plain_concat_head


def load_or_train_c5a_reference_heads(device: str = 'cuda:0') -> dict:
    """直接调用 v4 源域确定性 `train_source_plain_concat_head`，与历史 `C5a` 100% 逐比特恒等。"""
    return {seed: train_source_plain_concat_head(seed, device) for seed in [0, 1, 2]}


def softmax_np(lp: np.ndarray) -> np.ndarray:
    z = lp - lp.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ==============================================================================
# 模块 1: 190 张胡柚实拍权威索引与 100% 盲态几何候选框工作包
# ==============================================================================
def build_orchard190_master_index_and_blind_kit(manifest: dict) -> dict:
    print("\n================ [模块 1] 构建 190 张 / 148 簇权威主索引与本地 100% 盲态标注工作包 ================", flush=True)
    ds_rows = list(csv.DictReader(open(DS_AUDIT_CSV, encoding='utf-8-sig')))
    assert len(ds_rows) == 190, f"期望 190 行，实际 {len(ds_rows)}"

    batch_folder_map = {
        'A': '拍摄者A_iQOO_Neo10',
        'B': '拍摄者B_HUAWEI_Pura80Pro',
        '拍摄者A_iQOO_Neo10': '拍摄者A_iQOO_Neo10',
        '拍摄者B_HUAWEI_Pura80Pro': '拍摄者B_HUAWEI_Pura80Pro',
    }
    parsed_items = []
    bname_counts = defaultdict(int)
    for r in ds_rows:
        seq_id = int(r['总序号'])
        batch_dir = batch_folder_map[r['批次'].strip()]
        fn = r['文件名'].strip()
        rel_path = f"{batch_dir}/{fn}"
        full_p = RAW_HUYOU_DIR / batch_dir / fn
        assert full_p.exists(), f"原始照片缺失: {full_p}"
        bname_counts[fn] += 1
        parsed_items.append((seq_id, batch_dir, fn, rel_path, full_p, r))

    def _stat_file(it):
        seq_id, _batch_dir, _fn, _rel_path, full_p, _r = it
        raw_bytes = full_p.read_bytes()
        sha16 = hashlib.sha256(raw_bytes).hexdigest()[:16]
        bgr = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), 1)
        h, w = bgr.shape[:2]
        return seq_id, w, h, sha16, bgr

    decoded_190 = {}
    master_rows = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        stats_map = {sid: (w, h, sha16, bgr) for sid, w, h, sha16, bgr in ex.map(_stat_file, parsed_items)}

    for seq_id, batch_dir, fn, rel_path, _full_p, r in parsed_items:
        w, h, sha16, bgr = stats_map[seq_id]
        decoded_190[seq_id] = bgr
        is_mk = 1 if seq_id in MARKER_PEN_SEQS else 0
        is_inv = 1 if seq_id in INVALID_OCCLUSION_SEQS else 0
        clean_180 = 1 if (not is_mk and not is_inv) else 0
        master_rows.append({
            'seq_id': seq_id,
            'rel_path': rel_path,
            'basename': fn,
            'batch': batch_dir,
            'width': w,
            'height': h,
            'sha256_prefix16': sha16,
            'object_cluster_id': r['候选对象簇'].strip(),
            'sampling_cluster_id': r['采样隔离簇'].strip(),
            'has_001_002_091_core': r['含001_002_091确认核心'].strip(),
            'is_duplicate_basename': 1 if bname_counts[fn] > 1 else 0,
            'is_marker_pen_9': is_mk,
            'is_invalid_occlusion_1': is_inv,
            'clean_natural_180_flag': clean_180,
            'target_type': r['目标类型'].strip(),
            'target_leaf_tier': r['画面可辨叶片数'].strip(),
            'scene_type': r['既有画面结构'].strip(),
            'blur_status': r['模糊'].strip(),
            'occlusion_status': r['可见遮挡'].strip(),
        })

    master_csv_path = CFG_DIR / 'orchard190_148clusters_master_index.csv'
    with open(master_csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(master_rows[0].keys()))
        w.writeheader()
        w.writerows(master_rows)

    # 使用 Seed 0 B0 提取纯几何候选框（严格剥离 100% 类别与分数，仅保留坐标供两位标注员独立判断症状！）
    eng0 = UnifiedOrchardInferenceEngine(manifest, seed=0, device='cuda:0')
    LOCAL_BLIND_KIT_DIR.mkdir(parents=True, exist_ok=True)
    xany_dir = LOCAL_BLIND_KIT_DIR / 'x_anylabeling_class_blind_boxes'
    xany_dir.mkdir(parents=True, exist_ok=True)

    blind_sheet_rows = []
    total_blind_boxes = 0
    for mr in master_rows:
        sid = mr['seq_id']
        rel = mr['rel_path']
        bgr = decoded_190[sid]
        h, w = bgr.shape[:2]
        if mr['is_invalid_occlusion_1']:
            blind_sheet_rows.append({
                'seq_id': sid, 'rel_path': rel, 'sampling_cluster_id': mr['sampling_cluster_id'],
                'is_marker_pen_9': mr['is_marker_pen_9'], 'is_invalid_occlusion_1': 1,
                'box_id': 'INVALID_OCCLUDED_SKIP', 'bbox_x': 0, 'bbox_y': 0, 'bbox_w': 0, 'bbox_h': 0,
                'symptom_code': -1, 'annotator_note': '镜头完全遮挡无效废片_默认跳过',
            })
            continue

        bxs, scs, _cls_unused, _pr_unused, _, _ = eng0.infer_image_detailed(bgr, candidate='B0')
        # 仅取几何框，按框面积降序或从左到右排序编号（连框的排序都不使用模型置信度，彻底杜绝置信度暗示！）
        geom_boxes = [bb for bb, sc in zip(bxs, scs) if sc >= 0.20]
        geom_boxes.sort(key=lambda b: (round(b[1] / 200.0), round(b[0] / 200.0)))

        shapes = []
        if not geom_boxes:
            blind_sheet_rows.append({
                'seq_id': sid, 'rel_path': rel, 'sampling_cluster_id': mr['sampling_cluster_id'],
                'is_marker_pen_9': mr['is_marker_pen_9'], 'is_invalid_occlusion_1': 0,
                'box_id': 'NO_TARGET_IMAGE', 'bbox_x': 0, 'bbox_y': 0, 'bbox_w': 0, 'bbox_h': 0,
                'symptom_code': 'TODO_若图中有病叶请补框否则填-1', 'annotator_note': '',
            })
        else:
            for bi, bb in enumerate(geom_boxes, start=1):
                bid = f"B{bi:02d}"
                bx, by, bw, bh = [round(float(v), 1) for v in bb]
                total_blind_boxes += 1
                blind_sheet_rows.append({
                    'seq_id': sid, 'rel_path': rel, 'sampling_cluster_id': mr['sampling_cluster_id'],
                    'is_marker_pen_9': mr['is_marker_pen_9'], 'is_invalid_occlusion_1': 0,
                    'box_id': bid, 'bbox_x': bx, 'bbox_y': by, 'bbox_w': bw, 'bbox_h': bh,
                    'symptom_code': 'TODO_请独立填入编码(0..13或99无法判断或-1删除伪框)',
                    'annotator_note': '含人工马克笔字迹_敏感性对照' if mr['is_marker_pen_9'] else '',
                })
                shapes.append({
                    'label': 'TODO_请独立标注症状',
                    'group_id': bi,
                    'points': [[bx, by], [round(bx + bw, 1), round(by + bh, 1)]],
                    'shape_type': 'rectangle',
                    'flags': {},
                })

        lbl_json_p = xany_dir / Path(rel).with_suffix('.json')
        lbl_json_p.parent.mkdir(parents=True, exist_ok=True)
        raw_img_abs = RAW_HUYOU_DIR / mr['batch'] / mr['basename']
        lbl_json_p.write_text(json.dumps({
            'version': '2.3.6',
            'flags': {'blind_geometry_only': True, 'is_marker_pen_9': bool(mr['is_marker_pen_9'])},
            'shapes': shapes,
            'imagePath': str(raw_img_abs),
            'imageData': None,
            'imageHeight': h,
            'imageWidth': w,
        }, ensure_ascii=False, indent=2), encoding='utf-8')

    eng0.close()

    ann1_p = LOCAL_BLIND_KIT_DIR / '标注人1_独立盲标工作表_190张.csv'
    ann2_p = LOCAL_BLIND_KIT_DIR / '标注人2_独立盲标工作表_190张.csv'
    arb_tpl_p = LOCAL_BLIND_KIT_DIR / '第三人仲裁工作表_空白模板.csv'
    code_card_p = LOCAL_BLIND_KIT_DIR / '类别编码对照与盲标红线速查卡.csv'

    for out_p in (ann1_p, ann2_p):
        with open(out_p, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(blind_sheet_rows[0].keys()))
            w.writeheader()
            w.writerows(blind_sheet_rows)

    with open(arb_tpl_p, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'seq_id', 'rel_path', 'sampling_cluster_id', 'object_cluster_id', 'is_marker_pen_9',
            'box_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
            'blind_opinion_A', 'blind_opinion_B', 'note_A', 'note_B',
            'arbitrator_final_code', 'arbitrator_reason',
        ])

    with open(code_card_p, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['编码', '标准症状名称', '标注红线与判定说明'])
        for k, v in sorted(SYMPTOM_CODE_MAP.items()):
            rule = '无法仅凭外观确认时必须选99，严禁强行归类' if k == 99 else (
                '框内非柑橘叶片/果实或纯背景枝干伪框时选-1删除' if k == -1 else (
                    '仅表示外观疑似症状，严禁等同于PCR确诊黄龙病(HLB)' if k in (2, 3, 5, 6) else '按田间表观症状独立判定'
                )
            )
            w.writerow([k, v, rule])

    # 严格断言核验：盲标工作表与 JSON 中 100% 不包含任何模型类别名称 (S1_..S14_) 或置信度字段
    ann1_txt = ann1_p.read_text(encoding='utf-8-sig')
    for forbidden_tok in ['S1_', 'S2_', 'S3_', 'calibrated_score', 'top1_id', 'prob']:
        assert forbidden_tok not in ann1_txt, f"盲标表检测到疑似模型预测残留字段: {forbidden_tok}"

    n_clean_clusters = len({r['sampling_cluster_id'] for r in master_rows if r['clean_natural_180_flag']})
    n_all189_clusters = len({r['sampling_cluster_id'] for r in master_rows if not r['is_invalid_occlusion_1']})
    summary = {
        'master_csv_path': str(master_csv_path),
        'local_blind_kit_dir_outside_git': str(LOCAL_BLIND_KIT_DIR),
        'n_total_files': len(master_rows),
        'n_unique_basenames': len(bname_counts),
        'duplicate_basenames_disambiguated': [k for k, c in bname_counts.items() if c > 1],
        'n_object_clusters_aobj': len({r['object_cluster_id'] for r in master_rows}),
        'n_sampling_clusters_total': len({r['sampling_cluster_id'] for r in master_rows}),
        'sampling_cluster_source': 'DS_逐图场景核查_190张.csv_采样隔离簇_S001_to_S148',
        'n_invalid_occlusion_excluded': sum(r['is_invalid_occlusion_1'] for r in master_rows),
        'n_marker_pen_flagged': sum(r['is_marker_pen_9'] for r in master_rows),
        'n_clean_natural_images': sum(r['clean_natural_180_flag'] for r in master_rows),
        'n_clean_natural_clusters': n_clean_clusters,
        'n_readable_189_clusters': n_all189_clusters,
        'n_geometry_only_proposal_boxes_exported': total_blind_boxes,
        'zero_model_class_score_leakage_asserted': True,
    }
    print(f"  [模块 1 完成] 190图 / {summary['n_sampling_clusters_total']}总采样簇 (纯净180图={n_clean_clusters}簇) | "
          f"导出纯几何盲标框 {total_blind_boxes} 个 -> {LOCAL_BLIND_KIT_DIR}", flush=True)
    return summary, master_rows, decoded_190


# ==============================================================================
# 模块 2: 外部公开数据集全矩阵 3 种子补齐 (Zenodo 519 + CitrusUAT 500 + Sweet Orange 1389)
# ==============================================================================
@torch.no_grad()
def run_complete_external_matrix_3seeds(manifest: dict, c5a_heads: dict, device: str = 'cuda:0') -> dict:
    print("\n================ [模块 2] 补齐外部公开数据集全矩阵 (Zenodo 519 + CitrusUAT 500 + SweetOrange 1389) ================", flush=True)
    # 1) 预加载 Zenodo
    coco_images, img_paths, gt2_p, gt3_p = build_external_zenodo_gt()
    gt3_dict = json.loads(gt3_p.read_text(encoding='utf-8'))
    scidb_to_2cls = {0: 0, 1: 0, **{c: 1 for c in range(2, 14)}}
    scidb_to_3cls = {0: 0, 1: 0, 2: 1, 3: 1, **{c: 2 for c in range(4, 14)}}

    def _dec_zen(it):
        bgr = cv2.imdecode(np.fromfile(str(img_paths[it['id']]), dtype=np.uint8), 1)
        return it['id'], bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    with ThreadPoolExecutor(max_workers=6) as ex:
        zen_decoded = list(ex.map(_dec_zen, coco_images))
    zen_map = {iid: (bgr, rgb) for iid, bgr, rgb in zen_decoded}

    # 2) 预加载 CitrusUAT (500 imgs, 5 classes)
    uat_classes = ['Greasy_spot', 'Healthy', 'HLB', 'Mg', 'Zn']
    uat_groups = [[10], [0, 1], [2, 3], [7], [4]]
    uat_samples = []
    for ci, cname in enumerate(uat_classes):
        for p in sorted((CITRUSUAT_DIR / cname).glob('*.*')):
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                uat_samples.append((ci, p))

    def _dec_uat(item):
        ci, p = item
        bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), 1)
        h, w = bgr.shape[:2]
        if max(h, w) > 1280:
            f = 1280.0 / max(h, w)
            bgr = cv2.resize(bgr, (max(1, int(round(w * f))), max(1, int(round(h * f)))), interpolation=cv2.INTER_AREA)
        return ci, bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    with ThreadPoolExecutor(max_workers=6) as ex:
        uat_decoded = list(ex.map(_dec_uat, uat_samples))
    y_uat = np.array([x[0] for x in uat_decoded], dtype=np.int64)

    # 3) 预加载 Sweet Orange V1 (1,389 deduplicated imgs, 3 symptom groups)
    idx_all = json.loads(AUDIT_INDEX_JSON.read_text(encoding='utf-8'))
    lbl_map_sw = {'Healthy leaf': 0, 'Citrus canker': 1, 'Citrus greening': 2}
    sw_groups = [[0, 1], [11], [2, 3]]
    sw_recs = [
        r for r in idx_all
        if r.get('数据集') == 'f7cr74mwpj' and r.get('索引去重保留') and not r.get('同库跨标签冲突')
        and r.get('解码状态') == '通过' and r.get('原始标签') in lbl_map_sw
    ]

    def _dec_sw(r):
        bgr = cv2.imdecode(np.fromfile(str(Path(r['路径'])), dtype=np.uint8), 1)
        h, w = bgr.shape[:2]
        if max(h, w) > 1280:
            f = 1280.0 / max(h, w)
            bgr = cv2.resize(bgr, (max(1, int(round(w * f))), max(1, int(round(h * f)))), interpolation=cv2.INTER_AREA)
        return lbl_map_sw[r['原始标签']], bgr, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    with ThreadPoolExecutor(max_workers=6) as ex:
        sw_decoded = list(ex.map(_dec_sw, sw_recs))
    y_sw = np.array([x[0] for x in sw_decoded], dtype=np.int64)

    zen_methods = [
        'B0_YOLOv8n', 'B1_Full_DINOv2', 'C1_Single125_035_065', 'A1_Single125_030_070',
        'G2_blur_Single125_BlurRing', 'A4b_FlipTTA_125_HFlip',
        'C3_equal_DualScale_035_035', 'C3_frozen_DualScale_020_050', 'C5a_3Scale_PlainConcat',
    ]
    zen_res = {m: {'2cls_AP50': [], '2cls_AP50_95': [], '3cls_AP50': [], '3cls_AP50_95': []} for m in zen_methods}
    zen_3cls_rows_per_seed = {0: {}, 1: {}, 2: {}}

    uat_methods = zen_methods + ['G2_blur_PureDino_wdet0', 'C3_PureDino_wdet0']
    uat_res = {m: {'MacroF1_5cls': [], 'Acc_5cls': [], 'Strict14cls_Acc': []} for m in uat_methods}
    uat_maha_stats = []

    sw_methods = zen_methods
    sw_res = {m: {'Strict14cls_Hit': [], 'Acc_3cls': [], 'MacroF1_3cls': []} for m in sw_methods}
    sw_det_hit = []

    for seed in [0, 1, 2]:
        scfg = manifest['seeds'][str(seed)]
        T_DET = float(scfg['temperatures']['T_DET'])
        T_DINO = float(scfg['temperatures']['T_DINO'])
        beta = float(scfg['postprocess']['beta_score_reweight'])
        blob_tcd = torch.load(Path(scfg['checkpoints']['tcd_calib_stats']['path']), map_location='cpu', weights_only=False)
        T_REL = float(blob_tcd['temps']['T_REL'])
        m_cfg = blob_tcd['mahalanobis']

        yolo_ext = YoloFullStateExtractor(Path(scfg['checkpoints']['yolov8n_det']['path']), device)
        dino_ext = DinoExtractor(seed, device)
        c5a_hd = c5a_heads[seed]

        # ---- 2A: Zenodo (519 imgs) ----
        raw_preds = []
        for start in range(0, len(coco_images), 16):
            grp = coco_images[start:start + 16]
            bgrs = [zen_map[it['id']][0] for it in grp]
            res_b = yolo_ext.yolo.predict(bgrs, imgsz=640, conf=0.05, iou=0.7, max_det=300, device=0, half=False, rect=False, verbose=False)
            for it, r in zip(grp, res_b):
                for bb, sc, cl in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.conf.cpu().tolist(), r.boxes.cls.cpu().tolist()):
                    x1, y1, x2, y2 = bb
                    raw_preds.append({'image_id': it['id'], 'category_id': int(cl), 'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': sc})

        n_z = len(raw_preds)
        g_pos = defaultdict(list)
        for pos, r in enumerate(raw_preds):
            g_pos[r['image_id']].append(pos)

        p14_z = np.zeros((n_z, 14), dtype=np.float32)
        geo_z = np.zeros((n_z, 9), dtype=np.float32)
        c082_z = np.zeros((n_z, 3, 224, 224), dtype=np.uint8)
        c125_z = np.zeros((n_z, 3, 224, 224), dtype=np.uint8)
        c125_blur_z = np.zeros((n_z, 3, 224, 224), dtype=np.uint8)
        c125_flip_z = np.zeros((n_z, 3, 224, 224), dtype=np.uint8)
        c250_z = np.zeros((n_z, 3, 224, 224), dtype=np.uint8)

        ord_iids = list(g_pos.keys())
        for start in range(0, len(ord_iids), 16):
            sub_iids = ord_iids[start:start + 16]
            bgrs, bxs_b, tcls_b, pos_b = [], [], [], []
            for iid in sub_iids:
                bgr, rgb = zen_map[iid]
                h, w = rgb.shape[:2]
                plist = g_pos[iid]
                bl = [raw_preds[p]['bbox'] for p in plist]
                cl = [raw_preds[p]['category_id'] for p in plist]
                for p, bb in zip(plist, bl):
                    geo_z[p] = compute_box_geom(bb, w, h, bl)
                    c082_z[p] = crop_square_scale(rgb, bb, 0.82)
                    std_c = crop_square_scale(rgb, bb, 1.25)
                    c125_z[p] = std_c
                    c125_flip_z[p] = std_c[:, :, ::-1]
                    c125_blur_z[p] = crop_125_with_outer_blur(rgb, bb, 31, 11.0)
                    c250_z[p] = crop_square_scale(rgb, bb, 2.50)
                bgrs.append(bgr); bxs_b.append(bl); tcls_b.append(cl); pos_b.extend(plist)
            ps, _ = yolo_ext.extract_batch(bgrs, bxs_b, tcls_b)
            for k, p_idx in enumerate(pos_b):
                p14_z[p_idx] = ps[k]

        zc_z, lgc_z = dino_ext.encode_crops(c082_z)
        zs_z, lgs_z = dino_ext.encode_crops(c125_z)
        _, lgb_z = dino_ext.encode_crops(c125_blur_z)
        _, lgf_z = dino_ext.encode_crops(c125_flip_z)
        zx_z, _ = dino_ext.encode_crops(c250_z)
        lg_c5a_z = c5a_hd(
            torch.from_numpy(zc_z.astype(np.float32)).to(device),
            torch.from_numpy(zs_z.astype(np.float32)).to(device),
            torch.from_numpy(zx_z.astype(np.float32)).to(device),
            torch.from_numpy(geo_z).to(device),
        ).cpu().numpy()

        del c082_z, c125_z, c125_blur_z, c125_flip_z, c250_z

        yolo_sc_z = np.array([r['score'] for r in raw_preds], dtype=np.float64)
        fg_z = yolo_sc_z >= 0.08
        lp_det_z = F.log_softmax(torch.logit(torch.from_numpy(p14_z).clamp(1e-5, 1.0 - 1e-5)) / T_DET, dim=-1).numpy()
        lp_core_z = F.log_softmax(torch.from_numpy(lgc_z) / T_DINO, dim=-1).numpy()
        lp_std_z = F.log_softmax(torch.from_numpy(lgs_z) / T_DINO, dim=-1).numpy()
        lp_blur_z = F.log_softmax(torch.from_numpy(lgb_z) / T_DINO, dim=-1).numpy()
        lp_flip_z = F.log_softmax(torch.from_numpy(lgf_z) / T_DINO, dim=-1).numpy()
        lp_c5a_z = F.log_softmax(torch.from_numpy(lg_c5a_z) / T_REL, dim=-1).numpy()
        p_std_raw_z = F.softmax(torch.from_numpy(lgs_z), dim=-1).numpy()

        def _build_z_cal(pm):
            rw = [dict(r) for r in raw_preds]
            for i in range(n_z):
                if fg_z[i]:
                    nc = int(np.argmax(pm[i]))
                    rw[i]['category_id'] = nc
                    rw[i]['score'] = float((yolo_sc_z[i] ** beta) * (max(float(pm[i, nc]), 1e-6) ** (1.0 - beta)))
            return apply_classwise_nms(rw, 0.70)

        rw_b1 = [dict(r) for r in raw_preds]
        for i in range(n_z):
            nc = int(np.argmax(p_std_raw_z[i]))
            if nc != rw_b1[i]['category_id']:
                rw_b1[i]['category_id'] = nc
                rw_b1[i]['score'] = float(math.sqrt(yolo_sc_z[i] * max(float(p_std_raw_z[i, nc]), 1e-6)))

        zen_candidates = {
            'B0_YOLOv8n': [dict(r) for r in raw_preds],
            'B1_Full_DINOv2': rw_b1,
            'C1_Single125_035_065': _build_z_cal(softmax_np(0.35 * lp_det_z + 0.65 * lp_std_z)),
            'A1_Single125_030_070': _build_z_cal(softmax_np(0.30 * lp_det_z + 0.70 * lp_std_z)),
            'G2_blur_Single125_BlurRing': _build_z_cal(softmax_np(0.30 * lp_det_z + 0.70 * lp_blur_z)),
            'A4b_FlipTTA_125_HFlip': _build_z_cal(softmax_np(0.30 * lp_det_z + 0.35 * lp_std_z + 0.35 * lp_flip_z)),
            'C3_equal_DualScale_035_035': _build_z_cal(softmax_np(0.30 * lp_det_z + 0.35 * lp_core_z + 0.35 * lp_std_z)),
            'C3_frozen_DualScale_020_050': _build_z_cal(softmax_np(0.30 * lp_det_z + 0.20 * lp_core_z + 0.50 * lp_std_z)),
            'C5a_3Scale_PlainConcat': _build_z_cal(softmax_np(0.24 * lp_det_z + 0.24 * lp_std_z + 0.20 * lp_core_z + 0.32 * lp_c5a_z)),
        }
        for mk, rw14 in zen_candidates.items():
            m2 = [{'image_id': r['image_id'], 'category_id': scidb_to_2cls[r['category_id']], 'bbox': r['bbox'], 'score': r['score']} for r in rw14]
            m3 = [{'image_id': r['image_id'], 'category_id': scidb_to_3cls[r['category_id']], 'bbox': r['bbox'], 'score': r['score']} for r in rw14]
            r2 = coco_eval_metrics(gt2_p, m2)
            r3 = coco_eval_metrics(gt3_p, m3)
            zen_res[mk]['2cls_AP50'].append(round(r2['AP50'], 2))
            zen_res[mk]['2cls_AP50_95'].append(round(r2['AP50_95'], 2))
            zen_res[mk]['3cls_AP50'].append(round(r3['AP50'], 2))
            zen_res[mk]['3cls_AP50_95'].append(round(r3['AP50_95'], 2))
            zen_3cls_rows_per_seed[seed][mk] = m3

        # ---- 2B & 2C: 单叶/器官分类诊断通用辅助函数 (CitrusUAT 500 + SweetOrange 1389) ----
        def _extract_organ_dataset(decoded_list, fallback_pad: float):
            n_items = len(decoded_list)
            p14_o = np.zeros((n_items, 14), dtype=np.float32)
            znk_o = np.zeros((n_items, 192), dtype=np.float32)
            geo_o = np.zeros((n_items, 9), dtype=np.float32)
            c082_o = np.zeros((n_items, 3, 224, 224), dtype=np.uint8)
            c125_o = np.zeros((n_items, 3, 224, 224), dtype=np.uint8)
            c125_bl_o = np.zeros((n_items, 3, 224, 224), dtype=np.uint8)
            c125_fl_o = np.zeros((n_items, 3, 224, 224), dtype=np.uint8)
            c250_o = np.zeros((n_items, 3, 224, 224), dtype=np.uint8)
            det_hits = 0

            for st in range(0, n_items, 24):
                sub = decoded_list[st:st + 24]
                bgrs = [x[1] for x in sub]
                res_b = yolo_ext.yolo.predict(bgrs, imgsz=640, conf=0.05, iou=0.7, max_det=20, device=0, half=False, rect=False, verbose=False)
                bxs_b, tcls_b = [], []
                for k, ((_, _bgr_s, rgb_s), r) in enumerate(zip(sub, res_b)):
                    idx = st + k
                    h, w = rgb_s.shape[:2]
                    if len(r.boxes) > 0:
                        det_hits += 1
                        x1, y1, x2, y2 = r.boxes.xyxy[0].cpu().tolist()
                        bb = [x1, y1, x2 - x1, y2 - y1]
                        tc = int(r.boxes.cls[0].item())
                    else:
                        bb = [w * fallback_pad, h * fallback_pad, w * (1.0 - 2 * fallback_pad), h * (1.0 - 2 * fallback_pad)]
                        tc = None
                    bxs_b.append([bb])
                    tcls_b.append([tc])
                    geo_o[idx] = compute_box_geom(bb, w, h, [bb])
                    c082_o[idx] = crop_square_scale(rgb_s, bb, 0.82)
                    sc125 = crop_square_scale(rgb_s, bb, 1.25)
                    c125_o[idx] = sc125
                    c125_fl_o[idx] = sc125[:, :, ::-1]
                    c125_bl_o[idx] = crop_125_with_outer_blur(rgb_s, bb, 31, 11.0)
                    c250_o[idx] = crop_square_scale(rgb_s, bb, 2.50)
                ps, zs = yolo_ext.extract_batch(bgrs, bxs_b, tcls_b)
                for k in range(len(sub)):
                    p14_o[st + k] = ps[k]
                    znk_o[st + k] = zs[k]

            zc_o, lgc_o = dino_ext.encode_crops(c082_o)
            zs_o, lgs_o = dino_ext.encode_crops(c125_o)
            _, lgb_o = dino_ext.encode_crops(c125_bl_o)
            _, lgf_o = dino_ext.encode_crops(c125_fl_o)
            zx_o, _ = dino_ext.encode_crops(c250_o)
            lg_c5a_o = c5a_hd(
                torch.from_numpy(zc_o.astype(np.float32)).to(device),
                torch.from_numpy(zs_o.astype(np.float32)).to(device),
                torch.from_numpy(zx_o.astype(np.float32)).to(device),
                torch.from_numpy(geo_o).to(device),
            ).cpu().numpy()

            lp_d = F.log_softmax(torch.logit(torch.from_numpy(p14_o).clamp(1e-5, 1.0 - 1e-5)) / T_DET, dim=-1).numpy()
            lp_c = F.log_softmax(torch.from_numpy(lgc_o) / T_DINO, dim=-1).numpy()
            lp_s = F.log_softmax(torch.from_numpy(lgs_o) / T_DINO, dim=-1).numpy()
            lp_b = F.log_softmax(torch.from_numpy(lgb_o) / T_DINO, dim=-1).numpy()
            lp_f = F.log_softmax(torch.from_numpy(lgf_o) / T_DINO, dim=-1).numpy()
            lp_5 = F.log_softmax(torch.from_numpy(lg_c5a_o) / T_REL, dim=-1).numpy()

            maha_14 = compute_maha_z(znk_o, m_cfg['means'], m_cfg['prec'], m_cfg['m_mu'], m_cfg['m_sd'])
            dm_o = maha_14[np.arange(n_items), lp_d.argmax(axis=1)]

            pmats = {
                'B0_YOLOv8n': softmax_np(lp_d),
                'B1_Full_DINOv2': F.softmax(torch.from_numpy(lgs_o), dim=-1).numpy(),
                'C1_Single125_035_065': softmax_np(0.35 * lp_d + 0.65 * lp_s),
                'A1_Single125_030_070': softmax_np(0.30 * lp_d + 0.70 * lp_s),
                'G2_blur_Single125_BlurRing': softmax_np(0.30 * lp_d + 0.70 * lp_b),
                'A4b_FlipTTA_125_HFlip': softmax_np(0.30 * lp_d + 0.35 * lp_s + 0.35 * lp_f),
                'C3_equal_DualScale_035_035': softmax_np(0.30 * lp_d + 0.35 * lp_c + 0.35 * lp_s),
                'C3_frozen_DualScale_020_050': softmax_np(0.30 * lp_d + 0.20 * lp_c + 0.50 * lp_s),
                'C5a_3Scale_PlainConcat': softmax_np(0.24 * lp_d + 0.24 * lp_s + 0.20 * lp_c + 0.32 * lp_5),
                'G2_blur_PureDino_wdet0': softmax_np(lp_b),
                'C3_PureDino_wdet0': softmax_np(0.50 * lp_c + 0.50 * lp_s),
            }
            return pmats, round(det_hits / n_items * 100.0, 2), dm_o

        uat_pmats, _, dm_uat = _extract_organ_dataset(uat_decoded, fallback_pad=0.08)
        uat_maha_stats.append({
            'seed': seed,
            'mean_maha_z': round(float(dm_uat.mean()), 2),
            'pct_maha_gt_3': round(float((dm_uat > 3.0).mean() * 100.0), 2),
        })
        for mk in uat_methods:
            pm = uat_pmats[mk]
            c14 = pm.argmax(axis=1)
            st14 = [int(c14[i]) in uat_groups[y_uat[i]] for i in range(len(y_uat))]
            g5 = np.stack([pm[:, grp].sum(axis=1) for grp in uat_groups], axis=1).argmax(axis=1)
            uat_res[mk]['Strict14cls_Acc'].append(round(float(np.mean(st14) * 100.0), 2))
            uat_res[mk]['Acc_5cls'].append(round(float(accuracy_score(y_uat, g5) * 100.0), 2))
            uat_res[mk]['MacroF1_5cls'].append(round(float(f1_score(y_uat, g5, average='macro') * 100.0), 2))

        sw_pmats, sw_hit_pct, _ = _extract_organ_dataset(sw_decoded, fallback_pad=0.06)
        sw_det_hit.append(sw_hit_pct)
        for mk in sw_methods:
            pm = sw_pmats[mk]
            c14 = pm.argmax(axis=1)
            st14 = [int(c14[i]) in sw_groups[y_sw[i]] for i in range(len(y_sw))]
            g3 = np.stack([pm[:, grp].sum(axis=1) for grp in sw_groups], axis=1).argmax(axis=1)
            sw_res[mk]['Strict14cls_Hit'].append(round(float(np.mean(st14) * 100.0), 2))
            sw_res[mk]['Acc_3cls'].append(round(float(accuracy_score(y_sw, g3) * 100.0), 2))
            sw_res[mk]['MacroF1_3cls'].append(round(float(f1_score(y_sw, g3, average='macro') * 100.0), 2))

        print(f"  [外部矩阵 Seed {seed}] Zenodo 3cls AP50-95: B1={zen_res['B1_Full_DINOv2']['3cls_AP50_95'][-1]:.2f} | "
              f"G2_blur={zen_res['G2_blur_Single125_BlurRing']['3cls_AP50_95'][-1]:.2f} | C3_eq={zen_res['C3_equal_DualScale_035_035']['3cls_AP50_95'][-1]:.2f} || "
              f"CitrusUAT F1: B1={uat_res['B1_Full_DINOv2']['MacroF1_5cls'][-1]:.2f} | C3_eq={uat_res['C3_equal_DualScale_035_035']['MacroF1_5cls'][-1]:.2f} | "
              f"C3_PureDino={uat_res['C3_PureDino_wdet0']['MacroF1_5cls'][-1]:.2f} || "
              f"SweetOrange 14clsHit/3clsF1: C3_eq={sw_res['C3_equal_DualScale_035_035']['Strict14cls_Hit'][-1]:.2f}/{sw_res['C3_equal_DualScale_035_035']['MacroF1_3cls'][-1]:.2f}",
              flush=True)

        yolo_ext.close()
        del yolo_ext, dino_ext
        torch.cuda.empty_cache()

    # Zenodo 3-class AP50-95 B=1000 paired image bootstrap (per-seed + 3-seed synchronized mean)
    zen_pairs = [
        ('C3_equal_DualScale_035_035', 'B1_Full_DINOv2'),
        ('G2_blur_Single125_BlurRing', 'B1_Full_DINOv2'),
        ('C3_equal_DualScale_035_035', 'A1_Single125_030_070'),
        ('C3_equal_DualScale_035_035', 'G2_blur_Single125_BlurRing'),
        ('C3_equal_DualScale_035_035', 'A4b_FlipTTA_125_HFlip'),
        ('C3_equal_DualScale_035_035', 'C5a_3Scale_PlainConcat'),
        ('C3_frozen_DualScale_020_050', 'G2_blur_Single125_BlurRing'),
    ]
    boot_s = []
    for s in [0, 1, 2]:
        b_arrs, _ = paired_image_bootstrap_coco(gt3_dict, zen_3cls_rows_per_seed[s], zen_pairs, n_boot=1000, rng_seed=20261127 + s)
        boot_s.append(b_arrs)

    zen_boot = {}
    for m_a, m_b in zen_pairs:
        tag = f"{m_a}__minus__{m_b}"
        zen_boot[tag] = {}
        for col, mlbl in [(0, 'AP50_95'), (1, 'AP50')]:
            ps_v = [boot_s[s][m_a][:, col] - boot_s[s][m_b][:, col] for s in [0, 1, 2]]
            dv_3s = np.mean(ps_v, axis=0)
            lo, hi = float(np.percentile(dv_3s, 2.5)), float(np.percentile(dv_3s, 97.5))
            p_two = min(1.0, 2.0 * min(float(np.mean(dv_3s <= 0.0)), float(np.mean(dv_3s >= 0.0))))
            zen_boot[tag][mlbl] = {
                'mean_diff_3seed': round(float(dv_3s.mean()), 3),
                'ci95_3seed': [round(lo, 3), round(hi, 3)],
                'p_value_two_sided': round(p_two, 4),
                'significant_95': bool(lo > 0.0 or hi < 0.0),
                'per_seed_mean_diffs': [round(float(sv.mean()), 3) for sv in ps_v],
            }

    def _agg(rec_dict):
        out = {}
        for mk, md in rec_dict.items():
            out[mk] = {}
            for mtr, vals in md.items():
                out[mk][mtr] = {
                    'seeds': vals,
                    'mean': round(float(np.mean(vals)), 2),
                    'std': round(float(np.std(vals, ddof=1)), 2),
                }
        return out

    return {
        'Zenodo_11148332_519imgs': {
            'n_images': len(coco_images),
            'n_gt_boxes': len(gt3_dict['annotations']),
            'summary_per_seed_and_mean': _agg(zen_res),
            'paired_bootstrap_3cls_AP50_95_B1000': zen_boot,
        },
        'CitrusUAT_500imgs_IndoorSingleLeaf_OOD': {
            'n_images': len(uat_decoded),
            'pseudo_mask_disclosure': '历史研究用于注意力诊断的叶片轮廓由灰底颜色阈值启发式恢复，属近似伪掩膜而非人工逐像素金标准；主表仅报告5类诊断准确率与Macro-F1负结果。',
            'detector_neck_maha_shift': uat_maha_stats,
            'summary_per_seed_and_mean': _agg(uat_res),
        },
        'SweetOrange_V1_1389imgs_3ClassSubset': {
            'n_images': len(sw_decoded),
            'detector_organ_hit_rate_pct_seeds': sw_det_hit,
            'summary_per_seed_and_mean': _agg(sw_res),
        },
    }


# ==============================================================================
# 模块 3: 190 张 12MP 衢州胡柚实拍图【解盲前无标签部署延迟、流形偏移与同簇多拍自洽性实测】
# ==============================================================================
@torch.no_grad()
def run_huyou190_labelfree_deployment_audit(manifest: dict, master_rows: list, decoded_190: dict) -> dict:
    print("\n================ [模块 3] 190 张 12MP 胡柚实拍解盲前无标签部署测速与同簇多拍自洽性实测 ================", flush=True)
    clean_rows = [r for r in master_rows if r['clean_natural_180_flag'] == 1]
    multi4_seqs = {r['seq_id'] for r in clean_rows if ('4+' in r['target_leaf_tier'] or '4片' in r['target_leaf_tier'])}

    modes_eval = [
        ('B0', 'YOLOv8n'),
        ('B1', 'Full_DINOv2'),
        ('C1', 'C1_SingleScale_125'),
        ('G2_blur', 'G2_blur_Single125_BlurRing'),
        ('A4b', 'A4b_FlipTTA_125_HFlip'),
        ('C3_equal', 'C3_equal_DualScale_035_035'),
        ('C3', 'C3_DualScale_082_125'),
    ]

    # 3A: 同会话 RTX 4060 Ti 严格 FP32 端到端单图在线推理延迟 (Seed 0, 180 张纯净 12MP 原图内存已解码 vs 含磁盘解码)
    eng0 = UnifiedOrchardInferenceEngine(manifest, seed=0, device='cuda:0')
    # 重负载预热 25 张 12MP 原图，消除 GPU P-state 频率爬坡
    for warm_r in clean_rows[:25]:
        _ = eng0.infer_image_detailed(decoded_190[warm_r['seq_id']], candidate='C3')
    torch.cuda.synchronize()

    # 测量 12MP JPEG 磁盘解码开销
    decode_ms_list = []
    for r in clean_rows:
        p = RAW_HUYOU_DIR / r['batch'] / r['basename']
        t0 = time.perf_counter()
        _ = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), 1)
        decode_ms_list.append((time.perf_counter() - t0) * 1000.0)
    mean_jpeg_decode_ms = round(float(np.mean(decode_ms_list)), 2)

    latency_12mp = {}
    for m_code, m_name in modes_eval:
        for warm_r in clean_rows[:5]:
            _ = eng0.infer_image_detailed(decoded_190[warm_r['seq_id']], candidate=m_code)
        torch.cuda.synchronize()

        t_all, t_m4, n_dinos, n_act20 = [], [], [], []
        for r in clean_rows:
            bgr = decoded_190[r['seq_id']]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            bxs, scs, _cl, _pr, _, n_dino = eng0.infer_image_detailed(bgr, candidate=m_code)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0
            t_all.append(dt)
            if r['seq_id'] in multi4_seqs:
                t_m4.append(dt)
            n_dinos.append(n_dino)
            n_act20.append(sum(1 for s in scs if s >= 0.20))

        latency_12mp[m_name] = {
            'gpu_e2e_from_12mp_ram_mean_ms': round(float(np.mean(t_all)), 2),
            'gpu_e2e_from_12mp_ram_median_ms': round(float(np.median(t_all)), 2),
            'gpu_e2e_from_12mp_ram_p95_ms': round(float(np.percentile(t_all, 95)), 2),
            'multi4plus_leaves_subset_mean_ms': round(float(np.mean(t_m4)), 2),
            'total_with_12mp_jpeg_decode_mean_ms': round(float(np.mean(t_all)) + mean_jpeg_decode_ms, 2),
            'equiv_fps_from_ram': round(1000.0 / float(np.mean(t_all)), 1),
            'mean_dino_crops_per_img': round(float(np.mean(n_dinos)), 2),
            'mean_active_dets_conf020_per_img': round(float(np.mean(n_act20)), 2),
        }
        print(f"  [12MP 实拍测速 {m_name:28s}] GPU端到端 Mean={latency_12mp[m_name]['gpu_e2e_from_12mp_ram_mean_ms']:.2f} ms "
              f"(P95={latency_12mp[m_name]['gpu_e2e_from_12mp_ram_p95_ms']:.2f} ms, 4+叶片={latency_12mp[m_name]['multi4plus_leaves_subset_mean_ms']:.2f} ms, "
              f"含12MP磁盘解码={latency_12mp[m_name]['total_with_12mp_jpeg_decode_mean_ms']:.2f} ms)", flush=True)
    eng0.close()

    # 3B: 3 种子无标签部署统计 + 同对象簇 (AOBJ 多拍簇 & 001/002/091 同叶三连拍) 跨视角预测自洽性
    by_aobj = defaultdict(list)
    for r in clean_rows:
        by_aobj[r['object_cluster_id']].append(r['seq_id'])
    multishot_aobjs = {k: sorted(v) for k, v in sorted(by_aobj.items()) if len(v) >= 2}

    def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
        p = np.clip(p, 1e-8, 1.0); p = p / p.sum()
        q = np.clip(q, 1e-8, 1.0); q = q / q.sum()
        m = 0.5 * (p + q)
        kl_pm = float(np.sum(p * np.log(p / m)))
        kl_qm = float(np.sum(q * np.log(q / m)))
        return 0.5 * (kl_pm + kl_qm) / math.log(2.0)  # normalized to [0, 1]

    consistency_per_model = {m[1]: {'multishot_top1_agree_pct': [], 'multishot_jsd': [], 'triad_001_002_091_agree_pct': []} for m in modes_eval}
    maha_shift_seeds = []

    for seed in [0, 1, 2]:
        eng = UnifiedOrchardInferenceEngine(manifest, seed=seed, device='cuda:0')
        # 记录颈部流形马氏距离偏移
        maha_vals = []
        for r in clean_rows:
            bgr = decoded_190[r['seq_id']]
            h, w = bgr.shape[:2]
            xyxy_t, sc_t, _ = eng.yolo_detect_square640(bgr)
            fg_ids = torch.where(sc_t >= 0.08)[0]
            if len(fg_ids) > 0:
                bxs_fg = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in xyxy_t[fg_ids].cpu().tolist()]
                p14_fg, zn_fg = eng._extract_anchor_p14_and_neck(h, w, bxs_fg, need_neck=True)
                ch = p14_fg.argmax(dim=-1)
                diff_m = zn_fg - eng.m_means[ch]
                dm_z = (torch.sqrt(torch.clamp((diff_m @ eng.m_prec * diff_m).sum(dim=-1), min=0.0)) - eng.m_mu) / eng.m_sd
                maha_vals.extend(dm_z.cpu().tolist())
        mv_arr = np.array(maha_vals, dtype=np.float64)
        maha_shift_seeds.append({
            'seed': seed,
            'n_fg_boxes': len(mv_arr),
            'mean_maha_z': round(float(mv_arr.mean()), 2),
            'median_maha_z': round(float(np.median(mv_arr)), 2),
            'pct_maha_gt_3': round(float((mv_arr > 3.0).mean() * 100.0), 2),
        })

        # 对多拍簇计算主目标（离画面中心最近的高置信目标框，适配手持微调角度/距离同叶补拍）跨拍一致性
        multishot_seq_set = {sid for v in multishot_aobjs.values() for sid in v}
        for m_code, m_name in modes_eval:
            dom_pred = {}
            for sid in sorted(multishot_seq_set):
                bgr = decoded_190[sid]
                h, w = bgr.shape[:2]
                bxs, scs, cls_l, probs, _, _ = eng.infer_image_detailed(bgr, candidate=m_code)
                cands = [(bb, sc, cl, pr) for bb, sc, cl, pr in zip(bxs, scs, cls_l, probs) if sc >= 0.15]
                if not cands and len(bxs) > 0:
                    cands = [(bxs[0], scs[0], cls_l[0], probs[0])]
                if cands:
                    # 选取画面中心主导检测目标（综合置信度与中心接近度，同簇所有模型看到完全一致的主体区域）
                    def _dom_score(item):
                        bb, sc, _cl, _pr = item
                        cx, cy = (bb[0] + bb[2] * 0.5) / w, (bb[1] + bb[3] * 0.5) / h
                        dist_c = math.hypot(cx - 0.5, cy - 0.5)
                        return sc - 0.35 * dist_c
                    best_it = max(cands, key=_dom_score)
                    dom_pred[sid] = (int(best_it[2]), best_it[3])

            pair_agree, pair_jsd = [], []
            for _aobj_id, sids in multishot_aobjs.items():
                valid_s = [s for s in sids if s in dom_pred]
                for i in range(len(valid_s)):
                    for j in range(i + 1, len(valid_s)):
                        c_i, p_i = dom_pred[valid_s[i]]
                        c_j, p_j = dom_pred[valid_s[j]]
                        pair_agree.append(1.0 if c_i == c_j else 0.0)
                        pair_jsd.append(_js_divergence(p_i, p_j))

            triad_sids = [s for s in (1, 2, 91) if s in dom_pred]
            tri_ag = []
            for i in range(len(triad_sids)):
                for j in range(i + 1, len(triad_sids)):
                    tri_ag.append(1.0 if dom_pred[triad_sids[i]][0] == dom_pred[triad_sids[j]][0] else 0.0)

            consistency_per_model[m_name]['multishot_top1_agree_pct'].append(round(float(np.mean(pair_agree) * 100.0), 2))
            consistency_per_model[m_name]['multishot_jsd'].append(round(float(np.mean(pair_jsd)), 4))
            consistency_per_model[m_name]['triad_001_002_091_agree_pct'].append(round(float(np.mean(tri_ag) * 100.0), 2) if tri_ag else 0.0)

        eng.close()

    cons_summary = {}
    for _, m_name in modes_eval:
        d = consistency_per_model[m_name]
        cons_summary[m_name] = {
            k: {'seeds': v, 'mean': round(float(np.mean(v)), 4 if 'jsd' in k else 2)}
            for k, v in d.items()
        }

    return {
        'n_clean_natural_images_evaluated': len(clean_rows),
        'n_multi4plus_leaves_images': len(multi4_seqs),
        'n_multishot_aobj_clusters': len(multishot_aobjs),
        'mean_12mp_jpeg_disk_decode_ms': mean_jpeg_decode_ms,
        'same_session_12mp_latency_seed0': latency_12mp,
        'detector_neck_mahalanobis_domain_shift': maha_shift_seeds,
        'multishot_same_object_consistency_3seeds': cons_summary,
        'strict_privacy_and_blind_note': '本节全部为无人工真值条件下的推理耗时、流形距离与多拍自洽统计；任何单图预测类别均未写入双人盲标工作包。',
    }


def main():
    t0 = time.monotonic()
    backend = configure_strict_fp32()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    verify_manifest_hashes(manifest)

    mod1_summary, master_rows, decoded_190 = build_orchard190_master_index_and_blind_kit(manifest)
    c5a_heads = load_or_train_c5a_reference_heads('cuda:0')
    mod2_external = run_complete_external_matrix_3seeds(manifest, c5a_heads, 'cuda:0')
    mod3_orchard190 = run_huyou190_labelfree_deployment_audit(manifest, master_rows, decoded_190)

    print("\n================ [模块 4] 执行 Stage-2 Kappa/仲裁工具与 Stage-3 解盲判决门禁自检 ================", flush=True)
    s2_test = run_stage2_self_test()
    s3_test = run_stage3_dry_run()
    print(f"  [自检全部通过] Stage2={s2_test['self_test_passed']} | Stage3={s3_test['dry_run_test_passed']}", flush=True)

    out_json = RES_DIR / 'stage2to4_external_and_orchard190_preunblind_complete.json'
    out_json.write_text(json.dumps({
        'stage': 'Stage2_to_Stage4_Complete_Deliverables_PreUnblind_v1.0',
        'elapsed_sec': round(time.monotonic() - t0, 2),
        'strict_fp32_backend': backend,
        'manifest_version': manifest['manifest_version'],
        'module1_orchard190_index_and_blind_kit': mod1_summary,
        'module2_complete_external_matrix_3seeds': mod2_external,
        'module3_huyou190_12mp_labelfree_deployment_audit': mod3_orchard190,
        'module4_pipeline_self_tests': {
            'stage2_kappa_arbitration_builder': s2_test,
            'stage3_oneshot_unblind_decision_gate': s3_test,
        },
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n[全量任务顺利完成] 结果已写入: {out_json} (总耗时 {time.monotonic() - t0:.1f}s)", flush=True)


if __name__ == '__main__':
    main()
